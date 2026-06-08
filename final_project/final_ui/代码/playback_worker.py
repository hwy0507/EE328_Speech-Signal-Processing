import os
import sys
import tempfile
import gc

import numpy as np
import soundfile as sf


def _play_with_simpleaudio(segment, rate):
    import simpleaudio as sa

    # 确定通道数
    if segment.ndim == 1:
        channels = 1
        # 对于单声道，确保是连续数组
        segment = np.ascontiguousarray(segment, dtype=np.int16)
    else:
        channels = min(segment.shape[1], 2)  # 限制最多 2 个通道
        # 如果有超过 2 个通道，只取前两个
        if segment.shape[1] > 2:
            segment = segment[:, :2]
        # 确保是 C 连续的（行优先）
        segment = np.ascontiguousarray(segment, dtype=np.int16)

    # 验证数据
    if len(segment) == 0:
        raise ValueError("Audio segment is empty")
    
    # 简单检查数据范围
    max_val = np.max(np.abs(segment))
    if max_val > 32767:
        # 数据可能溢出，进行归一化
        segment = np.clip(segment, -32768, 32767).astype(np.int16)

    try:
        play_obj = sa.play_buffer(segment, channels, 2, int(rate))
        play_obj.wait_done()
    except Exception as e:
        # 如果 simpleaudio 失败，清理资源后重新抛出
        raise


def _play_with_winsound(segment, rate):
    import winsound
    import gc

    # 确保数据类型正确
    if segment.dtype != np.int16:
        segment = segment.astype(np.int16)
    
    # 确保内存连续
    segment = np.ascontiguousarray(segment)

    tmp_path = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        # 写入 WAV 文件
        try:
            sf.write(tmp_path, segment, int(rate), subtype="PCM_16")
        except Exception as e:
            print(f"ERR: failed to write temp wav: {e}")
            raise

        # 播放文件
        try:
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        except Exception as e:
            print(f"ERR: winsound playback failed: {e}")
            raise
            
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                # 等待一下确保文件不被占用
                import time
                time.sleep(0.1)
                os.remove(tmp_path)
            except OSError:
                pass
        
        # 清理内存
        gc.collect()


def main():
    if len(sys.argv) < 3:
        print("Usage: playback_worker.py <wav_path> <offset_ms>")
        return 2

    wav_path = sys.argv[1]
    try:
        offset_ms = int(sys.argv[2])
    except (ValueError, IndexError):
        print("ERR: invalid offset")
        return 2

    if not os.path.exists(wav_path):
        print(f"ERR: file not found: {wav_path}")
        return 3

    try:
        # 读取音频文件，确保使用 int16 格式
        data, rate = sf.read(wav_path, dtype="float64")  # 先读为浮点数避免溢出
        
        # 转换为 int16，自动处理范围
        if np.max(np.abs(data)) > 1.0:
            # 假设已经是 int16 范围
            data = np.clip(data, -32768, 32767).astype(np.int16)
        else:
            # 假设是浮点数范围 [-1, 1]
            data = (data * 32767).astype(np.int16)
        
        # 确保数据是连续的
        data = np.ascontiguousarray(data)
        
    except Exception as e:
        print(f"ERR: failed to read audio: {e}")
        return 3

    # 处理多通道 - 确保最多 2 个通道
    if data.ndim > 1 and data.shape[1] > 2:
        data = data[:, :2]
        data = np.ascontiguousarray(data)

    # 计算起始采样位置
    try:
        start_sample = int(max(offset_ms, 0) * rate / 1000.0)
        if start_sample >= len(data):
            print("ERR: offset out of range")
            return 4
    except (OverflowError, ValueError):
        print("ERR: invalid offset calculation")
        return 4

    # 提取片段，确保连续性
    segment = np.ascontiguousarray(data[start_sample:])
    
    if len(segment) == 0:
        print("ERR: empty segment")
        return 4

    # 优先尝试使用 winsound（Windows 原生），避免 simpleaudio 引发本地层崩溃
    try:
        print("INFO: attempting winsound playback...")
        _play_with_winsound(segment, rate)
        print("INFO: winsound playback finished")
        return 0
    except Exception as e_win:
        print(f"WARN: winsound playback failed: {e_win}")
        try:
            print("INFO: attempting simpleaudio playback...")
            _play_with_simpleaudio(segment, rate)
            print("INFO: simpleaudio playback finished")
            return 0
        except Exception as e_sa:
            print(f"ERR: playback failed (winsound: {e_win}, simpleaudio: {e_sa})")
            return 5


if __name__ == "__main__":
    raise SystemExit(main())
