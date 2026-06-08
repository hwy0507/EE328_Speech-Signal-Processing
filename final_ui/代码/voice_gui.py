import sys
import os
import time
import tempfile
import subprocess
import json
import shutil
from collections import deque
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pyworld as pw

try:
    import sounddevice as sd
    _HAVE_SOUNDDEVICE = True
except Exception:
    sd = None
    _HAVE_SOUNDDEVICE = False

try:
    import noisereduce as nr
    _HAVE_NOIREDUCE = True
except Exception:
    nr = None
    _HAVE_NOIREDUCE = False

try:
    import PyQt5
    _qt_plugins_dir = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_qt_plugins_dir):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _qt_plugins_dir)
        os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins_dir)
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QSlider,
    QProgressBar,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QMessageBox,
    QDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath

try:
    import simpleaudio as sa
    _HAVE_SIMPLEAUDIO = True
except Exception:
    sa = None
    _HAVE_SIMPLEAUDIO = False

try:
    import winsound
    _HAVE_WINSOUND = True
except Exception:
    winsound = None
    _HAVE_WINSOUND = False

try:
    from config import MIN_AUDIO_LENGTH, MAX_AUDIO_LENGTH, SUPPORTED_FORMATS
except Exception:
    MIN_AUDIO_LENGTH = 3.0
    MAX_AUDIO_LENGTH = 300.0
    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


def _supported_audio_text():
    return ", ".join(SUPPORTED_FORMATS)


def _is_supported_audio(path):
    return any(path.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


def _audio_duration_seconds(path):
    """返回音频时长（秒）。失败时抛出异常。"""
    # 先尝试 soundfile / librosa 的快速路径；失败后用解码后的帧长兜底
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        try:
            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            # 最后尝试使用本地 ffprobe/ffmpeg 解析 duration（优先项目内 ffmpeg）
            try:
                import subprocess, re
                # 先查找项目内 ffmpeg/ffprobe
                local_ffprobe = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffprobe.exe")
                local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
                cmd = None
                if os.path.exists(local_ffprobe):
                    cmd = [local_ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
                elif os.path.exists(local_ffmpeg):
                    # parse ffmpeg -i stderr
                    cmd = [local_ffmpeg, "-i", path]
                else:
                    # try system ffprobe/ffmpeg in PATH
                    from shutil import which
                    ffp = which("ffprobe") or which("ffmpeg")
                    if ffp:
                        if ffp.endswith("ffprobe") or ffp.lower().endswith("ffprobe.exe"):
                            cmd = [ffp, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
                        else:
                            cmd = [ffp, "-i", path]

                if cmd is None:
                    raise RuntimeError("未检测到可用的 ffprobe/ffmpeg，用于解析容器时长。请安装 ffmpeg 或把项目目录下的 ffmpeg 放在 ffmpeg/bin。")

                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                # first try to find a plain number (ffprobe output)
                m = re.search(r"^(\d+\.\d+)$", out.strip(), re.M)
                if m:
                    return float(m.group(1))
                # look for Duration: HH:MM:SS.ms
                m2 = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
                if m2:
                    hh = int(m2.group(1)); mm = int(m2.group(2)); ss = float(m2.group(3))
                    return float(hh * 3600 + mm * 60 + ss)
                raise RuntimeError(f"无法从 ffmpeg 输出解析时长。输出: {out[:300]}")
            except Exception as e:
                # 最后尝试完整解码（最慢）
                try:
                    y, sr = librosa.load(path, sr=None, mono=False)
                    if y is None:
                        raise RuntimeError("无法读取音频数据。")
                    return float(len(y)) / float(sr)
                except Exception:
                    raise RuntimeError(f"无法读取音频时长: {e}")


def _duration_status(path, role):
    duration = _audio_duration_seconds(path)
    warnings = []
    if duration < MIN_AUDIO_LENGTH:
        warnings.append(f"{role} 音频太短（{duration:.2f}s），建议至少 {MIN_AUDIO_LENGTH:.0f}s。")
    if duration > MAX_AUDIO_LENGTH:
        warnings.append(f"{role} 音频过长（{duration:.2f}s），建议不超过 {MAX_AUDIO_LENGTH:.0f}s。")
    return duration, warnings


def _convert_to_temp_wav(path, prefix):
    """将任意输入音频转成 16k 单声道临时 WAV，供 TTS 声线转换使用。"""
    # For non-WAV inputs, prefer ffmpeg to avoid audioread/mpg123 errors.
    ext = os.path.splitext(path)[1].lower()
    prefer_ffmpeg = ext != ".wav"

    def _ffmpeg_convert():
        # 优先使用项目内的 ffmpeg/bin/ffmpeg.exe，其次使用 PATH 中的 ffmpeg
        local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
        ffmpeg_cmd = None
        if os.path.exists(local_ffmpeg):
            ffmpeg_cmd = local_ffmpeg
        else:
            from shutil import which
            ffp = which("ffmpeg")
            if ffp:
                ffmpeg_cmd = ffp

        if ffmpeg_cmd is None:
            raise RuntimeError("无法读取音频且未检测到 ffmpeg，无法生成临时 WAV。请安装 ffmpeg 或在项目中放置 ffmpeg/bin/ffmpeg.exe。")

        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
        os.close(fd)
        # On Windows, some ffmpeg builds have issues with non-ANSI paths; try to use short (8.3) path
        try:
            if os.name == 'nt':
                import ctypes
                buf = ctypes.create_unicode_buffer(260)
                r = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
                if r:
                    ff_input = buf.value
                else:
                    ff_input = path
            else:
                ff_input = path
        except Exception:
            ff_input = path

        cmd = [ffmpeg_cmd, "-y", "-i", ff_input, "-ar", "16000", "-ac", "1", "-vn", temp_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(temp_path):
            # 清理并抛出
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg 转换失败 (rc={proc.returncode})，请检查 ffmpeg 可用性。stderr: {proc.stderr.decode(errors='ignore')}")
        return temp_path

    if prefer_ffmpeg:
        return _ffmpeg_convert()

    try:
        y, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
        os.close(fd)
        sf.write(temp_path, y, 16000)
        return temp_path
    except Exception:
        return _ffmpeg_convert()


def _load_audio_with_fallback(path, sr=None, dtype=np.float64, mono=True, temp_prefix="plot_temp_"):
    """读取音频；当 librosa 直接读取 mp3/m4a 失败时，先用 ffmpeg 转临时 WAV 再读取。"""
    temp_wav = None
    try:
        return librosa.load(path, sr=sr, dtype=dtype, mono=mono)
    except Exception as first_error:
        try:
            temp_wav = _convert_to_temp_wav(path, temp_prefix)
            return librosa.load(temp_wav, sr=sr, dtype=dtype, mono=mono)
        except Exception:
            raise first_error
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def apply_energy_vad(y, fs, frame_ms=20, threshold_ratio=0.05):
    """
    [纯手写] 基于短时能量的语音活动检测 (VAD) 与软降噪
    :param y: 输入音频信号 (1D numpy array, float64)
    :param fs: 采样率
    :param frame_ms: 帧长 (毫秒)
    :param threshold_ratio: 能量阈值比例 (越小越激进，越大越温和)
    :return: (y_clean, mask_expanded) - 处理后的音频和掩码
    """
    frame_length = int(fs * (frame_ms / 1000.0))
    hop_length = frame_length // 2
    
    # 1. 计算每一帧的短时能量 (STE)
    energies = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i : i + frame_length]
        energies.append(np.sum(frame ** 2))
    
    if len(energies) == 0:
        return y, np.zeros_like(y)
    
    energies = np.array(energies)
    
    # 2. 设定自适应阈值 (最大能量的 threshold_ratio 倍)
    threshold = np.max(energies) * threshold_ratio
    
    # 3. 生成二值掩码: 大于阈值为 1(人声), 小于为 0(噪音)
    mask = (energies > threshold).astype(float)
    
    # 4. 平滑掩码，防止声音断断续续 (Hangover 机制)
    smoothed_mask = np.copy(mask)
    for i in range(1, len(mask) - 1):
        if mask[i-1] == 1 and mask[i+1] == 1:
            smoothed_mask[i] = 1.0
    
    # 5. 将掩码还原回原始信号长度
    mask_expanded = np.repeat(smoothed_mask, hop_length)
    pad_length = len(y) - len(mask_expanded)
    if pad_length > 0:
        mask_expanded = np.pad(mask_expanded, (0, pad_length), 'edge')
    else:
        mask_expanded = mask_expanded[:len(y)]
    
    # 6. 施加软门限 (Soft Gate)
    # 人声部分保留 100%, 噪音部分保留 10% (不直接归零，听感更自然)
    final_gain = mask_expanded * 0.9 + 0.1
    y_clean = y * final_gain
    
    return y_clean, mask_expanded


def apply_phase2_noise_reduction(y, fs, mode="标准"):
    """第二阶段：基于 noisereduce 的谱减噪，提供三档力度。"""
    if not _HAVE_NOIREDUCE:
        return y

    mode = (mode or "标准").strip()
    settings = {
        "温和": {
            "stationary": False,
            "prop_decrease": 0.55,
            "time_constant_s": 1.5,
            "freq_mask_smooth_hz": 300,
            "time_mask_smooth_ms": 35,
            "thresh_n_mult_nonstationary": 2.2,
        },
        "标准": {
            "stationary": False,
            "prop_decrease": 0.75,
            "time_constant_s": 1.0,
            "freq_mask_smooth_hz": 500,
            "time_mask_smooth_ms": 50,
            "thresh_n_mult_nonstationary": 2.0,
        },
        "强力": {
            "stationary": False,
            "prop_decrease": 1.0,
            "time_constant_s": 0.8,
            "freq_mask_smooth_hz": 700,
            "time_mask_smooth_ms": 70,
            "thresh_n_mult_nonstationary": 1.7,
        },
    }.get(mode, {
        "stationary": False,
        "prop_decrease": 0.75,
        "time_constant_s": 1.0,
        "freq_mask_smooth_hz": 500,
        "time_mask_smooth_ms": 50,
        "thresh_n_mult_nonstationary": 2.0,
    })

    y_in = np.asarray(y, dtype=np.float64)
    if y_in.ndim > 1:
        y_in = np.mean(y_in, axis=1)

    with np.errstate(all="ignore"):
        y_out = nr.reduce_noise(y=y_in, sr=int(fs), **settings)

    if not np.isfinite(y_out).all():
        return y_in

    return np.asarray(y_out, dtype=np.float64)


def _analyze_and_convert(x, fs, strength=1.0):
    def _inner(x_in, fs_in, strength=1.0):
        _f0, t = pw.dio(x_in, fs_in)
        f0 = pw.stonemask(x_in, _f0, t, fs_in)
        sp = pw.cheaptrick(x_in, f0, t, fs_in)
        ap = pw.d4c(x_in, f0, t, fs_in)

        valid_f0 = f0[f0 > 0]
        if len(valid_f0) == 0:
            raise RuntimeError("未检测到有效的人声基频。")

        median_f0 = np.median(valid_f0)
        threshold = 165.0
        if median_f0 < threshold:
            detected = "男声"
            pitch_ratio = 220.0 / median_f0
            formant_ratio = 1.18
        else:
            detected = "女声"
            pitch_ratio = 120.0 / median_f0
            formant_ratio = 0.85

        # 生成修改后的 F0 与频谱
        modified_f0 = f0 * pitch_ratio
        modified_sp = np.zeros_like(sp)
        num_bins = sp.shape[1]
        old_freq_axis = np.arange(num_bins)
        for i in range(sp.shape[0]):
            new_freq_axis = old_freq_axis / formant_ratio
            modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])

        # 合成原始音与修改后音，按 strength 做线性混合，strength=1.0 完全修改
        y_orig = pw.synthesize(f0, sp, ap, fs_in)
        y_mod = pw.synthesize(modified_f0, modified_sp, ap, fs_in)

        strength = float(strength)
        strength = max(0.0, min(1.0, strength))
        y = y_mod * strength + y_orig * (1.0 - strength)
        return y, f0, modified_f0, median_f0, detected

    return _inner(x, fs, strength=float(strength))


def _extract_valid_f0_for_plot(audio_path):
    x, fs = _load_audio_with_fallback(audio_path, sr=None, dtype=np.float64, temp_prefix="f0_plot_")
    if x.size == 0:
        raise RuntimeError(f"音频为空: {os.path.basename(audio_path)}")
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, _ = pw.dio(x, int(fs), frame_period=5.0)
    valid_f0 = f0[f0 > 0]
    if len(valid_f0) == 0:
        raise RuntimeError(f"未检测到有效基频: {os.path.basename(audio_path)}")
    return valid_f0


def plot_f0_histogram(input_path, save_path, converted_path=None, bins=50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if converted_path and os.path.exists(converted_path):
        valid_f0_before = _extract_valid_f0_for_plot(input_path)
        valid_f0_after = _extract_valid_f0_for_plot(converted_path)
        title_after = "After Processing"
    else:
        x, fs = _load_audio_with_fallback(input_path, sr=None, dtype=np.float64, temp_prefix="f0_plot_")
        _, f0_before, f0_after, _, _ = _analyze_and_convert(x, fs)

        valid_f0_before = f0_before[f0_before > 0]
        valid_f0_after = f0_after[f0_after > 0]
        if len(valid_f0_before) == 0 or len(valid_f0_after) == 0:
            raise RuntimeError("未检测到有效基频，无法绘制直方图。")
        title_after = "After Conversion"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(valid_f0_before, bins=bins, color="#FF9800", edgecolor="#222", alpha=0.7)
    ax1.set_xlabel("F0 (Hz)")
    ax1.set_ylabel("Count")
    ax1.set_title("Before Conversion", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(valid_f0_after, bins=bins, color="#4CAF50", edgecolor="#222", alpha=0.7)
    ax2.set_xlabel("F0 (Hz)")
    ax2.set_ylabel("Count")
    ax2.set_title(title_after, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"F0 Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_mel_spectrogram_comparison(input_path, converted_path, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = _load_audio_with_fallback(input_path, sr=None, dtype=np.float32, temp_prefix="mel_input_")
    y, fs2 = _load_audio_with_fallback(converted_path, sr=fs, dtype=np.float32, temp_prefix="mel_converted_")
    if x.size == 0 or y.size == 0:
        raise RuntimeError("音频为空，无法绘制 Mel 对比图。")
    if fs2 != fs:
        raise RuntimeError("采样率不一致，无法绘制 Mel 对比图。")

    n_fft = 1024
    hop_length = 256
    n_mels = 128

    mel_before = librosa.feature.melspectrogram(
        y=x, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    mel_after = librosa.feature.melspectrogram(
        y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )

    mel_before_db = librosa.power_to_db(mel_before, ref=np.max)
    mel_after_db = librosa.power_to_db(mel_after, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
    img1 = librosa.display.specshow(
        mel_before_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[0],
    )
    axes[0].set_title("Mel Spectrogram - Before", fontweight="bold")

    librosa.display.specshow(
        mel_after_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        # 必须和左图同样使用 mel 频率坐标；否则在 sharey=True 时，
        # 右图会按 0~n_mels 的 bin 坐标绘制，被压缩到坐标轴底部。
        y_axis="mel",
        cmap="magma",
        ax=axes[1],
    )
    axes[1].set_title("Mel Spectrogram - After", fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    cbar = fig.colorbar(img1, ax=axes, format="%+2.0f dB", shrink=0.88, pad=0.02)
    cbar.set_label("dB")
    fig.suptitle(f"Mel Spectrogram Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.savefig(save_path, dpi=140)
    plt.close()


class VoiceConverterThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, enable_vad=True, threshold_ratio=0.05, frame_ms=20, enable_phase2=False, phase2_mode="标准", strength=1.0):
        super().__init__()
        self.input_path = input_path
        self.enable_vad = enable_vad
        self.threshold_ratio = threshold_ratio
        self.frame_ms = frame_ms
        self.enable_phase2 = enable_phase2
        self.phase2_mode = phase2_mode
        self.strength = float(strength)

    def run(self):
        try:
            self.log_signal.emit(f"正在加载音频: {os.path.basename(self.input_path)}")
            temp_wav = None
            try:
                x, fs = librosa.load(self.input_path, sr=None, dtype=np.float64)
            except Exception:
                self.log_signal.emit("⚠️ 直接解码失败，尝试转为临时 WAV...")
                temp_wav = _convert_to_temp_wav(self.input_path, "vc_temp_")
                x, fs = librosa.load(temp_wav, sr=None, dtype=np.float64)
            
            # ====== 核心拦截器：VAD 条件分支 ======
            mask = None
            if self.enable_vad:
                self.log_signal.emit(f"🔕 [VAD 启动] 正在进行动态能量分析与降噪 (阈值: {self.threshold_ratio*100:.0f}%, 帧长: {self.frame_ms}ms)...")
                x, mask = apply_energy_vad(x, fs, frame_ms=self.frame_ms, threshold_ratio=self.threshold_ratio)
                self.log_signal.emit("✅ 噪音与静音段抑制完成！")
            else:
                self.log_signal.emit("⚠️ [VAD 关闭] 原始音频直通 WORLD 声码器。")

            if self.enable_phase2:
                if _HAVE_NOIREDUCE:
                    self.log_signal.emit(f"🧼 [第二阶段] 启动谱减噪，模式: {self.phase2_mode}...")
                    x = apply_phase2_noise_reduction(x, fs, mode=self.phase2_mode)
                    self.log_signal.emit("✅ 第二阶段谱减噪完成！")
                else:
                    self.log_signal.emit("⚠️ 第二阶段降噪未启用：当前环境缺少 noisereduce。")
            # ===================================
            
            self.log_signal.emit("正在使用 WORLD 提取特征...")

            y, _, _, median_f0, detected = _analyze_and_convert(x, fs, strength=self.strength)
            self.log_signal.emit(f"基频中位数: {median_f0:.1f} Hz")
            self.log_signal.emit(f"识别结果: {detected}")
            
            # ====== 防爆音规范化：防止波形幅度溢出 ======
            peak = np.max(np.abs(y))
            if peak > 0.95:
                self.log_signal.emit("🔊 正在进行音频幅度规范化 (Normalization)...")
                y = y * (0.95 / peak)
            # =======================================

            output_dir = os.path.dirname(self.input_path)
            base_name = os.path.basename(self.input_path)
            stem, ext = os.path.splitext(base_name)
            if ext.lower() != ".wav":
                output_name = f"converted_{stem}.wav"
            else:
                output_name = f"converted_{base_name}"
            output_path = os.path.join(output_dir, output_name)
            sf.write(output_path, y, fs)

            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

            # 如果存在 VAD 掩码，保存到 sidecar 文件，便于播放时混音
            try:
                if mask is not None:
                    np.save(output_path + ".mask.npy", mask)
            except Exception:
                pass

            self.log_signal.emit("✅ 转换成功")
            self.finished_signal.emit(output_path)
        except Exception as e:
            self.log_signal.emit(f"❌ 发生异常: {e}")
            self.finished_signal.emit("")


class VoiceCloneThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, source_path, target_paths, engine="FreeVC", openvoice_tau=0.3, openvoice_vad=True):
        super().__init__()
        self.source_path = source_path
        self.engine = engine
        self.openvoice_tau = max(0.0, min(1.0, float(openvoice_tau)))
        self.openvoice_vad = bool(openvoice_vad)
        # target_paths 可为列表（多参考样本）或单个路径
        if isinstance(target_paths, (list, tuple)):
            self.target_paths = list(target_paths)
        else:
            self.target_paths = [target_paths]
        self._temp_files = []

    def _load_env_map(self):
        map_path = os.path.join(os.path.dirname(__file__), 'env_map.json')
        if not os.path.exists(map_path):
            return {}
        try:
            import json
            with open(map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_signal.emit(f"⚠️ 读取 env_map.json 失败: {e}")
            return {}

    def _run_similarity_eval(self, reference_path, candidate_path, output_path, env_map):
        try:
            openvoice_python = env_map.get('声线克隆（OpenVoice）')
            openvoice_ckpt = env_map.get('OpenVoice_Checkpoints')
            if not openvoice_python or not os.path.exists(openvoice_python):
                self.log_signal.emit("⚠️ 未配置可用的 OpenVoice Python，跳过 Speaker Similarity 评估")
                return

            if not openvoice_ckpt or not os.path.exists(openvoice_ckpt):
                self.log_signal.emit("⚠️ 未配置可用的 OpenVoice checkpoints，跳过 Speaker Similarity 评估")
                return

            runner = os.path.join(os.path.dirname(__file__), 'tools', 'speaker_similarity_runner.py')
            if not os.path.exists(runner):
                self.log_signal.emit("⚠️ 未找到 speaker_similarity_runner.py，跳过 Speaker Similarity 评估")
                return

            json_path = os.path.splitext(output_path)[0] + '_similarity.json'
            rep_target = self.target_paths[0] if len(self.target_paths) > 0 else reference_path
            cmd = [
                openvoice_python,
                runner,
                '--reference', reference_path,
                '--candidate', candidate_path,
                '--ckpt_dir', openvoice_ckpt,
                '--out_json', json_path,
                '--label_reference', os.path.basename(rep_target),
                '--label_candidate', os.path.basename(candidate_path),
            ]

            self.log_signal.emit("📊 正在评估 Speaker Similarity...")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()

            if stderr:
                for line in stderr.splitlines():
                    if line.strip():
                        self.log_signal.emit(line.strip())

            if proc.returncode != 0:
                self.log_signal.emit(f"⚠️ Speaker Similarity 评估失败，rc={proc.returncode}")
                return

            result = None
            if stdout:
                for line in stdout.splitlines()[::-1]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            import json
                            result = json.loads(line)
                            break
                        except Exception:
                            pass

            if result:
                score = result.get('similarity_score')
                cosine = result.get('cosine_similarity')
                if score is not None:
                    self.log_signal.emit(f"✅ Speaker Similarity: {score:.2f}/100")
                if cosine is not None:
                    self.log_signal.emit(f"ℹ️ Cosine Similarity: {cosine:.6f}")
                self.log_signal.emit(f"ℹ️ 评估结果已保存: {json_path}")
            else:
                self.log_signal.emit(f"ℹ️ Speaker Similarity 评估完成，结果文件: {json_path}")
        except Exception as e:
            self.log_signal.emit(f"⚠️ Speaker Similarity 评估异常: {e}")

    def _prepare_temp_wav(self, path, prefix):
        # path 可能是单路径，也可能是 list（多个目标）
        if isinstance(path, (list, tuple)):
            # 合并多个参考文件为单个临时 wav（顺序拼接）
            combined = []
            sr_target = 16000
            for p in path:
                try:
                    seg_temp = _convert_to_temp_wav(p, f"{prefix}seg_")
                    self._temp_files.append(seg_temp)
                    y, sr = sf.read(seg_temp, dtype="float32")
                    if y is None:
                        continue
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != sr_target:
                        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                    combined.append(y)
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 参考音频处理失败: {os.path.basename(p)} ({e})")
                    # 跳过有问题的文件
                    pass
            if len(combined) == 0:
                raise RuntimeError("无法从参考音频生成临时文件：无有效文件。")
            y_all = np.concatenate(combined, axis=0)
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix='.wav')
            os.close(fd)
            sf.write(temp_path, y_all, sr_target)
            self._temp_files.append(temp_path)
            return temp_path
        else:
            temp_path = _convert_to_temp_wav(path, prefix)
            self._temp_files.append(temp_path)
            return temp_path

    def run(self):
        source_temp = None
        target_temp = None
        try:
            self.log_signal.emit(f"正在加载源音频: {os.path.basename(self.source_path)}")
            # 如果有多个参考，显示第一个作为代表
            rep_target = self.target_paths[0] if len(self.target_paths) > 0 else ""
            self.log_signal.emit(f"正在加载目标音频: {os.path.basename(rep_target)}")

            source_temp = self._prepare_temp_wav(self.source_path, "voice_source_")
            target_temp = self._prepare_temp_wav(self.target_paths, "voice_target_")
            env_map = self._load_env_map()

            # 首先尝试使用外部映射到的 Python 环境来运行声线克隆（避免在 GUI env 安装 heavy 依赖）
            try:
                python_exec = None
                openvoice_ckpt = None
                if self.engine == "OpenVoice":
                    python_exec = env_map.get('声线克隆（OpenVoice）')
                    openvoice_ckpt = env_map.get('OpenVoice_Checkpoints')
                else:
                    python_exec = env_map.get('声线克隆（双音频）')

                if python_exec and os.path.exists(python_exec):
                    self.log_signal.emit(f"📡 使用外部 Python: {python_exec} 执行声线克隆")
                    if self.engine == "OpenVoice":
                        runner = os.path.join(os.path.dirname(__file__), 'tools', 'openvoice_runner.py')
                    else:
                        runner = os.path.join(os.path.dirname(__file__), 'tools', 'clone_runner.py')
                    # 组织输出路径
                    output_dir = os.path.dirname(self.source_path)
                    source_base = os.path.splitext(os.path.basename(self.source_path))[0]
                    rep_target = self.target_paths[0] if len(self.target_paths) > 0 else "target"
                    target_base = os.path.splitext(os.path.basename(rep_target))[0]
                    output_path = os.path.join(output_dir, f"vc_{source_base}_to_{target_base}.wav")

                    cmd = [python_exec, runner, '--source', source_temp, '--target', target_temp, '--out', output_path]
                    if self.engine == "OpenVoice" and openvoice_ckpt:
                        cmd.extend(["--ckpt_dir", openvoice_ckpt])
                    if self.engine == "OpenVoice":
                        cmd.extend(["--tau", f"{self.openvoice_tau:.2f}"])
                        if not self.openvoice_vad:
                            cmd.append("--no_vad")
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    # stream stdout/stderr back to GUI
                    for line in proc.stdout:
                        self.log_signal.emit(line.rstrip())
                    err = proc.stderr.read()
                    if err:
                        for l in err.splitlines():
                            self.log_signal.emit(l)

                    rc = proc.wait()
                    if rc == 0 and os.path.exists(output_path):
                        self.log_signal.emit("✅ 声线转换成功 (外部运行)")
                        self._run_similarity_eval(target_temp, output_path, output_path, env_map)
                        self.finished_signal.emit(output_path)
                        return
                    else:
                        self.log_signal.emit(f"⚠️ 外部克隆进程失败，rc={rc}")

            except Exception as e:
                self.log_signal.emit(f"⚠️ 外部克隆尝试出错: {e}")

            # 外部运行不可用或失败，尝试在当前环境内直接使用 TTS（如果可用）
            try:
                from TTS.api import TTS
                import torch
                model_name = "voice_conversion_models/multilingual/vctk/freevc24"
                self.log_signal.emit(f"在当前环境加载声线转换模型: {model_name}")
                t0 = time.time()
                tts = TTS(model_name, gpu=torch.cuda.is_available())
                t1 = time.time()
                self.log_signal.emit("正在执行声线转换...")

                output_dir = os.path.dirname(self.source_path)
                source_base = os.path.splitext(os.path.basename(self.source_path))[0]
                rep_target = self.target_paths[0] if len(self.target_paths) > 0 else "target"
                target_base = os.path.splitext(os.path.basename(rep_target))[0]
                output_path = os.path.join(output_dir, f"vc_{source_base}_to_{target_base}.wav")

                tts.voice_conversion_to_file(
                    source_wav=source_temp,
                    target_wav=target_temp,
                    file_path=output_path,
                )
                t2 = time.time()

                self.log_signal.emit(f"模型加载耗时: {t1 - t0:.2f}s")
                self.log_signal.emit(f"声线转换耗时: {t2 - t1:.2f}s")
                self.log_signal.emit(f"总耗时: {t2 - t0:.2f}s")
                self.log_signal.emit("✅ 声线转换成功 (当前环境)")
                self._run_similarity_eval(target_temp, output_path, output_path, env_map)
                self.finished_signal.emit(output_path)
                return
            except Exception as e:
                self.log_signal.emit(f"❌ 声线转换失败: {e}")
                self.finished_signal.emit("")
        finally:
            for temp_path in self._temp_files:
                try:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass


class AnonymizationThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)

    def __init__(self, source_path, denoise_preset="standard"):
        super().__init__()
        self.source_path = source_path
        self.denoise_preset = denoise_preset

    def run(self):
        try:
            import json

            project_dir = os.path.dirname(__file__)
            runner = os.path.join(project_dir, "tools", "anonymization_runner.py")
            external_project = os.path.join(project_dir, "_external", "EE328_Speech-Signal-Processing")
            if not os.path.exists(runner):
                raise FileNotFoundError(f"未找到匿名化 runner: {runner}")
            if not os.path.exists(external_project):
                raise FileNotFoundError(f"未找到匿名化模块目录: {external_project}")

            output_dir = os.path.dirname(self.source_path)
            source_base = os.path.splitext(os.path.basename(self.source_path))[0]
            out_json = os.path.join(output_dir, f"anon_{source_base}_result.json")
            # Keep the work directory short on Windows.  The teammate pipeline nests
            # candidate filenames deeply; putting work_root under the cloned repo can
            # exceed MAX_PATH and make scipy wavfile.write fail with FileNotFoundError.
            work_root = os.path.join(project_dir, "_anon_work", str(int(time.time())))

            cmd = [
                sys.executable,
                runner,
                "--source",
                self.source_path,
                "--project-root",
                external_project,
                "--work-root",
                work_root,
                "--out-dir",
                output_dir,
                "--out-json",
                out_json,
                "--denoise-preset",
                self.denoise_preset,
            ]
            self.log_signal.emit("📡 正在执行匿名化处理...")
            self.log_signal.emit(" ".join(cmd))

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.log_signal.emit(line)
            err = proc.stderr.read() if proc.stderr is not None else ""
            if err:
                for line in err.splitlines():
                    if line.strip():
                        self.log_signal.emit(line.strip())

            rc = proc.wait()
            if rc != 0:
                self.log_signal.emit(f"❌ 匿名化进程失败，rc={rc}")
                self.finished_signal.emit(None)
                return

            if not os.path.exists(out_json):
                self.log_signal.emit(f"❌ 未生成匿名化结果 JSON: {out_json}")
                self.finished_signal.emit(None)
                return

            with open(out_json, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.log_signal.emit("✅ 匿名化完成")
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"❌ 匿名化失败: {e}")
            self.finished_signal.emit(None)


class PlotF0Thread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, save_path, converted_path=None):
        super().__init__()
        self.input_path = input_path
        self.save_path = save_path
        self.converted_path = converted_path

    def run(self):
        try:
            self.log_signal.emit(f"绘制 F0 对比图: {os.path.basename(self.input_path)}")
            plot_f0_histogram(self.input_path, self.save_path, self.converted_path)
            self.log_signal.emit("F0 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            reason = str(e) or e.__class__.__name__
            self.log_signal.emit(f"F0 绘图失败: {reason}")
            self.finished_signal.emit("")


class PlotMelThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, converted_path, save_path):
        super().__init__()
        self.input_path = input_path
        self.converted_path = converted_path
        self.save_path = save_path

    def run(self):
        try:
            self.log_signal.emit("绘制 Mel 语谱图对比...")
            plot_mel_spectrogram_comparison(self.input_path, self.converted_path, self.save_path)
            self.log_signal.emit("Mel 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            reason = str(e) or e.__class__.__name__
            self.log_signal.emit(f"Mel 绘图失败: {reason}")
            self.finished_signal.emit("")


class SimilarityEvalThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)

    def __init__(self, source_path, target_paths):
        super().__init__()
        self.source_path = source_path
        if isinstance(target_paths, (list, tuple)):
            self.target_paths = list(target_paths)
        else:
            self.target_paths = [target_paths]
        self._temp_files = []

    def _load_env_map(self):
        map_path = os.path.join(os.path.dirname(__file__), 'env_map.json')
        if not os.path.exists(map_path):
            return {}
        try:
            import json
            with open(map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_signal.emit(f"⚠️ 读取 env_map.json 失败: {e}")
            return {}

    def _prepare_temp_wav(self, path, prefix):
        if isinstance(path, (list, tuple)):
            combined = []
            sr_target = 16000
            for p in path:
                try:
                    seg_temp = _convert_to_temp_wav(p, f"{prefix}seg_")
                    self._temp_files.append(seg_temp)
                    y, sr = sf.read(seg_temp, dtype="float32")
                    if y is None:
                        continue
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != sr_target:
                        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                    combined.append(y)
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 参考音频处理失败: {os.path.basename(p)} ({e})")
            if len(combined) == 0:
                raise RuntimeError("无法从参考音频生成临时文件：无有效文件。")
            y_all = np.concatenate(combined, axis=0)
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix='.wav')
            os.close(fd)
            sf.write(temp_path, y_all, sr_target)
            self._temp_files.append(temp_path)
            return temp_path
        temp_path = _convert_to_temp_wav(path, prefix)
        self._temp_files.append(temp_path)
        return temp_path

    def run(self):
        try:
            self.log_signal.emit(f"正在加载待评估音频: {os.path.basename(self.source_path)}")
            rep_target = self.target_paths[0] if len(self.target_paths) > 0 else ""
            self.log_signal.emit(f"正在加载参考音频: {os.path.basename(rep_target)}")

            candidate_temp = self._prepare_temp_wav(self.source_path, "similarity_candidate_")
            reference_temp = self._prepare_temp_wav(self.target_paths, "similarity_reference_")

            env_map = self._load_env_map()
            openvoice_python = env_map.get('声线克隆（OpenVoice）')
            openvoice_ckpt = env_map.get('OpenVoice_Checkpoints')
            runner = os.path.join(os.path.dirname(__file__), 'tools', 'speaker_similarity_runner.py')

            if not openvoice_python or not os.path.exists(openvoice_python):
                raise RuntimeError("未配置可用的 OpenVoice Python。")
            if not openvoice_ckpt or not os.path.exists(openvoice_ckpt):
                raise RuntimeError("未配置可用的 OpenVoice checkpoints。")
            if not os.path.exists(runner):
                raise RuntimeError("未找到 speaker_similarity_runner.py。")

            output_dir = os.path.dirname(self.source_path)
            source_base = os.path.splitext(os.path.basename(self.source_path))[0]
            target_base = os.path.splitext(os.path.basename(rep_target))[0] if rep_target else "target"
            json_path = os.path.join(output_dir, f"similarity_{source_base}_vs_{target_base}.json")

            cmd = [
                openvoice_python,
                runner,
                '--reference', reference_temp,
                '--candidate', candidate_temp,
                '--ckpt_dir', openvoice_ckpt,
                '--out_json', json_path,
                '--label_reference', os.path.basename(rep_target) if rep_target else 'reference',
                '--label_candidate', os.path.basename(self.source_path),
            ]

            self.log_signal.emit("📊 正在评估 Speaker Similarity...")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()

            if stderr:
                for line in stderr.splitlines():
                    if line.strip():
                        self.log_signal.emit(line.strip())

            if proc.returncode != 0:
                self.log_signal.emit(f"⚠️ Speaker Similarity 评估失败，rc={proc.returncode}")
                self.finished_signal.emit(None)
                return

            result = None
            if stdout:
                for line in stdout.splitlines()[::-1]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            import json
                            result = json.loads(line)
                            break
                        except Exception:
                            pass

            if result is not None:
                result['result_json'] = json_path
                score = result.get('similarity_score')
                cosine = result.get('cosine_similarity')
                if score is not None:
                    self.log_signal.emit(f"✅ Speaker Similarity: {score:.2f}/100")
                if cosine is not None:
                    self.log_signal.emit(f"ℹ️ Cosine Similarity: {cosine:.6f}")
                self.log_signal.emit(f"ℹ️ 评估结果已保存: {json_path}")
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"❌ Speaker Similarity 独立评估失败: {e}")
            self.finished_signal.emit(None)
        finally:
            for temp_path in self._temp_files:
                try:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass


class AudioRecordThread(QThread):
    log_signal = pyqtSignal(str)
    level_signal = pyqtSignal(float)
    elapsed_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(str)

    def __init__(self, output_path, samplerate=16000, channels=1, blocksize=1024):
        super().__init__()
        self.output_path = output_path
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        if not _HAVE_SOUNDDEVICE:
            self.log_signal.emit("❌ 当前环境缺少 sounddevice，无法录音。请先安装: pip install sounddevice")
            self.finished_signal.emit("")
            return

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            started_at = time.time()
            self.log_signal.emit(f"🎙️ 开始录音，保存为 WAV: {self.output_path}")
            with sf.SoundFile(
                self.output_path,
                mode="w",
                samplerate=self.samplerate,
                channels=self.channels,
                subtype="PCM_16",
            ) as wav_file:
                with sd.InputStream(
                    samplerate=self.samplerate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.blocksize,
                ) as stream:
                    while not self._stop_requested:
                        data, overflowed = stream.read(self.blocksize)
                        if overflowed:
                            self.log_signal.emit("⚠️ 录音输入缓冲区溢出，当前片段可能有轻微丢帧。")
                        wav_file.write(data)
                        level = float(np.max(np.abs(data))) if data.size else 0.0
                        self.level_signal.emit(max(0.0, min(1.0, level)))
                        self.elapsed_signal.emit(time.time() - started_at)

            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 44:
                self.log_signal.emit(f"✅ 录音已保存: {self.output_path}")
                self.finished_signal.emit(self.output_path)
            else:
                try:
                    if os.path.exists(self.output_path):
                        os.remove(self.output_path)
                except Exception:
                    pass
                self.log_signal.emit("⚠️ 录音为空，未保存有效文件。")
                self.finished_signal.emit("")
        except Exception as e:
            try:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
            except Exception:
                pass
            self.log_signal.emit(f"❌ 录音失败: {e}")
            self.finished_signal.emit("")


class RecordDialog(QDialog):
    recording_finished = pyqtSignal(str)

    def __init__(self, parent, role_text, output_path, samplerate=16000):
        super().__init__(parent)
        self.role_text = role_text
        self.output_path = output_path
        self.record_thread = None
        self._has_started = False
        self._finished_path = ""

        self.setWindowTitle(f"录制{role_text}")
        self.setModal(True)
        self.resize(460, 220)

        layout = QVBoxLayout(self)
        title = QLabel(f"🎙️ 录制{role_text}（WAV / {samplerate}Hz / 单声道）")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#333;")
        layout.addWidget(title)

        self.path_label = QLabel(f"保存位置: {output_path}")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size:12px; color:#666;")
        layout.addWidget(self.path_label)

        self.time_label = QLabel("录音时长: 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size:22px; font-weight:bold; color:#D32F2F; padding:8px;")
        layout.addWidget(self.time_label)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setFormat("输入音量 %p%")
        layout.addWidget(self.level_bar)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始录音")
        self.btn_start.setStyleSheet("font-size:14px; background-color:#4CAF50; color:white; padding:7px;")
        self.btn_start.clicked.connect(lambda: self.start_recording(samplerate=samplerate))
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止并选中")
        self.btn_stop.setStyleSheet("font-size:14px; background-color:#F44336; color:white; padding:7px;")
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    def start_recording(self, samplerate=16000):
        if not _HAVE_SOUNDDEVICE:
            QMessageBox.warning(self, "缺少录音依赖", "当前环境缺少 sounddevice，无法录音。\n请先执行: pip install sounddevice")
            return
        if self.record_thread and self.record_thread.isRunning():
            return
        self._has_started = True
        self._finished_path = ""
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.record_thread = AudioRecordThread(self.output_path, samplerate=samplerate, channels=1)
        self.record_thread.level_signal.connect(self._update_level)
        self.record_thread.elapsed_signal.connect(self._update_elapsed)
        self.record_thread.log_signal.connect(self._relay_log)
        self.record_thread.finished_signal.connect(self._recording_done)
        self.record_thread.start()

    def stop_recording(self):
        if self.record_thread and self.record_thread.isRunning():
            self.btn_stop.setEnabled(False)
            self.record_thread.stop()

    def _relay_log(self, msg):
        parent = self.parent()
        if parent and hasattr(parent, "log_message"):
            parent.log_message(msg)

    def _update_level(self, level):
        self.level_bar.setValue(int(max(0.0, min(1.0, float(level))) * 100.0))

    def _update_elapsed(self, seconds):
        seconds = int(seconds)
        self.time_label.setText(f"录音时长: {seconds // 60:02d}:{seconds % 60:02d}")

    def _recording_done(self, path):
        self._finished_path = path or ""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        if path:
            self.recording_finished.emit(path)
            self.accept()
        else:
            QMessageBox.warning(self, "录音失败", "没有生成有效 WAV 文件，请检查麦克风权限和输入设备。")

    def reject(self):
        if self.record_thread and self.record_thread.isRunning():
            self.record_thread.stop()
            self.record_thread.wait(1500)
        if not self._finished_path and os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
            except Exception:
                pass
        super().reject()


class DragDropLabel(QLabel):
    file_dropped = pyqtSignal(str)
    file_list_dropped = pyqtSignal(list)

    def __init__(self, prompt_text=None):
        super().__init__()
        hint = f"\n\n拖动音频文件到这里\n支持格式：{_supported_audio_text()}\n（会自动转换为 WAV）\n\n"
        self.setText(prompt_text or hint)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #555;
            }
            """
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            paths = [u.toLocalFile() for u in urls]
            valid_paths = [p for p in paths if _is_supported_audio(p)]
            if not valid_paths:
                event.ignore()
                self.setStyleSheet(
                    "border: 2px dashed #aaa; background-color: #fff5f5; font-size: 16px;"
                )
                return
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #4CAF50; background-color: #e8f5e9; font-size: 16px;"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")

    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")
        urls = event.mimeData().urls()
        if urls:
            # 支持多文件拖放
            paths = [u.toLocalFile() for u in urls]
            valid_paths = [p for p in paths if _is_supported_audio(p)]
            if len(valid_paths) == 0:
                self.setText(f"\n\n请拖入支持的音频格式\n支持格式：{_supported_audio_text()}\n\n")
                QMessageBox.warning(
                    self.window(),
                    "格式不支持",
                    f"请拖入以下格式的音频文件：\n{_supported_audio_text()}\n\n当前拖入的文件没有被识别为支持的音频格式。",
                )
                return
            if len(valid_paths) == 1:
                self.file_dropped.emit(valid_paths[0])
            else:
                # 多文件回调
                self.file_list_dropped.emit(valid_paths)


class WaveformStrip(QWidget):
    def __init__(self, parent=None, history_size=240):
        super().__init__(parent)
        self.history = deque([0.0] * history_size, maxlen=history_size)
        self.setMinimumHeight(56)
        self.setMaximumHeight(72)

    def push_level(self, level):
        level = max(0.0, min(1.0, float(level)))
        self.history.append(level)
        self.update()

    def reset(self):
        self.history = deque([0.0] * self.history.maxlen, maxlen=self.history.maxlen)
        self.update()

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        mid = h / 2.0
        pad = 4.0

        painter.fillRect(self.rect(), QColor("#101418"))

        grid_pen = QPen(QColor("#25303a"), 1)
        painter.setPen(grid_pen)
        painter.drawLine(0, int(mid), w, int(mid))

        line_pen = QPen(QColor("#00e676"), 1.8)
        painter.setPen(line_pen)

        values = list(self.history)
        if len(values) < 2:
            return

        step_x = w / float(len(values) - 1)
        amp = max(mid - pad, 1.0)

        upper = QPainterPath()
        lower = QPainterPath()
        for i, v in enumerate(values):
            x = i * step_x
            y_up = mid - v * amp
            y_dn = mid + v * amp
            if i == 0:
                upper.moveTo(x, y_up)
                lower.moveTo(x, y_dn)
            else:
                upper.lineTo(x, y_up)
                lower.lineTo(x, y_dn)

        painter.drawPath(upper)
        painter.drawPath(lower)


class VoiceChangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_filepath = None
        self.target_filepaths = []
        self.converted_filepath = None
        self.anonymized_variants = {}
        self.anonymized_variant_order = []
        self.last_anonymization_result = None
        self.last_similarity_result = None
        self._temp_play_path = None

        self.play_obj = None
        self.play_proc = None
        self.play_data = None
        self.play_rate = 0
        self.play_channels = 0
        self.play_total_ms = 0
        self.play_start_ms = 0
        self.play_start_epoch = 0.0
        self.current_play_path = None

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(80)
        self.play_timer.timeout.connect(self._on_playback_timer)

        self.mode_names = ["WORLD 声学特征转换", "声线克隆（双音频）", "说话人相似度评估", "语音隐私匿名化"]

        self.initUI()

    def initUI(self):
        self.setWindowTitle("智能语音隐私保护与声线转换系统 - 融合增强版")
        self.resize(760, 560)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        mode_row = QHBoxLayout()
        mode_label = QLabel("工作模式:")
        mode_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 80px;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(self.mode_names)
        self.combo_mode.setCurrentIndex(0)
        self.combo_mode.setStyleSheet("QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 220px; }")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.combo_mode)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.drop_label = DragDropLabel("\n\n拖动源音频文件到这里\n\n")
        self.drop_label.file_dropped.connect(self.load_file)
        layout.addWidget(self.drop_label)

        # 源文件选择按钮行
        src_btn_row = QHBoxLayout()
        self.btn_choose_source = QPushButton("选择源文件")
        self.btn_choose_source.setStyleSheet("font-size:13px; padding:6px;")
        self.btn_choose_source.clicked.connect(self.choose_source_file)
        src_btn_row.addWidget(self.btn_choose_source)
        self.btn_record_source = QPushButton("录制源音频")
        self.btn_record_source.setStyleSheet("font-size:13px; padding:6px; background-color:#E8F5E9;")
        self.btn_record_source.setToolTip("直接用麦克风录制一段 WAV，并自动作为源音频加载。")
        self.btn_record_source.clicked.connect(lambda: self.record_audio_for_role("source"))
        src_btn_row.addWidget(self.btn_record_source)
        src_btn_row.addStretch(1)
        layout.addLayout(src_btn_row)

        self.source_info_label = QLabel("源音频: 未选择")
        self.source_info_label.setStyleSheet("font-size: 12px; color: #666; padding: 4px 0px;")
        layout.addWidget(self.source_info_label)

        self.target_title_label = QLabel("目标音频（声线克隆模式）")
        self.target_title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #333; padding: 8px 0px 4px 0px;")
        layout.addWidget(self.target_title_label)

        self.target_drop_label = DragDropLabel("\n\n拖动目标音频文件到这里\n\n")
        self.target_drop_label.file_dropped.connect(self.load_target_file)
        self.target_drop_label.file_list_dropped.connect(self.load_target_files)
        layout.addWidget(self.target_drop_label)

        # 目标文件选择按钮行
        tgt_btn_row = QHBoxLayout()
        self.btn_choose_target = QPushButton("选择目标文件")
        self.btn_choose_target.setStyleSheet("font-size:13px; padding:6px;")
        self.btn_choose_target.clicked.connect(self.choose_target_file)
        tgt_btn_row.addWidget(self.btn_choose_target)
        self.btn_record_target = QPushButton("录制目标/参考音频")
        self.btn_record_target.setStyleSheet("font-size:13px; padding:6px; background-color:#E3F2FD;")
        self.btn_record_target.setToolTip("直接用麦克风录制一段 WAV，并自动作为目标/参考音频加载。")
        self.btn_record_target.clicked.connect(lambda: self.record_audio_for_role("target"))
        tgt_btn_row.addWidget(self.btn_record_target)
        tgt_btn_row.addStretch(1)
        layout.addLayout(tgt_btn_row)

        self.target_info_label = QLabel("目标音频: 未选择")
        self.target_info_label.setStyleSheet("font-size: 12px; color: #666; padding: 4px 0px;")
        layout.addWidget(self.target_info_label)
        
        clone_engine_row = QHBoxLayout()
        self.clone_engine_label = QLabel("克隆引擎:")
        self.clone_engine_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        self.combo_clone_engine = QComboBox()
        self.combo_clone_engine.addItems(["FreeVC", "OpenVoice"])
        self.combo_clone_engine.setCurrentText("FreeVC")
        self.combo_clone_engine.setStyleSheet("QComboBox { font-size: 13px; padding: 4px; min-width: 160px; }")
        clone_engine_row.addWidget(self.clone_engine_label)
        clone_engine_row.addWidget(self.combo_clone_engine)
        clone_engine_row.addStretch(1)
        layout.addLayout(clone_engine_row)

        tau_row = QHBoxLayout()
        self.openvoice_tau_label = QLabel("OpenVoice tau:")
        self.openvoice_tau_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        self.openvoice_tau_tip_left = QLabel("保守")
        self.openvoice_tau_tip_left.setStyleSheet("font-size: 11px; color: #5C6BC0; font-weight: bold;")
        self.slider_openvoice_tau = QSlider(Qt.Horizontal)
        self.slider_openvoice_tau.setRange(0, 100)
        self.slider_openvoice_tau.setValue(30)
        self.slider_openvoice_tau.setTickPosition(QSlider.TicksBelow)
        self.slider_openvoice_tau.setTickInterval(10)
        self.slider_openvoice_tau.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #5C6BC0;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.input_openvoice_tau = QLineEdit("0.30")
        self.input_openvoice_tau.setStyleSheet("""
            QLineEdit {
                font-size: 13px;
                font-weight: bold;
                color: #5C6BC0;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                min-width: 60px;
                max-width: 70px;
            }
        """)
        self.input_openvoice_tau.setAlignment(Qt.AlignCenter)
        self.input_openvoice_tau.setToolTip("仅对 OpenVoice 生效。建议先尝试 0.20 ~ 0.40。")
        self.openvoice_tau_tip_right = QLabel("更像目标")
        self.openvoice_tau_tip_right.setStyleSheet("font-size: 11px; color: #8E24AA; font-weight: bold;")
        self.slider_openvoice_tau.valueChanged.connect(self._update_openvoice_tau_from_slider)
        self.input_openvoice_tau.textChanged.connect(self._on_openvoice_tau_input_changed)
        tau_row.addWidget(self.openvoice_tau_label)
        tau_row.addWidget(self.openvoice_tau_tip_left)
        tau_row.addWidget(self.slider_openvoice_tau, 1)
        tau_row.addWidget(self.input_openvoice_tau)
        tau_row.addWidget(self.openvoice_tau_tip_right)
        layout.addLayout(tau_row)

        self.similarity_label = QLabel("Speaker Similarity: --")
        self.similarity_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1565C0; background-color: #E3F2FD; "
            "border: 1px solid #90CAF9; border-radius: 6px; padding: 8px 10px;"
        )
        self.similarity_label.setToolTip("显示目标参考音频与克隆结果之间的说话人相似度评分。")
        layout.addWidget(self.similarity_label)

        # [新增] VAD 预处理复选框
        self.chk_vad = QCheckBox("🔕 开启 VAD 预处理 (过滤环境底噪与静音)")
        self.chk_vad.setChecked(True)  # 默认开启
        self.chk_vad.setStyleSheet("font-size: 13px; padding: 5px;")
        self.chk_vad.setToolTip(
            "使用短时能量检测算法，智能压低非人声片段的背景噪音，\n"
            "提升 WORLD 声码器的转换纯净度和听感质量。"
        )
        layout.addWidget(self.chk_vad)

        # ===== [新增] VAD 参数调节区域：能量阈值 =====
        self.vad_title = QLabel("⚙️ VAD 参数微调")
        self.vad_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 10px 0px 5px 0px;")
        layout.addWidget(self.vad_title)

        # 能量阈值行
        vad_threshold_layout = QHBoxLayout()
        
        self.threshold_label = QLabel("能量阈值 (%):")
        self.threshold_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        
        self.label_aggressive = QLabel("激进降噪")
        self.label_aggressive.setStyleSheet("font-size: 11px; color: #ff6b6b; font-weight: bold;")
        
        self.slider_vad_threshold = QSlider(Qt.Horizontal)
        self.slider_vad_threshold.setRange(1, 15)  # 1% - 15%
        self.slider_vad_threshold.setValue(5)      # 默认 5%
        self.slider_vad_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_vad_threshold.setTickInterval(1)
        self.slider_vad_threshold.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #4CAF50;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        # 改成 QLineEdit 可输入百分比
        self.input_vad_threshold = QLineEdit("5")
        self.input_vad_threshold.setStyleSheet("""
            QLineEdit {
                font-size: 13px; 
                font-weight: bold;
                color: #4CAF50;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                min-width: 50px;
                max-width: 60px;
            }
        """)
        self.input_vad_threshold.setAlignment(Qt.AlignCenter)
        # 添加输入验证：只允许 1-15 的数字
        self.input_vad_threshold.textChanged.connect(self._on_threshold_input_changed)
        
        self.label_gentle = QLabel("温和降噪")
        self.label_gentle.setStyleSheet("font-size: 11px; color: #4ecdc4; font-weight: bold;")
        
        # 绑定滑块和输入框的同步
        self.slider_vad_threshold.valueChanged.connect(self._update_threshold_from_slider)
        
        vad_threshold_layout.addWidget(self.threshold_label)
        vad_threshold_layout.addWidget(self.label_aggressive)
        vad_threshold_layout.addWidget(self.slider_vad_threshold, 1)
        vad_threshold_layout.addWidget(self.input_vad_threshold)
        vad_threshold_layout.addWidget(self.label_gentle)
        layout.addLayout(vad_threshold_layout)

        # 帧长参数行
        vad_frame_layout = QHBoxLayout()
        
        self.frame_label = QLabel("帧长时间 (ms):")
        self.frame_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        
        self.label_short = QLabel("灵敏")
        self.label_short.setStyleSheet("font-size: 11px; color: #9b59b6; font-weight: bold;")
        
        self.slider_frame_ms = QSlider(Qt.Horizontal)
        self.slider_frame_ms.setRange(10, 50)     # 10ms - 50ms
        self.slider_frame_ms.setValue(20)         # 默认 20ms
        self.slider_frame_ms.setTickPosition(QSlider.TicksBelow)
        self.slider_frame_ms.setTickInterval(5)
        self.slider_frame_ms.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #9b59b6;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        self.input_frame_ms = QLineEdit("20")
        self.input_frame_ms.setStyleSheet("""
            QLineEdit {
                font-size: 13px; 
                font-weight: bold;
                color: #9b59b6;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                min-width: 50px;
                max-width: 60px;
            }
        """)
        self.input_frame_ms.setAlignment(Qt.AlignCenter)
        self.input_frame_ms.textChanged.connect(self._on_frame_input_changed)
        
        self.label_long = QLabel("稳定")
        self.label_long.setStyleSheet("font-size: 11px; color: #3498db; font-weight: bold;")
        
        self.slider_frame_ms.valueChanged.connect(self._update_frame_from_slider)
        
        vad_frame_layout.addWidget(self.frame_label)
        vad_frame_layout.addWidget(self.label_short)
        vad_frame_layout.addWidget(self.slider_frame_ms, 1)
        vad_frame_layout.addWidget(self.input_frame_ms)
        vad_frame_layout.addWidget(self.label_long)
        layout.addLayout(vad_frame_layout)

        # ===== 转换强度：控制 WORLD 合成与原声的混合比例 =====
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("转换强度:")
        self.strength_label.setStyleSheet("font-size:13px; font-weight:bold; color:#555; min-width:120px;")
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)
        self.slider_strength.setTickPosition(QSlider.TicksBelow)
        self.slider_strength.setTickInterval(10)
        self.input_strength = QLineEdit("100")
        self.input_strength.setStyleSheet("font-size:13px; color:#4CAF50; background-color:#f5f5f5;")
        self.input_strength.setMaximumWidth(60)
        self.input_strength.setAlignment(Qt.AlignCenter)
        # 同步
        self.slider_strength.valueChanged.connect(lambda v: self.input_strength.setText(str(v)))
        self.input_strength.textChanged.connect(lambda t: self._on_strength_input_changed(t))
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.slider_strength, 1)
        strength_layout.addWidget(self.input_strength)
        layout.addLayout(strength_layout)

        # 连接复选框信号到启用/禁用滑块和输入框的槽函数
        self.chk_vad.stateChanged.connect(self._on_vad_toggled)

        self.chk_phase2 = QCheckBox("🧼 开启第二阶段谱减噪 (noisereduce)")
        self.chk_phase2.setChecked(True)
        self.chk_phase2.setStyleSheet("font-size: 13px; padding: 5px;")
        self.chk_phase2.setToolTip(
            "在 VAD 后继续做谱减噪，进一步压制稳定背景噪声。\n"
            "可选温和 / 标准 / 强力 三档。"
        )
        layout.addWidget(self.chk_phase2)

        phase2_row = QHBoxLayout()
        self.phase2_label = QLabel("降噪模式:")
        self.phase2_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")

        self.combo_phase2_mode = QComboBox()
        self.combo_phase2_mode.addItems(["温和", "标准", "强力"])
        self.combo_phase2_mode.setCurrentText("标准")
        self.combo_phase2_mode.setStyleSheet(
            "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; }"
        )
        self.combo_phase2_mode.setToolTip("温和保留更多原音，强力更适合背景噪声明显的录音。")

        self.chk_phase2.stateChanged.connect(self._on_phase2_toggled)

        phase2_row.addWidget(self.phase2_label)
        phase2_row.addWidget(self.combo_phase2_mode)
        phase2_row.addStretch(1)
        layout.addLayout(phase2_row)

        # 降噪策略：无 / 仅VAD / 仅谱减 / 两者
        strategy_row = QHBoxLayout()
        self.strat_label = QLabel("降噪策略:")
        self.strat_label.setStyleSheet("font-size:13px; font-weight:bold; color:#555; min-width:120px;")
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["无", "仅VAD", "仅谱减", "VAD->谱减"])
        self.combo_strategy.setCurrentText("VAD->谱减")
        self.combo_strategy.setStyleSheet("QComboBox { font-size:13px; padding:4px; min-width:180px; }")
        strategy_row.addWidget(self.strat_label)
        strategy_row.addWidget(self.combo_strategy)
        strategy_row.addStretch(1)
        layout.addLayout(strategy_row)

        # 播放时是否保留背景噪声（仅对播放转换后有效）
        self.chk_keep_noise = QCheckBox("播放转换后时保留背景噪声")
        self.chk_keep_noise.setChecked(False)
        self.chk_keep_noise.setToolTip("启用后播放转换后音频时会把原始录音中的背景噪声混回，便于A/B听感比较。")
        layout.addWidget(self.chk_keep_noise)

        self.btn_convert = QPushButton("开始转换")
        self.btn_convert.setMinimumHeight(38)
        self.btn_convert.setStyleSheet(
            "font-size: 15px; font-weight: bold; background-color: #4CAF50; color: white;"
        )
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        layout.addWidget(self.btn_convert)

        plot_row = QHBoxLayout()
        self.btn_plot_f0 = QPushButton("保存 F0 对比图")
        self.btn_plot_f0.setStyleSheet("font-size: 14px; background-color: #2196F3; color: white;")
        self.btn_plot_f0.clicked.connect(self.on_plot_f0_clicked)
        self.btn_plot_f0.setEnabled(False)
        plot_row.addWidget(self.btn_plot_f0)

        self.btn_plot_mel = QPushButton("保存 Mel 对比图")
        self.btn_plot_mel.setStyleSheet("font-size: 14px; background-color: #FF7043; color: white;")
        self.btn_plot_mel.clicked.connect(self.on_plot_mel_clicked)
        self.btn_plot_mel.setEnabled(False)
        plot_row.addWidget(self.btn_plot_mel)
        layout.addLayout(plot_row)

        player_row = QHBoxLayout()
        self.btn_play_original = QPushButton("播放原声")
        self.btn_play_original.clicked.connect(self.play_original)
        self.btn_play_original.setEnabled(False)
        player_row.addWidget(self.btn_play_original)

        self.btn_play_converted = QPushButton("播放转换后")
        self.btn_play_converted.clicked.connect(self.play_converted)
        self.btn_play_converted.setEnabled(False)
        player_row.addWidget(self.btn_play_converted)

        self.combo_anon_variant = QComboBox()
        self.combo_anon_variant.setToolTip("选择要试听/另存的匿名化方法输出")
        self.combo_anon_variant.setStyleSheet("QComboBox { font-size: 13px; padding: 3px; min-width: 170px; }")
        self.combo_anon_variant.setEnabled(False)
        self.combo_anon_variant.currentIndexChanged.connect(self._sync_selected_anonymized_output)
        player_row.addWidget(self.combo_anon_variant)

        self.btn_play_anon_male = QPushButton("播放所选匿名结果")
        self.btn_play_anon_male.clicked.connect(self.play_selected_anonymized)
        self.btn_play_anon_male.setEnabled(False)
        self.btn_play_anon_male.setStyleSheet("font-size: 13px; background-color: #3F51B5; color: white;")
        player_row.addWidget(self.btn_play_anon_male)

        self.btn_save_anon_selected = QPushButton("另存匿名结果")
        self.btn_save_anon_selected.clicked.connect(self.save_selected_anonymized)
        self.btn_save_anon_selected.setEnabled(False)
        self.btn_save_anon_selected.setStyleSheet("font-size: 13px; background-color: #607D8B; color: white;")
        player_row.addWidget(self.btn_save_anon_selected)

        self.btn_stop = QPushButton("停止播放")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        player_row.addWidget(self.btn_stop)
        layout.addLayout(player_row)

        self.anonymization_summary_label = QTextEdit()
        self.anonymization_summary_label.setReadOnly(True)
        self.anonymization_summary_label.setMinimumHeight(115)
        self.anonymization_summary_label.setMaximumHeight(180)
        self.anonymization_summary_label.setStyleSheet(
            "QTextEdit {font-size: 13px; color: #0D47A1; background-color: #E3F2FD; "
            "border: 1px solid #90CAF9; border-radius: 6px; padding: 8px 10px; "
            "font-family: 'Microsoft YaHei', Consolas;}"
        )
        self.anonymization_summary_label.setToolTip(
            "按阶段展示匿名化 pipeline 指标：预处理、baseline/三方法候选、声学质量、输出文件。"
        )
        self.anonymization_summary_label.setText("匿名化结果: --")
        layout.addWidget(self.anonymization_summary_label)

        self.slider_position = QSlider(Qt.Horizontal)
        self.slider_position.setRange(0, 0)
        self.slider_position.sliderMoved.connect(self.seek_position)
        layout.addWidget(self.slider_position)

        self.label_time = QLabel("00:00 / 00:00")
        self.label_time.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label_time)

        wave_row = QHBoxLayout()
        self.label_wave = QLabel("实时振幅")
        self.wave_meter = QProgressBar()
        self.wave_meter.setRange(0, 100)
        self.wave_meter.setValue(0)
        self.wave_meter.setTextVisible(False)
        self.wave_meter.setMaximumWidth(110)
        self.wave_meter.setStyleSheet(
            "QProgressBar {border: 1px solid #666; border-radius: 4px; background: #1f1f1f;}"
            "QProgressBar::chunk {background-color: #00c853;}"
        )
        self.label_wave_value = QLabel("0%")
        self.label_wave_value.setMinimumWidth(40)
        self.label_wave_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.waveform_strip = WaveformStrip(self)

        wave_row.addWidget(self.label_wave)
        wave_row.addWidget(self.waveform_strip, 1)
        wave_row.addWidget(self.wave_meter)
        wave_row.addWidget(self.label_wave_value)
        layout.addLayout(wave_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: Consolas;")
        layout.addWidget(self.log_text)

        self.combo_clone_engine.currentIndexChanged.connect(self._refresh_clone_engine_controls)
        self._on_mode_changed()

    def load_file(self, filepath):
        # 加载新文件前，先停止任何播放
        if self.play_proc is not None:
            self.stop_playback()
        
        self.current_filepath = filepath
        self.anonymized_variants = {}
        self.anonymized_variant_order = []
        self.last_anonymization_result = None
        self._clear_anonymization_display()
        self._clear_similarity_display()

        self.drop_label.setText(f"\n\n源音频已加载:\n{os.path.basename(filepath)}\n（内部会自动转 WAV）\n\n")
        self.source_info_label.setText(f"源音频: {os.path.basename(filepath)}")
        self._report_audio_duration(filepath, "源音频")

        self.log_message(f"✅ 已准备好源文件: {filepath}")

        self._refresh_action_state()
        self.btn_plot_f0.setEnabled(True)
        self.btn_plot_mel.setEnabled(False)
        self.btn_play_anon_male.setEnabled(False)
        self.combo_anon_variant.setEnabled(False)
        self.btn_save_anon_selected.setEnabled(False)
        self.btn_play_original.setEnabled(True)
        self.btn_play_converted.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def load_target_file(self, filepath):
        # 单个目标文件
        self.target_filepaths = [filepath]
        self._clear_similarity_display()
        self.target_drop_label.setText(f"\n\n目标音频已加载:\n{os.path.basename(filepath)}\n（内部会自动转 WAV）\n\n")
        self.target_info_label.setText(f"目标音频: {os.path.basename(filepath)}")
        self._report_audio_duration(filepath, "目标音频")
        self.log_message(f"✅ 已准备好目标文件: {filepath}")
        self._refresh_action_state()

    def load_target_files(self, file_list):
        # 多个目标文件
        self.target_filepaths = list(file_list)
        self._clear_similarity_display()
        names = [os.path.basename(p) for p in self.target_filepaths]
        display = "\n\n目标音频已加载 (%d):\n%s\n（内部会自动转 WAV）\n\n" % (len(names), " | ".join(names))
        self.target_drop_label.setText(display)
        self.target_info_label.setText(f"目标音频: {len(names)} 个参考样本")
        for p in self.target_filepaths:
            self._report_audio_duration(p, "目标音频参考")
        self.log_message(f"✅ 已准备好 {len(names)} 个目标参考文件")
        self._refresh_action_state()

    def choose_source_file(self):
        # 打开文件选择对话框并加载为源音频
        try:
            exts = ' '.join(['*' + e for e in SUPPORTED_FORMATS])
            filter_str = f"音频文件 ({exts});;所有文件 (*)"
            path, _ = QFileDialog.getOpenFileName(self, "选择源音频", "", filter_str)
            if path:
                self.load_file(path)
        except Exception as e:
            self.log_message(f"⚠️ 打开源文件对话框失败: {e}")

    def choose_target_file(self):
        # 打开文件选择对话框并加载为目标音频
        try:
            exts = ' '.join(['*' + e for e in SUPPORTED_FORMATS])
            filter_str = f"音频文件 ({exts});;所有文件 (*)"
            paths, _ = QFileDialog.getOpenFileNames(self, "选择目标音频（可多选）", "", filter_str)
            if paths:
                if len(paths) == 1:
                    self.load_target_file(paths[0])
                else:
                    self.load_target_files(paths)
        except Exception as e:
            self.log_message(f"⚠️ 打开目标文件对话框失败: {e}")

    def _recordings_dir(self):
        path = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(path, exist_ok=True)
        return path

    def _new_recording_path(self, role):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_role = "source" if role == "source" else "target"
        return os.path.join(self._recordings_dir(), f"recorded_{safe_role}_{timestamp}.wav")

    def record_audio_for_role(self, role):
        if not _HAVE_SOUNDDEVICE:
            QMessageBox.warning(
                self,
                "缺少录音依赖",
                "当前环境缺少 sounddevice，无法使用麦克风录音。\n请先执行: pip install sounddevice",
            )
            self.log_message("❌ 录音功能不可用：缺少 sounddevice。")
            return

        is_source = role == "source"
        role_text = "源音频" if is_source else "目标/参考音频"
        output_path = self._new_recording_path(role)
        dialog = RecordDialog(self, role_text, output_path, samplerate=16000)
        if is_source:
            dialog.recording_finished.connect(self.load_file)
        else:
            dialog.recording_finished.connect(self.load_target_file)
        dialog.exec_()

    def log_message(self, msg):
        self.log_text.append(msg)

    def _on_mode_changed(self, *args):
        mode_index = self.combo_mode.currentIndex()
        is_clone_mode = mode_index == 1
        is_similarity_mode = mode_index == 2
        is_anonymization_mode = mode_index == 3
        is_dual_audio_mode = mode_index in (1, 2)

        if is_similarity_mode:
            self.target_title_label.setText("参考音频（说话人相似度模式）")
        elif is_anonymization_mode:
            self.target_title_label.setText("匿名化模式无需目标音频")
        else:
            self.target_title_label.setText("目标音频（声线克隆模式）")

        self.target_title_label.setVisible(is_dual_audio_mode)
        self.target_drop_label.setVisible(is_dual_audio_mode)
        self.target_info_label.setVisible(is_dual_audio_mode)
        self.similarity_label.setVisible(is_dual_audio_mode)
        self.anonymization_summary_label.setVisible(is_anonymization_mode)

        self.btn_play_converted.setText("播放最佳匿名" if is_anonymization_mode else "播放转换后")
        self.combo_anon_variant.setVisible(is_anonymization_mode)
        self.btn_play_anon_male.setVisible(is_anonymization_mode)
        self.btn_save_anon_selected.setVisible(is_anonymization_mode)

        world_controls = [
            self.vad_title,
            self.threshold_label,
            self.label_aggressive,
            self.label_gentle,
            self.chk_vad,
            self.slider_vad_threshold,
            self.input_vad_threshold,
            self.frame_label,
            self.label_short,
            self.label_long,
            self.slider_frame_ms,
            self.input_frame_ms,
            self.strength_label,
            self.slider_strength,
            self.input_strength,
            self.chk_phase2,
            self.phase2_label,
            self.combo_phase2_mode,
            self.strat_label,
            self.combo_strategy,
            self.chk_keep_noise,
        ]
        for widget in world_controls:
            widget.setVisible(mode_index == 0)

        # 目标选择按钮也只在声线克隆模式可见
        try:
            self.btn_choose_target.setVisible(is_dual_audio_mode)
            self.btn_record_target.setVisible(is_dual_audio_mode)
        except Exception:
            pass
        
        try:
            self.clone_engine_label.setVisible(is_clone_mode)
            self.combo_clone_engine.setVisible(is_clone_mode)
        except Exception:
            pass

        self._refresh_clone_engine_controls()
        self._refresh_action_state()

    def _refresh_clone_engine_controls(self):
        is_vc_mode = self.combo_mode.currentIndex() == 1
        is_openvoice = False
        try:
            is_openvoice = self.combo_clone_engine.currentText() == "OpenVoice"
        except Exception:
            pass

        show_tau = is_vc_mode and is_openvoice
        for widget in [
            self.openvoice_tau_label,
            self.openvoice_tau_tip_left,
            self.slider_openvoice_tau,
            self.input_openvoice_tau,
            self.openvoice_tau_tip_right,
        ]:
            widget.setVisible(show_tau)

    def _update_openvoice_tau_from_slider(self):
        value = self.slider_openvoice_tau.value() / 100.0
        self.input_openvoice_tau.blockSignals(True)
        self.input_openvoice_tau.setText(f"{value:.2f}")
        self.input_openvoice_tau.blockSignals(False)

    def _on_openvoice_tau_input_changed(self, text):
        try:
            value = float(text)
        except ValueError:
            return
        value = max(0.0, min(1.0, value))
        slider_value = int(round(value * 100))
        self.slider_openvoice_tau.blockSignals(True)
        self.slider_openvoice_tau.setValue(slider_value)
        self.slider_openvoice_tau.blockSignals(False)

    def _clear_similarity_display(self):
        self.last_similarity_result = None
        self.similarity_label.setText("Speaker Similarity: --")
        self.similarity_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1565C0; background-color: #E3F2FD; "
            "border: 1px solid #90CAF9; border-radius: 6px; padding: 8px 10px;"
        )

    def _update_similarity_display(self, result):
        self.last_similarity_result = result
        score = result.get("similarity_score")
        cosine = result.get("cosine_similarity")
        ref_label = result.get("reference_label") or "reference"

        if score is None:
            self.similarity_label.setText("Speaker Similarity: 无结果")
            return

        score_val = float(score)
        if score_val >= 85:
            color = "#2E7D32"
            bg = "#E8F5E9"
            border = "#81C784"
            level = "高相似"
        elif score_val >= 70:
            color = "#EF6C00"
            bg = "#FFF3E0"
            border = "#FFB74D"
            level = "中等相似"
        else:
            color = "#C62828"
            bg = "#FFEBEE"
            border = "#E57373"
            level = "较低相似"

        cosine_text = "--"
        if cosine is not None:
            cosine_text = f"{float(cosine):.6f}"

        self.similarity_label.setText(
            f"Speaker Similarity: {score_val:.2f}/100  |  {level}  |  参考: {ref_label}  |  Cosine: {cosine_text}"
        )
        self.similarity_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color}; background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 6px; padding: 8px 10px;"
        )

    def _load_similarity_from_output(self, output_path):
        try:
            json_path = os.path.splitext(output_path)[0] + '_similarity.json'
            if not os.path.exists(json_path):
                return None
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_message(f"⚠️ 读取相似度结果失败: {e}")
            return None

    def _refresh_action_state(self):
        mode_index = self.combo_mode.currentIndex()
        if mode_index == 1:
            can_convert = bool(self.current_filepath and self.target_filepaths)
            self.btn_convert.setText("开始克隆")
            self.btn_plot_f0.setEnabled(bool(self.current_filepath))
            self.btn_plot_mel.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
            self.btn_play_converted.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
        elif mode_index == 2:
            can_convert = bool(self.current_filepath and self.target_filepaths)
            self.btn_convert.setText("开始评估")
            self.btn_plot_f0.setEnabled(False)
            self.btn_plot_mel.setEnabled(False)
            self.btn_play_converted.setEnabled(False)
        elif mode_index == 3:
            can_convert = bool(self.current_filepath)
            self.btn_convert.setText("开始匿名化")
            self.btn_plot_f0.setEnabled(bool(self.current_filepath))
            self.btn_plot_mel.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
            self.btn_play_converted.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
            has_variants = bool(self._available_anonymized_variants())
            self.combo_anon_variant.setEnabled(has_variants)
            self.btn_play_anon_male.setEnabled(has_variants)
            self.btn_save_anon_selected.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
        else:
            can_convert = bool(self.current_filepath)
            self.btn_convert.setText("开始转换")
            self.btn_plot_f0.setEnabled(bool(self.current_filepath))
            self.btn_plot_mel.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
            self.btn_play_converted.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
            self.combo_anon_variant.setEnabled(False)
            self.btn_play_anon_male.setEnabled(False)
            self.btn_save_anon_selected.setEnabled(False)
        self.btn_convert.setEnabled(can_convert)

    def _clear_anonymization_display(self):
        try:
            self.anonymization_summary_label.setText("匿名化结果: --")
        except Exception:
            pass
        try:
            self.combo_anon_variant.blockSignals(True)
            self.combo_anon_variant.clear()
            self.combo_anon_variant.blockSignals(False)
            self.combo_anon_variant.setEnabled(False)
        except Exception:
            pass

    def _variant_display_name(self, variant):
        names = {
            "male_leaning": "低沉稳重匿名声线",
            "female_leaning": "柔和清亮匿名声线",
            "lab9_montage_pool": "综合自然匿名声线",
            "freevc_baseline": "FreeVC optimized male",
            "metric_clarity": "Metric+clarity optimized male",
            "ppg_tone": "PPG-tone optimized male",
            "baseline": "原始参考基线",
        }
        return names.get(variant, variant or "匿名版本")

    def _variant_label(self, variant, item=None):
        item = item or (self.anonymized_variants or {}).get(variant) or {}
        # 对同学 Web UI 三方法保留完全一致的英文显示名；其他旧版策略继续用中文别名。
        friendly_name = self._variant_display_name(variant)
        label = friendly_name if friendly_name != (variant or "匿名版本") else (item.get("label") or item.get("display_name") or friendly_name)
        return str(label)

    def _available_anonymized_variants(self):
        order = list(self.anonymized_variant_order or [])
        for variant in (self.anonymized_variants or {}).keys():
            if variant not in order:
                order.append(variant)
        return [variant for variant in order if self._get_anonymized_path(variant)]

    def _selected_anonymized_variant(self):
        try:
            data = self.combo_anon_variant.currentData()
            if data:
                return str(data)
        except Exception:
            pass
        variants = self._available_anonymized_variants()
        return variants[0] if variants else ""

    def _sync_selected_anonymized_output(self, *args):
        variant = self._selected_anonymized_variant()
        path = self._get_anonymized_path(variant) if variant else ""
        if path:
            self.converted_filepath = path
            self.btn_save_anon_selected.setEnabled(True)
            self.btn_play_anon_male.setEnabled(True)

    def _get_anonymized_path(self, variant):
        item = (self.anonymized_variants or {}).get(variant) or {}
        path = (
            item.get("copied_output")
            or item.get("final_output")
            or item.get("selected_candidate")
            or item.get("audio_path")
            or item.get("output_path")
            or item.get("path")
            or item.get("wav_path")
        )
        if path and os.path.exists(path):
            return path
        return ""

    def _format_float(self, value, digits=3):
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "--"

    def _format_percent(self, value, digits=1):
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except Exception:
            return "--"

    def _format_metric_percent(self, value, digits=1):
        """Format metrics that may be stored either as 0-1 ratios or 0-100 percentages."""
        try:
            numeric = float(value)
            if abs(numeric) <= 1.0:
                numeric *= 100.0
            return f"{numeric:.{digits}f}%"
        except Exception:
            return "--"

    def _metric_value(self, *objects, keys=()):
        """Find the first metric value from several possibly nested result dictionaries."""
        key_set = {str(k).lower() for k in keys}

        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if str(key).lower() in key_set and value is not None:
                        return value
                for value in obj.values():
                    found = visit(value)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for value in obj:
                    found = visit(value)
                    if found is not None:
                        return found
            return None

        for obj in objects:
            found = visit(obj)
            if found is not None:
                return found
        return None

    def _display_method_name(self, variant, item=None):
        item = item or {}
        names = {
            "freevc_baseline": "FreeVC optimized male",
            "metric_clarity": "Metric+clarity optimized male",
            "ppg_tone": "PPG-tone optimized male",
            "male_leaning": "低沉稳重匿名声线",
            "female_leaning": "柔和清亮匿名声线",
            "lab9_montage_pool": "综合自然匿名声线",
            "baseline": "原始参考基线",
        }
        if variant in names:
            return names[variant]
        label = item.get("label") or item.get("display_name") or item.get("method") or item.get("name")
        if label:
            return str(label)
        return names.get(variant, self._variant_display_name(variant))

    def _variant_report_data(self, report, variant):
        if not isinstance(report, dict):
            return {}
        matched = {}
        for container_name in ("variants", "vc_variants", "results", "outputs"):
            container = report.get(container_name)
            if isinstance(container, dict) and isinstance(container.get(variant), dict):
                matched.update(container.get(variant) or {})
            if isinstance(container, list):
                for row in container:
                    if not isinstance(row, dict):
                        continue
                    row_variant = row.get("variant") or row.get("method") or row.get("name")
                    if row_variant == variant:
                        matched.update(row)

        # 同学新版 recording_demo_ui.py 会把四项即时评估指标写入
        # recording_evaluation.metrics（部分历史版本叫 rows），而不是
        # results/variants 里。GUI 之前没读这里，所以 ASR-WER / 内容保留率 /
        # 源相似度 / 相似度下降会全部显示 --。
        evaluation = report.get("recording_evaluation") or {}
        rows = None
        if isinstance(evaluation, dict):
            rows = evaluation.get("metrics") or evaluation.get("rows")
        if isinstance(rows, list):
            display_name = self._display_method_name(variant, matched)
            candidate_names = {
                str(variant),
                str(display_name),
                str(matched.get("method") or ""),
                str(matched.get("label") or ""),
                str(matched.get("display_name") or ""),
            }
            candidate_names = {name for name in candidate_names if name}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_names = {
                    str(row.get("method") or ""),
                    str(row.get("variant") or ""),
                    str(row.get("label") or ""),
                    str(row.get("display_name") or ""),
                }
                if candidate_names.intersection(name for name in row_names if name):
                    merged = dict(matched)
                    merged.update(row)
                    return merged
        return matched

    def _baseline_source_similarity(self, report):
        if not isinstance(report, dict):
            return None
        baseline = report.get("baseline") or {}
        source_vs_source = baseline.get("source_vs_source") if isinstance(baseline, dict) else None
        return self._metric_value(
            source_vs_source,
            baseline,
            report.get("baseline_metrics"),
            keys=("source_similarity", "source_mean_score", "target_mean_score", "speaker_similarity"),
        )

    def _variant_core_metrics(self, report, variant, item, best=None):
        variant_data = self._variant_report_data(report, variant)
        metric_sources = [item, best, variant_data]
        privacy = variant_data.get("privacy") if isinstance(variant_data, dict) else None
        utility = variant_data.get("utility") if isinstance(variant_data, dict) else None

        source_similarity = self._metric_value(
            *metric_sources,
            privacy,
            keys=(
                "source_similarity",
                "source_sim",
                "similarity_to_source",
                "source_speaker_similarity",
                "protected_similarity",
                "target_mean_score",
            ),
        )
        similarity_drop = self._metric_value(
            *metric_sources,
            privacy,
            keys=(
                "similarity_drop",
                "source_similarity_reduction",
                "speaker_similarity_drop",
                "privacy_gain",
                "deidentification_rate",
                "deid_rate",
            ),
        )
        if similarity_drop is None and source_similarity is not None:
            baseline_similarity = self._baseline_source_similarity(report)
            try:
                baseline_similarity = float(baseline_similarity)
                source_similarity_f = float(source_similarity)
                if baseline_similarity > 0:
                    similarity_drop = max(0.0, (baseline_similarity - source_similarity_f) / baseline_similarity)
            except Exception:
                pass

        asr_wer = self._metric_value(
            *metric_sources,
            utility,
            keys=("asr_wer", "wer", "word_error_rate", "char_error_rate", "cer"),
        )
        content_kept = self._metric_value(
            *metric_sources,
            utility,
            keys=("content_kept", "content_retention", "content_preservation", "utility_kept", "intelligibility"),
        )
        if content_kept is None and asr_wer is not None:
            try:
                wer_f = float(asr_wer)
                content_kept = max(0.0, 1.0 - wer_f) if wer_f <= 1.0 else max(0.0, 100.0 - wer_f)
            except Exception:
                pass

        return {
            "method": self._display_method_name(variant, item),
            "asr_wer": asr_wer,
            "content_kept": content_kept,
            "source_similarity": source_similarity,
            "similarity_drop": similarity_drop,
        }

    def _core_metrics_table_lines(self, report, variant_order):
        rows = []
        missing_count = 0
        for rank, variant in enumerate(variant_order, start=1):
            item = self.anonymized_variants.get(variant) or {}
            if not item:
                continue
            _, best = self._find_variant_evaluation(report, variant, item)
            metrics = self._variant_core_metrics(report, variant, item, best)
            if all(metrics.get(key) is None for key in ("asr_wer", "content_kept", "source_similarity", "similarity_drop")):
                missing_count += 1
            rows.append((rank, variant, metrics))
        if not rows:
            return []

        lines = [
            "核心评估指标",
        ]
        for rank, _variant, metrics in rows:
            method = metrics.get("method") or "--"
            lines.append(
                f"{rank}. {method}: "
                f"ASR-WER {self._format_metric_percent(metrics.get('asr_wer'), 1)}  |  "
                f"内容保留率 {self._format_metric_percent(metrics.get('content_kept'), 1)}  |  "
                f"源相似度 {self._format_metric_percent(metrics.get('source_similarity'), 1)}  |  "
                f"相似度下降 {self._format_metric_percent(metrics.get('similarity_drop'), 1)}"
            )

        reason = self._metric_missing_reason(report)
        if missing_count:
            lines.append(f"说明：有 {missing_count} 个方法仍为 --。原因：{reason}")
        else:
            lines.append("说明：这些指标来自同学项目 recording_evaluation.rows 的即时评估结果。")
        return lines

    def _metric_missing_reason(self, report):
        if not isinstance(report, dict):
            return "没有读到 pipeline_report，无法取得同学项目写出的评估 JSON。"
        evaluation = report.get("recording_evaluation")
        if not isinstance(evaluation, dict):
            return "当前报告没有 recording_evaluation 字段，说明本次流程没有运行即时评估。"
        if evaluation.get("available") is False:
            error = evaluation.get("error") or evaluation.get("note") or "未知错误"
            return f"同学项目即时评估失败：{error}。通常是 speechbrain / faster_whisper / 本地模型路径缺失。"
        rows = evaluation.get("metrics") or evaluation.get("rows")
        if not isinstance(rows, list) or not rows:
            return "recording_evaluation 中没有 metrics/rows，可能评估被跳过或结果文件未完整写入。"
        return "评估 rows 已存在，但方法名或字段名未匹配到当前 GUI 候选。"

    def _basename_or_dash(self, path):
        try:
            return os.path.basename(path) if path else "--"
        except Exception:
            return "--"

    def _read_json_file(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _probe_brief(self, probe):
        if not isinstance(probe, dict):
            return "--"
        fmt = probe.get("format") or {}
        streams = [s for s in (probe.get("streams") or []) if s.get("codec_type") == "audio"]
        stream = streams[0] if streams else ((probe.get("streams") or [{}])[0] if probe.get("streams") else {})
        duration = fmt.get("duration")
        duration_text = self._format_float(duration, 2) + "s" if duration is not None else "--"
        sr = stream.get("sample_rate") or "--"
        channels = stream.get("channels") or "--"
        codec = stream.get("codec_name") or "--"
        container = fmt.get("format_name") or "--"
        return f"时长 {duration_text}｜采样率 {sr}Hz｜声道 {channels}｜编码 {codec}｜容器 {container}"

    def _audio_stats_brief(self, stats):
        if not isinstance(stats, dict):
            return "--"
        return (
            f"时长 {self._format_float(stats.get('duration_sec'), 2)}s｜"
            f"F0中位 {self._format_float(stats.get('median_f0_hz'), 1)} Hz｜"
            f"有声帧 {self._format_percent(stats.get('voiced_ratio'), 1)}"
        )

    def _first_mapping_value(self, mapping):
        if isinstance(mapping, dict) and mapping:
            return next(iter(mapping.values()))
        return None

    def _load_anonymization_report(self, result):
        report_path = result.get("pipeline_report") if isinstance(result, dict) else None
        if not report_path or not os.path.exists(report_path):
            return None
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _find_variant_evaluation(self, report, variant, variant_item=None):
        if not isinstance(report, dict):
            return None, None
        variant_data = (report.get("vc_variants") or {}).get(variant) or {}
        evaluation_summary = variant_data.get("evaluation_summary") or {}
        source_name = (variant_item or {}).get("source_name")
        if source_name and source_name in evaluation_summary:
            summary = evaluation_summary.get(source_name)
        else:
            summary = self._first_mapping_value(evaluation_summary)
        if not isinstance(summary, dict):
            return variant_data, None
        best = summary.get("best_candidate") or self._first_mapping_value(summary.get("all_candidates") or {})
        return variant_data, best

    def _variant_metric_lines(self, report, variant, item):
        lines = []
        display = self._variant_display_name(variant)
        variant_data, best = self._find_variant_evaluation(report, variant, item)
        profile = (best or {}).get("profile") or item.get("profile") or {}
        backend = profile.get("backend") or item.get("backend") or "--"

        lines.append(f"【{display}】")
        lines.append(
            f"  选择结果：backend {backend}｜"
            f"目标池 {profile.get('target_pool_name') or profile.get('name') or '--'}｜"
            f"参考数 {profile.get('target_reference_count', '--')}｜策略 {profile.get('target_strategy', '--')}"
        )

        output_path = (
            item.get("copied_output")
            or item.get("final_output")
            or item.get("selected_candidate")
            or item.get("audio_path")
            or item.get("output_path")
            or item.get("path")
            or item.get("wav_path")
        )
        lines.append(f"  文件：{self._basename_or_dash(output_path)}")
        return lines

    def _build_anonymization_summary_text(self, result):
        report = self._load_anonymization_report(result)
        selected_variant = result.get("selected_variant") or "best"
        selected_output = result.get("selected_output")

        lines = [
            f"匿名化结果: {self._variant_display_name(selected_variant)}  |  文件: {self._basename_or_dash(selected_output)}",
        ]

        if report:
            lines.extend([f"报告文件: {self._basename_or_dash(result.get('pipeline_report'))}"])
            preprocess = (report.get("preprocess_outputs") or [{}])[0]
            metadata = self._read_json_file(preprocess.get("metadata_json")) or {}
            denoised = report.get("denoised_path") or preprocess.get("denoised_wav")
            lines.append(
                f"降噪: {report.get('denoise_preset') or metadata.get('denoise_preset') or '--'}  |  "
                f"降噪文件: {self._basename_or_dash(denoised)}"
            )
            if metadata:
                lines.append(f"音频信息: {self._probe_brief(metadata.get('denoised_probe') or metadata.get('normalized_probe') or metadata.get('source_probe'))}")
        else:
            lines.append("报告解析: 未找到 pipeline_report，当前只显示 runner 返回的简要结果。")

        variant_order = self._available_anonymized_variants()
        if not variant_order:
            variant_order = list(self.anonymized_variant_order or self.anonymized_variants.keys())

        core_metric_lines = self._core_metrics_table_lines(report, variant_order)
        if core_metric_lines:
            lines.extend([""] + core_metric_lines)

        for variant in variant_order:
            lines.append("")
            item = self.anonymized_variants.get(variant)
            if item:
                lines.extend(self._variant_metric_lines(report, variant, item))
            else:
                lines.append(f"{self._variant_display_name(variant)}: 未生成该方向的匿名化结果。")

        lines.extend([
            "",
            "操作: 下拉框可切换三种匿名方法；播放/另存按钮作用于当前选中的匿名文件。",
        ])
        return "\n".join(lines)

    def _update_anonymization_display(self, result):
        variants = result.get("variants") or []
        self.anonymized_variants = {}
        self.anonymized_variant_order = []
        for item in variants:
            variant = item.get("variant") or "variant"
            self.anonymized_variants[variant] = item
            self.anonymized_variant_order.append(variant)
        self.combo_anon_variant.blockSignals(True)
        self.combo_anon_variant.clear()
        for variant in self._available_anonymized_variants():
            self.combo_anon_variant.addItem(self._variant_label(variant), variant)
        selected_variant = result.get("selected_variant")
        if selected_variant:
            idx = self.combo_anon_variant.findData(selected_variant)
            if idx >= 0:
                self.combo_anon_variant.setCurrentIndex(idx)
        self.combo_anon_variant.blockSignals(False)
        self.combo_anon_variant.setEnabled(self.combo_anon_variant.count() > 0)
        self._sync_selected_anonymized_output()
        self.anonymization_summary_label.setText(self._build_anonymization_summary_text(result))

    def _report_audio_duration(self, filepath, role):
        try:
            duration, warnings = _duration_status(filepath, role)
            self.log_message(f"ℹ️ {role}时长: {duration:.2f}s")
            show_warnings = role.startswith("目标")
            if show_warnings:
                show_popup = self.combo_mode.currentIndex() == 1
                for warn in warnings:
                    self.log_message(f"⚠️ {warn}")
                    if show_popup:
                        QMessageBox.warning(self, "时长提醒", warn)
            return duration
        except Exception as e:
            self.log_message(f"⚠️ 无法检测{role}时长: {e}")
            return None

    def _update_threshold_from_slider(self):
        """滑块变化时更新输入框"""
        value = self.slider_vad_threshold.value()
        self.input_vad_threshold.blockSignals(True)
        self.input_vad_threshold.setText(str(value))
        self.input_vad_threshold.blockSignals(False)

    def _on_threshold_input_changed(self, text):
        """输入框变化时同步滑块"""
        try:
            value = int(text)
            if 1 <= value <= 15:
                self.slider_vad_threshold.blockSignals(True)
                self.slider_vad_threshold.setValue(value)
                self.slider_vad_threshold.blockSignals(False)
            else:
                # 超出范围时不更新滑块
                pass
        except ValueError:
            # 非数字输入，忽略
            pass

    def _update_frame_from_slider(self):
        """帧长滑块变化时更新输入框"""
        value = self.slider_frame_ms.value()
        self.input_frame_ms.blockSignals(True)
        self.input_frame_ms.setText(str(value))
        self.input_frame_ms.blockSignals(False)

    def _on_frame_input_changed(self, text):
        """帧长输入框变化时同步滑块"""
        try:
            value = int(text)
            if 10 <= value <= 50:
                self.slider_frame_ms.blockSignals(True)
                self.slider_frame_ms.setValue(value)
                self.slider_frame_ms.blockSignals(False)
            else:
                # 超出范围时不更新滑块
                pass
        except ValueError:
            # 非数字输入，忽略
            pass

    def _on_strength_input_changed(self, text):
        try:
            value = int(text)
            if 0 <= value <= 100:
                self.slider_strength.blockSignals(True)
                self.slider_strength.setValue(value)
                self.slider_strength.blockSignals(False)
            else:
                pass
        except ValueError:
            pass

    def _on_vad_toggled(self):
        """当 VAD 复选框状态改变时，启用/禁用滑块和输入框"""
        is_checked = self.chk_vad.isChecked()
        self.slider_vad_threshold.setEnabled(is_checked)
        self.input_vad_threshold.setEnabled(is_checked)
        self.slider_frame_ms.setEnabled(is_checked)
        self.input_frame_ms.setEnabled(is_checked)
        
        # 如果禁用 VAD，所有控件变灰
        if not is_checked:
            gray_style_slider = """
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #aaa;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """
            self.slider_vad_threshold.setStyleSheet(gray_style_slider)
            self.slider_frame_ms.setStyleSheet(gray_style_slider)
            
            gray_input_style = """
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #aaa;
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """
            self.input_vad_threshold.setStyleSheet(gray_input_style)
            self.input_frame_ms.setStyleSheet(gray_input_style)
        else:
            # 恢复原来的样式
            self.slider_vad_threshold.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #4CAF50;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """)
            self.slider_frame_ms.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #9b59b6;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """)
            self.input_vad_threshold.setStyleSheet("""
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #4CAF50;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """)
            self.input_frame_ms.setStyleSheet("""
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #9b59b6;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """)

    def _on_phase2_toggled(self):
        is_checked = self.chk_phase2.isChecked()
        self.combo_phase2_mode.setEnabled(is_checked)
        if not is_checked:
            self.combo_phase2_mode.setStyleSheet(
                "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; color: #888; }"
            )
        else:
            self.combo_phase2_mode.setStyleSheet(
                "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; }"
            )


    def start_conversion(self):
        if not self.current_filepath:
            return

        self._clear_similarity_display()

        mode_index = self.combo_mode.currentIndex()
        is_vc_mode = mode_index == 1
        is_similarity_mode = mode_index == 2
        is_anonymization_mode = mode_index == 3
        if (is_vc_mode or is_similarity_mode) and not self.target_filepaths:
            QMessageBox.warning(self, "缺少参考音频", "当前模式需要同时选择两段音频。")
            self.log_message("⚠️ 当前模式缺少参考音频。")
            return

        # 在开始转换前，先停止任何正在进行的播放
        if self.play_proc is not None:
            self.log_message("⏹️ 停止现有播放以开始转换...")
            self.stop_playback()

        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("正在处理中...")

        # 时长检查：克隆模式过短直接提醒并停止；单音模式仅提示
        try:
            source_duration, source_warnings = _duration_status(self.current_filepath, "源音频")
            # 源音频不再做时长限制与提醒
        except Exception as e:
            self.log_message(f"⚠️ 源音频时长检测失败: {e}")
            QMessageBox.warning(self, "时长检测失败", f"源音频时长检测失败：{e}")
            self.btn_convert.setEnabled(True)
            self._refresh_action_state()
            return

        if is_vc_mode or is_similarity_mode:
            try:
                for idx, target_path in enumerate(self.target_filepaths, start=1):
                    target_duration, target_warnings = _duration_status(target_path, f"目标音频{idx}")
                    if target_duration < MIN_AUDIO_LENGTH:
                        warn_text = target_warnings[0] if target_warnings else f"目标音频{idx}太短，至少需要 {MIN_AUDIO_LENGTH:.0f}s。"
                        QMessageBox.warning(self, "时长不足", warn_text)
                        self.log_message(f"⚠️ {warn_text}")
                        self.btn_convert.setEnabled(True)
                        self._refresh_action_state()
                        return
                    for warn_text in target_warnings:
                        self.log_message(f"⚠️ {warn_text}")
                        QMessageBox.warning(self, "时长提醒", warn_text)
            except Exception as e:
                self.log_message(f"⚠️ 目标音频时长检测失败: {e}")
                QMessageBox.warning(self, "时长检测失败", f"目标音频时长检测失败：{e}")
                self.btn_convert.setEnabled(True)
                self._refresh_action_state()
                return

        if is_anonymization_mode:
            self.converted_filepath = None
            self.anonymized_variants = {}
            self.anonymized_variant_order = []
            self.last_anonymization_result = None
            self._clear_anonymization_display()
            self.anonymization_thread = AnonymizationThread(self.current_filepath, denoise_preset="standard")
            self.anonymization_thread.log_signal.connect(self.log_message)
            self.anonymization_thread.finished_signal.connect(self.anonymization_finished)
            self.anonymization_thread.start()
            return

        if is_similarity_mode:
            self.converted_filepath = None
            self.similarity_thread = SimilarityEvalThread(self.current_filepath, self.target_filepaths)
            self.similarity_thread.log_signal.connect(self.log_message)
            self.similarity_thread.finished_signal.connect(self.similarity_eval_finished)
            self.similarity_thread.start()
            return

        if is_vc_mode:
            engine = "FreeVC"
            try:
                engine = self.combo_clone_engine.currentText()
            except Exception:
                pass

            if engine == "OpenVoice" and source_duration < MIN_AUDIO_LENGTH:
                warn_text = (
                    f"OpenVoice 对源音频较敏感，当前仅 {source_duration:.2f}s，可能偏机械。\n"
                    f"系统会自动补长后再提取说话人特征；建议 tau 先调到 0.20~0.25。\n"
                    f"如果仍不稳定，优先换更长、更清晰的源音频再试。\n"
                    f"建议源音频时长至少 {MIN_AUDIO_LENGTH:.0f}s。"
                )
                self.log_message(f"⚠️ {warn_text}")
                QMessageBox.warning(self, "时长提醒", warn_text)

            openvoice_tau = 0.3
            if engine == "OpenVoice":
                try:
                    openvoice_tau = max(0.0, min(1.0, float(self.input_openvoice_tau.text().strip())))
                except Exception:
                    openvoice_tau = 0.3

            self.thread = VoiceCloneThread(
                self.current_filepath,
                self.target_filepaths,
                engine=engine,
                openvoice_tau=openvoice_tau,
            )
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.conversion_finished)
            self.thread.start()
            return

        # WORLD 单音频模式：继续使用原有 VAD / 谱减噪流程
        try:
            threshold_value = int(self.input_vad_threshold.text())
            threshold_ratio = max(1, min(15, threshold_value)) / 100.0
        except ValueError:
            threshold_ratio = 0.05

        try:
            frame_value = int(self.input_frame_ms.text())
            frame_ms = max(10, min(50, frame_value))
        except ValueError:
            frame_ms = 20

        strat = self.combo_strategy.currentText().strip()
        if strat == "无":
            use_vad = False
            use_phase2 = False
        elif strat == "仅VAD":
            use_vad = True
            use_phase2 = False
        elif strat == "仅谱减":
            use_vad = False
            use_phase2 = True
        else:
            use_vad = True
            use_phase2 = True

        phase2_mode = self.combo_phase2_mode.currentText().strip()

        self.thread = VoiceConverterThread(
            self.current_filepath,
            enable_vad=use_vad,
            threshold_ratio=threshold_ratio,
            frame_ms=frame_ms,
            enable_phase2=use_phase2,
            phase2_mode=phase2_mode,
            strength=max(0.0, min(1.0, float(self.slider_strength.value()) / 100.0)),
        )
        self.thread.log_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.conversion_finished)
        self.thread.start()

    def conversion_finished(self, output_path):
        self._refresh_action_state()
        if output_path:
            self.converted_filepath = output_path
            self.log_message(f"文件已保存至: {output_path}")
            if self.combo_mode.currentIndex() == 1:
                similarity_result = self._load_similarity_from_output(output_path)
                if similarity_result:
                    self._update_similarity_display(similarity_result)
                else:
                    self.similarity_label.setText("Speaker Similarity: 本次未生成结果")
            self.log_message("-" * 32)
            self.btn_play_converted.setEnabled(True)
            self.btn_plot_mel.setEnabled(True)
        else:
            if self.combo_mode.currentIndex() == 1:
                self.similarity_label.setText("Speaker Similarity: 转换失败，未生成结果")
            self.log_message("转换失败。")

    def anonymization_finished(self, result):
        self._refresh_action_state()
        if not result:
            self.converted_filepath = None
            self.log_message("匿名化失败。")
            return

        selected_output = result.get("selected_output")
        if not selected_output or not os.path.exists(selected_output):
            self.converted_filepath = None
            self.log_message("匿名化完成，但未找到可播放的输出文件。")
            return

        self.converted_filepath = selected_output
        self.last_anonymization_result = result
        self._update_anonymization_display(result)
        selected_variant = result.get("selected_variant") or "best"
        self.log_message(f"匿名化最佳结果: {self._variant_display_name(selected_variant)}")
        for item in result.get("variants") or []:
            variant = item.get("variant") or "variant"
            copied = item.get("copied_output") or item.get("final_output")
            if copied:
                self.log_message(f"- {self._variant_display_name(variant)}: {copied}")
        self.log_message(f"匿名化结果已保存至: {selected_output}")
        report = result.get("pipeline_report")
        if report:
            self.log_message(f"完整 pipeline 报告: {report}")
        self.log_message("-" * 32)
        self.btn_play_converted.setEnabled(True)
        has_variants = bool(self._available_anonymized_variants())
        self.combo_anon_variant.setEnabled(has_variants)
        self.btn_play_anon_male.setEnabled(has_variants)
        self.btn_save_anon_selected.setEnabled(True)
        self.btn_plot_f0.setEnabled(True)
        self.btn_plot_mel.setEnabled(True)

    def similarity_eval_finished(self, result):
        self._refresh_action_state()
        if result:
            self.converted_filepath = None
            self._update_similarity_display(result)
            result_json = result.get('result_json')
            if result_json:
                self.log_message(f"相似度结果已保存至: {result_json}")
            self.log_message("-" * 32)
        else:
            self.similarity_label.setText("Speaker Similarity: 评估失败，未生成结果")

    def on_plot_f0_clicked(self):
        if not self.current_filepath:
            return
        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_f0_hist.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 F0 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_f0.setEnabled(False)
        self.log_message(f"正在保存 F0 对比图: {save_path}")
        compare_path = self.converted_filepath if self.converted_filepath and os.path.exists(self.converted_filepath) else None
        self.plot_f0_thread = PlotF0Thread(self.current_filepath, save_path, compare_path)
        self.plot_f0_thread.log_signal.connect(self.log_message)
        self.plot_f0_thread.finished_signal.connect(self.plot_f0_finished)
        self.plot_f0_thread.start()

    def plot_f0_finished(self, path):
        self.btn_plot_f0.setEnabled(True)
        if path:
            self.log_message(f"F0 图像已保存: {path}")
        else:
            self.log_message("F0 绘图未完成或失败。")

    def on_plot_mel_clicked(self):
        if not self.current_filepath:
            return
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("请先完成转换，再生成 Mel 对比图。")
            return

        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_mel_compare.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Mel 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_mel.setEnabled(False)
        self.log_message(f"正在保存 Mel 对比图: {save_path}")
        self.plot_mel_thread = PlotMelThread(self.current_filepath, self.converted_filepath, save_path)
        self.plot_mel_thread.log_signal.connect(self.log_message)
        self.plot_mel_thread.finished_signal.connect(self.plot_mel_finished)
        self.plot_mel_thread.start()

    def plot_mel_finished(self, path):
        self.btn_plot_mel.setEnabled(True)
        if path:
            self.log_message(f"Mel 图像已保存: {path}")
        else:
            self.log_message("Mel 绘图未完成或失败。")

    def _remember_temp_play_path(self, path):
        """记录播放链路产生的临时 WAV，避免后续 sf.read 再读 m4a/mp3。"""
        if not path:
            return
        if not hasattr(self, "_temp_play_paths"):
            self._temp_play_paths = []
        self._temp_play_path = path
        self._temp_play_paths.append(path)

    def _cleanup_temp_play_paths(self):
        """清理播放/拼接用临时 WAV。"""
        paths = []
        if hasattr(self, "_temp_play_paths"):
            paths.extend(self._temp_play_paths or [])
        if hasattr(self, "_temp_play_path") and self._temp_play_path:
            paths.append(self._temp_play_path)

        seen = set()
        for temp_path in paths:
            if not temp_path or temp_path in seen:
                continue
            seen.add(temp_path)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

        self._temp_play_path = None
        self._temp_play_paths = []

    def _ensure_playable_wav(self, file_path, prefix="play_tmp_"):
        """返回 soundfile/playback_worker 可稳定读取的 WAV 路径。"""
        if not file_path:
            raise RuntimeError("音频路径为空。")
        if file_path.lower().endswith(".wav"):
            return file_path
        tmp = _convert_to_temp_wav(file_path, prefix=prefix)
        self._remember_temp_play_path(tmp)
        return tmp

    def _play_file(self, file_path, tag):
        if not file_path or not os.path.exists(file_path):
            self.log_message(f"❌ {tag}文件不存在，无法播放。")
            return

        # 显示文件信息
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.log_message(f"📂 准备播放 {tag}: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
        
        play_path = file_path
        # 如果不是 WAV，先转换为临时 WAV；后续拖动进度也必须使用该 WAV，不能再读 m4a/mp3。
        try:
            play_path = self._ensure_playable_wav(file_path, prefix="play_tmp_")
        except Exception as e:
            self.log_message(f"⚠️ 无法为播放生成临时 WAV，尝试直接播放原文件: {e}")

        self.current_play_path = play_path
        ok = self._start_playback(play_path, 0)
        if ok:
            self.log_message(f"▶️ 正在播放 {tag}: {os.path.basename(file_path)}")
        else:
            self.log_message(f"❌ 播放启动失败。")

    def play_original(self):
        self._play_file(self.current_filepath, "原声")

    def play_converted(self):
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("❌ 转换后文件不存在，无法播放。")
            return

        play_path = self.converted_filepath
        # 如果用户选择播放时保留背景噪声，生成临时混合文件
        if self.combo_mode.currentIndex() == 0 and self.chk_keep_noise.isChecked():
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                tmp_path = tmp.name
                tmp.close()

                # 读取 converted 和原始音频。原始输入可能是 m4a/mp3，必须先转 WAV 再给 soundfile 读。
                conv_read_path = self._ensure_playable_wav(self.converted_filepath, prefix="mix_conv_")
                orig_read_path = self._ensure_playable_wav(self.current_filepath, prefix="mix_orig_")
                y_conv, sr_conv = sf.read(conv_read_path, dtype='float64')
                x_orig, sr_orig = sf.read(orig_read_path, dtype='float64')
                if sr_conv != sr_orig:
                    # 重采样原始到 converted 的采样率
                    x_orig = librosa.resample(x_orig, orig_sr=sr_orig, target_sr=sr_conv)
                    sr = sr_conv
                else:
                    sr = sr_conv

                # 载入 mask（优先使用 sidecar），否则生成一个简单 VAD mask
                mask_path = self.converted_filepath + ".mask.npy"
                mask = None
                if os.path.exists(mask_path):
                    try:
                        mask = np.load(mask_path)
                    except Exception:
                        mask = None

                if mask is None:
                    try:
                        _, mask = apply_energy_vad(x_orig, sr, frame_ms=int(self.input_frame_ms.text()), threshold_ratio=max(1,min(15,int(self.input_vad_threshold.text())))/100.0)
                    except Exception:
                        mask = np.ones_like(x_orig)

                # 调整长度
                minlen = min(len(y_conv), len(x_orig), len(mask))
                y_conv = np.asarray(y_conv[:minlen], dtype=np.float64)
                x_orig = np.asarray(x_orig[:minlen], dtype=np.float64)
                mask = np.asarray(mask[:minlen], dtype=np.float64)

                # 混合：语音部分使用转换后的，噪声部分从原始音频取一部分（gain 0.3）
                background_gain = 0.35
                mixed = y_conv * mask + x_orig * (1.0 - mask) * background_gain
                # 防止溢出
                peak = np.max(np.abs(mixed))
                if peak > 1.0:
                    mixed = mixed / peak * 0.95

                sf.write(tmp_path, mixed, int(sr))
                play_path = tmp_path
                # 记录临时文件用于后续清理
                self._remember_temp_play_path(tmp_path)
            except Exception as e:
                self.log_message(f"⚠️ 生成带背景混合文件失败，使用原始转换文件播放: {e}")
                play_path = self.converted_filepath

        self._play_file(play_path, "转换后")

    def play_anonymized_variant(self, variant):
        path = self._get_anonymized_path(variant)
        if not path:
            self.log_message(f"❌ {self._variant_display_name(variant)}文件不存在，无法播放。")
            return
        self.converted_filepath = path
        self._play_file(path, self._variant_display_name(variant))

    def play_selected_anonymized(self):
        variant = self._selected_anonymized_variant()
        if not variant:
            self.log_message("❌ 没有可播放的匿名化版本。")
            return
        self.play_anonymized_variant(variant)

    def save_selected_anonymized(self):
        if self.combo_mode.currentIndex() == 3:
            self._sync_selected_anonymized_output()
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("❌ 没有可另存的匿名化结果。")
            return
        suggested = os.path.basename(self.converted_filepath)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "另存匿名化结果", suggested, "WAV Files (*.wav);;All Files (*)"
        )
        if not save_path:
            return
        try:
            shutil.copy2(self.converted_filepath, save_path)
            self.log_message(f"✅ 匿名化结果已另存为: {save_path}")
        except Exception as e:
            self.log_message(f"❌ 另存匿名化结果失败: {e}")

    def stop_playback(self):
        """完全停止播放并清理资源"""
        # 终止播放进程
        if self.play_proc is not None:
            try:
                # 强制终止进程
                self.play_proc.terminate()
                # 等待进程完全退出（最多 500ms）
                try:
                    self.play_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.play_proc.kill()
                    self.play_proc.wait()
            except Exception as e:
                self.log_message(f"⚠️ 清理播放进程时出错: {e}")
            finally:
                self.play_proc = None
        
        # 停止定时器
        self.play_timer.stop()
        
        # 清空播放相关的缓存
        self.play_data = None
        self.play_obj = None
        
        # 重置波形显示
        self._reset_wave_meter()
        
        self.log_message("⏹️ 播放已停止。")

        # 删除播放/混合用临时 WAV（如果有）
        self._cleanup_temp_play_paths()

    def seek_position(self, value_ms):
        if not self.current_play_path:
            return
        ok = self._start_playback(self.current_play_path, value_ms)
        if not ok:
            self.log_message("拖动定位失败，已停止播放。")

    def _start_playback(self, file_path, offset_ms):
        """启动音频播放，offset_ms 为从文件开始的偏移量（毫秒）"""
        # 首先完全停止任何正在进行的播放
        if self.play_proc is not None:
            try:
                self.play_proc.terminate()
                try:
                    self.play_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.play_proc.kill()
                    self.play_proc.wait()
            except Exception:
                pass
            self.play_proc = None
        
        try:
            # 读取音频文件进行时间计算
            data, rate = sf.read(file_path, dtype="int16")
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]

            # simpleaudio 对通道和内存布局比较敏感，先规整数据。
            if channels > 2:
                data = data[:, :2]
                channels = 2

            data = np.ascontiguousarray(data, dtype=np.int16)

            # 验证偏移量
            start_sample = int(max(offset_ms, 0) * rate / 1000)
            if start_sample >= len(data):
                self.log_message("⚠️ 播放位置超出音频长度。")
                return False

            # 存储播放状态
            self.play_data = data
            self.play_rate = int(rate)
            self.play_channels = channels
            self.play_total_ms = int(len(data) * 1000 / rate)
            self.play_start_ms = int(offset_ms)
            self.play_start_epoch = time.time()

            self.slider_position.setRange(0, self.play_total_ms)

            # 获取 playback_worker.py 的路径
            worker_path = os.path.join(os.path.dirname(__file__), "playback_worker.py")
            if not os.path.exists(worker_path):
                self.log_message("❌ 播放脚本 playback_worker.py 不存在。")
                return False

            # Windows 下创建隐藏进程
            creation_flags = 0
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            # 启动播放进程（subprocess 隔离）
            self.play_proc = subprocess.Popen(
                [sys.executable, worker_path, file_path, str(int(offset_ms))],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
            
            # 启动定时器用于进度追踪
            self.play_timer.start()
            return True
            
        except Exception as e:
            # 播放失败，清理资源
            self.play_proc = None
            self.play_timer.stop()
            self._reset_wave_meter()
            self.log_message(f"❌ 播放异常: {e}")
            return False

    def _on_playback_timer(self):
        """处理播放进度更新和进程监控"""
        if self.play_proc is None:
            self.play_timer.stop()
            self._reset_wave_meter()
            return

        # 计算当前播放位置
        elapsed_ms = int((time.time() - self.play_start_epoch) * 1000)
        current_ms = min(self.play_start_ms + elapsed_ms, self.play_total_ms)

        # 更新进度条
        self.slider_position.blockSignals(True)
        self.slider_position.setValue(current_ms)
        self.slider_position.blockSignals(False)
        
        # 更新时间显示
        self.label_time.setText(f"{self._fmt_ms(current_ms)} / {self._fmt_ms(self.play_total_ms)}")
        
        # 更新波形显示
        self._update_wave_meter(current_ms)

        # 检查播放进程状态
        rc = self.play_proc.poll()
        if rc is not None:
            # 进程已退出
            self.play_timer.stop()
            
            # 读取进程输出用于调试
            try:
                stdout, stderr = self.play_proc.communicate(timeout=0.1)
                if stderr:
                    error_msg = stderr.decode('utf-8', errors='ignore').strip()
                    if error_msg:
                        self.log_message(f"🔧 [播放进程] {error_msg}")
            except:
                pass
            
            self.play_proc = None
            self._reset_wave_meter()
            # 清理播放/混合用临时 WAV（如果存在）
            self._cleanup_temp_play_paths()
            
            if rc != 0:
                # 进程异常退出
                if rc == 3221225477:
                    # 特殊处理这个特定的崩溃代码
                    self.log_message(f"⚠️ 播放进程崩溃 (返回码: {rc}) - 可能是音频格式问题或内存错误")
                else:
                    self.log_message(f"⚠️ 播放进程异常退出，返回码: {rc}")

    def _update_wave_meter(self, current_ms):
        if self.play_data is None or self.play_rate <= 0:
            self._reset_wave_meter()
            return

        sample_idx = int(current_ms * self.play_rate / 1000)
        window_size = max(int(self.play_rate * 0.025), 1)
        left = max(sample_idx - window_size // 2, 0)
        right = min(sample_idx + window_size // 2, len(self.play_data))
        if right <= left:
            self._reset_wave_meter()
            return

        segment = self.play_data[left:right]
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim > 1:
            segment = np.max(np.abs(segment), axis=1)
        else:
            segment = np.abs(segment)

        peak = float(np.max(segment)) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(segment))) / 32768.0)
        level = min(1.0, 0.55 * (peak ** 0.5) + 0.45 * (rms ** 0.5))
        meter_value = int(level * 100.0)
        self.wave_meter.setValue(meter_value)
        self.label_wave_value.setText(f"{meter_value}%")
        self.waveform_strip.push_level(level)

    def _reset_wave_meter(self):
        self.wave_meter.setValue(0)
        self.label_wave_value.setText("0%")
        self.waveform_strip.reset()

    @staticmethod
    def _fmt_ms(ms):
        seconds = int(ms / 1000)
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VoiceChangerApp()
    ex.show()
    sys.exit(app.exec_())
