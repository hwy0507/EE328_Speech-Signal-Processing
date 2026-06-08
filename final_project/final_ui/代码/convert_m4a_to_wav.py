#!/usr/bin/env python3
"""批量将目录中的 .m4a 文件转换为 16kHz PCM16 单声道 WAV
依赖：ffmpeg 可执行程序在 PATH 中，或安装 pydub（仍需 ffmpeg）。
用法：
  python convert_m4a_to_wav.py --src_dir . --out_dir converted --sr 16000 --channels 1
"""
import argparse
import os
import subprocess
from pathlib import Path


def convert_ffmpeg(in_path: Path, out_path: Path, sr=16000, channels=1):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(in_path),
        '-ar', str(sr),
        '-ac', str(channels),
        '-vn',
        '-acodec', 'pcm_s16le',
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode(errors='ignore')


def find_m4a_files(src_dir: Path):
    return list(src_dir.rglob('*.m4a')) + list(src_dir.rglob('*.mp4'))


def main():
    p = argparse.ArgumentParser(description='Convert .m4a/.mp4 to 16k PCM WAV')
    p.add_argument('--src_dir', default='.', help='搜索 .m4a 的根目录')
    p.add_argument('--out_dir', default='converted', help='输出目录')
    p.add_argument('--sr', type=int, default=16000, help='目标采样率')
    p.add_argument('--channels', type=int, default=1, help='目标声道数')
    args = p.parse_args()

    src = Path(args.src_dir).resolve()
    out = Path(args.out_dir).resolve()
    files = find_m4a_files(src)
    if not files:
        print('未找到 .m4a 或 .mp4 文件于', src)
        return
    print(f'找到 {len(files)} 个文件，开始转换到 {out} (sr={args.sr}, channels={args.channels})')
    results = []
    for f in files:
        rel = f.relative_to(src)
        out_file = out / rel.with_suffix('.wav')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ok, err = convert_ffmpeg(f, out_file, sr=args.sr, channels=args.channels)
        results.append((f, out_file, ok, err))
        print(f'{"OK" if ok else "ERR"}: {f} -> {out_file}')
        if err:
            print('  error:', err[:200])

    print('\n转换完成。')

if __name__ == '__main__':
    main()
