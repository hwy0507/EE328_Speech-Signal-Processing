#!/usr/bin/env python3
"""录音脚本：支持定时录音和交互录音（按回车开始、Ctrl+C 停止）。
依赖：sounddevice, soundfile
用法示例：
  pip install sounddevice soundfile
  python record_chinese.py --duration 5 --out myrec.wav
  python record_chinese.py --interactive --out myrec.wav
"""

import argparse
import sys

try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    print("缺少依赖：", e)
    print("请先安装：pip install sounddevice soundfile")
    sys.exit(1)


def list_devices():
    print(sd.query_devices())


def record_duration(out, duration, samplerate=16000, channels=1):
    print(f"开始定时录音：{duration}s → {out} （采样率 {samplerate}）")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()
    sf.write(out, data, samplerate, subtype='PCM_16')
    print("录音已保存：", out)


def record_interactive(out, samplerate=16000, channels=1, blocksize=1024):
    print("交互录音模式：按回车开始录音，Ctrl+C 停止并保存。")
    input("准备好按回车开始... ")
    try:
        with sf.SoundFile(out, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as f:
            with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32', blocksize=blocksize) as stream:
                print('录音中，按 Ctrl+C 停止...')
                while True:
                    data, overflowed = stream.read(blocksize)
                    if overflowed:
                        print('警告：音频输入溢出', file=sys.stderr)
                    f.write(data)
    except KeyboardInterrupt:
        print('\n已停止录音。')
    except Exception as e:
        print('录音失败：', e)
        sys.exit(1)
    print('录音已保存：', out)


def main():
    p = argparse.ArgumentParser(description='录音脚本：支持定时或交互录音。')
    p.add_argument('--list-devices', action='store_true', help='列出可用音频设备')
    p.add_argument('--duration', type=float, default=None, help='定时录音时长（秒）')
    p.add_argument('--interactive', action='store_true', help='交互录音：按回车开始，Ctrl+C 停止')
    p.add_argument('--out', default='record.wav', help='输出文件路径（默认 record.wav）')
    p.add_argument('--sr', type=int, default=16000, help='采样率，默认 16000')
    p.add_argument('--channels', type=int, default=1, help='声道数，默认 1（单声道）')
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.duration is None and not args.interactive:
        p.print_help()
        print('\n说明：请使用 --duration 或 --interactive 之一来开始录音。')
        return

    if args.duration:
        record_duration(args.out, args.duration, samplerate=args.sr, channels=args.channels)
    else:
        record_interactive(args.out, samplerate=args.sr, channels=args.channels)


if __name__ == '__main__':
    main()
