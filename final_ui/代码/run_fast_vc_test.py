import subprocess, time
from pathlib import Path
from TTS.api import TTS

work = Path(r'd:/大三下/语音pro')
ffmpeg = work / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
source = work / 'converted' / '中文test1.wav'
target_mp3 = work / '5月24日.mp3'
target_wav = work / 'converted' / '5月24日.wav'
source_short = work / 'converted' / '中文test1_short3s.wav'
target_short = work / 'converted' / '5月24日_short3s.wav'
out = work / 'out_fast_vc.wav'


def run(cmd):
    print('RUN:', ' '.join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)


def trim(inp, outp, seconds=3):
    run([ffmpeg, '-y', '-i', inp, '-t', str(seconds), '-ar', '16000', '-ac', '1', '-vn', '-acodec', 'pcm_s16le', outp])


def main():
    if not target_wav.exists():
        trim(target_mp3, target_wav, 3)
    trim(source, source_short, 3)
    trim(target_wav, target_short, 3)

    model = 'voice_conversion_models/multilingual/vctk/freevc24'
    print('Loading VC model:', model)
    t0 = time.time()
    tts = TTS(model)
    t1 = time.time()
    tts.voice_conversion_to_file(source_wav=str(source_short), target_wav=str(target_short), file_path=str(out))
    t2 = time.time()
    print('load_sec', round(t1 - t0, 2))
    print('convert_sec', round(t2 - t1, 2))
    print('elapsed_sec', round(t2 - t0, 2))
    print('output', out)


if __name__ == '__main__':
    main()
