import time
from pathlib import Path
from TTS.api import TTS

work = Path(r'd:/大三下/语音pro')
source = work / 'converted' / '中文test1.wav'
target = work / 'converted' / '5月24日.wav'
out = work / 'out_full_vc.wav'
model_name = 'voice_conversion_models/multilingual/vctk/freevc24'


def main():
    t0 = time.time()
    print('Loading model:', model_name)
    tts = TTS(model_name)
    t1 = time.time()
    print('Running voice conversion...')
    tts.voice_conversion_to_file(source_wav=str(source), target_wav=str(target), file_path=str(out))
    t2 = time.time()
    print('load_sec', round(t1 - t0, 2))
    print('convert_sec', round(t2 - t1, 2))
    print('elapsed_sec', round(t2 - t0, 2))
    print('output', out)


if __name__ == '__main__':
    main()
