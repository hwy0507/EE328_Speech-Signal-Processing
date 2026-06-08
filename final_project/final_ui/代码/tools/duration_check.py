import sys
from pathlib import Path
p = Path(r"D:/大三下/语音pro/中文test1.m4a")
print('Testing file:', p)
print('Exists:', p.exists())

try:
    import soundfile as sf
    try:
        info = sf.info(str(p))
        print('soundfile.info:', info)
    except Exception as e:
        print('soundfile.info failed:', repr(e))
except Exception as e:
    print('import soundfile failed:', repr(e))

try:
    import librosa
    try:
        d = librosa.get_duration(path=str(p))
        print('librosa.get_duration:', d)
    except Exception as e:
        print('librosa.get_duration failed:', repr(e))
    try:
        y, sr = librosa.load(str(p), sr=None, mono=False)
        print('librosa.load: loaded, sr=', sr, 'shape=', getattr(y, 'shape', type(y)))
    except Exception as e:
        print('librosa.load failed:', repr(e))
except Exception as e:
    print('import librosa failed:', repr(e))

try:
    import audioread
    try:
        with audioread.audio_open(str(p)) as fh:
            dur = fh.duration
            print('audioread duration:', dur)
    except Exception as e:
        print('audioread failed:', repr(e))
except Exception as e:
    print('import audioread failed:', repr(e))

# Quick ffmpeg probe using subprocess if available
import subprocess
try:
    res = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(p)], capture_output=True, text=True)
    print('ffmpeg probe returncode', res.returncode)
    if res.stderr:
        print('ffmpeg stderr:', res.stderr.strip())
except Exception as e:
    print('ffmpeg probe failed:', repr(e))

print('\nDone')
