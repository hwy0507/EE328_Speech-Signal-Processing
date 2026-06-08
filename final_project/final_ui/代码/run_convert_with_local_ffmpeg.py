import os, subprocess
from pathlib import Path
python = r"C:\Users\zhangs\.conda\envs\tts_py310\python.exe"
work = Path('d:/大三下/语音pro').resolve()
ffbin = str(work / 'ffmpeg' / 'bin')
env = os.environ.copy()
env['PATH'] = ffbin + os.pathsep + env.get('PATH','')
cmd = [python, str(work / 'convert_m4a_to_wav.py'), '--src_dir', str(work), '--out_dir', str(work / 'converted'), '--sr', '16000', '--channels', '1']
print('Running:', ' '.join(cmd))
ret = subprocess.run(cmd, env=env)
print('Returncode', ret.returncode)
