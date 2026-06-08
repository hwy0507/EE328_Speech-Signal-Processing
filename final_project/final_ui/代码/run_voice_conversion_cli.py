import subprocess, time
from pathlib import Path
work = Path('d:/大三下/语音pro')
ffmpeg = work / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
source = work / 'converted' / '中文test1.wav'
target_mp3 = work / '5月24日.mp3'
target_wav = work / 'converted' / '5月24日.wav'
out = work / 'out_clone_vc.wav'
cli = Path(r"C:\Users\zhangs\.conda\envs\tts_py310\Scripts\tts.exe")

# convert target mp3 -> wav
print('Converting target mp3 to wav...')
cmd = [str(ffmpeg), '-y', '-i', str(target_mp3), '-ar', '16000', '-ac', '1', '-vn', '-acodec', 'pcm_s16le', str(target_wav)]
subprocess.run(cmd, check=True)
print('Target converted:', target_wav)

# run voice conversion CLI
vc_model = 'voice_conversion_models/multilingual/vctk/freevc24'
cmd = [str(cli), '--model_name', vc_model, '--in_path', str(source), '--speaker_wav', str(target_wav), '--out_path', str(out)]
print('Running VC CLI:', ' '.join(cmd))
start = time.time()
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
end = time.time()
print('Returncode', proc.returncode)
print(proc.stdout[:2000])
print('Elapsed', end-start)
print('Output file:', out)
