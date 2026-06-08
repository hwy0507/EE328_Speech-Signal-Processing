import urllib.request, zipfile, os, shutil, sys
from pathlib import Path

url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
work = Path('d:/大三下/语音pro').resolve()
zip_path = work / 'ffmpeg.zip'
extract_dir = work / 'ffmpeg_temp'
final_dir = work / 'ffmpeg'

print('Downloading', url)
urllib.request.urlretrieve(url, zip_path)
print('Downloaded to', zip_path)

if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir()

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

# Find the extracted top-level dir containing bin\ffmpeg.exe
candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
if not candidates:
    print('No extracted dir found')
    sys.exit(1)

# Prefer folder that contains 'bin/ffmpeg.exe'
ffmpeg_folder = None
for c in candidates:
    if (c / 'bin' / 'ffmpeg.exe').exists():
        ffmpeg_folder = c
        break

if ffmpeg_folder is None:
    ffmpeg_folder = candidates[0]

# Move or rename to final_dir
if final_dir.exists():
    shutil.rmtree(final_dir)
shutil.move(str(ffmpeg_folder), str(final_dir))
# Clean temp and zip
shutil.rmtree(extract_dir)
zip_path.unlink()

print('FFmpeg prepared at', final_dir)
print('ffmpeg executable:', final_dir / 'bin' / 'ffmpeg.exe')
