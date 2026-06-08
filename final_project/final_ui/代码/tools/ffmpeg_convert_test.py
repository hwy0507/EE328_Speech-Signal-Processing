import os
import subprocess
import sys

def main(inp, out):
    here = os.path.dirname(os.path.dirname(__file__))
    ff = os.path.join(here, 'ffmpeg', 'bin', 'ffmpeg.exe')
    if not os.path.exists(ff):
        ff = 'ffmpeg'
    # On Windows, try to use the short (8.3) path for the input to avoid encoding issues
    ff_input = inp
    if os.name == 'nt':
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            r = ctypes.windll.kernel32.GetShortPathNameW(inp, buf, len(buf))
            if r:
                ff_input = buf.value
        except Exception:
            pass

    cmd = [ff, '-y', '-i', ff_input, '-ar', '16000', '-ac', '1', '-vn', out]
    print('Running:', ' '.join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('rc=', p.returncode)
    print('stdout=', p.stdout.decode(errors='ignore')[:1000])
    print('stderr=', p.stderr.decode(errors='ignore')[:1000])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ffmpeg_convert_test.py <input> <output>')
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
