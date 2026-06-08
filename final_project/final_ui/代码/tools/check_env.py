import sys
import importlib.util
print('executable:', sys.executable)
print('version:', sys.version.splitlines()[0])
print('TTS_installed =', importlib.util.find_spec('TTS') is not None)
