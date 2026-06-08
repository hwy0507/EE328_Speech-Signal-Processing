import sys, os
cwd = os.getcwd()
print('cwd', cwd)
if cwd not in sys.path:
    sys.path.insert(0, cwd)
print('sys.path[0]=', sys.path[0])
print('Trying import voice_gui')
try:
    import voice_gui
    print('imported voice_gui')
    print('calling _audio_duration_seconds...')
    print(voice_gui._audio_duration_seconds(r'D:/大三下/语音pro/中文test1.m4a'))
except Exception as e:
    print('import or call failed:', repr(e))
