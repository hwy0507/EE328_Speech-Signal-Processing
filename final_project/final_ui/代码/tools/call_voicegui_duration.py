from voice_gui import _audio_duration_seconds
p = r"D:/大三下/语音pro/中文test1.m4a"
print('Calling _audio_duration_seconds on', p)
try:
    d = _audio_duration_seconds(p)
    print('Duration:', d)
except Exception as e:
    print('Error:', repr(e))
