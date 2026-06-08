import time, torch
from TTS.api import TTS

t0 = time.time()
print('loading model...')
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
t1 = time.time()
text = "你好，这是一次GPU性能测试。"
out = "d:/大三下/语音pro/out_gpu_test.wav"
print('synthesizing...')
speaker = None
try:
    if hasattr(tts, 'speakers') and tts.speakers:
        # speakers may be a list or dict
        if isinstance(tts.speakers, dict):
            speaker = list(tts.speakers.keys())[0]
        else:
            speaker = tts.speakers[0]
        print('selected speaker', speaker)
except Exception:
    speaker = None

if speaker:
    # try to auto-select language if model is multilingual
    language = None
    try:
        if hasattr(tts, 'languages') and tts.languages:
            if isinstance(tts.languages, dict):
                language = list(tts.languages.keys())[0]
            else:
                language = tts.languages[0]
        elif hasattr(tts, 'langs') and tts.langs:
            if isinstance(tts.langs, dict):
                language = list(tts.langs.keys())[0]
            else:
                language = tts.langs[0]
        print('selected language', language)
    except Exception:
        language = None

    if language:
        tts.tts_to_file(text=text, file_path=out, speaker=speaker, language=language)
    else:
        tts.tts_to_file(text=text, file_path=out, speaker=speaker)
else:
    # fallback: try with speaker_wav if available or raise
    tts.tts_to_file(text=text, file_path=out)
if torch.cuda.is_available():
    torch.cuda.synchronize()

t2 = time.time()
print('load_time', t1 - t0)
print('synthesis_time', t2 - t1)
print('total_time', t2 - t0)
print('cuda_available', torch.cuda.is_available(), torch.version.cuda)
print('out_file', out)
try:
    print('memory_allocated', torch.cuda.memory_allocated(0))
    print('max_memory_allocated', torch.cuda.max_memory_allocated(0))
except Exception as e:
    print('mem_err', e)
