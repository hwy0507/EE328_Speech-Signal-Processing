import time, torch
from pathlib import Path
from TTS.api import TTS

work = Path('d:/大三下/语音pro')
speaker_wav = work / 'converted' / '中文test1.wav'
out = work / 'out_clone.wav'
text = '今天天气很好，我在做语音转换测试。'

def attempt_model(model_name, kwargs):
    print('Trying model:', model_name)
    t0 = time.time()
    tts = TTS(model_name)
    t1 = time.time()
    try:
        tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), file_path=str(out), **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        print('model', model_name, 'load_time', t1-t0, 'synthesis_time', t2-t1, 'total', t2-t0)
        return True
    except Exception as e:
        print('model', model_name, 'failed:', e)
        return False

# Try multilingual YourTTS first
ok = attempt_model('tts_models/multilingual/multi-dataset/your_tts', {'language':'zh'})
if not ok:
    # Fallback: try a Chinese model (may not support speaker_wav). Try a list of known Chinese models.
    chinese_candidates = [
        'tts_models/zh-CN/baker/tacotron2',
        'tts_models/zh-CN/tacotron2-DDC',
        'tts_models/zh-CN/vits',
    ]
    for m in chinese_candidates:
        ok = attempt_model(m, {})
        if ok:
            break

print('Output file (if generated):', out)
print('CUDA available:', torch.cuda.is_available(), torch.version.cuda)
