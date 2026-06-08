# Project Handoff

This project uses two Python environments:

1) Single-audio WORLD mode (local venv in this repo)
2) Voice clone mode (external Python environments for Coqui TTS and OpenVoice)

## 1) Single-audio WORLD environment

From the project root:

```cmd
python -m venv voice_env_clean
voice_env_clean\Scripts\activate
pip install -r requirements.txt
```

Start GUI:

```cmd
run_gui.bat
```

## 2) Voice clone environments (external)

Install Coqui TTS in a separate Python 3.10 environment (example: conda env `tts_py310`).
Install OpenVoice in another Python 3.10 environment (example: conda env `openvoice_py310`).

Then update env_map.json to point to your Python and checkpoints:

```json
{
  "声线克隆（双音频）": "C:\\PATH\\TO\\tts_py310\\python.exe",
  "声线克隆（OpenVoice）": "C:\\PATH\\TO\\openvoice_py310\\python.exe",
  "OpenVoice_Checkpoints": ".\\checkpoints_v2"
}
```

After that, the GUI will run clone mode via tools/clone_runner.py or tools/openvoice_runner.py in the external env.

## 3) Git LFS (required for checkpoints)

The OpenVoice checkpoints include files larger than 100MB. GitHub requires Git LFS for those.

```cmd
git lfs install
git lfs track "*.pth"
git lfs track "*.zip"
git add .gitattributes
```

## 4) Verification checklist

Use this checklist after a fresh clone:

```cmd
:: A) GUI (WORLD)
run_gui.bat

:: B) OpenVoice CLI (replace paths if needed)
<openvoice_py310>\python.exe tools\openvoice_runner.py --source converted\openvoice_source.wav --target converted\openvoice_target.wav --out converted\openvoice_result.wav --ckpt_dir checkpoints_v2

:: C) GUI (OpenVoice)
:: Open GUI -> 选择“声线克隆” -> 引擎选 OpenVoice -> 选择源/目标 -> 开始转换
```

## Notes

- Do NOT package or copy any virtual environment folders (voice_env_clean, voice_env, .venv*).
- This repo already contains ffmpeg/bin for format conversion.
- If you move the project, relative paths still work. Only env_map.json must be updated.
