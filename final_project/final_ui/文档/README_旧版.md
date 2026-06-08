# 语音变声与声线克隆系统

本项目是一个面向课程展示和实验验证的本地语音处理系统，集成了 **WORLD 声码器变声**、**FreeVC 声线克隆**、**OpenVoice 声线克隆**、**说话人相似度评估**，并融合了同学项目中的 **语音匿名化 / 隐私保护评估流程**。

项目提供 PyQt5 图形界面，支持拖拽音频、录音、试听、转换、相似度评估和结果报告生成，适合用于语音信号处理、电创课程展示和答辩演示。

---

## 1. 功能概览

| 模块 | 说明 | 主要文件 |
|---|---|---|
| WORLD 单音频变声 | 基于 WORLD 声码器提取 F0、频谱包络、非周期成分，实现男声/女声方向的音高与共振峰变换 | `voice_gui.py`, `high_quality_voice_changer.py` |
| FreeVC 声线克隆 | 使用 Coqui TTS FreeVC 模型，将源语音转换为目标参考音色 | `voice_gui.py`, `tools/clone_runner.py` |
| OpenVoice 声线克隆 | 使用 OpenVoice V2 tone color converter 进行零样本音色迁移 | `voice_gui.py`, `tools/openvoice_runner.py` |
| Speaker Similarity | 使用 OpenVoice speaker embedding 计算两段语音的说话人相似度 | `tools/speaker_similarity_runner.py` |
| 匿名化 / 隐私保护 | 调用融合后的同学项目流程，生成匿名化候选音频与评估 JSON | `tools/anonymization_runner.py`, `_external/EE328_Speech-Signal-Processing` |
| GUI 播放与录音 | 支持音频播放、进度条、录音、音量显示、临时格式转换 | `voice_gui.py`, `playback_worker.py`, `record_chinese.py` |

---

## 2. 项目结构

```text
语音pro/
├── README.md                         # 项目说明
├── requirements.txt                  # 主 GUI / WORLD 环境依赖
├── requirements_tts.txt              # 可选：FreeVC / Coqui TTS 环境依赖
├── requirements_openvoice.txt        # 可选：OpenVoice 环境依赖
├── env_map.json                      # 外部 Python 与模型路径配置
├── run_gui.bat / run_gui.ps1         # GUI 启动脚本
├── voice_gui.py                      # 主图形界面与流程调度
├── high_quality_voice_changer.py     # WORLD 变声核心逻辑
├── playback_worker.py                # 音频播放工作线程
├── record_chinese.py                 # 录音工具
├── tools/
│   ├── clone_runner.py               # FreeVC 外部环境调用脚本
│   ├── openvoice_runner.py           # OpenVoice 外部环境调用脚本
│   ├── speaker_similarity_runner.py  # 说话人相似度评估脚本
│   └── anonymization_runner.py       # 匿名化融合调用脚本
├── checkpoints_v2/                   # OpenVoice V2 checkpoints，本地大文件，不建议直接上传 GitHub
├── ffmpeg/                           # 本地 ffmpeg，可选
└── _external/
    └── EE328_Speech-Signal-Processing/  # 同学项目，可选，用于匿名化融合
```

> 注意：`voice_env_clean/`、`.venv/`、`checkpoints_v2/`、输出音频、临时目录等通常不应提交到 GitHub。

---

## 3. 环境建议

建议使用 **3 套独立环境**，避免依赖冲突：

| 环境 | Python 建议 | 用途 |
|---|---|---|
| 主 GUI / WORLD 环境 | Python 3.10 ~ 3.12 | 运行 `voice_gui.py`、WORLD 变声、播放、录音、画图 |
| FreeVC 环境 | Python 3.10 | 运行 Coqui TTS / FreeVC 声线克隆 |
| OpenVoice 环境 | Python 3.10 | 运行 OpenVoice 转换与 speaker similarity |

如果只演示 WORLD 变声，只安装主环境即可；如果要演示声线克隆和相似度评估，需要额外配置 FreeVC / OpenVoice 环境。

---

## 4. 快速启动：主 GUI / WORLD 环境

### 4.1 创建并激活虚拟环境

Windows CMD：

```cmd
cd /d "d:\大三下\语音pro"
python -m venv voice_env_clean
voice_env_clean\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd "d:\大三下\语音pro"
python -m venv voice_env_clean
.\voice_env_clean\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 启动 GUI

推荐直接运行：

```cmd
run_gui.bat
```

或 PowerShell：

```powershell
.\run_gui.ps1
```

也可以手动启动：

```cmd
python voice_gui.py
```

---

## 5. FreeVC 声线克隆环境（可选）

FreeVC 依赖 Coqui TTS，建议单独环境安装。

```cmd
conda create -n tts_py310 python=3.10 -y
conda activate tts_py310
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_tts.txt
```

安装完成后，在 `env_map.json` 中配置该环境 Python 路径：

```json
{
  "声线克隆（双音频）": "C:\\PATH\\TO\\tts_py310\\python.exe"
}
```

FreeVC 调用链路：

```text
voice_gui.py
  -> tools/clone_runner.py
    -> TTS.api.TTS("voice_conversion_models/multilingual/vctk/freevc24")
      -> tts.voice_conversion_to_file(...)
```

---

## 6. OpenVoice 环境（可选）

OpenVoice 用于 OpenVoice 声线克隆和 Speaker Similarity 说话人相似度评估。

```cmd
conda create -n openvoice_py310 python=3.10 -y
conda activate openvoice_py310
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_openvoice.txt
```

OpenVoice 本体和 checkpoints 可能需要按实际来源安装，例如：

```cmd
pip install -e path\to\OpenVoice
```

或者如果你已经把 OpenVoice 项目放到本地，请保证 `openvoice_py310` 环境中可以正常执行：

```cmd
python -c "import openvoice; print('OpenVoice OK')"
```

`env_map.json` 示例：

```json
{
  "声线克隆（双音频）": "C:\\PATH\\TO\\tts_py310\\python.exe",
  "声线克隆（OpenVoice）": "C:\\PATH\\TO\\openvoice_py310\\python.exe",
  "OpenVoice_Checkpoints": ".\\checkpoints_v2"
}
```

`checkpoints_v2` 目录需要包含：

```text
checkpoints_v2/
└── converter/
    ├── config.json
    └── checkpoint.pth
```

---

## 7. GUI 使用说明

### 7.1 WORLD 单音频变声

输入一段音频，系统使用 WORLD 声码器进行分析与重合成。

```text
输入音频
  -> 可选 VAD 静音/噪声抑制
  -> 可选第二阶段谱减噪
  -> WORLD 特征提取：F0 / 频谱包络 / 非周期成分
  -> 修改音高和共振峰
  -> WORLD 合成
  -> peak 防爆音归一化
  -> 输出 converted_*.wav
```

输出文件默认保存在输入音频同目录，命名为：`converted_<原文件名>.wav`。

### 7.2 声线克隆（双音频）

输入：

- 源音频：提供说话内容；
- 目标参考音频：提供目标说话人音色。

支持两种引擎：

| 引擎 | 特点 |
|---|---|
| FreeVC | 由 Coqui TTS 调用，使用 `voice_conversion_models/multilingual/vctk/freevc24` |
| OpenVoice | 支持 `tau` 参数控制音色迁移强度，支持 VAD 提取 speaker embedding |

输出文件默认命名为：`vc_<源音频名>_to_<目标音频名>.wav`。声线克隆完成后，系统会尝试自动计算 Speaker Similarity。

### 7.3 Speaker Similarity 说话人相似度

输入两段音频：参考音频与待评估音频。系统会提取 speaker embedding，计算 cosine similarity，并映射为 0~100 分。

| 分数 | 解释 |
|---|---|
| 85 ~ 100 | 高相似 |
| 70 ~ 85 | 中等相似 |
| < 70 | 相似度较低 |

结果会显示在 GUI 中，同时保存为：`similarity_<reference>_vs_<candidate>.json`。

### 7.4 匿名化 / 隐私保护模式

该模块调用融合后的同学项目流程，用于降低说话人可识别性，同时尽量保持语音可懂度。

典型输出：

```text
anon_<源音频名>_result.json
anon_<源音频名>_*.wav
```

如果未克隆或未放置同学项目，匿名化模式会提示缺少：`_external/EE328_Speech-Signal-Processing`。

---

## 8. 项目实际归一化策略

本项目里需要区分 **WORLD 变声输出归一化** 和 **声线克隆输出归一化**。

### 8.1 WORLD 变声模块

WORLD 变声模块实际使用的是 **peak 防爆音归一化**：

```python
peak = np.max(np.abs(y))
if peak > 0.95:
    y = y * (0.95 / peak)
```

含义：

- 只在最大峰值超过 `0.95` 时缩小音频；
- 不会主动放大小音量；
- 主要目的是防止削波、爆音和保存 WAV 时失真。

严格来说，这是 **Peak Protection / Peak Limiting**，不是 RMS 或 LUFS 响度统一。

### 8.2 FreeVC / OpenVoice 声线克隆模块

当前声线克隆输出没有额外项目级归一化：

- FreeVC：直接调用 `tts.voice_conversion_to_file(...)` 写出；
- OpenVoice：直接调用 `tcc.convert(...)` 写出。

因此声线克隆输出音量主要由模型内部决定。

### 8.3 播放混音临时文件

GUI 播放混音时，为避免混音结果溢出，会在临时播放文件中做 peak 保护：

```python
if peak > 1.0:
    mixed = mixed / peak * 0.95
```

这只影响播放临时文件，不代表原始声线克隆输出文件被修改。

---

## 9. 参数建议

### 9.1 WORLD 变声参数

| 参数 | 推荐范围 | 说明 |
|---|---|---|
| `pitch_ratio` | 1.65 ~ 1.90 | 控制基频升高倍数 |
| `formant_ratio` | 1.12 ~ 1.18 | 控制共振峰平移幅度 |

推荐组合：

```python
convert_gender_high_quality(
    input_wav_path="input.wav",
    output_wav_path="output.wav",
    pitch_ratio=1.75,
    formant_ratio=1.15,
)
```

女声转男声可使用较低比例：

```python
convert_gender_high_quality(
    input_wav_path="input.wav",
    output_wav_path="output.wav",
    pitch_ratio=0.65,
    formant_ratio=0.85,
)
```

### 9.2 OpenVoice tau 参数

| tau | 效果 |
|---|---|
| 0.20 ~ 0.25 | 更稳、更自然 |
| 0.30 | 默认平衡值 |
| 0.35 ~ 0.40 | 更接近目标音色，但可能增加机械感 |

---

## 10. 常见问题

### Q1：只想运行 GUI，需要装 FreeVC 和 OpenVoice 吗？

不需要。只运行 WORLD 单音频变声时，安装 `requirements.txt` 即可。

### Q2：为什么 README 里建议多个环境？

因为 PyQt5、WORLD、Coqui TTS、OpenVoice 的依赖版本差异较大。拆成多个环境可以减少冲突，方便答辩时稳定运行。

### Q3：声线克隆失败怎么办？

检查：

1. `env_map.json` 中 Python 路径是否正确；
2. 对应环境是否能 `import TTS` 或 `import openvoice`；
3. `checkpoints_v2/converter/config.json` 和 `checkpoint.pth` 是否存在；
4. 输入音频是否能被 ffmpeg / librosa 正常读取。

### Q4：Speaker Similarity 为什么会失败？

常见原因：OpenVoice 环境未配置、checkpoints 缺失、音频太短、音频格式无法解码。项目中已经对过短音频做了自动补长兜底，但仍建议参考音频不少于 3~6 秒。

### Q5：哪些文件不建议上传 GitHub？

不建议上传：

```text
voice_env_clean/
.venv/
.venv-1/
voice_env/
checkpoints_v2/
*.wav
*.mp3
*.m4a
*.json
*_v2_*/
_anon_work/
```

建议通过 `.gitignore` 排除本地环境、模型权重和实验输出。

---

## 11. 答辩说明口径

可以这样介绍：

> 本系统不是简单调速或变调，而是把语音分成声源和声道两部分处理。WORLD 模块通过 F0、频谱包络和非周期成分解耦，实现可解释的传统声学变声；FreeVC 和 OpenVoice 模块提供深度学习声线克隆能力；Speaker Similarity 模块用说话人嵌入向量进行客观评估；匿名化模块则进一步评估在保护说话人隐私的同时保持语音可用性的效果。

技术亮点：

1. 传统声码器与深度学习声线转换融合；
2. GUI 一体化调度多环境；
3. 自动格式转换、录音、播放与可视化；
4. Speaker Similarity 客观评估闭环；
5. 融合同学项目的匿名化与隐私保护流程。

---

## 12. 参考

- WORLD Vocoder: Morise et al., WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications
- Coqui TTS / FreeVC
- OpenVoice / OpenVoice V2
- Speaker Embedding / Cosine Similarity
- VoicePrivacy / Speaker Anonymization 相关任务

---

## 13. 项目状态

- 主 GUI / WORLD 变声：可运行
- FreeVC 声线克隆：需单独环境
- OpenVoice 声线克隆：需单独环境与 checkpoints
- Speaker Similarity：依赖 OpenVoice 环境
- 匿名化融合：依赖 `_external/EE328_Speech-Signal-Processing`

最后更新：2026-06-05
