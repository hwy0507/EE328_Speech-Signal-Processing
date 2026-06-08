# 🎤 WORLD 声码器高质量变声项目 - 项目结构与快速导航

## 📁 项目文件导航

```
语音pro/
├── 📄 high_quality_voice_changer.py   ⭐ 核心库文件
│   ├─ convert_gender_high_quality()   转换单个文件
│   └─ batch_convert()                 批量转换多个文件
│
├── 📄 interactive_demo.py              🎯 交互式界面（推荐新手）
│   └─ 友好的菜单驱动界面
│
├── 📄 examples.py                      📚 示例代码集合
│   ├─ 示例 1：基础转换
│   ├─ 示例 2：自定义参数
│   ├─ 示例 3：批量处理
│   ├─ 示例 4：女→男转换
│   └─ 示例 5：参数对比
│
├── 📄 config.py                        ⚙️  项目配置文件
│   ├─ 默认参数设置
│   ├─ 预设参数组合
│   └─ 验证函数
│
├── 📄 README.md                        📖 完整项目文档
│   ├─ 项目介绍
│   ├─ 使用说明
│   ├─ 参数调节指南
│   ├─ 技术原理
│   └─ FAQ
│
├── 📄 QUICK_START.md                   🚀 本文件 - 快速开始
│
├── 📄 requirements.txt                 📦 依赖包列表
│
├── 📁 voice_env_clean/                 🐍 Python 虚拟环境（单音 WORLD）
│   └─ (pyworld, librosa, 等)
│
├── 📁 input_audio/                     📥 输入音频目录（自行创建）
│   └─ place_your_wav_files_here.txt
│
└── 📁 output_audio/                    📤 输出音频目录（自动生成）
    └─ (转换后的文件会保存在这里)
```

## ⚡ 快速开始（3 步）

### 步骤 1️⃣：创建并激活虚拟环境

```bash
cd "d:\大三下\语音pro"
python -m venv voice_env_clean
voice_env_clean\Scripts\activate
pip install -r requirements.txt
```

### 步骤 1.5️⃣：声线克隆环境（可选）

如果要用声线克隆（FreeVC / OpenVoice），需要额外创建两个 Python 3.10 环境：

```cmd
conda create -n tts_py310 python=3.10
conda activate tts_py310
pip install -r requirements_tts.txt

conda create -n openvoice_py310 python=3.10
conda activate openvoice_py310
pip install -r requirements_openvoice.txt
```

然后更新 `env_map.json` 为你本机的 Python 路径与 checkpoints 路径。

你会看到命令行前面出现 `(voice_env_clean)`，说明激活成功。

### 步骤 2️⃣：选择使用方式

#### 方式 A：🎯 交互式界面（推荐新手）

```bash
python interactive_demo.py
```

然后按照菜单选择操作。

#### 方式 B：💻 在 Python 脚本中使用

创建一个文件 `my_convert.py`：

```python
from high_quality_voice_changer import convert_gender_high_quality

# 单个文件转换
convert_gender_high_quality(
    input_wav_path="my_voice.wav",      # 替换成你的文件
    output_wav_path="my_voice_female.wav",
    pitch_ratio=1.75,                   # 推荐默认值
    formant_ratio=1.15                  # 推荐默认值
)
```

然后运行：
```bash
python my_convert.py
```

#### 方式 C：📚 查看示例代码

```bash
python examples.py
```

查看 5 个不同使用场景的示例。

### 步骤 3️⃣：查看结果

转换后的音频文件会保存在当前目录（或你指定的输出路径）。

## 🎚️ 参数速查表

### 最常用的参数组合

| 效果 | pitch_ratio | formant_ratio | 代码示例 |
|------|-----------|--------------|---------|
| 轻微女化 | 1.65 | 1.12 | `convert_gender_high_quality(input, output, 1.65, 1.12)` |
| **自然女声** ⭐ | **1.75** | **1.15** | `convert_gender_high_quality(input, output)` # 默认值 |
| 高亢少女 | 1.90 | 1.18 | `convert_gender_high_quality(input, output, 1.90, 1.18)` |
| 女→男 | 0.65 | 0.85 | `convert_gender_high_quality(input, output, 0.65, 0.85)` |

## 🔧 调试和常见问题

### Q: 为什么转换很慢？

**A:** 这是正常的！WORLD 声码器需要：
- 提取基频（DIO + StoneMask）
- 提取频谱包络（CheapTrick）
- 提取非周期成分（D4C）
- 重新合成音频

处理时间通常是音频时长的 10-30 倍。

### Q: 输入音频的要求？

**A:**
- ✅ 格式：WAV, MP3, FLAC（librosa 自动支持）
- ✅ 采样率：任意（16/44.1/48 kHz 都可）
- ✅ 时长：建议 >3 秒
- ❌ 避免：背景噪音、回声

### Q: 听起来还是不够自然？

**A:** 调参！试试：
1. 降低 `pitch_ratio`（如 1.70）让声音不那么尖
2. 增加 `formant_ratio`（如 1.18）让音色更女性
3. 多试几个值找到"甜区"

### Q: 如何恢复原始声音？

**A:**
```python
convert_gender_high_quality(input, output, pitch_ratio=1.0, formant_ratio=1.0)
```

## 📖 文档导航

| 文档 | 用途 | 推荐阅读对象 |
|-----|------|-----------|
| **README.md** | 完整项目文档、原理、FAQ | 所有人 |
| **QUICK_START.md** | 快速上手指南（本文件） | 新手 |
| **high_quality_voice_changer.py** | 源代码 + 详细注释 | 开发者 |
| **examples.py** | 5 个实际使用示例 | 想要学习的人 |
| **config.py** | 参数配置和预设 | 高级用户 |

## 🚀 下一步：项目扩展方向

核心功能已经完成 ✅，这些是可以考虑的扩展方向：

### 1. 🎵 实时变声（直播/会议）
- 支持麦克风实时输入输出
- 集成到 Discord、OBS 等软件

### 2. 🎨 GUI 界面
- 拖拽文件进行转换
- 实时参数调节和效果预览
- 频谱可视化

### 3. 🎙️ 高级功能
- 多种变声预设（不仅是性别，还有年龄、气质等）
- 自动参数优化
- 音质评估

### 4. 🔊 性能优化
- GPU 加速
- 多线程并行处理
- 流式处理（分块转换）

### 5. 📊 分析工具
- 频谱可视化
- MOS 评分（音质自动评估）
- 参数推荐系统

## 💡 有用的命令

```bash
# 激活虚拟环境
voice_env_clean\Scripts\activate

# 退出虚拟环境
deactivate

# 检查已安装的包
pip list

# 更新依赖包
pip install --upgrade -r requirements.txt

# 运行测试
python -c "import pyworld; print('WORLD 可用')"

## 声线克隆（双音频）说明

声线克隆依赖独立的 Python 环境（推荐 Python 3.10 + Coqui TTS），不与 `voice_env_clean` 混装。
通过 `env_map.json` 配置克隆环境的 Python 路径。

# 查看帮助信息
python high_quality_voice_changer.py --help
```

## 🎓 学习资源

### 技术原理
- 📚 WORLD 原始论文：Morise et al., 2016
- 🔗 Librosa 文档：https://librosa.org/
- 🔗 NumPy 文档：https://numpy.org/

### 相关概念
- 基频提取（F0 Extraction）
- 频谱包络（Spectral Envelope）
- 共振峰（Formants）
- 声码器（Vocoder）

## 📞 支持

如有问题，可以：
1. 查看 README.md 的 FAQ 部分
2. 查看源代码中的详细注释
3. 运行 examples.py 查看实际例子
4. 调整 config.py 中的参数

---

**祝你使用愉快！🎉**

最后更新：2026年5月18日
