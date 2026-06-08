# 中文语音音色匿名化项目

本项目实现一个面向中文语音的音色匿名化 demo：用户在网页端录音，系统将原始说话人的音色转换成目标男声音色，同时尽量保持真人听感和语义内容完整。

最终版本采用三条处理链路：

1. **FreeVC baseline**：用 FreeVC 将原始语音的内容特征与目标男声 speaker embedding 结合，生成基础匿名音频。
2. **Metric+clarity**：在 FreeVC 输出上加入目标音色池评分和清晰度保护，优先选择隐私性与保真度更平衡的候选音色。
3. **PPG-tone**：在 Metric+clarity 输出上进一步做 PPG-inspired 内容约束与中文声调自然化，改善吞字、语调突变和机械感。

## 目录结构

```text
期末proj/
├── 代码/                  # 最终可运行核心代码
├── 文档/                  # 交接文档、实验说明、结果图和展示音频
├── README.md              # 当前项目说明
└── .gitignore
```

核心代码在 `代码/` 下：

| 文件 | 作用 |
|---|---|
| `recording_demo_ui.py` | Web 录音、匿名化、即时评估主入口 |
| `audio_preprocess.py` | 输入音频预处理、降噪、格式统一 |
| `vc_candidate_builder.py` | 调用 FreeVC 生成目标音色候选 |
| `privacy_target_optimizer.py` | 对目标音色池打分，选择 Balance 最优候选 |
| `build_metric_attack_variants.py` | `clarity_guard` 清晰度保护处理 |
| `ppg_tone_naturalizer.py` | PPG-tone 中文声调自然化处理 |
| `evaluate_voiceprivacy.py` | 说话人相似度、ASR WER 等评估工具 |
| `generate_report_evaluation.py` | 固定 benchmark 图表生成 |
| `prepare_external_male_targets.py` | 下载/整理 CMU ARCTIC 男声音色池 |
| `run_privacy_optimized_recording.py` | 对本地音频复现 Web UI 的匿名化流程 |
| `vc_target_pool_male_external.json` | 男声目标音色池配置 |

文档和展示材料在 `文档/` 下：

| 路径 | 内容 |
|---|---|
| `文档/HANDOFF.md` | 项目交接说明 |
| `文档/report_evaluation_male/` | 最终 benchmark 图表与指标表 |
| `文档/音频/绿色.m4a` | 原始展示录音 |
| `文档/展示音频/` | PPT 可用的原始/匿名化音频导出 |
| `文档/README_旧版.md` | 清理前的 README 归档 |

## 运行 Web Demo

进入代码目录：

```bash
cd 代码
```

启动网页：

```bash
python recording_demo_ui.py --port 7862
```

浏览器打开：

```text
http://127.0.0.1:7862/
```

如果想减少处理时间，可以降低每次评估的目标音色数量：

```bash
python recording_demo_ui.py --port 7862 --max-targets 5
```

## 目标音色池

当前目标池是男声池，配置文件为：

```text
代码/vc_target_pool_male_external.json
```

其中包含：

- 6 个 CMU ARCTIC 公共男声参考：`bdl`、`rms`、`jmk`、`awb`、`ksp`、`rxr`
- 3 个本地 lab 男声参考：`s2.wav`、`s5.wav`、`s6.wav`

如果本地没有 CMU 目标音频，可以重新生成：

```bash
cd 代码
/opt/anaconda3/envs/speech-anon310/bin/python prepare_external_male_targets.py --speakers bdl rms jmk awb ksp rxr --utterances-per-speaker 5 --target-seconds 9
```

生成的目标音频会放在：

```text
代码/external_voice_targets/male/
```

这个目录是本地数据目录，不提交到 Git。

## 评分逻辑

网页端会展示 Top 3 候选音色，并用 Balance 分数自动选择第一名作为默认 target：

```text
Balance = 0.50 * Privacy + 0.50 * Fidelity
Privacy = clamp((70 - standard_score) / 20)
Fidelity = 0.60 * naturalness + 0.25 * duration_match + 0.15 * processing_simplicity
```

含义：

- `Privacy`：匿名性，原声音色和匿名音频越不像越高。
- `Fidelity`：保真度，声音越自然、时长越稳定、处理越不过度越高。
- `Balance`：隐私性和保真度的折中分数，越高越适合作为目标音色。

即时评估中：

- `source_similarity` 越低，说明匿名后越不像原说话人。
- `standard_similarity_score` 是把 cosine similarity 映射到 0-100 后的相似度分数，越低越匿名。
- `ASR-WER` 在本项目中越低越好，表示内容保留越好。

## 固定 Benchmark

最终报告图表保存在：

```text
文档/report_evaluation_male/
```

如需重新生成：

```bash
cd 代码
python generate_report_evaluation.py --target-gender male --output-dir ../文档/report_evaluation_male --skip-embedding-heatmap
```

## 展示音频

PPT 可直接使用的音频在：

```text
文档/展示音频/green_m4a_demo/
```

包含原始音频、降噪音频，以及三种匿名化方法的输出音频。

## 备注

本项目的网页 demo 是“录音结束后再处理”，不是低延迟实时流式匿名化。单条网页录音不能得到严格可靠的 ASV-EER，因此展示时更推荐使用 source similarity、standard similarity score、ASR-WER 和固定 benchmark 图表来说明匿名效果与内容保留效果。
