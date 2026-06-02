# EE328 Speech Signal Processing Project

本仓库是 EE328《语音信号处理》课程项目：**语音音色匿名化**。项目目标不是破坏语义内容，而是在尽量保持可懂度、自然度和真人音色的同时，让输出语音不再容易被识别为原说话人。

当前主线分支：

```text
codex/ppg-chinese-tone-naturalness
```

## 当前成果

已经完成：

- 基于 FreeVC 的语音匿名化主流程
- 男声目标的浏览器录音 Demo UI
- `FreeVC baseline`、`Metric+phone`、`PPG-tone` 三种方法对比
- 本次网页录音的即时评估
- 固定测试集的报告级评估图表
- PPG-inspired 中文声调自然化实验与调参记录
- 外部公共男声目标池和隐私/自然度联合候选选择

当前主推荐展示方式是：

1. 用 Web UI 现场录音。
2. 自动生成三条匿名语音。
3. 页面直接展示本次录音的音色匿名效果和内容保留效果。
4. 固定 benchmark 只作为报告参考，不当作本次录音结果。

## 方法概览

| 方法 | 含义 | 适合展示什么 |
| --- | --- | --- |
| `FreeVC baseline male` | 直接用 FreeVC 做男声目标转换 | 基础音色匿名能力 |
| `Metric+phone male` | FreeVC 候选筛选 + 干净电话信道后处理 | 更强的声纹隐藏，但可能损失可懂度 |
| `PPG-tone male` | 在 Metric+phone 基础上加入 PPG-inspired 中文声调自然化 | 中文音调自然度与内容保留折中 |

`Metric+phone` 可以理解为 FreeVC 的进阶增强版：先基于指标挑选匿名效果更好的 FreeVC 候选，再叠加 `phone_clean` 通话信道式后处理。

`PPG-tone` 是本分支的创新点：它不是完整神经 PPG 模型，而是轻量的 content bottleneck + Mandarin tone contour naturalization，用于让中文匿名语音的声调轮廓更自然。

## 外部男声目标池

为避免匿名效果受单一参考音色限制，当前 Web UI 默认使用：

```text
vc_target_pool_male_external.json
```

该目标池包含 6 个 CMU ARCTIC 公共男声参考和 3 个本地 lab 男声参考。音频文件由脚本下载/整理到 `external_voice_targets/male/`，该目录不提交到 Git。

准备或刷新目标池：

```bash
/opt/anaconda3/envs/speech-anon310/bin/python prepare_external_male_targets.py --speakers bdl rms jmk awb ksp rxr --utterances-per-speaker 5 --target-seconds 9
```

每次处理录音时，`privacy_target_optimizer.py` 会在目标池中选择声纹相似度低、但语速/音高变化不过度的候选。报告数值见：

```text
EXTERNAL_MALE_TARGET_POOL_RESULTS.md
```

## Web UI 使用

启动：

```bash
python recording_demo_ui.py --port 7862
```

当前默认就是 9 个男声参考的全池搜索。如果临时想加快处理，可以降低为：

```bash
python recording_demo_ui.py --port 7862 --max-targets 5
```

打开：

```text
http://127.0.0.1:7862
```

使用流程：

1. 点击 `开始录音`。
2. 说一段 3-10 秒中文。
3. 点击 `结束录音`。
4. 点击 `匿名化处理`。
5. 等待页面生成三条匿名化音频和即时评估结果。

输出保存在：

```text
work_recording_demo/
```

## Web UI 指标解释

网页端即时评估针对**本次录音**计算，重点是音色匿名和内容保留：

| 指标 | 方向 | 含义 |
| --- | --- | --- |
| `Source similarity` | 越低越好 | 匿名输出和本次原录音的声纹相似度 |
| `Similarity drop` | 越高越好 | 相似度下降比例，越高表示音色越不像原人 |
| `ASR WER` | 越低越好 | 匿名后内容被 ASR 听错的比例，越低表示内容保留越好 |
| `Content kept` | 越高越好 | `1 - WER`，内容保留比例 |
| `Timbre index` | 越高越好 | 综合音色匿名和内容保留的本地指数 |

注意：单条网页录音不能可靠计算 ASV EER。EER 需要一组 target / non-target trials，因此网页端用 `Source similarity` 和 `Similarity drop` 作为即时音色匿名证据。

## 固定 Benchmark

固定测试集用于报告，不代表每次网页录音的结果。

生成男声报告结果：

```bash
python generate_report_evaluation.py --target-gender male --output-dir report_evaluation_male --skip-embedding-heatmap
```

主要输出：

```text
report_evaluation_male/REPORT_EVALUATION_SUMMARY.md
report_evaluation_male/method_metrics.csv
report_evaluation_male/method_metrics.json
report_evaluation_male/identity_privacy_index_bar.png
```

固定 benchmark 当前结论：

- 原始语音 ASV EER 为 `0.000`，说明可被稳定识别回原说话人。
- 男声增强方法的 ASV EER 可达 `0.583`。
- `PPG-tone male` 的 source-speaker similarity 约为 `0.060`，相对原始 source similarity `0.716` 下降约 `91.5%`。
- 在 EER 都为 `0.583` 的同档位里，细排顺序为 `PPG-tone male`、`Metric+phone male`、`Raw metric male`。

## PPG-tone 调参

调参脚本：

```bash
python tune_ppg_tone_parameters.py
```

结果文档：

```text
PPG_TONE_TUNING_RESULTS.md
ppg_tone_tuning_report/tuning_metrics.csv
ppg_tone_tuning_report/figures/
```

调参结论：

- `balanced_phone_clean_male + strength=0.4` 仍是隐私优先默认。
- `strength=1.0/1.3/1.6` 主要改善 tone/naturalness proxy，但没有稳定提高 ASV/WER。
- 因此报告中可把 `strength=1.0` 作为自然度 ablation，不替换主结果。

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `recording_demo_ui.py` | Web 录音、匿名化、即时评估 |
| `prepare_external_male_targets.py` | 下载/整理 CMU ARCTIC 公共男声目标池 |
| `privacy_target_optimizer.py` | 在男声池中按隐私和自然度联合选择最佳候选 |
| `run_privacy_optimized_recording.py` | 对单个本地录音复现 Web UI 优化流程 |
| `vc_candidate_builder.py` | FreeVC 候选生成 |
| `build_metric_attack_variants.py` | Metric+phone 等后处理候选 |
| `ppg_tone_naturalizer.py` | PPG-inspired 中文声调自然化 |
| `run_ppg_tone_experiment.py` | PPG-tone 批量实验 |
| `evaluate_voiceprivacy.py` | 固定测试集 VoicePrivacy 风格评估 |
| `generate_report_evaluation.py` | 报告图表与表格生成 |
| `VOICEPRIVACY_RESULTS.md` | 指标结果解释 |
| `EXTERNAL_MALE_TARGET_POOL_RESULTS.md` | 外部男声池和最新低相似度结果 |
| `PPG_TONE_EXPERIMENT.md` | PPG-tone 实验说明 |
| `PPG_TONE_TUNING_RESULTS.md` | PPG-tone 调参结论 |
| `HANDOFF.md` | 给队友/后续 AI 的交接说明 |

## 环境说明

本机默认使用：

```text
/opt/anaconda3/envs/speech-anon310/bin/python3.10
```

主要依赖：

- Python 3.10
- ffmpeg / ffprobe
- FreeVC 相关环境
- SpeechBrain ECAPA speaker model
- faster-whisper small
- numpy / scipy / pandas / matplotlib / seaborn / flask

模型默认使用本地缓存，`evaluate_voiceprivacy.py` 和 `recording_demo_ui.py` 会设置离线 HuggingFace 环境变量。

## 结果解释原则

报告和展示中建议使用下面这套表述：

```text
本项目的目标是音色/声纹匿名化，而不是破坏语义关键词。好的结果应该同时满足：
1. 匿名输出与原说话人的 speaker similarity 明显下降；
2. 匿名输出仍保持较低 ASR WER，即语义内容尽量可懂；
3. 听感仍像真人录音，没有明显机械音或严重失真。
```

不要把“ASR 关键词被扰乱”单独当作成功目标。它最多说明 ASR 受影响，但如果 WER 过高，反而表示内容保留变差。
