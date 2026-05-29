# EE328 Speech Signal Processing Project

本仓库包含 EE328《语音信号处理》课程项目的当前代码与阶段性结果。项目目标是对输入语音进行匿名化处理，在尽量保持可懂度和自然度的同时，让输出音色不再对应到一个明确、可识别的单一说话人。

## 当前状态

目前项目已经实现：

- 音频预处理（统一采样率 / 单声道 / 基础降噪 / 响度规整）
- 基于 FreeVC 的匿名音色转换
- 匿名目标配置化
- `female_leaning` / `male_leaning` 两类输出版本
- 候选评估与自动选择
- 语调平滑度惩罚项（用于避免句内音高跳变过大的候选被选中）

当前阶段最重要的结论是：

- 匿名性已经明显提升；
- female / male 双版本已经可以稳定生成；
- 当前最主要的剩余问题不是“像某个人”，而是**句内语调连续性仍不够自然**，尤其是 male 分支更容易出现跳变。

## 核心脚本

- `run_pipeline.py`：主流程入口
- `audio_preprocess.py`：预处理、降噪、静噪抑制
- `vc_anonymizer.py`：FreeVC 后端调用
- `vc_candidate_builder.py`：VC 候选构建
- `evaluate_anonymization.py`：候选评估与排序
- `evaluate_voiceprivacy.py`：VoicePrivacy 风格匿名性 / 可懂度评估
- `naturalness_postprocess.py`：后处理与自然度修正
- `ppg_tone_naturalizer.py`：PPG-inspired 中文声调自然化后处理
- `run_ppg_tone_experiment.py`：生成 PPG-tone 男声 / 女声实验输出
- `generate_report_evaluation.py`：生成报告用评估表格和可视化图
- `recording_demo_ui.py`：浏览器录音并自动匿名化的交互式 Demo UI

## 目标音色配置

- `vc_target_pool_female.json`：偏女声配置
- `vc_target_pool_male.json`：偏男声配置
- `vc_target_pool.json`：较早阶段的混池配置（保留作实验记录）

当前默认流程会读取：
- `vc_target_pool_female.json`
- `vc_target_pool_male.json`

并分别输出 female / male 两套结果。

## 运行方式

### 环境说明
当前代码默认在本地 `speech-anon310` 环境中运行，并依赖：

- Python 3.10
- `numpy`
- `scipy`
- `matplotlib`
- `ffmpeg`
- Coqui TTS (`TTS`)

### 直接运行
在项目根目录下执行：

```bash
python run_pipeline.py
```

默认会：
1. 读取项目中的测试输入；
2. 做预处理；
3. 分别生成 female / male 候选；
4. 对各候选打分；
5. 输出每个版本当前最优的匿名化结果。

也可以指定输出目录，例如：

```bash
python run_pipeline.py --work-root work_smooth_verify
```

### VoicePrivacy 风格评估
当前仓库也提供一套本地可复现的 VoicePrivacy 风格评估脚本：

```bash
python evaluate_voiceprivacy.py
```

默认会输出：
- `voiceprivacy_style_results.json`

说明：这里沿用了 VoicePrivacy 的两条核心指标轴：
- Privacy：ASV EER
- Utility：ASR WER

但由于当前项目只有一个受保护说话人、两条 source utterance，且中文文本参考来自 source 音频的 Whisper 转写，因此该结果适合做**项目内横向比较**，不应直接当作官方榜单分数理解。

## 输出目录说明

### 历史实验目录
- `work/`
- `work_pooled_verify/`
- `work_dual_verify/`
- `work_standard_probe/`

### 当前推荐结果目录
- `work_smooth_verify/`

其中当前最推荐试听的结果位于：

- `work_smooth_verify/final/preferred_variants/test_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/test_denoised_male_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_male_leaning.wav`

## VoicePrivacy 风格评估结果

当前对 `work_smooth_verify/final/preferred_variants/` 的结果，使用本地离线模型做了 VoicePrivacy 风格评估：
- ASV：SpeechBrain ECAPA，指标为 EER（越高表示越难被重新识别）
- ASR：faster-whisper small，指标为 WER（越低表示可懂度越好）

### 当前结果

| Variant | ASV EER ↑ | ASR WER ↓ | 解读 |
| --- | ---: | ---: | --- |
| source baseline | 0.000 | - | 原始语音可被稳定验证回原说话人 |
| `female_leaning` | 0.000 | 0.379 | 当前可懂度尚可，但在这组小样本上仍容易被 ASV 连回原说话人 |
| `male_leaning` | 0.417 | 0.345 | 当前在匿名性上更强，同时可懂度与 female 版本接近 |

### 逐条样例

- `test.wav`
  - female 版本 WER：0.550
  - male 版本 WER：0.500
- `绿色.m4a`
  - female 版本 WER：0.000
  - male 版本 WER：0.000

完整原始结果见：
- `voiceprivacy_style_results.json`

## 指标增强实验结果

根据“保持真人音色，同时尽量提高 ASV EER + ASR WER”的目标，新增了一条指标增强实验分支：

- `build_metric_attack_variants.py`：从 `work_smooth_verify` 的全部 FreeVC 候选生成 raw 组合和真实通话式后处理组合。
- `export_metric_attack_results.py`：导出推荐试听版本。
- 完整评估结果：`work_metric_attack/voiceprivacy_metric_attack_results.json`
- 推荐输出目录：`work_metric_attack/final/recommended/`

当前推荐版本：

| Variant | ASV EER ↑ | ASR WER ↑ | 说明 |
| --- | ---: | ---: | --- |
| previous `male_leaning` | 0.417 | 0.345 | 旧推荐结果 |
| `raw_metric_male` | 0.583 | 0.414 | 男声候选重选，不加通道失真 |
| `raw_metric_female` | 0.500 | 0.621 | 女声候选重选，不加通道失真 |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 男声当前推荐，干净电话通道式后处理 |
| `balanced_phone_clean_female` | 0.500 | 1.414 | 女声当前推荐，干净电话通道式后处理 |
| `mixed_metric_reference` | 0.583 | 1.310 | 跨性别混合指标参考，不作为双版本交付默认 |
| `max_metric_vowel_mask_reference` | 0.542 | 4.241 | 指标上限对照，容易触发 ASR 长段幻觉，听感风险更高 |

最推荐试听：

- `work_metric_attack/final/recommended/balanced_phone_clean_male/test_denoised_balanced_phone_clean_male.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_male/绿色_denoised_balanced_phone_clean_male.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/test_denoised_balanced_phone_clean_female.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/绿色_denoised_balanced_phone_clean_female.wav`

指标上限对照：

- `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/test_denoised_max_metric_vowel_mask_reference.wav`
- `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/绿色_denoised_max_metric_vowel_mask_reference.wav`

复现实验：

```bash
python build_metric_attack_variants.py
python evaluate_voiceprivacy.py --selection-glob 'work_metric_attack/final/preferred_variants/*_selections.json' --output-path work_metric_attack/voiceprivacy_metric_attack_results.json
python export_metric_attack_results.py
```

说明：这里的 ASR WER 按当前目标解释为“越高越能扰乱 ASR”。这和 VoicePrivacy 里通常把 WER 当 utility、越低越好的方向相反。

## PPG-inspired 中文声调自然化分支

当前 `codex/ppg-chinese-tone-naturalness` 分支新增了一条 PPG-inspired 后处理实验：

- 从 `balanced_phone_clean_male` / `balanced_phone_clean_female` 出发；
- 构造轻量 content posterior bottleneck；
- 分析中文 tone-like F0 contour；
- 对匿名化结果做保守的谱包络和声调能量自然化。

复现实验：

```bash
python run_ppg_tone_experiment.py
python evaluate_voiceprivacy.py --selection-glob 'work_ppg_tone/final/recommended/*/*_selections.json' --output-path work_ppg_tone/voiceprivacy_ppg_tone_results.json
```

当前结果：

| Variant | ASV EER ↑ | ASR WER ↑ | 说明 |
| --- | ---: | ---: | --- |
| `ppg_tone_male` | 0.583 | 0.655 | 男声指标基本保持，并加入声调自然化后处理 |
| `ppg_tone_female` | 0.500 | 0.690 | 女声匿名性保持，但 WER 低于原推荐女声 |

完整说明见 `PPG_TONE_EXPERIMENT.md`。

## 报告用可视化评估

如果需要为课程报告生成直观图表，可以运行：

```bash
python generate_report_evaluation.py
```

输出目录：

- `report_evaluation/`

其中包含：

- `asv_eer_bar.png`：ASV EER 隐私提升柱状图
- `asr_wer_bar.png`：ASR WER 扰动柱状图
- `source_similarity_reduction_bar.png`：与原说话人的 speaker similarity 下降比例
- `speaker_similarity_heatmap.png`：ECAPA speaker embedding 相似度热力图
- `privacy_utility_scatter.png`：隐私 / ASR 扰动折中散点图
- `green_f0_contours.png`：`绿色.m4a` 的 F0 / 声调曲线对照
- `REPORT_EVALUATION_SUMMARY.md`：可直接摘进报告的结果摘要

报告中最直观的结论是：原始语音 ASV EER 为 `0.000`，匿名化后最高达到 `0.583`；与原说话人的 speaker similarity 从 `0.716` 降到约 `0.060`，下降约 `91.5%`。

## 录音 Demo UI

如果需要课堂展示“开始录音 / 结束录音 / 自动匿名化”，可以运行：

```bash
python recording_demo_ui.py
```

然后打开：

```text
http://127.0.0.1:7860
```

浏览器会请求麦克风权限。录音结束后，后端会自动生成：

- `Metric+phone male`
- `Metric+phone female`
- `PPG-tone male`
- `PPG-tone female`

输出会保存在 `work_recording_demo/`。这是“录完后处理”的准实时展示，不是边说边输出的低延迟 streaming voice conversion。

## 当前技术路线总结

项目在匿名目标上经历了两个阶段：

### 第一阶段：直接 mixed-pool target
将多个参考说话人的片段合成为 pooled target，再直接送入 FreeVC。

**结果：**
- 匿名性更强；
- 但句内语调连续性明显下降；
- 会出现较重的“拼接感”和音高突变。

### 第二阶段：single-reference group + smoothness selection
当前采用的方案是：
- female / male 各自维护一组参考说话人；
- 每个参考分别生成 VC 候选；
- 在评估中加入基频跳变惩罚；
- 从同组候选中选择更平滑的输出。

**当前效果：**
- 比直接 pooled target 更稳定；
- female 分支效果优于 male 分支；
- 仍然存在进一步优化空间。

## 当前已知问题

目前最主要的问题是：

- 句子内部的语调还不够连贯；
- 某些音节连接处仍有明显的 pitch 跳变；
- male 分支的 prosody continuity 仍不如 female 分支自然。

这说明当前瓶颈更多来自 VC 后端在匿名目标条件下的 prosody 保真，而不是简单的后处理或降噪不足。

## 下一步工作

下一步应重点继续解决：

1. **句内语调突变**
2. **male 分支的流畅度**
3. **更稳定的匿名目标策略**
4. **必要时尝试更适合匿名化场景的 VC 后端**

## 阶段汇报

更详细的阶段性总结见：

- `STAGE_PROGRESS_REPORT.md`
