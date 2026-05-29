# 项目交接说明（给新会话 / 其他 AI）

这份文档的目标是：让一个**完全不知道前文上下文的新会话**，或者另一个 AI 编程助手，在进入本项目后可以立刻接着当前工作继续做，而不需要重新摸索项目状态。

---

## 1. 这个项目在做什么

这是 EE328《语音信号处理》课程项目，目标是做**语音匿名化**：

- 输入一段原始语音；
- 输出一段仍然可懂、尽量自然；
- 但**音色不再对应到一个明确、可识别的单一说话人**的匿名语音。

当前后端主要基于：
- Coqui TTS 的 FreeVC

当前整体目标不是“完美拟真某个新说话人”，而是：
1. 匿名性明显增强；
2. 语义内容尽量保留；
3. 句内语调和连贯性尽量自然。

---

## 2. 先看什么文件

如果你是新 AI，建议按下面顺序读：

1. `README.md`
   - 看项目整体说明、运行方式、当前结果和目录结构。
2. `STAGE_PROGRESS_REPORT.md`
   - 看阶段性总结、当前结论、已知问题、下一步建议。
3. `HANDOFF.md`
   - 看当前交接信息、哪些结论已经验证、哪些方向不建议再走回头路。
4. `run_pipeline.py`
   - 看主流程怎么串起来。
5. `vc_candidate_builder.py`
   - 看匿名目标的构造策略和 female / male 双分支候选生成方式。
6. `evaluate_anonymization.py`
   - 看当前候选打分逻辑，尤其是 prosody 平滑度惩罚。
7. `evaluate_voiceprivacy.py`
   - 看当前 VoicePrivacy 风格评估怎么做。
8. `audio_preprocess.py`
   - 看降噪、预处理、轻量静噪抑制。
9. `naturalness_postprocess.py`
   - 看后处理和 source detail 回灌逻辑。
10. `PPG_TONE_EXPERIMENT.md`
   - 看 PPG-inspired 中文声调自然化分支的实现、结果和局限。
11. `ppg_tone_naturalizer.py` / `run_ppg_tone_experiment.py`
   - 看本分支新增的声调自然化代码与批量实验入口。

---

## 3. 当前主结论

### 已经做成的东西

目前已经完成：

- 统一采样率 / 单声道 / 基础降噪 / 响度规整
- 基于 FreeVC 的匿名音色转换
- 匿名目标配置化
- `female_leaning` / `male_leaning` 双版本输出
- 候选评估与自动选择
- 针对句内 pitch 跳变的 prosody smoothness penalty
- VoicePrivacy 风格的本地评估（ASV EER + ASR WER）

### 当前最重要的结论

当前最重要的已验证结论是：

1. **匿名性已经比最初版本明显更强。**
2. **female / male 双版本已经可以稳定生成。**
3. 如果目标改为“提高 ASV EER + ASR WER”，当前最有效的新分支是：
   - 先从全部 FreeVC single-reference 候选中做 metric-driven re-selection；
   - 再叠加真实通话式、非机器人化的轻量 channel postprocess。
4. 当前最主要的自然度剩余问题仍然是：
   - 句子内部语调连续性仍不够自然；
   - 某些音节连接处仍有明显突变；
   - `male_leaning` 分支更容易出现不够流畅的问题。
5. 2026-05-29 新增 PPG-inspired 中文声调自然化分支：
   - 分支名：`codex/ppg-chinese-tone-naturalness`
   - 新增 `ppg_tone_naturalizer.py` 和 `run_ppg_tone_experiment.py`
   - 它不是完整 neural PPG extractor，而是轻量 content posterior bottleneck + Mandarin tone contour 后处理。

---

## 4. 哪些技术路线已经验证过

### 路线 A：直接 mixed-pool target

做法：
- 把多个参考说话人的片段拼成 pooled target；
- 直接送进 FreeVC 作为 target。

结果：
- 匿名性确实更强；
- 但**句内 prosody continuity 明显下降**；
- 听感容易出现拼接感、音高跳变、说话一卡一卡。

结论：
- **这个方向不适合继续作为默认主线。**
- 可以保留为实验对照，但不要轻易再把它切回默认方案。

### 路线 B：single-reference group + smoothness selection

当前默认主线：
- female / male 各维护一个参考说话人组；
- 每个参考分别生成一个 VC candidate；
- 再用评估脚本从同组候选中挑更平滑的输出。

结果：
- 比直接 mixed-pool 更稳定；
- prosody continuity 更好；
- female 分支效果优于 male 分支。

结论：
- **这是当前推荐继续迭代的主线。**

---

## 5. 当前默认配置与推荐输出

### 目标配置文件

当前默认流程读取：

- `vc_target_pool_female.json`
- `vc_target_pool_male.json`

其中：
- `female_leaning` 当前更稳定地选中 `s3`
- `male_leaning` 当前更稳定地选中 `s2`

### 当前推荐试听结果

位于：

- `work_smooth_verify/final/preferred_variants/test_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/test_denoised_male_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_male_leaning.wav`

### 当前最推荐结果目录

- `work_smooth_verify/`

### 指标增强推荐结果目录

新增的 metric-attack 推荐输出位于：

- `work_metric_attack/final/recommended/balanced_phone_clean_male/test_denoised_balanced_phone_clean_male.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_male/绿色_denoised_balanced_phone_clean_male.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/test_denoised_balanced_phone_clean_female.wav`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/绿色_denoised_balanced_phone_clean_female.wav`

对照版本：

- `work_metric_attack/final/recommended/raw_metric_male/`
- `work_metric_attack/final/recommended/raw_metric_female/`
- `work_metric_attack/final/recommended/mixed_metric_reference/`
- `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/`

旧实验目录仅作参考：
- `work/`
- `work_pooled_verify/`
- `work_dual_verify/`
- `work_standard_probe/`

### PPG-tone 实验输出目录

PPG-inspired 中文声调自然化输出位于：

- `work_ppg_tone/final/recommended/ppg_tone_male/`
- `work_ppg_tone/final/recommended/ppg_tone_female/`

完整说明：

- `PPG_TONE_EXPERIMENT.md`
- `work_ppg_tone/voiceprivacy_ppg_tone_results.json`

---

## 6. 当前评估结果（VoicePrivacy 风格）

当前项目已经补充了本地 VoicePrivacy 风格评估，结果见：

- `voiceprivacy_style_results.json`

评估轴：
- Privacy：ASV EER
- Utility：ASR WER

当前结果：

| Variant | ASV EER ↑ | ASR WER ↓ | 解读 |
| --- | ---: | ---: | --- |
| source baseline | 0.000 | - | 原始语音可被稳定验证回原说话人 |
| `female_leaning` | 0.000 | 0.379 | 可懂度尚可，但在这组小样本上匿名性仍偏弱 |
| `male_leaning` | 0.417 | 0.345 | 当前匿名性更强，可懂度与 female 版本接近 |

逐条样例：
- `test.wav`：female WER = 0.550，male WER = 0.500
- `绿色.m4a`：female WER = 0.000，male WER = 0.000

### 如何理解这些数字

注意：
- 这套评估**沿用了 VoicePrivacy 的核心指标方向**；
- 但当前数据集只有 1 个受保护说话人、2 条 source utterance；
- 中文参考文本来自 source 音频的 Whisper 转写；
- 所以这些数字适合做**项目内横向比较**，不等同于官方挑战榜单分数。

### 目前可直接得出的结论

- 当前版本里，`male_leaning` 在匿名性上优于 `female_leaning`。
- `female_leaning` 仍然容易被 ASV 连回原说话人。
- 长句 `test.wav` 仍然是当前可懂度和流畅度的主要难点。

### 2026-05-28 metric-attack 结果

本轮用户目标改为“保持真人音色，同时尽量提高 ASV EER + ASR WER”。因此 ASR WER 在这里按攻击成功指标解释，方向为越高越好，不再按 utility 越低越好的常规 VoicePrivacy 解读。

完整结果：

- `work_metric_attack/voiceprivacy_metric_attack_results.json`
- `work_metric_attack/final/recommended/recommended_summary.json`

关键结果：

| Variant | ASV EER ↑ | ASR WER ↑ | 解读 |
| --- | ---: | ---: | --- |
| previous `male_leaning` | 0.417 | 0.345 | 旧推荐结果 |
| `raw_metric_male` | 0.583 | 0.414 | 男声候选重选，不加通道失真 |
| `raw_metric_female` | 0.500 | 0.621 | 女声候选重选，不加通道失真 |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 男声当前推荐，干净电话通道式后处理 |
| `balanced_phone_clean_female` | 0.500 | 1.414 | 女声当前推荐，干净电话通道式后处理 |
| `mixed_metric_reference` | 0.583 | 1.310 | 跨性别混合指标参考，不作为双版本交付默认 |
| `max_metric_vowel_mask_reference` | 0.542 | 4.241 | 指标上限对照，容易触发 ASR 长段幻觉 |

推荐优先试听：

- `work_metric_attack/final/recommended/balanced_phone_clean_male/`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/`

如果只看指标上限，可试听：

- `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/`

但注意 `max_metric_vowel_mask_reference` 的短句会触发 Whisper 很长的幻觉文本，听感风险比 `balanced_phone_clean_*` 高。

### 2026-05-29 PPG-tone 结果

本分支新增了 `PPG_TONE_EXPERIMENT.md`，用于记录 PPG-inspired 中文声调自然化实验。默认从 `balanced_phone_clean_male/female` 出发，使用 `--strength 0.4` 的保守后处理。

当前保存的 VoicePrivacy 风格结果：

| Variant | ASV EER ↑ | ASR WER ↑ | 解读 |
| --- | ---: | ---: | --- |
| `ppg_tone_male` | 0.583 | 0.655 | 男声指标基本保持，同时加入声调自然化后处理 |
| `ppg_tone_female` | 0.500 | 0.690 | 女声匿名性保持，但 WER 低于原 `balanced_phone_clean_female` |

结论：如果主打指标最大化，仍优先展示 `balanced_phone_clean_male/female`；如果主打“中文声调自然度 / PPG-inspired 创新点”，展示 `ppg_tone_male/female` 作为对照实验。

---

## 7. 关键代码结构

### 主流程

`run_pipeline.py`

职责：
- 读取输入；
- 跑预处理；
- 为多个 target config 生成 VC candidates；
- 跑候选评估；
- 输出每个 variant 的最终 preferred 结果。

### 预处理

`audio_preprocess.py`

职责：
- 采样率/声道统一；
- 降噪；
- 响度规整；
- 轻量静噪抑制。

重点：
- 这里已经有可调 denoise preset；
- 之前过强的 gate 会把连续性破坏掉，所以不要轻易把低活动段抑制得太狠。

### VC 候选构建

`vc_candidate_builder.py`

职责：
- 读取 `vc_target_pool*.json`；
- 根据 target strategy 生成候选；
- 当前主要使用 `single_ref_group`；
- 也保留过 `montage_pool` 的能力，作为实验记录。

### VC 后端

`vc_anonymizer.py`

职责：
- 调 Coqui TTS / FreeVC 完成 voice conversion。

### 候选打分

`evaluate_anonymization.py`

职责：
- 计算频谱距离、时长、响度、voiced ratio；
- 额外计算：
  - `p95_f0_step_st`
  - `f0_jump_ratio`
- 用 prosody penalty 压制那些“虽然音色变了，但说话很不连贯”的候选。

### 后处理

`naturalness_postprocess.py`

职责：
- 在 candidate 上做自然度修正；
- 控制 source detail 的回灌。

重点：
- 之前试过过强的 output gating，会导致说话一卡一卡；
- 当前判断：**后处理不是主要瓶颈**，不要把优化重点完全放在 postprocess 上。

### VoicePrivacy 风格评估

`evaluate_voiceprivacy.py`

职责：
- 读取 `work_smooth_verify/final/preferred_variants/*_selections.json`
- 用本地 cached SpeechBrain ECAPA 做 ASV embedding
- 用本地 cached faster-whisper small 做 ASR 转写
- 输出 `voiceprivacy_style_results.json`

当前已修复：
- SpeechBrain 读取 HuggingFace cache 时不能在只读 cache 目录写 lock 文件的问题；
- 现在通过 `--speaker-savedir-root` 使用 `/private/tmp/speechbrain_ecapa_eval` 作为可写 savedir；
- 批量评估大量组合时会缓存 ASR 转写，避免重复转写相同音频。

### Metric-attack 生成与导出

`build_metric_attack_variants.py`

职责：
- 读取 `work_smooth_verify/evaluation_vc/*/summary.json`；
- 组合所有 single-reference VC 候选；
- 生成 raw 组合和 6 类通话式后处理组合；
- 输出到 `work_metric_attack/final/preferred_variants/`。

`export_metric_attack_results.py`

职责：
- 从评估结果中导出当前推荐版本；
- 输出到 `work_metric_attack/final/recommended/`。

---

## 8. 当前环境与依赖

项目默认环境：
- `speech-anon310`

关键依赖：
- Python 3.10
- `numpy`
- `scipy`
- `matplotlib`
- `ffmpeg`
- Coqui TTS (`TTS`)

VoicePrivacy 风格评估额外依赖：
- 本地可用的 SpeechBrain ECAPA 模型缓存
- 本地可用的 faster-whisper small 模型缓存

在当前机器上，评估脚本默认假设：
- 主评估 Python: 当前 shell 的 `python`（当前为 `/opt/anaconda3/bin/python`，有 `faster_whisper`）
- ASV Python: `/opt/anaconda3/envs/speech-anon310/bin/python`
- Whisper model cache: `~/.cache/huggingface/hub/models--Systran--faster-whisper-small/...`
- SpeechBrain model cache: `~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/...`
- SpeechBrain writable savedir: `/private/tmp/speechbrain_ecapa_eval`

如果换机器，`evaluate_voiceprivacy.py` 里的默认路径可能需要调整。

---

## 9. 现在怎么跑

### 跑主流程

```bash
python run_pipeline.py
```

或者指定输出目录：

```bash
python run_pipeline.py --work-root work_smooth_verify
```

### 跑 VoicePrivacy 风格评估

```bash
python evaluate_voiceprivacy.py
```

输出：

- `voiceprivacy_style_results.json`

### 跑 metric-attack 指标增强实验

```bash
python build_metric_attack_variants.py
python evaluate_voiceprivacy.py --selection-glob 'work_metric_attack/final/preferred_variants/*_selections.json' --output-path work_metric_attack/voiceprivacy_metric_attack_results.json
python export_metric_attack_results.py
```

输出：

- `work_metric_attack/voiceprivacy_metric_attack_results.json`
- `work_metric_attack/final/recommended/`

---

## 10. 当前最值得继续做的事情

### 优先级最高

1. **如果目标是攻击指标：优先试听并比较 `balanced_phone_clean_male` / `balanced_phone_clean_female` 与对应 raw 版本**
2. **如果目标是听感自然：继续解决句内语调突变**
3. **如果目标是极限指标：保留 `max_metric_vowel_mask_reference` 作上限对照，但不要默认作为最终听感版本**

### 推荐继续尝试的方向

1. 在 `single_ref_group` 框架内继续调优，不要回退到 mixed-pool 作为默认主线。
2. 继续优化 candidate selection，让评估更偏向 prosody continuity。
3. 针对 `male_leaning` 单独调参，而不是强行和 female 用完全相同的策略。
4. 如果 FreeVC 的 prosody 保真始终成为瓶颈，可以评估替换或补充 VC 后端。

### 不建议优先投入的方向

1. 不要把主要精力继续堆到更激进的降噪 gate 上。
   - 之前已经验证：太强会破坏说话连续性。
2. 不要把 mixed-pool target 直接切回默认方案。
   - 之前已经验证：会明显增加拼接感和句内 pitch 跳变。
3. 不要默认认为“后处理再加强一点”就能根治问题。
   - 当前判断：主瓶颈更可能在 VC 后端的 prosody 保真。

---

## 11. 如果你是新 AI，建议的第一步动作

如果要继续这个项目，建议先做下面几件事：

1. 读 `README.md`、`STAGE_PROGRESS_REPORT.md`、`HANDOFF.md`
2. 读 `run_pipeline.py`、`vc_candidate_builder.py`、`evaluate_anonymization.py`
3. 确认当前默认输出目录 `work_smooth_verify/` 下的最终结果和选择 JSON
4. 读 `voiceprivacy_style_results.json`
5. 如果要继续调优，先围绕：
   - `male_leaning`
   - prosody continuity
   - candidate selection
   这三点开始，而不是从 pooled target 重做一遍

---

## 12. 可直接给新 AI 的起手 prompt

如果你要在新会话里继续，可以直接把下面这段发给新的 AI：

```text
请接手这个语音匿名化项目：/Users/hwy/Desktop/个人/26春/语音信号处理/期末proj

先按顺序阅读：
1. README.md
2. STAGE_PROGRESS_REPORT.md
3. HANDOFF.md
4. run_pipeline.py
5. vc_candidate_builder.py
6. evaluate_anonymization.py
7. evaluate_voiceprivacy.py

项目当前主线不是 mixed-pool target，而是 single_ref_group + smoothness selection。之前已经验证过：直接 mixed-pool 虽然匿名性更强，但会明显破坏句内 prosody continuity，导致拼接感和音高跳变变重，所以不要默认切回那个方案。

当前阶段最主要剩余问题不是匿名性不足，而是句内语调连续性仍不够自然，尤其 male_leaning 分支更明显。当前最推荐继续优化的方向是：male_leaning、prosody continuity、candidate selection。

当前 VoicePrivacy 风格评估结果见 voiceprivacy_style_results.json：female_leaning 的 ASV EER 为 0.000、WER 为 0.379；male_leaning 的 ASV EER 为 0.417、WER 为 0.345。这个结果适合做项目内横向比较，不等同于官方榜单分数。

请先基于现有代码理解当前主流程和评估逻辑，再提出最值得继续做的 2~3 个具体改进点，并优先围绕 prosody continuity 推进，而不是只继续加强降噪或后处理 gate。
```

---

## 13. 补充说明

- 当前 `.gitignore` 忽略了 `work/`、`work_pooled_verify/`、`work_dual_verify/`、`work_standard_probe/`、`work_smooth_verify/`、`work_metric_attack/`、`work_metric_probe/`。
- 如果需要把最终试听样例上传到 GitHub，需要显式 `git add -f`。
- 如果要做课程汇报，建议优先展示：
  - `female_leaning` 最终版
  - `male_leaning` 最终版
  - 再说明当前瓶颈是 prosody continuity，而不是匿名性不足。

---

## 14. 一句话总结

当前项目已经从“能做匿名化”推进到了“能稳定输出 female / male 双版本匿名语音”，接下来最值得继续投入的方向，是**围绕 prosody continuity 继续优化，尤其是 male_leaning 分支**。
