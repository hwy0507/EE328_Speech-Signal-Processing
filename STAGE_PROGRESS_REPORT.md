# 语音匿名化项目阶段性进展汇报

## 1. 当前做到的程度

截至目前，项目已经完成了一个可运行的匿名语音处理主链路，并且拿到了可试听的阶段性成果。

### 已完成内容
1. **音频预处理链路搭建完成**
   - 输入音频会统一转为单声道、16 kHz PCM WAV。
   - 已加入基础降噪、响度规整和轻量静噪抑制。
   - 主流程入口为 `run_pipeline.py`，预处理逻辑在 `audio_preprocess.py`。

2. **匿名音色生成链路完成**
   - 已接入 Coqui TTS 的 FreeVC (`voice_conversion_models/multilingual/vctk/freevc24`) 作为 VC 后端。
   - 不再默认直接使用“单一固定说话人”作为目标音色，而是支持按配置生成不同匿名目标策略。

3. **“偏女声 / 偏男声”双版本已经可生成**
   - 当前项目支持分别输出：
     - `female_leaning`
     - `male_leaning`
   - 现在的实现不是直接把混池 wav 喂给 VC，而是：
     - 先在同一性别倾向组里生成多个单参考候选；
     - 再通过评估脚本选出更稳定、更平滑的候选。

4. **已经定位出当前最关键的问题来源**
   - 之前明显的“拼接感”和“句内语调突变”，主要不是后处理造成的；
   - 根因更接近于：**直接把 mixed pool 目标送入 FreeVC，会让 VC 自身的基频轨迹变得不稳定**；
   - 因此当前策略已经调整为：
     - `single_ref_group` 候选生成
     - 评估时加入语调平滑度惩罚项

5. **评估逻辑已扩展**
   - 除原本的频谱距离、包络相关性、时长/能量等指标外；
   - 现在会额外计算：
     - `p95_f0_step_st`
     - `f0_jump_ratio`
   - 用来惩罚句内语调跳变过大的候选，避免“音色变了但人声不流畅”的结果被选中。

6. **已补充 VoicePrivacy 风格评估**
   - 新增 `evaluate_voiceprivacy.py`；
   - 用 ASV EER 观察匿名性；
   - 用 ASR WER 观察可懂度；
   - 当前结果已经写出到 `voiceprivacy_style_results.json`，可作为项目内横向对比基线。

---

## 2. 当前推荐结果

### 当前最推荐试听的输出
位于：

- `work_smooth_verify/final/preferred_variants/test_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/test_denoised_male_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_female_leaning.wav`
- `work_smooth_verify/final/preferred_variants/绿色_denoised_male_leaning.wav`

### 当前自动选中的稳定参考
- `female_leaning`：当前更稳定地选中 `s3`
- `male_leaning`：当前更稳定地选中 `s2`

### VoicePrivacy 风格结果摘要
当前对 `work_smooth_verify/final/preferred_variants/` 的结果，补充了本地 VoicePrivacy 风格评估：

| Variant | ASV EER ↑ | ASR WER ↓ | 当前解读 |
| --- | ---: | ---: | --- |
| source baseline | 0.000 | - | 原始语音可被稳定验证回原说话人 |
| `female_leaning` | 0.000 | 0.379 | 当前这版匿名性仍偏弱，但可懂度尚可 |
| `male_leaning` | 0.417 | 0.345 | 当前匿名性更强，可懂度与 female 版本接近 |

其中：
- `test.wav`：female WER = 0.550，male WER = 0.500
- `绿色.m4a`：female WER = 0.000，male WER = 0.000

完整结果文件：
- `voiceprivacy_style_results.json`

说明：这里延续了 VoicePrivacy 的核心指标轴（ASV EER / ASR WER），但当前数据集规模很小，因此结果用于项目内横向比较，不等同于官方挑战榜单分数。

---

## 3. 各阶段输出文件目录

### 原始旧版实验输出
- `work/`

### 混池匿名化 + 增强降噪实验输出
- `work_pooled_verify/`

### 首次双版本（female/male pool）实验输出
- `work_dual_verify/`

### 标准降噪顺滑度探针输出
- `work_standard_probe/`

### 当前最推荐的“平滑度优先”输出
- `work_smooth_verify/`

其中本阶段最关键的最终文件在：
- `work_smooth_verify/final/preferred_variants/`

---

## 4. 目前仍存在的问题

虽然匿名性已经明显增强，而且 female / male 两类输出已经分开生成，但项目当前仍然存在一个核心问题：

### 主要未解决问题
**句子内部的语调连续性还不够自然。**

具体表现为：
- 某些音节连接处会有明显的音高突变；
- 听感上仍可能出现“像拼接出来的”不连续感；
- 这个问题在 `male_leaning` 分支上通常比 `female_leaning` 更明显。

### 当前判断
- 后处理不是主要矛盾；
- 真正的瓶颈更可能是 **FreeVC 在匿名目标条件变化下对 prosody 的保真不足**；
- 因此后续优化重点应该继续放在：
  1. 让候选选择更偏向 prosody 平稳；
  2. 为 male/female 分支固定更稳定的参考策略；
  3. 如有必要，考虑替换或补充 VC 后端，而不只是继续堆后处理。

---

## 5. 下一步建议

### 2026-05-28 指标增强补充

按照“保持真人音色，同时尽量提高 ASV EER + ASR WER”的新目标，已补充一条 `metric_attack` 实验分支：

- 新增 `build_metric_attack_variants.py`，从 `work_smooth_verify` 的全部 FreeVC 候选生成 36 个 raw 组合和 216 个通话式后处理组合。
- 修复 `evaluate_voiceprivacy.py` 在当前沙箱下的 SpeechBrain savedir 问题，并加入 ASR 转写缓存。
- 新增 `export_metric_attack_results.py`，导出推荐版本到 `work_metric_attack/final/recommended/`。

当前指标结果：

| Variant | ASV EER ↑ | ASR WER ↑ | 解读 |
| --- | ---: | ---: | --- |
| previous `male_leaning` | 0.417 | 0.345 | 旧推荐结果 |
| `raw_metric_male` | 0.583 | 0.414 | 男声候选重选，不加通道失真 |
| `raw_metric_female` | 0.500 | 0.621 | 女声候选重选，不加通道失真 |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 男声当前推荐，干净电话通道式后处理 |
| `balanced_phone_clean_female` | 0.500 | 1.414 | 女声当前推荐，干净电话通道式后处理 |
| `mixed_metric_reference` | 0.583 | 1.310 | 跨性别混合指标参考，不作为双版本交付默认 |
| `max_metric_vowel_mask_reference` | 0.542 | 4.241 | 指标上限对照，ASR 会长段幻觉，听感风险更高 |

推荐试听目录：

- `work_metric_attack/final/recommended/balanced_phone_clean_male/`
- `work_metric_attack/final/recommended/balanced_phone_clean_female/`
- `work_metric_attack/final/recommended/raw_metric_male/`
- `work_metric_attack/final/recommended/raw_metric_female/`

注意：这里将 ASR WER 当作攻击成功指标，因此方向是越高越好；这和 VoicePrivacy 常规定义里 utility WER 越低越好的使用方式不同。

1. **优先继续解决句内语调突变**
   - 重点优化 `male_leaning` 分支；
   - 尝试固定更稳定的单参考基底，再做非常轻量的去个人化处理；
   - 避免再回到直接 mixed-pool target 的做法。

2. **保留 female/male 双版本框架**
   - 当前双版本结构已经具备，后续只需要在各自分支内继续调优即可。

3. **如果课程汇报需要展示阶段结果**
   - 建议优先展示：
     - `female_leaning` 最终版
     - `male_leaning` 最终版
     - 并说明当前技术瓶颈是“prosody continuity 而不是匿名性不足”。

---

## 6. 结论

当前项目已经从“能做匿名化”推进到了“能稳定输出带有明显匿名特征、并区分偏女声/偏男声的两个版本”。

目前的阶段性结论是：
- **匿名性目标已经基本达成**；
- **双版本输出已经达成**；
- **最主要剩余问题是句内语调突变与流畅度不足**；
- 当前最值得继续投入的方向，是围绕 prosody continuity 进一步优化，而不是再单纯加强降噪或继续堆混池 target。
