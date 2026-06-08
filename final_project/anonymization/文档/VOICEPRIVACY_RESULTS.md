# VoicePrivacy 风格评估结果说明

更新时间：2026-05-30

本文件汇总当前项目中各版本的 VoicePrivacy 风格评估结果，并解释这些数值代表什么。

注意：本文包含早期“ASR 攻击增强”阶段的固定 benchmark 记录。当前最终展示目标已经调整为**音色匿名化 + 内容保留**：网页端和报告主线中，`ASR WER` 应按内容保留解释，越低越好；过高 WER 表示内容损失，不应单独当作成功。

## 1. 指标怎么读

### ASV EER

ASV EER 用来衡量匿名性，越高表示越难被说话人验证系统重新识别为原说话人。

- `0.000`：匿名性很弱，ASV 可以稳定区分目标/非目标。
- `0.4` 左右：匿名性明显增强。
- `0.5` 左右：在当前小样本协议下已经接近随机判断，匿名性较强。

注意：当前数据只有 2 条 source utterance、1 个受保护说话人，所以 ASV EER 适合作为项目内横向比较，不等同于官方 VoicePrivacy 榜单结果。

为什么很多版本都是 `0.583`：

当前本地协议只有 `2` 个 target trials 和 `12` 个 non-target trials，因此 EER 不是连续变化的指标，而是会按少数几个档位跳变。多个方法同为 `0.583` 不代表它们完全一样，只代表它们落在同一个 EER 档位。为了在同档位里继续比较匿名化优劣，本项目新增以下细排指标：

- `source_target_mean_score`：匿名语音和原说话人的 ECAPA 余弦相似度，越低越匿名。
- `source_similarity_reduction`：相似度相对原始语音下降比例，越高越匿名。
- `identity_privacy_index`：结合 EER 和相似度下降的身份隐私指数，用于 EER 同档位细排。
- `privacy_rank`：先按 EER 排，再按 `identity_privacy_index` / 相似度下降细排。

### ASR WER

ASR WER 表示 ASR 转写和参考文本的差异。

在常规 VoicePrivacy utility 评价里，WER 越低越好，表示内容越清楚。

早期 metric-attack 阶段曾把 WER 当作攻击指标看：

- WER 越高：ASR 越难正确识别内容，但也可能意味着内容保留变差。
- WER 过高：可能说明内容已经被明显破坏，甚至触发 ASR 幻觉。

当前主线不再追求 WER 最大化。推荐结果不能只看 WER，还要看 source similarity 是否下降、内容是否保留、听感是否仍像真人录音。

## 2. 总体结果

| 版本 | ASV EER ↑ | 总 ASR WER | 结果评价 |
| --- | ---: | ---: | --- |
| source baseline | 0.000 | - | 原始语音可被稳定识别回原说话人，不匿名。 |
| `female_leaning` | 0.000 | 0.379 | 女声旧版可懂度较好，但匿名性弱。 |
| `male_leaning` | 0.417 | 0.345 | 男声旧版匿名性明显好于女声旧版，但 ASR 仍能较多恢复内容。 |
| `raw_metric_male` | 0.583 | 0.414 | 男声候选重选后，匿名性很强；不加通道失真，听感风险较低。 |
| `raw_metric_female` | 0.500 | 0.621 | 女声候选重选后，匿名性较强，但非当前 Web UI 主展示。 |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 历史男声增强版本；匿名性强，但内容保留需要结合 WER 判断。 |
| `balanced_phone_clean_female` | 0.500 | 1.414 | 历史女声增强版本；匿名性较强，但短句内容破坏更重。 |
| `mixed_metric_reference` | 0.583 | 1.310 | 跨性别混合指标参考，不作为男/女双版本默认交付。 |
| `max_metric_vowel_mask_reference` | 0.542 | 4.241 | 指标上限对照；WER 很高，但短句会触发 ASR 长段幻觉，听感风险高。 |

### 当前男声主报告细排

男声主报告见 `report_evaluation_male/REPORT_EVALUATION_SUMMARY.md`。在男声条件下，`raw_metric_male`、`balanced_phone_clean_male`、`ppg_tone_male` 的 ASV EER 都是 `0.583`，因此用相似度下降和身份隐私指数继续细排：

| 版本 | ASV EER ↑ | ASR WER | 源说话人相似度 ↓ | 相似度下降 ↑ | 身份隐私指数 ↑ | 细排 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppg_tone_male` | 0.583 | 0.655 | 0.060 | 91.5% | 0.966 | 1 |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 0.061 | 91.4% | 0.966 | 2 |
| `raw_metric_male` | 0.583 | 0.414 | 0.076 | 89.4% | 0.958 | 3 |
| `male_leaning` | 0.417 | 0.345 | 0.159 | 77.8% | 0.811 | 4 |

因此，虽然前三个男声方法 EER 都显示为 `0.583`，更细的身份相似度指标仍能说明：`ppg_tone_male` 最接近当前主目标，`balanced_phone_clean_male` 次之，`raw_metric_male` 匿名性也强但固定集 WER 较低、内容保留更好。

## 3. 每条录音结果

说明：ASV EER 是对应版本在两条录音上的整体 privacy 分数，不是单条录音单独计算出来的 EER。单条录音主要看 ASR WER。

### `test.wav`

参考文本：

```text
南方科技大学电子与电器工程系信息工程专业
```

| 版本 | 输出文件 | 所属版本 ASV EER ↑ | 本条 WER | 转写表现 | 评价 |
| --- | --- | ---: | ---: | --- | --- |
| `female_leaning` | `work_smooth_verify/final/preferred_variants/test_denoised_female_leaning.wav` | 0.000 | 0.550 | “男人放科幣大學電子電池工程系,信息工程專業” | ASR 已有错误，但匿名性仍弱。 |
| `male_leaning` | `work_smooth_verify/final/preferred_variants/test_denoised_male_leaning.wav` | 0.417 | 0.500 | “南方科技大学电子应该变成协议你看 工程协议 信息工程专业” | 匿名性明显提升，ASR 仍能识别大量关键词。 |
| `raw_metric_male` | `work_metric_attack/final/recommended/raw_metric_male/test_denoised_raw_metric_male.wav` | 0.583 | 0.600 | “南方科技大學電子裡面寫工程寫,實際上是工程專業” | 男声 raw 版本匿名性强，内容仍保留部分结构。 |
| `raw_metric_female` | `work_metric_attack/final/recommended/raw_metric_female/test_denoised_raw_metric_female.wav` | 0.500 | 0.550 | “南方科技大学,电子里面的系业工程的系业,心理系的工程专业” | 女声 raw 版本匿名性较强，ASR 仍能保留部分结构。 |
| `balanced_phone_clean_male` | `work_metric_attack/final/recommended/balanced_phone_clean_male/test_denoised_balanced_phone_clean_male.wav` | 0.583 | 0.650 | “南方科技大學電子應變企業工程企業,實際是工程卵業” | 推荐男声版；隐私强，ASR 错误增加，听感应仍接近通话人声。 |
| `balanced_phone_clean_female` | `work_metric_attack/final/recommended/balanced_phone_clean_female/test_denoised_balanced_phone_clean_female.wav` | 0.500 | 0.800 | “南方科技大學電子裏面的協議還有工程協議心理學工程專業” | 历史女声增强版；WER 更高，但内容保留更少。 |
| `max_metric_vowel_mask_reference` | `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/test_denoised_max_metric_vowel_mask_reference.wav` | 0.542 | 0.750 | “南北方科技大學電子電子企業工程企業,心理事業工程卵業” | 指标上限对照；长句 WER 高，但不是最稳定的听感版本。 |

本条录音结论：

- 旧版本里，`male_leaning` 比 `female_leaning` 更匿名。
- 男声推荐版 `balanced_phone_clean_male` 把 EER 提到 `0.583`，WER 提到 `0.650`，是比较均衡的结果。
- 女声增强版 `balanced_phone_clean_female` 的 WER 更高，但内容破坏也更明显。

### `绿色.m4a`

参考文本：

```text
我选择的颜色是绿色
```

| 版本 | 输出文件 | 所属版本 ASV EER ↑ | 本条 WER | 转写表现 | 评价 |
| --- | --- | ---: | ---: | --- | --- |
| `female_leaning` | `work_smooth_verify/final/preferred_variants/绿色_denoised_female_leaning.wav` | 0.000 | 0.000 | “我选择的颜色是绿色” | 内容完全可识别，且匿名性弱。 |
| `male_leaning` | `work_smooth_verify/final/preferred_variants/绿色_denoised_male_leaning.wav` | 0.417 | 0.000 | “我选择的颜色是绿色” | 匿名性较好，但 ASR 完全识别内容。 |
| `raw_metric_male` | `work_metric_attack/final/recommended/raw_metric_male/绿色_denoised_raw_metric_male.wav` | 0.583 | 0.000 | “我选择的颜色是绿色。” | 男声 raw 版本匿名性强，但短句 ASR 仍完全识别。 |
| `raw_metric_female` | `work_metric_attack/final/recommended/raw_metric_female/绿色_denoised_raw_metric_female.wav` | 0.500 | 0.778 | “我先做点这些谜色” | 女声 raw 版本匿名性较强，但短句内容已有偏移。 |
| `balanced_phone_clean_male` | `work_metric_attack/final/recommended/balanced_phone_clean_male/绿色_denoised_balanced_phone_clean_male.wav` | 0.583 | 0.667 | “我選擇的一項是鋁鎖” | 推荐男声版；短句 ASR 已被扰乱，但仍像一句中文短语。 |
| `balanced_phone_clean_female` | `work_metric_attack/final/recommended/balanced_phone_clean_female/绿色_denoised_balanced_phone_clean_female.wav` | 0.500 | 2.778 | “Wash into the air, this is Musso” | 历史女声增强版；ASR 被误导成英文，内容破坏明显。 |
| `max_metric_vowel_mask_reference` | `work_metric_attack/final/recommended/max_metric_vowel_mask_reference/绿色_denoised_max_metric_vowel_mask_reference.wav` | 0.542 | 12.000 | ASR 输出很长的幻觉文本 | 指标极高但不推荐作主结果，说明 ASR 已严重幻觉。 |

本条录音结论：

- 旧版 `female_leaning` / `male_leaning` 的短句 WER 都是 `0.000`，说明 ASR 完全能听懂。
- `balanced_phone_clean_male` 将短句 WER 提到 `0.667`，同时 EER 保持 `0.583`，说明匿名性较强，但内容保留需要人工确认。
- `balanced_phone_clean_female` 的 WER 达到 `2.778`，已经出现跨语言误识别，因此不适合作为当前主展示。

## 4. 推荐使用方式

如果课程展示需要男声和女声各一版，建议主展示：

- 男声：`work_metric_attack/final/recommended/balanced_phone_clean_male/`
- 女声：`work_metric_attack/final/recommended/balanced_phone_clean_female/`

如果更强调“自然、少通道后处理”，可以展示：

- 男声：`work_metric_attack/final/recommended/raw_metric_male/`
- 女声：`work_metric_attack/final/recommended/raw_metric_female/`

如果只是展示指标上限，可以把 `max_metric_vowel_mask_reference` 放在附录或对照实验里，不建议作为主试听结果。

## 5. 总结

当前最合理的结论是：

1. 旧女声版本匿名性不足，`ASV EER = 0.000`。
2. 旧男声版本匿名性较好，`ASV EER = 0.417`。
3. 新的男声增强版本达到 `ASV EER = 0.583`，说明匿名性强，但仍需要结合 WER 判断内容保留。
4. 女声增强版本达到 `ASV EER = 0.500`，但整体 WER 达到 `1.414`，短句内容破坏更强，因此不再作为当前 Web UI 主展示。
5. 在当前小样本评估下，`0.5` 左右的 EER 已经说明 ASV 难以稳定识别原说话人；WER 过高要警惕内容被过度破坏。
