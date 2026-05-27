# 语音匿名化工程实现思路

更新时间：2026-05-28

本文件说明当前匿名化工程是如何实现的，以及为什么这样设计。

## 1. 项目目标

本项目的目标是把输入语音转换成匿名语音：

1. 仍然像真人说话；
2. 尽量保留语句节奏和基本语义；
3. 让说话人验证系统更难把输出语音识别回原说话人；
4. 在当前攻击目标下，也尽量让 ASR 更难正确转写内容。

当前不是追求“合成一个完美的新说话人”，而是追求：

- 原说话人身份特征被削弱；
- 输出仍像真实录音，而不是机械变声；
- 男声 / 女声两个匿名版本都能稳定生成。

## 2. 总体流程

主流程入口是 `run_pipeline.py`。整体链路如下：

```text
原始音频
  -> 音频预处理
  -> FreeVC 语音转换候选生成
  -> 自然度后处理
  -> 候选评估与选择
  -> 男声 / 女声匿名结果
  -> VoicePrivacy 风格评估
  -> metric-attack 指标增强版本
```

核心脚本：

| 脚本 | 作用 |
| --- | --- |
| `audio_preprocess.py` | 统一采样率、单声道、降噪、响度规整 |
| `vc_candidate_builder.py` | 根据目标说话人配置生成 FreeVC 候选 |
| `vc_anonymizer.py` | 调用 Coqui TTS / FreeVC 做 voice conversion |
| `naturalness_postprocess.py` | 做轻量自然度修正和细节回灌 |
| `evaluate_anonymization.py` | 计算候选分数，选择更平滑、更匿名的候选 |
| `evaluate_voiceprivacy.py` | 计算 ASV EER 和 ASR WER |
| `build_metric_attack_variants.py` | 生成指标增强候选 |
| `export_metric_attack_results.py` | 导出当前推荐男声 / 女声结果 |

## 3. 第一步：音频预处理

预处理由 `audio_preprocess.py` 完成。

输入文件：

- `test.wav`
- `绿色.m4a`

预处理目标：

- 统一为 16 kHz；
- 单声道；
- PCM WAV；
- 降低背景噪声；
- 规整响度；
- 轻量压低长静音段噪声。

关键设计：

```text
原始音频
  -> ffmpeg 统一格式
  -> highpass / lowpass
  -> afftdn 降噪
  -> loudnorm 响度规整
  -> suppress_low_activity_noise 轻量低活动段抑制
```

这里没有使用很强的 gate，因为之前实验发现过强静噪会导致语音变得一卡一卡，破坏句内连续性。

## 4. 第二步：匿名目标设计

匿名化后端使用 FreeVC：

```text
voice_conversion_models/multilingual/vctk/freevc24
```

FreeVC 需要两个输入：

- source：原始说话内容；
- target：目标参考音色。

项目目前保留两组目标配置：

| 配置 | 作用 |
| --- | --- |
| `vc_target_pool_female.json` | 生成偏女声匿名候选 |
| `vc_target_pool_male.json` | 生成偏男声匿名候选 |

当前采用的策略是 `single_ref_group`：

```text
同一组内有多个参考说话人
  -> 每个参考说话人单独生成一个 FreeVC 候选
  -> 不直接把多个参考拼成一个 target
  -> 最后再从候选里选择最合适的一条
```

这样做的原因是：之前试过 mixed-pool target，也就是把多个参考说话人片段拼起来直接送进 FreeVC。它确实会增强匿名性，但会明显破坏语调连续性，容易出现拼接感和音高突变。因此当前默认主线改成单参考候选组。

## 5. 第三步：FreeVC 候选生成

候选生成由 `vc_candidate_builder.py` 完成。

对每条预处理后的 source 音频：

1. 读取目标配置；
2. 遍历目标参考音频；
3. 对每个目标参考调用 `vc_anonymizer.py`；
4. 得到多个 FreeVC 输出候选；
5. 对每个候选执行 `humanize_candidate` 后处理；
6. 写出 manifest，供后续评估。

例如男声分支会生成类似：

```text
test_denoised_freevc_male_leaning_s2.wav
test_denoised_freevc_male_leaning_s5.wav
test_denoised_freevc_male_leaning_s6.wav
```

女声分支会生成类似：

```text
test_denoised_freevc_female_leaning_s1.wav
test_denoised_freevc_female_leaning_s3.wav
test_denoised_freevc_female_leaning_s4.wav
```

## 6. 第四步：自然度后处理

后处理由 `naturalness_postprocess.py` 完成。

它不是重新合成语音，而是对 FreeVC 输出做轻量修正：

- 匹配 source 的能量包络；
- 根据 voiced / unvoiced mask 区分浊音和清音；
- 给少量 source 高频细节回灌；
- 限制峰值，避免削波。

这样做的目的：

- 保留一些发音边界和辅音清晰度；
- 减少 VC 输出过糊的问题；
- 让输出更像真人录音。

但这里的 source detail 回灌很克制，因为回灌太多会把原说话人的身份特征带回来。

## 7. 第五步：候选评估与选择

候选评估由 `evaluate_anonymization.py` 完成。

它会对每个候选计算：

| 指标 | 作用 |
| --- | --- |
| spectral distance | 衡量频谱差异，差异越大通常越匿名 |
| envelope correlation | 衡量能量包络是否跟原句节奏一致 |
| duration penalty | 防止时长偏差过大 |
| voiced ratio penalty | 防止语音结构异常 |
| median F0 shift | 观察基频是否发生变化 |
| p95 F0 step | 检测音高跳变 |
| F0 jump ratio | 检测句内 pitch 是否不连续 |

评分思想是：

```text
希望音色和原说话人不同
但不希望语音节奏、时长、能量、prosody 连续性完全坏掉
```

所以候选分数不是单纯“变得越不像越好”，而是在匿名性和自然度之间折中。

早期的推荐结果就是通过这个评分选出的：

- `female_leaning`
- `male_leaning`

## 8. 第六步：VoicePrivacy 风格评估

评估脚本是 `evaluate_voiceprivacy.py`。

它评估两个核心指标：

| 指标 | 含义 |
| --- | --- |
| ASV EER | 说话人验证错误率，越高越匿名 |
| ASR WER | ASR 转写错误率，常规 utility 越低越好；本轮攻击目标里越高表示 ASR 越被扰乱 |

ASV 使用：

```text
SpeechBrain ECAPA speaker embedding
```

ASR 使用：

```text
faster-whisper small
```

当前协议：

1. 用原始 source 语音求平均 embedding，作为受保护说话人的 enrollment；
2. 匿名输出作为 target trials；
3. lab9 的参考说话人作为 non-target trials；
4. 计算 target / non-target 分数并得到 EER；
5. 用 source 的 Whisper 转写作为 pseudo ground truth；
6. 对匿名输出再次转写，计算字符级 WER。

注意：当前数据量很小，所以这个评估是项目内横向比较，不是官方 VoicePrivacy challenge 协议。

## 9. 第七步：指标增强分支

用户后续要求是：

```text
保持真人音色的同时，尽量提高 ASV EER + ASR WER。
```

这和传统 VoicePrivacy utility 目标不同，因为传统目标通常希望 WER 越低越好；这里希望 ASR 更难识别，所以 WER 越高越符合攻击目标。

为此新增了 `metric_attack` 分支。

### 9.1 候选重选

脚本：

```text
build_metric_attack_variants.py
```

它会读取 `work_smooth_verify/evaluation_vc/*/summary.json` 中已有的所有 FreeVC 候选。

对两条录音分别组合：

- 男声候选组合；
- 女声候选组合；
- raw 组合；
- phone channel 组合；
- vowel mask 组合；
- 其他轻量通道失真组合。

这样做的目的不是重新训练模型，而是从已有候选里找到更符合 ASV / ASR 指标目标的组合。

### 9.2 干净电话通道后处理

当前主推荐是 `phone_clean`：

```text
highpass
lowpass
equalizer
compressor
loudnorm
```

它模拟真实通话链路：

- 带宽变窄；
- 部分 formant 区域被削弱；
- 动态范围被压缩；
- 但不加入明显机械音效；
- 不加入强噪声。

这样可以让 ASV / ASR 更难处理，同时保留“像真人录音”的感觉。

### 9.3 导出推荐结果

脚本：

```text
export_metric_attack_results.py
```

当前默认导出：

| 导出名 | 含义 |
| --- | --- |
| `balanced_phone_clean_male` | 当前推荐男声指标增强版 |
| `balanced_phone_clean_female` | 当前推荐女声指标增强版 |
| `raw_metric_male` | 男声 raw 对照，不加电话通道 |
| `raw_metric_female` | 女声 raw 对照，不加电话通道 |
| `mixed_metric_reference` | 跨性别混合指标参考，不作为主交付 |
| `max_metric_vowel_mask_reference` | 指标上限参考，不作为主听感版本 |

## 10. 当前推荐结果

当前主推荐是男声 / 女声各一套：

```text
work_metric_attack/final/recommended/balanced_phone_clean_male/
work_metric_attack/final/recommended/balanced_phone_clean_female/
```

对应指标：

| 版本 | ASV EER ↑ | ASR WER ↑ | 评价 |
| --- | ---: | ---: | --- |
| `balanced_phone_clean_male` | 0.583 | 0.655 | 匿名性强，ASR 扰动提升，听感风险较低 |
| `balanced_phone_clean_female` | 0.500 | 1.414 | 匿名性较强，ASR 扰动更强，但短句内容破坏更明显 |

如果更强调自然度，可以使用 raw 版本：

```text
work_metric_attack/final/recommended/raw_metric_male/
work_metric_attack/final/recommended/raw_metric_female/
```

如果只展示攻击上限，可以使用：

```text
work_metric_attack/final/recommended/max_metric_vowel_mask_reference/
```

但它会触发 ASR 长段幻觉，不建议作为主试听版本。

## 11. 为什么不是简单变调

简单变调的问题是：

- 容易听起来不自然；
- ASV 可能仍能利用说话节奏、频谱包络、发音习惯识别原说话人；
- 对 ASR 的扰动不稳定；
- 很容易变成“机械变声器”。

当前方案的核心是：

```text
用 VC 改变说话人 timbre
用候选选择保证 prosody 不崩
用轻量后处理增加真实录音链路扰动
用 ASV / ASR 指标反向筛选最终结果
```

这比单纯调 pitch 或 EQ 更稳，也更容易解释为一个完整的语音匿名化系统。

## 12. 当前局限

1. 数据集很小，只有两条 source utterance，所以 EER 数值只能做项目内比较。
2. FreeVC 的 prosody 保真仍不是完美的，某些候选会出现句内音高跳变。
3. ASR WER 被提高后，语义可懂度会下降；这符合当前攻击目标，但不符合传统 utility 目标。
4. 女声指标增强版对短句 ASR 攻击很强，但内容破坏也更明显，需要人工试听确认。
5. `max_metric_vowel_mask_reference` 指标很激进，但会触发 ASR 幻觉，因此只适合作为对照。

## 13. 复现命令

运行基础匿名化：

```bash
python run_pipeline.py --work-root work_smooth_verify
```

运行旧版 VoicePrivacy 风格评估：

```bash
python evaluate_voiceprivacy.py
```

运行指标增强实验：

```bash
python build_metric_attack_variants.py
python evaluate_voiceprivacy.py --selection-glob 'work_metric_attack/final/preferred_variants/*_selections.json' --output-path work_metric_attack/voiceprivacy_metric_attack_results.json
python export_metric_attack_results.py
```

查看结果说明：

```text
VOICEPRIVACY_RESULTS.md
```

## 14. 一句话总结

当前工程的思路是：先用 FreeVC 做真实语音转换，再用多候选筛选避免语调崩坏，最后用干净通话式后处理和 VoicePrivacy 风格指标筛出男声 / 女声两套匿名结果。
