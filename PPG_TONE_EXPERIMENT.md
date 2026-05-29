# PPG-inspired 中文声调自然化实验

更新时间：2026-05-29

当前分支：`codex/ppg-chinese-tone-naturalness`

## 1. 实验目标

这个分支尝试在现有 FreeVC 匿名化结果之上加入一个轻量的 PPG-inspired 中文声调自然化后处理：

- 保留男声、女声两套输出。
- 尽量维持原来较高的 ASV EER / ASR WER 攻击指标。
- 减少匿名化语音里声调轮廓和谱细节的突兀感，让中文听起来更自然。

注意：本分支没有下载或训练真正的神经 PPG extractor。实现方式是“PPG 风格内容后验 + 声调后验 + 轻量谱瓶颈后处理”，适合作为课程项目里的创新性实验分支，不应表述成完整的 neural PPG anonymizer。

## 2. 方法思路

核心代码在：

- `ppg_tone_naturalizer.py`
- `run_ppg_tone_experiment.py`

处理流程：

1. 读取原始 denoised source 和已有匿名化 candidate。
2. 用自相关估计帧级 F0，得到 voiced mask、F0 jump ratio、p95 F0 step。
3. 用 Mel-like 频带能量构造 PPG-inspired content posterior，作为轻量内容瓶颈。
4. 将 voiced 片段划分为 tone-like segment，并分类成 `level / rising / falling / dipping / neutral`。
5. 根据声调段的趋势误差，对 candidate 做保守的谱包络平滑、能量包络匹配和轻微声调能量塑形。
6. 输出新的男声/女声 selection JSON，可直接接入 `evaluate_voiceprivacy.py`。

默认后处理强度为 `--strength 0.4`。我也测试过更强的 `1.0` 和 `1.25`，但女声短句更容易被 ASR 读懂，所以最终保留弱强度版本。

## 3. 运行命令

生成 PPG-tone 输出：

```bash
python run_ppg_tone_experiment.py
```

运行 VoicePrivacy 风格评估：

```bash
python evaluate_voiceprivacy.py \
  --selection-glob 'work_ppg_tone/final/recommended/*/*_selections.json' \
  --output-path work_ppg_tone/voiceprivacy_ppg_tone_results.json
```

## 4. 输出文件

男声：

- `work_ppg_tone/final/recommended/ppg_tone_male/test_denoised_ppg_tone_male.wav`
- `work_ppg_tone/final/recommended/ppg_tone_male/绿色_denoised_ppg_tone_male.wav`

女声：

- `work_ppg_tone/final/recommended/ppg_tone_female/test_denoised_ppg_tone_female.wav`
- `work_ppg_tone/final/recommended/ppg_tone_female/绿色_denoised_ppg_tone_female.wav`

评估结果：

- `work_ppg_tone/voiceprivacy_ppg_tone_results.json`
- `work_ppg_tone/ppg_tone_experiment_summary.json`

## 5. VoicePrivacy 风格结果

| 版本 | 输入底座 | ASV EER ↑ | 总 ASR WER ↑ | 结果评价 |
| --- | --- | ---: | ---: | --- |
| `balanced_phone_clean_male` | 原推荐男声 | 0.583 | 0.655 | 原男声推荐，指标强。 |
| `ppg_tone_male` | `balanced_phone_clean_male` | 0.583 | 0.655 | 男声指标基本保持，同时加入声调自然化后处理。 |
| `balanced_phone_clean_female` | 原推荐女声 | 0.500 | 1.414 | 原女声推荐，WER 很高但短句破坏较重。 |
| `ppg_tone_female` | `balanced_phone_clean_female` | 0.500 | 0.690 | 女声匿名性保持，但 WER 下降，说明自然化会提高一部分可懂度。 |

当前保存的逐条 WER：

| 版本 | 录音 | WER ↑ | ASR 输出摘要 |
| --- | --- | ---: | --- |
| `ppg_tone_male` | `test.wav` | 0.650 | “南方科技大学电子应变企业工程企业...” |
| `ppg_tone_male` | `绿色.m4a` | 0.667 | “我选择的一项是铝锁” |
| `ppg_tone_female` | `test.wav` | 0.850 | “南方科技大学电子里面的协议还有工程...” |
| `ppg_tone_female` | `绿色.m4a` | 0.333 | “我现在的颜色是蜜色” |

说明：本项目只有两条 source utterance，ASR WER 对短句非常敏感，`绿色.m4a` 的 Whisper 输出在重复运行中会有波动。因此结果更适合作为本项目内部横向比较，不是官方 VoicePrivacy 排名。

## 6. 声调自然化效果

从生成的 metadata 看，PPG-tone 后处理确实让 tone contour error 有所下降：

| 版本 | 录音 | tone error before ↓ | tone error after ↓ | F0 jump before ↓ | F0 jump after ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| `ppg_tone_male` | `test.wav` | 0.994 | 0.909 | 0.268 | 0.269 |
| `ppg_tone_male` | `绿色.m4a` | 1.297 | 1.271 | 0.238 | 0.238 |
| `ppg_tone_female` | `test.wav` | 0.843 | 0.799 | 0.225 | 0.221 |
| `ppg_tone_female` | `绿色.m4a` | 1.358 | 1.288 | 0.237 | 0.237 |

结论：

1. 男声 `ppg_tone_male` 是目前这个分支最稳的结果：EER/WER 没有掉，且加入了声调自然化分析与轻量后处理。
2. 女声 `ppg_tone_female` 匿名性保持，但 WER 明显低于原 `balanced_phone_clean_female`，说明自然化和可懂度之间存在冲突。
3. 如果课程展示主打“指标最大化”，仍建议把 `balanced_phone_clean_male/female` 作为主结果；如果展示“PPG-inspired 中文声调自然化创新点”，则展示本分支的 `ppg_tone_male/female` 作为对照实验。
