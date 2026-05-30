# Handoff: EE328 Speech Anonymization Project

更新时间：2026-05-30

当前主线分支：

```text
codex/ppg-chinese-tone-naturalness
```

远端仓库：

```text
https://github.com/hwy0507/EE328_Speech-Signal-Processing
```

## 1. 项目目标

本项目做的是**语音音色匿名化**，不是关键词扰乱。

正确目标：

1. 匿名输出的音色 / 声纹尽量不像原说话人。
2. 语义内容尽量保留，ASR WER 应尽量低。
3. 听感仍像真人语音，不要明显机械音、断裂或严重失真。

一句话给报告：

```text
本项目基于 FreeVC 构建语音匿名化系统，并加入指标驱动候选选择、干净通话信道后处理和 PPG-inspired 中文声调自然化，使输出语音在保持真人听感和内容可懂度的同时显著降低与原说话人的声纹相似度。
```

## 2. 当前已经完成

已经完成的主功能：

- FreeVC baseline 匿名化
- Metric+phone 增强方法
- PPG-tone 中文声调自然化方法
- Web 端录音、结束录音、匿名化处理按钮
- 本次网页录音的即时评估
- 固定测试集报告图表
- PPG-tone 参数 sweep
- GitHub 分支提交和推送

当前推荐展示方式：

1. 现场用网页录音。
2. 自动生成三种男声目标匿名结果。
3. 网页直接展示本次录音的声纹相似度下降和内容保留指标。
4. 固定 benchmark 图表只作为报告参考。

## 3. 三种方法怎么理解

| 方法 | 工程名称 | 含义 | 评价重点 |
| --- | --- | --- | --- |
| FreeVC baseline | `freevc_baseline` | 直接用 FreeVC 做目标男声音色转换 | 基线匿名效果 |
| Metric+phone | `metric_phone` / `balanced_phone_clean_male` | FreeVC 候选筛选 + 干净电话通道后处理 | 更强声纹隐藏，但可能损失内容保留 |
| PPG-tone | `ppg_tone_male` | Metric+phone 基础上加入中文声调自然化 | 中文自然度、内容保留和匿名化折中 |

`Metric+phone` 是 FreeVC 的进阶增强版，不是全新模型。

`PPG-tone` 是本分支创新点，但要诚实描述：它是 PPG-inspired lightweight content bottleneck，不是真正训练出的 neural PPG extractor。

## 4. Web UI 怎么用

启动：

```bash
python recording_demo_ui.py --port 7862
```

打开：

```text
http://127.0.0.1:7862
```

操作：

1. 点击 `开始录音`。
2. 说 3-10 秒中文。
3. 点击 `结束录音`。
4. 点击 `匿名化处理`。
5. 等待生成三条音频和即时评估表格。

生成结果目录：

```text
work_recording_demo/<session_id>/
```

每次 session 里一般会有：

```text
recording.webm
preprocess/denoised/recording_denoised.wav
raw_freevc/demo_male_freevc.wav
metric_phone/demo_male_metric_phone.wav
ppg_tone/demo_male_ppg_tone.wav
demo_summary.json
recording_evaluation.json
```

## 5. Web UI 指标怎么分析

网页即时评估是针对**刚刚录的这条音频**，不是固定 benchmark。

| 指标 | 方向 | 解释 |
| --- | --- | --- |
| `Source similarity` | 越低越好 | 匿名输出和本次原录音的 ECAPA 声纹相似度 |
| `Similarity drop` | 越高越好 | 声纹相似度下降比例，表示音色匿名强度 |
| `ASR WER` | 越低越好 | 匿名输出相对本次原录音转写的错误率 |
| `Content kept` | 越高越好 | `1 - WER`，内容保留比例 |
| `Timbre index` | 越高越好 | 综合音色匿名和内容保留的本地指数 |

重要解释：

```text
单条网页录音不能可靠计算 ASV EER，因为 EER 需要一组 target / non-target trials。因此网页端用 source-to-output speaker similarity 作为即时音色匿名证据。
```

不要再把 `ASR WER` 解释成越高越好。对于现在的项目目标，WER 越低说明内容保留越好。

## 6. 一次最新网页测试如何解读

用户最近网页测试中出现过类似结果：

| 方法 | Similarity drop | Content kept | 结论 |
| --- | ---: | ---: | --- |
| FreeVC baseline | 约 87% | 约 73% | 音色去除最强，综合分最高 |
| PPG-tone | 约 85% | 约 77% | 内容保留最好，更适合强调中文自然度 |
| Metric+phone | 约 85% | 约 68% | 匿名仍强，但内容损失更大 |

报告里可以这样写：

```text
在实时录音样例中，三种方法均能将源说话人相似度降低 85% 以上，说明音色匿名化有效。其中 PPG-tone 在保持较高匿名度的同时取得更好的内容保留，体现了中文声调自然化后处理的价值。
```

如果某次录音中 FreeVC baseline 排第一，也不矛盾。实时单条样例会受录音内容、麦克风、句长影响；报告中应强调 PPG-tone 的创新点和稳定折中，而不是声称它每条录音都绝对第一。

## 7. 固定 Benchmark 怎么用

固定 benchmark 用于课程报告，不代表每次网页录音。

生成命令：

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

当前固定 benchmark 结论：

| 方法 | ASV EER | ASR WER | Source similarity | Similarity drop | Privacy rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| FreeVC male | 0.417 | 0.345 | 0.159 | 77.8% | 4 |
| Raw metric male | 0.583 | 0.414 | 0.076 | 89.4% | 3 |
| Metric+phone male | 0.583 | 0.655 | 0.061 | 91.4% | 2 |
| PPG-tone male | 0.583 | 0.655 | 0.060 | 91.5% | 1 |

EER 解释：

```text
固定测试集只有 2 个 target trials 和 12 个 non-target trials，所以 EER 是粗台阶指标。多个方法同为 0.583 不代表完全一样，需要再看 source similarity 和 similarity drop。
```

## 8. PPG-tone 调参结论

脚本：

```bash
python tune_ppg_tone_parameters.py
```

结果：

```text
PPG_TONE_TUNING_RESULTS.md
ppg_tone_tuning_report/tuning_metrics.csv
ppg_tone_tuning_report/figures/
```

结论：

- 默认仍用 `strength=0.4`。
- 提高到 `1.0/1.3/1.6` 主要改善 tone/naturalness proxy。
- 高 strength 没有稳定提升 ASV/WER，也会略弱化 source similarity reduction。
- `strength=1.0` 可以作为自然度 ablation，不替换主结果。

## 9. 关键文件

| 文件 | 作用 |
| --- | --- |
| `recording_demo_ui.py` | Web 录音、匿名化、即时评估 |
| `vc_candidate_builder.py` | FreeVC 单候选生成 |
| `build_metric_attack_variants.py` | Metric+phone 后处理候选 |
| `ppg_tone_naturalizer.py` | PPG-inspired 中文声调自然化 |
| `run_ppg_tone_experiment.py` | PPG-tone 批量生成 |
| `evaluate_voiceprivacy.py` | 固定测试集 ASV/ASR 评估 |
| `generate_report_evaluation.py` | 报告图表生成 |
| `tune_ppg_tone_parameters.py` | PPG-tone 参数 sweep |
| `VOICEPRIVACY_RESULTS.md` | 指标解释 |
| `README.md` | 项目入口说明 |

## 10. 常见误区

误区 1：把 ASR WER 越高当作目标。

修正：现在目标是音色匿名 + 内容保留，所以 WER 越低越好。WER 过高表示内容损失。

误区 2：把固定 benchmark 图当作网页录音结果。

修正：网页默认上方是本次录音即时评估；固定 benchmark 在下方折叠区域，只作为报告参考。

误区 3：单条录音也要算 ASV EER。

修正：单条录音不能可靠计算 EER。网页端看 source similarity；报告固定集看 EER + similarity drop。

误区 4：说 PPG-tone 是完整 PPG 神经网络。

修正：应写成 PPG-inspired lightweight content bottleneck + Mandarin tone contour naturalization。

## 11. 后续建议

优先级最高：

1. 扩大评估集：多录 10-20 条中文 source utterances，使 ASV EER 更有区分度。
2. 加人工参考文本输入框：网页端 WER 可以用人工文本而不是 Whisper pseudo-reference。
3. 加主观听感评分：自然度、内容清晰度、是否像原说话人。
4. 对 PPG-tone 做更细的听感 ablation：`strength=0.4` vs `1.0`。

不建议：

- 不要继续追求 ASR WER 最大化。
- 不要把强噪声或强失真版本作为主结果。
- 不要用 `max_metric_vowel_mask_reference` 作为最终展示，它容易触发 ASR 幻觉。

## 12. Git 状态

当前主线已经推送到：

```text
origin/codex/ppg-chinese-tone-naturalness
```

关键提交：

```text
806d0aa Score demo for timbre privacy and content preservation
7b407f6 Evaluate current recording in demo UI
6774936 Add EER tie-break evaluation metrics
d15b530 Add PPG-tone tuning sweep
```

准备合并到 `main` 时，以该分支向 `main` 开 Pull Request。
