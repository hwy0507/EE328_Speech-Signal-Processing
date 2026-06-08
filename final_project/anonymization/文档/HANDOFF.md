# Handoff: EE328 Speech Anonymization Project

更新时间：2026-06-08

## 0. 清理后的目录结构

项目根目录现在只保留两类主要材料：

```text
期末proj/
├── 代码/      # 最终可运行核心代码
├── 文档/      # 交接文档、实验说明、结果图和展示音频
└── README.md  # 新版项目总说明
```

运行 Web Demo 时先进入 `代码/`：

```bash
cd 代码
python recording_demo_ui.py --port 7862
```

最终 benchmark 图表在 `文档/report_evaluation_male/`，PPT 展示音频在 `文档/展示音频/green_m4a_demo/`。

远端仓库：

```text
https://github.com/hwy0507/EE328_Speech-Signal-Processing
```

最终交付主线：

```text
main
```

原始开发分支：

```text
codex/ppg-chinese-tone-naturalness
```

## 1. 项目目标

本项目做的是**语音音色 / 声纹匿名化**，不是关键词扰乱。

正确目标：

1. 匿名输出的音色 / 声纹尽量不像原说话人。
2. 语义内容尽量保留，ASR WER 应尽量低。
3. 听感仍像真人语音，不要明显机械音、断裂、吞字或严重失真。

报告里可以这样概括：

```text
本项目基于 FreeVC 构建中文语音音色匿名化系统，并加入外部男声目标池、指标驱动候选选择、清晰度保护后处理和 PPG-inspired 中文声调自然化，使输出语音在保持真人听感和内容可懂度的同时显著降低与原说话人的声纹相似度。
```

## 2. 当前已经完成

已经完成的主功能：

- FreeVC baseline 匿名化。
- Metric+clarity 保真增强方法。
- PPG-tone 中文声调自然化方法。
- 男声目标池：6 个 CMU ARCTIC 公共男声 + 3 个本地 lab 男声。
- Web 端录音 Demo：开始录音、结束录音、匿名化处理。
- 本次网页录音即时评估：source similarity、similarity drop、ASR WER、content kept、timbre index。
- Top 3 候选音色展示，以及隐私/保真 1:1 平衡分。
- 固定测试集报告图表。
- PPG-tone 参数 sweep 和结果记录。
- README 和交接文档同步。

当前推荐展示方式：

1. 现场用网页录音。
2. 系统自动搜索男声目标池，选择一个 shared base target。
3. 基于同一个 base target 生成三条匿名语音：FreeVC baseline、Metric+clarity、PPG-tone。
4. 网页直接展示本次录音的音色匿名效果和内容保留效果。
5. 固定 benchmark 图表只作为报告参考，不要说成是本次网页录音结果。

## 3. 三种方法怎么理解

| 方法 | 工程名称 | 含义 | 评价重点 |
| --- | --- | --- | --- |
| FreeVC baseline | `freevc_baseline` | 直接用 FreeVC 做男声目标转换 | 基线匿名效果 |
| Metric+clarity | `metric_clarity` | FreeVC 候选筛选 + `clarity_guard` 宽频轻压缩后处理 | 在匿名化和真人清晰度之间折中，减少吞字 |
| PPG-tone | `ppg_tone` | 在 Metric+clarity 基础上加入 PPG-inspired 中文声调自然化 | 中文音调自然度、内容保留和匿名化折中 |

`Metric+clarity` 是当前 Web UI 的主展示增强版。它替代了早期的 `Metric+phone` 网页主线，因为 `phone_clean` 窄带通话信道虽然可能让声纹更不像原人，但更容易导致吞字和 ASR 内容损失。

`PPG-tone` 是本分支的创新点，但要诚实描述：它是 PPG-inspired lightweight content bottleneck + Mandarin tone contour naturalization，不是真正训练出的 neural PPG extractor。

## 4. 为什么三张方法卡 target 一样

网页里三种方法显示同一个 target 是正常现象。

流程是：

```text
先从男声池选出 1 个 shared base target
→ FreeVC baseline 直接输出
→ Metric+clarity 在同一个底座音频上做清晰度保护后处理
→ PPG-tone 再在同一个底座音频上做中文声调自然化
```

所以三张方法卡里的 `Shared base target` 和 `Base pre-score` 可能一样。它们表示的是“共同底座候选”的选择结果，不是三种方法最终效果完全一样。

真正比较三种方法，要看网页下方的即时评估表：

- `Source similarity` / `Standard score`：越低越匿名。
- `Similarity drop`：越高越不像原说话人。
- `ASR WER`：越低越说明内容没被破坏。
- `Content kept`：越高越好。
- `Timbre index`：综合音色匿名和内容保留的本地指数。

## 5. Web UI 怎么用

启动：

```bash
python recording_demo_ui.py --port 7862
```

当前默认就是 9 个男声参考的全池搜索。如果临时想加快处理，可以降低搜索数：

```bash
python recording_demo_ui.py --port 7862 --max-targets 5
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
5. 等待生成三条音频、Top 3 候选音色和即时评估表格。

每次 session 结果在：

```text
work_recording_demo/<session_id>/
```

常见输出：

```text
recording.webm
preprocess/denoised/recording_denoised.wav
target_optimization/raw_freevc_pool/*.wav
metric_clarity/demo_male_metric_clarity.wav
ppg_tone/demo_male_ppg_tone.wav
demo_summary.json
recording_evaluation.json
```

## 6. Top 3 候选音色和平衡公式

网页会展示系统排名前 3 的候选音色。当前平衡分权重是 1:1：

```text
Balance = 0.50 * Privacy + 0.50 * Fidelity
Privacy = clamp((70 - standard_score) / 20)
Fidelity = 0.60 * naturalness + 0.25 * duration_match + 0.15 * processing_simplicity
```

解释：

- `Privacy` 越高，说明候选音色和原说话人越不像。
- `Fidelity` 越高，说明候选更自然、更接近原始时长、没有过度处理。
- `Balance` 不是最终论文指标，只是网页端为“隐私 vs 保真”做的可解释排序分。

## 7. Web UI 指标怎么分析

网页即时评估针对**刚刚录的这条音频**，不是固定 benchmark。

| 指标 | 方向 | 解释 |
| --- | --- | --- |
| `Source similarity` | 越低越好 | 匿名输出和本次原录音的 ECAPA 声纹相似度 |
| `Standard score` | 越低越好 | 按 `standard.docx` 公式转换后的相似度分数：`(cosine + 1) / 2 * 100` |
| `Similarity drop` | 越高越好 | 声纹相似度下降比例 |
| `ASR WER` | 越低越好 | 匿名输出相对本次原录音转写的错误率 |
| `Content kept` | 越高越好 | `1 - WER`，内容保留比例 |
| `Timbre index` | 越高越好 | 综合音色匿名和内容保留的本地指数 |

重要解释：

```text
单条网页录音不能可靠计算 ASV EER，因为 EER 需要一组 target / non-target trials。因此网页端用 source-to-output speaker similarity 作为即时音色匿名证据。
```

不要把 `ASR WER` 解释成越高越好。对于现在的项目目标，WER 越低说明内容保留越好。WER 过高不是成功，而是内容损失。

## 8. 当前最新样例结果

用 `绿色.m4a` 快速验证当前 Web UI 后端流程：

```bash
python run_privacy_optimized_recording.py /Users/hwy/Desktop/个人/26春/语音信号处理/期末proj/绿色.m4a --max-targets 3
```

对应 session：

```text
work_recording_demo/batch_20260603_024353/
```

选中的 shared base target：

```text
male:cmu_arctic_rxr_male_ref/plain
```

本次录音参考文本：

```text
我选择的颜色是绿色
```

关键结果：

| 方法 | Standard score ↓ | Similarity drop ↑ | ASR WER ↓ | Content kept ↑ | Timbre index ↑ | 解读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FreeVC optimized male | 55.880 | 88.24% | 0.778 | 22.22% | 0.684 | 匿名强，但内容识别损失大 |
| Metric+clarity optimized male | 57.625 | 84.75% | 0.000 | 100.00% | 0.893 | 当前最适合网页展示：内容完整、真人清晰度更好 |
| PPG-tone optimized male | 57.637 | 84.73% | 0.556 | 44.44% | 0.726 | 有声调自然化尝试，但该样例内容损失较大 |

报告里建议这样说：

```text
在实时录音样例中，Metric+clarity 在保持 84% 以上 source similarity drop 的同时取得 0.000 ASR WER，说明该版本更适合作为现场展示的“保真优先匿名化”结果。PPG-tone 保留为中文声调自然化创新路径，但单条录音结果可能受内容、录音质量和 ASR 影响，不保证每次都优于 Metric+clarity。
```

## 9. 外部男声目标池

目标池配置：

```text
vc_target_pool_male_external.json
```

包含：

- 6 个 CMU ARCTIC 公共男声参考：`bdl`、`rms`、`jmk`、`awb`、`ksp`、`rxr`。
- 3 个本地 lab 男声参考：`s2.wav`、`s5.wav`、`s6.wav`。

准备或刷新目标池：

```bash
/opt/anaconda3/envs/speech-anon310/bin/python prepare_external_male_targets.py --speakers bdl rms jmk awb ksp rxr --utterances-per-speaker 5 --target-seconds 9
```

注意：

- `external_voice_targets/male/` 是下载/整理出来的音频目录，不提交到 Git。
- Web UI 默认使用 `--max-targets 9` 全池搜索。
- 如果电脑处理慢，可以临时用 `--max-targets 3` 或 `--max-targets 5`。

## 10. 固定 Benchmark 怎么用

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

当前固定 benchmark 历史结论：

| 方法 | ASV EER | ASR WER | Source similarity | Similarity drop | Privacy rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| FreeVC male | 0.417 | 0.345 | 0.159 | 77.8% | 4 |
| Raw metric male | 0.583 | 0.414 | 0.076 | 89.4% | 3 |
| Metric+phone male | 0.583 | 0.655 | 0.061 | 91.4% | 2 |
| PPG-tone male | 0.583 | 0.655 | 0.060 | 91.5% | 1 |

说明：

```text
固定 benchmark 是旧批量实验结果，其中 Metric+phone 是隐私优先对照方法。当前 Web UI 已改用 Metric+clarity，因为最终展示更重视真人清晰度和低 WER。
```

EER 解释：

```text
固定测试集只有 2 个 target trials 和 12 个 non-target trials，所以 EER 是粗台阶指标。多个方法同为 0.583 不代表完全一样，需要再看 source similarity、similarity drop、ASR WER 和听感。
```

## 11. PPG-tone 调参结论

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

## 12. 关键文件

| 文件 | 作用 |
| --- | --- |
| `recording_demo_ui.py` | Web 录音、匿名化、即时评估 |
| `prepare_external_male_targets.py` | 下载/整理 CMU ARCTIC 公共男声目标池 |
| `privacy_target_optimizer.py` | 在男声池中按隐私和自然度联合选择最佳候选 |
| `run_privacy_optimized_recording.py` | 对单个本地录音复现 Web UI 优化流程 |
| `vc_candidate_builder.py` | FreeVC 单候选生成 |
| `build_metric_attack_variants.py` | `clarity_guard`、`phone_clean` 等后处理候选 |
| `ppg_tone_naturalizer.py` | PPG-inspired 中文声调自然化 |
| `run_ppg_tone_experiment.py` | PPG-tone 批量生成 |
| `evaluate_voiceprivacy.py` | 固定测试集 ASV/ASR 评估 |
| `generate_report_evaluation.py` | 报告图表生成 |
| `tune_ppg_tone_parameters.py` | PPG-tone 参数 sweep |
| `VOICEPRIVACY_RESULTS.md` | 指标解释 |
| `EXTERNAL_MALE_TARGET_POOL_RESULTS.md` | 外部男声池和最新低相似度结果 |
| `README.md` | 项目入口说明 |

## 13. 环境说明

本机 FreeVC 默认使用：

```text
/opt/anaconda3/envs/speech-anon310/bin/python3.10
```

Web UI 用当前 shell 的 `python` 启动，内部会调用上面的 FreeVC Python。

主要依赖：

- Python 3.10
- ffmpeg / ffprobe
- FreeVC 相关环境
- SpeechBrain ECAPA speaker model
- faster-whisper small
- numpy / scipy / pandas / matplotlib / seaborn / flask

模型默认使用本地缓存，`evaluate_voiceprivacy.py` 和 `recording_demo_ui.py` 会设置离线 HuggingFace 环境变量。

## 14. 常见误区

误区 1：把 ASR WER 越高当作目标。

修正：现在目标是音色匿名 + 内容保留，所以 WER 越低越好。WER 过高表示内容损失。

误区 2：把三张方法卡的 shared target 当作最终结果。

修正：shared target 是三种方法共用的底座音色。最终效果要看下面的即时评估表和实际音频听感。

误区 3：把固定 benchmark 图当作网页录音结果。

修正：网页默认上方是本次录音即时评估；固定 benchmark 在下方区域，只作为报告参考。

误区 4：单条录音也要算 ASV EER。

修正：单条录音不能可靠计算 EER。网页端看 source similarity；报告固定集看 EER + similarity drop。

误区 5：说 PPG-tone 是完整 PPG 神经网络。

修正：应写成 PPG-inspired lightweight content bottleneck + Mandarin tone contour naturalization。

## 15. 后续建议

优先级最高：

1. 扩大评估集：多录 10-20 条中文 source utterances，使 ASV EER 和平均 WER 更有区分度。
2. 加人工参考文本输入框：网页端 WER 可以用人工文本而不是 Whisper pseudo-reference。
3. 加主观听感评分：自然度、内容清晰度、是否像原说话人。
4. 对 PPG-tone 做更细的听感 ablation：`strength=0.4` vs `1.0`。

不建议：

- 不要继续追求 ASR WER 最大化。
- 不要把强噪声或强失真版本作为主结果。
- 不要用 `max_metric_vowel_mask_reference` 作为最终展示，它容易触发 ASR 幻觉。

## 16. Git 状态

本轮开发分支：

```text
origin/codex/ppg-chinese-tone-naturalness
```

最新关键提交：

```text
5e744c3 Improve web demo clarity balance
7d77d3c Show top target candidates with balance score
32e0080 Expand male target pool for privacy optimization
806d0aa Score demo for timbre privacy and content preservation
7b407f6 Evaluate current recording in demo UI
```

本交接文档更新后应合并到 `main`，以 `main` 作为最终交付版本。
