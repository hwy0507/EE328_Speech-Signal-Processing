# External Male Target Pool Results

更新时间：2026-06-03

本轮优化目标：扩大男声匿名目标池，在保持真人音色和中文内容可懂度的前提下，尽量降低 `standard.docx` 中的 speaker embedding similarity。

## 数据来源

外部男声参考音色来自 CMU ARCTIC 公共语音库：

- 官方页面：https://www.festvox.org/cmu_arctic/
- 论文/说明：https://www.cs.cmu.edu/~awb/papers/ssw5/arctic.pdf

本项目只把生成脚本和目标池配置提交到 Git。下载后的本地音频在：

```text
external_voice_targets/male/
```

该目录已经写入 `.gitignore`，不直接提交音频文件，方便后续回退和复现。

## 目标池

当前男声池共 9 个参考：

| 类别 | 参考音色 |
| --- | --- |
| CMU ARCTIC male | `bdl`, `rms`, `jmk`, `awb`, `ksp`, `rxr` |
| 本地 lab male | `s2`, `s5`, `s6` |

配置文件：

```text
vc_target_pool_male_external.json
```

重新准备外部男声池：

```bash
/opt/anaconda3/envs/speech-anon310/bin/python prepare_external_male_targets.py --speakers bdl rms jmk awb ksp rxr --utterances-per-speaker 5 --target-seconds 9
```

## 选择策略

新增脚本：

```text
privacy_target_optimizer.py
```

流程：

1. 对源语音和全部目标参考抽取 ECAPA speaker embedding。
2. 先按源音频与目标参考的 embedding similarity 做预筛选。
3. 对候选目标逐个运行 FreeVC。
4. 在 FreeVC 输出上生成轻量变体：`plain`、`tempo_096`、`tempo_104`、`pitch_096`、`pitch_104`。
5. 用综合分数选择候选：speaker similarity 越低越好，同时惩罚过强语速/音高变化、时长偏移和削波。

这样做不是把声音处理到越怪越好，而是选择“声纹不像原人，但仍像真人语音”的折中点。

## 最新测试

测试音频：

```text
/Users/hwy/Desktop/个人/26春/语音信号处理/期末proj/绿色.m4a
```

命令：

```bash
python run_privacy_optimized_recording.py /Users/hwy/Desktop/个人/26春/语音信号处理/期末proj/绿色.m4a --max-targets 9
```

输出 session：

```text
work_recording_demo/batch_20260603_015207/
```

选中的原始 FreeVC 候选：

| 项目 | 数值 |
| --- | ---: |
| Selected target | `cmu_arctic_bdl_male_ref/plain` |
| Target pool size | 9 |
| Evaluated target count | 9 |
| Standard score | 54.450 / 100 |
| Raw cosine similarity | 0.089 |
| Naturalness proxy | 0.993 |

即时评估结果：

| 方法 | Raw cosine ↓ | Standard score ↓ | Similarity drop ↑ | ASR WER ↓ | Content kept ↑ | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FreeVC optimized male | 0.089 | 54.450 | 91.10% | 0.000 | 100.00% | 内容完整，声纹已明显远离原人 |
| Metric+phone optimized male | 0.081 | 54.026 | 91.95% | 0.222 | 77.78% | 声纹更低，但语义有轻微损失 |
| PPG-tone optimized male | 0.079 | 53.935 | 92.13% | 0.222 | 77.78% | 本次最低相似度，仍保持真人听感 |

ASR reference：

```text
我选择的颜色是绿色
```

FreeVC hypothesis：

```text
我選擇的顏色是綠色
```

代码已经加入繁简归一化，因此这类“简体/繁体差异”不会再被误算为内容错误。

## 如何解释这个数字

`standard.docx` 的分数可以理解为：

```text
standard score = (cosine + 1) / 2 * 100
```

所以：

- `100` 表示 embedding 方向几乎完全一致，极像同一个人。
- `50` 表示 cosine 接近 0，接近“不相关说话人”的状态。
- 本轮最优 PPG-tone 的 `53.935/100` 对应 raw cosine `0.079`，已经接近随机不相关，而不是旧结果里约 `0.7` 的高相似状态。

报告里建议写：

```text
扩大男声目标池并引入隐私/自然度联合选择后，绿色样例的源说话人 raw cosine similarity 降至约 0.079；按课程标准公式换算为 53.94/100，接近不相关说话人的 50 分基线。同时 ASR 仍能保留主要中文语义，说明系统实现的是音色匿名化，而不是简单破坏语音内容。
```

