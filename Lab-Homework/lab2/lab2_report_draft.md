# 语音信号处理实验二报告

## 基本信息
- 课程：语音信号处理
- 实验题目：LAB II Fundamentals of Human Speech Production
- 姓名：`（填写）`
- 学号：`（填写）`
- 日期：`（填写）`

---

## 一、Problem 1

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 用 ARPAbet 写出 “Oak is strong and also gives shade” 的音标转写 | 1.1 |
| 2 | 用 plotting/listening 对 `s5.wav` 分段并标注（含首尾静音、可能中间静音） | 1.2, 1.3 |
| 3 | 把所有元音区间置零并听感知结果 | 1.4 |
| 4 | 把所有辅音区间置零并听感知结果 | 1.5 |
| 5 | 判断两种修改语音哪一个更可懂 | 1.6 |

### 1.1 ARPAbet 转写
句子：`Oak is strong and also gives shade.`

ARPAbet：
`OW K | IH Z | S T R AO NG | AE N D | AO L S OW | G IH V Z | SH EY D`

### 1.2 分段与标注方法
- 输入音频：`s5.wav`
- 结合波形显示与试听对语音进行 25 段标注（含首尾静音）。
- 结果保存到：`problem1_segments.csv`

### 1.3 分段与标注结果表（段之间相差一个采样周期）

| 段号 | 时间区间(s) | 标签 | 类型 | 单词 |
|---|---|---|---|---|
| 1 | 0.000000 - 0.159750 | SIL | silence | - |
| 2 | 0.159875 - 0.279750 | OW | vowel | Oak |
| 3 | 0.279875 - 0.344750 | K | consonant | Oak |
| 4 | 0.344875 - 0.579750 | IH | vowel | is |
| 5 | 0.579875 - 0.724750 | Z | consonant | is |
| 6 | 0.724875 - 0.769750 | S | consonant | strong |
| 7 | 0.769875 - 0.784750 | T | consonant | strong |
| 8 | 0.784875 - 0.804750 | R | consonant | strong |
| 9 | 0.804875 - 0.864750 | AO | vowel | strong |
| 10 | 0.864875 - 1.194750 | NG | consonant | strong |
| 11 | 1.194875 - 1.214750 | AE | vowel | and |
| 12 | 1.214875 - 1.274750 | N | consonant | and |
| 13 | 1.274875 - 1.319750 | D | consonant | and |
| 14 | 1.319875 - 1.369750 | AO | vowel | also |
| 15 | 1.369875 - 1.499750 | L | consonant | also |
| 16 | 1.499875 - 1.589750 | S | consonant | also |
| 17 | 1.589875 - 1.684750 | OW | vowel | also |
| 18 | 1.684875 - 1.779750 | G | consonant | gives |
| 19 | 1.779875 - 1.884750 | IH | vowel | gives |
| 20 | 1.884875 - 1.934750 | V | consonant | gives |
| 21 | 1.934875 - 1.999750 | Z | consonant | gives |
| 22 | 1.999875 - 2.094750 | SH | consonant | shade |
| 23 | 2.094875 - 2.349750 | EY | vowel | shade |
| 24 | 2.349875 - 2.429750 | D | consonant | shade |
| 25 | 2.429875 - 2.999875 | SIL | silence | - |

- 分割标注图（蓝色元音，红色辅音）

<img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312145157004.png" alt="image-20260312145157004" style="zoom:50%;" />

### 1.4 元音置零结果
- 处理方式：将标注为元音（vowel）的区间样本置零。

- 输出文件：`problem1_no_vowels.wav`

- 置零结果图：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312145339386.png" alt="image-20260312145339386" style="zoom:50%;" />

### 1.5 辅音置零结果
- 处理方式：将标注为辅音（consonant）的区间样本置零。

- 输出文件：`problem1_no_consonants.wav`

- 置零结果图：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312145411695.png" alt="image-20260312145411695" style="zoom:50%;" />

### 1.6 可懂度判断
- 比较对象：
  - 缺失元音版本（`problem1_no_vowels.wav`）
  - 缺失辅音版本（`problem1_no_consonants.wav`）
  
- 结论：**两种处理后语音都明显降低可懂度。对比发现，元音置零后仍能较多保留单词轮廓；辅音置零后单词边界与区分信息下降更明显。说明在本实验语句中，辅音对词汇区分（可懂度）贡献更大，而元音更多承载能量与韵律信息。**

  

---

## 二、Problem 2

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 在 MATLAB 中用 `audiowrite` 和 `audiorecorder` 录制 /a/ /i/ /u/ | 2.1 |
| 2 | 使用 FFT 分析录制元音频谱 | 2.2 |
| 3 | 绘制 dB 频谱并找谱包络峰值频率 | 2.3 |
| 4 | 用 Praat 重复录音和分析，并展示结果 | 2.4 |
| 5 | 比较 FFT 与 Praat 提取的 formant frequencies | 2.5 |

### 2.1 MATLAB 录音结果
- 采样率：16 kHz
- 每个元音录制时长：1.5 s
- 输出文件：
  - `problem2_a.wav`（AA）
  - `problem2_i.wav`（IY）
  - `problem2_u.wav`（UW）

### 2.2 FFT 分析方法
- 取稳态 40 ms 帧进行分析。
- 加 Hamming 窗，`nfft = 8192`。
- 计算单边幅度谱并转 dB。

### 2.3 dB 频谱与谱包络峰值
- 绘图内容：灰线 FFT、红线平滑包络、蓝点 F1/F2/F3。

- 同时叠加标准参考点（绿菱形 C1/C2/C3）用于可视化对比。

- 由于包络线的极值点不太好判断，我在本实验中先确定了理论共振峰（见下图）位置，再在邻域寻找所录制音频的共振峰

- 理论共振峰：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150041136.png" alt="image-20260312150041136" style="zoom: 33%;" />

MATLAB（FFT）提取结果（来自 `problem2_formants_fft.csv`）：

| 元音 | F1 (Hz) | F2 (Hz) | F3 (Hz) |
|---|---:|---:|---:|
| AA | 451.17 | 912.11 | 1921.88 |
| IY | 189.45 | 2669.92 | 2783.20 |
| UW | 216.80 | 490.23 | 1720.70 |

- 图3：三元音 FFT 频谱图

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150450279.png" alt="image-20260312150450279" style="zoom:50%;" />

  #### 具体结论如下（后续用Praat进一步分析）：

  - 与理论值存在明显偏差，尤其是 AA 和 UW 的 F3 偏低、UW 的 F2 偏低最明显。
  - 但元音总体模式仍正确：IY 的 F2 最高、UW 的 F2 最低、AA 的 F1 相对更高，说明类别区分方向是对的。
  - AA 的 F1 明显低于理论，说明发音可能不够“开口”（更接近中后元音）；UW 的 F2 很低，说明圆唇和后舌位较强；IY 的 F2 偏高，说明前舌位更靠前。

### 2.4 Praat 分析结果
- 使用 Praat 对 `problem2_a.wav / problem2_i.wav / problem2_u.wav` 做频谱分析。

- 在稳态中段读出 F1/F2/F3。

- 音标AA分析图：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150701306.png" alt="image-20260312150701306" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150722870.png" alt="image-20260312150722870" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150746137.png" alt="image-20260312150746137" style="zoom:25%;" />

- 音标IY分析图：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150911327.png" alt="image-20260312150911327" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150925458.png" alt="image-20260312150925458" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312150944610.png" alt="image-20260312150944610" style="zoom:25%;" />

  

- 音标UW分析图：

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312151211048.png" alt="image-20260312151211048" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312151225622.png" alt="image-20260312151225622" style="zoom:25%;" />

  <img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312151242187.png" alt="image-20260312151242187" style="zoom:25%;" />

- Praat 结果表：

| 元音 | F1 (Hz) | F2 (Hz) | F3 (Hz) |
|---|---:|---:|---:|
| AA | 579.9 | 959.7 | 2816 |
| IY | 354.6 | 2810 | 3424 |
| UW | 342.6 | 809.1 | 2526 |

- 对比表（Δ=FFT-Praat）：

| 元音 | F1_FFT | F1_Praat | ΔF1 | F2_FFT | F2_Praat | ΔF2 | F3_FFT | F3_Praat | ΔF3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AA | 451.17 | 579.90 | -128.73 | 912.11 | 959.70 | -47.59 | 1921.88 | 2816.00 | -894.12 |
| IY | 189.45 | 354.60 | -165.15 | 2669.92 | 2810.00 | -140.08 | 2783.20 | 3424.00 | -640.80 |
| UW | 216.80 | 342.60 | -125.80 | 490.23 | 809.10 | -318.87 | 1720.70 | 2526.00 | -805.30 |

- 与标准值对比（标准值：AA=[730,1090,2440]，IY=[270,2290,3010]，UW=[300,870,2240]）：

| 元音 | ΔF1_FFT-Std | ΔF1_Praat-Std | ΔF2_FFT-Std | ΔF2_Praat-Std | ΔF3_FFT-Std | ΔF3_Praat-Std |
|---|---:|---:|---:|---:|---:|---:|
| AA | -278.83 | -150.10 | -177.89 | -130.30 | -518.12 | +376.00 |
| IY | -80.55 | +84.60 | +379.92 | +520.00 | -226.80 | +414.00 |
| UW | -83.20 | +42.60 | -379.77 | -60.90 | -519.30 | +286.00 |

- Problem 2 结论：**本次实测 formant 与理论均值存在系统偏差，但三元音相对分布关系与理论一致，说明元音类别识别有效；偏差主要来自个体发音差异与估计方法误差。**

---

## 三、Problem 3

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 对语音文件做高通滤波，去除潜在 DC offset 和/或 60 Hz hum | 3.1, 3.3 |
| 2 | 设计高通滤波器，设置 stop/passband edge，并保证 60 Hz 至少衰减 40 dB | 3.2, 3.3 |
| 3 | 建议使用线性相位 FIR | 3.2 |
| 4 | 滤波后生成新文件（实验要求标注为 create a new file） | 3.4 |

### 3.1 任务说明
- 输入文件：`test_16k.wav`
- 目标：抑制 DC 偏置和 60 Hz 干扰。

### 3.2 滤波器设计
- 滤波器类型：线性相位 FIR 高通

- stopband edge：60 Hz

- passband edge：120 Hz

- 设计目标：60 Hz 分量衰减 ≥ 40 dB

  ![image-20260312151553731](/Users/hwy/Library/Application Support/typora-user-images/image-20260312151553731.png)

### 3.3 结果与量化验证（MATLAB输出结果）

<img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260312151450763.png" alt="image-20260312151450763" style="zoom:50%;" />

无论是实测结果还是理论设计，在60Hz处的reduction均满足要求。

![image-20260312151926251](/Users/hwy/Library/Application Support/typora-user-images/image-20260312151926251.png)

### 3.4 输出文件
- 生成新文件：`test_16k_hp.wav`

---

## 四、结论

本次实验完整完成了 Problem 1-3 的全部要求，结论如下：

1. Problem 1：已完成 ARPAbet 转写、`s5.wav` 分段标注、元音置零与辅音置零试听比较。两种处理均降低可懂度；在本句中辅音对词汇区分信息贡献更显著，元音更多承载能量与韵律信息。
2. Problem 2：已完成 MATLAB 录音、FFT 频谱与谱包络峰值提取，并完成 Praat Formant 分析及对比。结果显示实测 formant 与理论均值存在偏差，但三元音相对分布关系与理论一致，元音类别识别有效。
3. Problem 3：已完成线性相位 FIR 高通滤波设计与实现，60 Hz 衰减达到 40 dB 以上（理论 40.62 dB，实测窄带功率下降 40.80 dB），DC 偏置显著降低，并生成新文件 `test_16k_hp.wav`。

综合来看，本实验验证了“语音分段与标注—共振峰分析—低频噪声抑制”这条完整处理链路；方法与结果均满足题目要求。
