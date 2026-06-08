# 语音信号处理实验三报告

## 基本信息
- 课程：语音信号处理
- 实验题目：LAB III Hearing and Speech Perception
- 姓名：`（填写）`
- 学号：`（填写）`
- 日期：`（填写）`

---

## 一、Problem 1（bonus）

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 在 Lab 2 中采样元音 /a/ /i/ /u/ 的谱包络 | 1.1 |
| 2 | 用纯音合成 /a/ /i/ /u/（男声 F0=150 Hz，女声 F0=250 Hz） | 1.2 |
| 3 | 绘制三元音的波形/频谱/语谱图 | 1.3 |
| 4 | 将合成元音 wav 文件随报告提交 | 见“六、附件说明” |

### 1.1 谱包络采样
- 对 /a/、/i/、/u/ 各取稳态段，计算单边频谱并在 dB 域平滑，得到谱包络。
- 包络采样结果导出为 `csv` 表格用于后续谐波加权。

### 1.2 纯音合成方法
- 采样率：16 kHz。
- 对每个元音，以基频 `F0` 的整数倍生成谐波纯音，并按谱包络在各谐波频率处的幅值进行加权叠加。
- 条件设置：
  - 男声：`F0 = 150 Hz`
  - 女声：`F0 = 250 Hz`

### 1.3 波形/频谱/语谱图结果
- 男声（F0=150 Hz）

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem1/P1_male_wave_spec_specgram.png" alt="P1_male" style="zoom:50%;" />

- 女声（F0=250 Hz）

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem1/P1_female_wave_spec_specgram.png" alt="P1_female" style="zoom:50%;" />

### 1.4 关键代码段（MATLAB）
```matlab
% 提取稳态段并估计谱包络（dB 平滑）
[fEnv, envDb] = estimateSpectralEnvelope(xv, fsv);

% 纯音谐波叠加合成（F0=150/250）
y = synthVowelFromEnvelope(fEnv, envDb, fsSyn, f0, durSec);

% 语谱图
spectrogram(y, round(0.02*fs), round(0.01*fs), 1024, fs, "yaxis");
```

### 1.5 小结
- 合成信号频谱呈明显谐波结构；
- 三个元音在谱包络形状上可区分；
- 语谱图中的谐波轨迹平稳，符合稳态元音合成信号特征。

---

## 二、Problem 2

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 完成在线听觉测试（sensitivity、equal loudness contour） | 2.1 |
| 2 | 截图并展示你得到的等响度曲线 | 2.2 |

### 2.1 在线测试说明
- 测试网址：<http://www.phys.unsw.edu.au/jw/hearing.html>
- 测试内容：
  - 听觉灵敏度（sensitivity）
  - 等响度曲线（equal loudness contour）

### 2.2 测试结果截图
<img src="/Users/hwy/Library/Application Support/typora-user-images/image-20260314110903108.png" alt="image-20260314110903108" style="zoom:50%;" />

### 2.3 简要结论
- 低频和高频区域通常需要更高声压级才能感知为与中频相同响度。
- 本次测试曲线整体呈典型 U 形：约 1--2 kHz 附近所需电平最低（最敏感），向低频端（30--125 Hz）和高频端（8--16 kHz）逐渐上升。
- 该结果与人耳等响度规律一致，即中频最敏感，低频和高频需要更大声压级才能达到与中频相同的主观响度。

---

## 三、Problem 3

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 对 `mhint_01_01.wav` 添加白噪声，SNR=5 dB 和 0 dB | 3.1 |
| 2 | 将带噪语音能量归一化到干净语音能量 | 3.1 |
| 3 | 绘制 clean 与 equalized noisy 的波形和频谱 | 3.2 |
| 4 | 绘制 clean 与 noisy（5/0 dB）的语谱图 | 3.3 |

### 3.1 加噪与能量归一化
- 对 clean speech 分别生成 5 dB 和 0 dB 白噪语音。
- 使用 `y_eq = y / norm(y) * norm(clean)` 进行能量均衡。
- 实测 SNR 与目标值一致（约 5 dB 和 0 dB）。

### 3.2 波形与频谱结果
- 波形图（clean + equalized noisy）

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem3/P3_waveforms_clean_eq_noisy.png" alt="P3_wave" style="zoom:50%;" />

- 频谱图（分开显示 clean、5 dB、0 dB）

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem3/P3_spectra_clean_eq_noisy.png" alt="P3_spectrum" style="zoom:50%;" />

### 3.3 语谱图结果

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem3/P3_spectrograms_clean_noisy.png" alt="P3_specgram" style="zoom:50%;" />

### 3.4 关键代码段（MATLAB）
```matlab
% 按目标 SNR 加白噪
[yNoisy, snrActual] = addWhiteNoiseAtSNR(clean, snrTarget);

% 能量均衡到 clean
yEq = yNoisy / (norm(yNoisy) + eps) * norm(clean);

% 频谱分开绘制（clean / 5 dB / 0 dB）
[fc, mc] = oneSidedSpectrum(clean, fs);
[f5, m5] = oneSidedSpectrum(noisyEq{1}, fs);
[f0, m0] = oneSidedSpectrum(noisyEq{2}, fs);
```

### 3.5 结果分析
- 0 dB 条件下噪声明显，但语音结构信息（节律与主要频谱特征）仍部分可辨；
- 随 SNR 降低，背景噪声增强，语谱图中语音细节可见度下降。

---

## 四、Problem 4（Objective Speech Quality Evaluation）

### 题目原始要求
| 要求编号 | 题目要求 | 本报告对应位置 |
|---|---|---|
| 1 | 对 clean speech 添加白噪声（-5,-3,-1,1,3,5 dB） | 4.1 |
| 2 | 将 noisy speech 能量归一化到 clean speech | 4.1 |
| 3 | 运行 PESQ 并绘制 PESQ-SNR 曲线 | 4.2, 4.3 |

### 4.1 数据构造
- 噪声条件：`-5, -3, -1, 1, 3, 5 dB`。
- 每个 SNR 条件下先加白噪，再做能量均衡，然后计算 PESQ。

### 4.2 PESQ 计算方法
- 使用本地 `pesq.m`：`scores = pesq(ref_wav, deg_wav)`。
- 当前 16 kHz（wideband）模式下使用返回的 MOS-LQO 作为 `PESQ` 指标。

### 4.3 PESQ 结果表与曲线
- 结果表（来自 `P4_pesq_vs_snr.csv`）：

| SNR_Target_dB | SNR_Actual_dB | PESQ |
|---:|---:|---:|
| -5 | -5.0000 | 1.0741 |
| -3 | -3.0000 | 1.0821 |
| -1 | -1.0000 | 1.0908 |
| 1  | 1.0000  | 1.1023 |
| 3  | 3.0000  | 1.1164 |
| 5  | 5.0000  | 1.1346 |

- PESQ-SNR 曲线图：

<img src="/Users/hwy/Desktop/个人/26春/语音信号处理/lab3/lab3_outputs/problem4/P4_pesq_vs_snr.png" alt="P4_pesq" style="zoom:50%;" />

### 4.4 关键代码段（MATLAB）
```matlab
% 使用本地 pesq.m
addpath(fileparts(pesqFile), '-begin');
scores = pesq(char(refWav), char(degWav));

% wideband / narrowband 统一取最终 MOS-LQO
pesqVal(i) = double(scores(end));

% 绘制 PESQ-SNR 曲线
plot(snrList, pesqVal, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
```

### 4.5 结果分析
- 随 SNR 从 -5 dB 提升到 5 dB，PESQ 单调上升；
- 说明在本实验条件下，噪声减弱会稳定提升客观语音质量评分。

---

## 五、结论

1. Problem 1（bonus）：完成了谱包络采样、纯音合成及波形/频谱/语谱图分析。
2. Problem 2：已给出在线测试流程，截图与个人观察。
3. Problem 3：完成了 5 dB/0 dB 噪声构造、能量均衡及时域/频域/时频域分析。
4. Problem 4：完成了多 SNR 条件下的 PESQ 评估与趋势分析。

---

## 六、附件说明

本报告正文中已省略所有音频文件细节。与实验相关的全部音频（原始、加噪、均衡、合成）统一随作业附件提交。
