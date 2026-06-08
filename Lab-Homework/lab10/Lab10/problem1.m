% Problem 1: mu-law Compression and Quantization Error Analysis

%% Part (a): 绘制不同 mu 值下的压缩特性曲线
x = -1:0.001:1;
mu_values = [1, 20, 50, 100, 255, 500];

clr_a = [0.12 0.47 0.71;   % 蓝
         0.20 0.63 0.17;   % 绿
         0.89 0.10 0.11;   % 红
         1.00 0.50 0.05;   % 橙
         0.42 0.24 0.60;   % 紫
         0.65 0.34 0.16];  % 棕

figure('Name', 'μ律压缩特性');
hold on;
ls = {'-', '--', ':', '-.', '-', '--'};
for i = 1:length(mu_values)
    plot(x, mulaw(x, mu_values(i)), ls{i}, ...
        'Color', clr_a(i,:), 'LineWidth', 2);
end
hold off;
title('μ律压缩特性曲线');
xlabel('输入信号 x');
ylabel('压缩输出 y');
legend('μ=1','μ=20','μ=50','μ=100','μ=255','μ=500','Location','southeast');
grid on; box on;

%% Part (b): 对 s5.wav 片段进行 μ=255 压缩
[audio_data, fs] = audioread('s5.wav');
x_speech = audio_data(1300:18800);
mu = 255;

y_speech = mulaw(x_speech, mu);

figure('Name', 'μ律压缩输出');
subplot(2, 1, 1);
plot(y_speech, 'Color', [0.17 0.51 0.73], 'LineWidth', 0.8);
title('μ律压缩后时域波形  (μ=255)');
xlabel('样本序号');
ylabel('幅度');
axis tight; grid on;

subplot(2, 1, 2);
histogram(y_speech, 40, 'FaceColor', [0.30 0.69 0.29], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);
title('压缩输出样本直方图');
xlabel('幅度');
ylabel('频次');
grid on;

%% Part (c): 验证压缩-扩展闭环精度
v = mulawinv(mulaw(2, 255), 255);
fprintf('=== Part (c): 闭环验证 ===\n');
fprintf('原始输入  : 2\n');
fprintf('恢复值    : %.8f\n', v);
fprintf('绝对误差  : %.2e\n\n', abs(v - 2));

%% Part (d): 10/8/4-bit 量化误差分析
x_test    = x_speech(1:8000);
bits_list = [10, 8, 4];

clr_d = [0.00 0.62 0.75;   % 10-bit 
         0.93 0.35 0.05;   % 8-bit  
         0.13 0.59 0.20];  % 4-bit  

fig_wave = figure('Name', '量化误差波形');
fig_hist = figure('Name', '误差概率分布');
fig_spec = figure('Name', '量化噪声功率谱密度');

bit_labels = {'10-bit','8-bit','4-bit'};

for i = 1:length(bits_list)
    B = bits_list(i);

    % 压缩 → 均匀量化  扩展
    y_comp  = mulaw(x_test, mu);
    y_quant = fxquant(y_comp, B, 'round', 'sat');
    x_hat   = mulawinv(y_quant, mu);
    [~, e]  = SNR(x_hat, x_test);

    % 误差波形
    figure(fig_wave);
    subplot(3, 1, i);
    plot(e, 'Color', clr_d(i,:), 'LineWidth', 0.7);
    title(['量化误差波形  ', bit_labels{i}]);
    xlabel('样本序号'); ylabel('误差幅度');
    ylim([-0.3 0.3]); grid on;

    % 误差直方图（30 bins，半透明填充）
    figure(fig_hist);
    subplot(3, 1, i);
    histogram(e, 30, 'FaceColor', clr_d(i,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.8);
    title(['误差概率分布  ', bit_labels{i}]);
    xlabel('误差幅度'); ylabel('频次');
    xlim([-0.3 0.3]); grid on;

    % 量化噪声功率谱（Welch，汉明窗）
    figure(fig_spec);
    hold on;
    [pxx, f] = pwelch(e, 256, 128, 512, fs);
    plot(f, 10*log10(pxx), 'Color', clr_d(i,:), ...
        'LineWidth', 1.8, 'DisplayName', bit_labels{i});
end

figure(fig_spec);
title('μ律量化噪声功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');
legend('Location', 'best');
grid on; hold off;
