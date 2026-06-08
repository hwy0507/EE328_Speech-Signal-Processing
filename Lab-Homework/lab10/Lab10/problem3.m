% Problem 3: Adaptive Quantization via Local Variance Estimation
% Uses: s5.wav

[audio_data, ~] = audioread('s5.wav');

% 去除直流分量
x = audio_data - mean(audio_data);

% 绘图区间：样本 2700 ~ 6699
idx    = 2700:6699;
x_plot = x(idx);

% 静音段防除零偏置
epsilon = 1e-6;
clr_bg = [0.75 0.80 0.85];

%% Part (a): IIR 滤波器（指数衰减窗）估计局部方差
% 差分方程：sigma^2[n] = alpha*sigma^2[n-1] + (1-alpha)*x^2[n-1]

alphas    = [0.9, 0.99];
clr_iir   = [0.85 0.33 0.10;   % α=0.9  包络
             0.47 0.67 0.19];  % α=0.99 包络
clr_eq_iir = [0.00 0.45 0.74;  % α=0.9  均衡信号
              0.64 0.08 0.18]; % α=0.99 均衡信号

for i = 1:length(alphas)
    alpha = alphas(i);

    var_iir = zeros(size(x));
    for n = 2:length(x)
        var_iir(n) = alpha * var_iir(n-1) + (1 - alpha) * x(n-1)^2;
    end
    sigma_iir  = sqrt(var_iir);
    sigma_plot = sigma_iir(idx);
    x_eq       = x_plot ./ (sigma_plot + epsilon);

    figure('Name', sprintf('IIR  α=%.2f', alpha));

    % 语音波形 + 标准差包络
    subplot(2, 1, 1);
    plot(idx, x_plot, 'Color', clr_bg, 'LineWidth', 0.8);
    hold on;
    plot(idx,  sigma_plot, 'Color', clr_iir(i,:), 'LineWidth', 2.0);
    plot(idx, -sigma_plot, 'Color', clr_iir(i,:), 'LineWidth', 2.0);
    hold off;
    title(sprintf('语音信号与 IIR 标准差包络（α = %.2f）', alpha));
    ylabel('幅度'); xlabel('样本序号');
    legend('x[n]', '+σ[n]', '-σ[n]', 'Location', 'northeast');
    grid on; axis tight;

    % 增益均衡后信号
    subplot(2, 1, 2);
    plot(idx, x_eq, 'Color', clr_eq_iir(i,:), 'LineWidth', 0.8);
    title('增益均衡后信号  x[n] / σ[n]');
    ylabel('归一化幅度'); xlabel('样本序号');
    grid on; axis tight;
end

%% Part (b): FIR 矩形窗估计局部方差
% sigma^2[n] = (1/M) * sum_{k=0}^{M-1} x^2[n-k]

M_values   = [10, 100];
clr_fir    = [0.49 0.18 0.56;   % M=10  
              0.19 0.63 0.53];  % M=100 
clr_eq_fir = [0.93 0.69 0.13;   % M=10  
              0.30 0.75 0.93];  % M=100 

for i = 1:length(M_values)
    M = M_values(i);

    % filter 实现因果滑动平均
    var_fir   = filter(ones(1, M) / M, 1, x.^2);
    sigma_fir  = sqrt(var_fir);
    sigma_plot = sigma_fir(idx);
    x_eq       = x_plot ./ (sigma_plot + epsilon);

    figure('Name', sprintf('FIR  M=%d', M));

    % 语音波形 + 标准差包络
    subplot(2, 1, 1);
    plot(idx, x_plot, 'Color', clr_bg, 'LineWidth', 0.8);
    hold on;
    plot(idx,  sigma_plot, 'Color', clr_fir(i,:), 'LineWidth', 2.0);
    plot(idx, -sigma_plot, 'Color', clr_fir(i,:), 'LineWidth', 2.0);
    hold off;
    title(sprintf('语音信号与 FIR 标准差包络（M = %d）', M));
    ylabel('幅度'); xlabel('样本序号');
    legend('x[n]', '+σ[n]', '-σ[n]', 'Location', 'northeast');
    grid on; axis tight;

    % 增益均衡后信号
    subplot(2, 1, 2);
    plot(idx, x_eq, 'Color', clr_eq_fir(i,:), 'LineWidth', 0.8);
    title('增益均衡后信号  x[n] / σ[n]');
    ylabel('归一化幅度'); xlabel('样本序号');
    grid on; axis tight;
end
