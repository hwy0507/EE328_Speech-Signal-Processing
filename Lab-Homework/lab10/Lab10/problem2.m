% Problem 2: SNR Comparison - Uniform Quantization vs mu-law Quantization
% Uses: mulaw.m, mulawinv.m, fxquant.m, SNR.m

[audio_data, ~] = audioread('s5.wav');
x_original = audio_data(1300:18800);

%% Part (a): 验证 6 dB/bit 规律（8-bit vs 9-bit 均匀量化）
xh_8 = fxquant(x_original, 8, 'round', 'sat');
xh_9 = fxquant(x_original, 9, 'round', 'sat');

snr_8 = SNR(xh_8, x_original);
snr_9 = SNR(xh_9, x_original);

fprintf('=== Part (a): 6 dB/bit 规律验证 ===\n');
fprintf('8-bit 均匀量化 SNR : %.2f dB\n', snr_8);
fprintf('9-bit 均匀量化 SNR : %.2f dB\n', snr_9);
fprintf('SNR 增量           : %.2f dB（理论值 ≈ 6 dB）\n\n', snr_9 - snr_8);

%% Part (b): SNR vs 1/σx 半对数曲线（动态范围 64:1）
% 衰减因子：2^0 → 2^-12，共 13 个电平
factors = 2.^(0:-1:-12);

bits_uniform = [10, 9, 8, 7, 6];
mu_values    = [100, 255, 500];
B_mu         = 6;

snr_uniform = zeros(length(bits_uniform), length(factors));
snr_mulaw   = zeros(length(mu_values),    length(factors));
inv_sigma_x = zeros(1, length(factors));

for i = 1:length(factors)
    x_scaled       = x_original * factors(i);
    inv_sigma_x(i) = 1 / rms(x_scaled);

    for b = 1:length(bits_uniform)
        xh = fxquant(x_scaled, bits_uniform(b), 'round', 'sat');
        snr_uniform(b, i) = SNR(xh, x_scaled);
    end

    for m = 1:length(mu_values)
        mu      = mu_values(m);
        y_comp  = mulaw(x_scaled, mu);
        y_quant = fxquant(y_comp, B_mu, 'round', 'sat');
        xh_mu   = mulawinv(y_quant, mu);
        snr_mulaw(m, i) = SNR(xh_mu, x_scaled);
    end
end

% --- 绘图 ---
figure('Name', 'SNR 动态范围对比');
hold on;

clr_uni = [0.09 0.47 0.69;  
           0.20 0.63 0.17;   
           0.55 0.34 0.71;   
           0.89 0.47 0.06;   
           0.68 0.18 0.18];  

for b = 1:length(bits_uniform)
    semilogx(inv_sigma_x, snr_uniform(b, :), '-^', ...
        'Color', clr_uni(b, :), 'LineWidth', 1.6, 'MarkerSize', 5, ...
        'DisplayName', sprintf('%d-bit 均匀量化', bits_uniform(b)));
end

% μ律量化
clr_mu = [0.00 0.75 0.75;   %  μ=100
          0.95 0.60 0.00;   %  μ=255
          0.80 0.20 0.60];  %  μ=500

mu_labels = {'μ=100','μ=255','μ=500'};
for m = 1:length(mu_values)
    semilogx(inv_sigma_x, snr_mulaw(m, :), '-.d', ...
        'Color', clr_mu(m, :), 'LineWidth', 2.0, 'MarkerSize', 6, ...
        'DisplayName', sprintf('6-bit μ律 (%s)', mu_labels{m}));
end

hold off;
grid on; box on;
title('均匀量化与 μ 律量化 SNR 随动态范围的变化');
xlabel('1/\sigma_x（对数刻度）');
ylabel('信噪比 SNR (dB)');
legend('Location', 'southwest', 'NumColumns', 2);
