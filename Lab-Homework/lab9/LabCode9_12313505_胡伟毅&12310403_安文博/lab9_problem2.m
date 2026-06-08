%% LAB IX - Problem 2: Uniform quantization of speech
% This script uses fxquant.m to study quantizer characteristics,
% quantization error sequences, histograms, and error power spectra.

clear; close all; clc;

outputDir = fullfile(pwd, 'lab9_output');
figureDir = fullfile(outputDir, 'figures');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
if ~exist(figureDir, 'dir'), mkdir(figureDir); end

%% Problem 2(a)
xin = (-1:0.001:1).';
bits = 4;
xRound = fxquant(xin, bits, 'round', 'sat');
xTrunc = fxquant(xin, bits, 'trunc', 'sat');
eRound = xRound - xin;
eTrunc = xTrunc - xin;

figure('Name', 'Problem 2(a) - Quantizer characteristics');
plot(xin, xRound, 'LineWidth', 1.2); hold on;
plot(xin, xTrunc, '--', 'LineWidth', 1.2);
grid on;
xlabel('Input x[n]');
ylabel('Quantized output');
title('4-bit quantizer characteristic, lmode = sat');
legend('rmode = round', 'rmode = trunc', 'Location', 'best');
saveas(gcf, fullfile(figureDir, 'p2a_quantizer_characteristic.png'));

fprintf('Problem 2(a): truncation error range = [%.6f, %.6f]\n', min(eTrunc), max(eTrunc));

%% Problem 2(b)
[s5, fs] = audioread('s5.wav');
if size(s5, 2) > 1
    s5 = mean(s5, 2);
end
speechRange = 1300:18800;
speech = s5(speechRange);
bitRates = [10 8 4];
errors = cell(size(bitRates));
quantized = cell(size(bitRates));
snrDb = zeros(size(bitRates));
noiseVariance = zeros(size(bitRates));

for k = 1:numel(bitRates)
    quantized{k} = fxquant(speech, bitRates(k), 'round', 'sat');
    errors{k} = quantized{k} - speech;
    snrDb(k) = 10 * log10(mean(speech .^ 2) / mean(errors{k} .^ 2));
    noiseVariance(k) = var(errors{k}, 1);

    figure('Name', sprintf('Problem 2(b) - Error strips, %d bits', bitRates(k)));
    strips(errors{k}(1:8000), 2000 / fs, fs);
    title(sprintf('First 8000 quantization-error samples, %d bits', bitRates(k)));
    saveas(gcf, fullfile(figureDir, sprintf('p2b_error_strips_%dbit.png', bitRates(k))));
end

figure('Name', 'Problem 2(b) - Error histograms');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for k = 1:numel(bitRates)
    nexttile;
    histogram(errors{k}, 60);
    grid on;
    xlabel('Quantization error');
    ylabel('Count');
    title(sprintf('%d bits', bitRates(k)));
end
saveas(gcf, fullfile(figureDir, 'p2b_error_histograms.png'));

%% Problem 2(c)
Nfft = 2048;
Nwin = 256;
[Ps, F] = pspect(speech, fs, Nfft, Nwin);

figure('Name', 'Problem 2(c) - Speech and quantization-noise spectra');
plot(F, 10 * log10(Ps + eps), 'k', 'LineWidth', 1.4); hold on;
noiseSpectraDb = zeros(numel(bitRates), length(F));
for k = 1:numel(bitRates)
    [Pe, Fe] = pspect(errors{k}, fs, Nfft, Nwin);
    noiseSpectraDb(k, :) = 10 * log10(Pe + eps);
    plot(Fe, noiseSpectraDb(k, :), 'LineWidth', 1.1);
end
grid on;
xlabel('Frequency (Hz)');
ylabel('Power spectrum (dB)');
title('Original speech and quantization-noise spectra');
legend('Original speech', '10-bit noise', '8-bit noise', '4-bit noise', 'Location', 'best');
saveas(gcf, fullfile(figureDir, 'p2c_noise_spectra.png'));

noise10minus8Db = mean(noiseSpectraDb(1, :) - noiseSpectraDb(2, :));

fid = fopen(fullfile(outputDir, 'problem2_stats.txt'), 'w');
fprintf(fid, 'Problem 2(a)\n');
fprintf(fid, 'Truncation error range: [%.10f, %.10f]\n', min(eTrunc), max(eTrunc));
fprintf(fid, 'Rounding error range: [%.10f, %.10f]\n\n', min(eRound), max(eRound));
fprintf(fid, 'Problem 2(b)\n');
for k = 1:numel(bitRates)
    fprintf(fid, '%d bits: noise variance = %.12g, SNR = %.6f dB\n', bitRates(k), noiseVariance(k), snrDb(k));
end
fprintf(fid, '\nProblem 2(c)\n');
fprintf(fid, 'Mean spectral difference, 10-bit noise minus 8-bit noise: %.6f dB\n', noise10minus8Db);
fprintf(fid, 'Absolute 8-bit minus 10-bit noise-level increase: %.6f dB\n', -noise10minus8Db);
fclose(fid);

fprintf('Problem 2 statistics saved in %s\n', outputDir);
