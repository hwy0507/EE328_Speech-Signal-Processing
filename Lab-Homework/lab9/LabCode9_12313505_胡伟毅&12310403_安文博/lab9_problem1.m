%% LAB IX - Problem 1: Statistical model of speech
% This script creates the required concatenated speech files, computes
% statistics, histograms, and long-term average power spectra.

clear; close all; clc;

outputDir = fullfile(pwd, 'lab9_output');
figureDir = fullfile(outputDir, 'figures');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
if ~exist(figureDir, 'dir'), mkdir(figureDir); end

files = {'s1.wav','s2.wav','s3.wav','s4.wav','s5.wav','s6.wav'};
femaleFiles = {'s1.wav','s3.wav'};
maleFiles = {'s2.wav','s4.wav','s5.wav','s6.wav'};

[out_s1_s6, fs] = concatenate_trimmed(files);
[out_female, fsFemale] = concatenate_trimmed(femaleFiles);
[out_male, fsMale] = concatenate_trimmed(maleFiles);
assert(fs == fsFemale && fs == fsMale, 'Input WAV files must use the same sampling rate.');

audiowrite(fullfile(outputDir, 'out_s1_s6.wav'), out_s1_s6, fs);
audiowrite(fullfile(outputDir, 'out_female.wav'), out_female, fs);
audiowrite(fullfile(outputDir, 'out_male.wav'), out_male, fs);

stats.mean = mean(out_s1_s6);
stats.variance = var(out_s1_s6, 1);
stats.minimum = min(out_s1_s6);
stats.maximum = max(out_s1_s6);
stats.numSamples = length(out_s1_s6);
stats.durationSeconds = length(out_s1_s6) / fs;

fprintf('Problem 1 statistics for out_s1_s6.wav\n');
fprintf('Mean: %.8f\n', stats.mean);
fprintf('Variance: %.8f\n', stats.variance);
fprintf('Minimum: %.8f\n', stats.minimum);
fprintf('Maximum: %.8f\n', stats.maximum);
fprintf('Samples: %d\n', stats.numSamples);
fprintf('Duration: %.3f s\n', stats.durationSeconds);

fid = fopen(fullfile(outputDir, 'problem1_stats.txt'), 'w');
fprintf(fid, 'Problem 1 statistics for out_s1_s6.wav\n');
fprintf(fid, 'Mean: %.10f\n', stats.mean);
fprintf(fid, 'Variance: %.10f\n', stats.variance);
fprintf(fid, 'Minimum: %.10f\n', stats.minimum);
fprintf(fid, 'Maximum: %.10f\n', stats.maximum);
fprintf(fid, 'Samples: %d\n', stats.numSamples);
fprintf(fid, 'Duration: %.6f s\n', stats.durationSeconds);
fclose(fid);

figure('Name', 'Problem 1(a) - Amplitude histograms');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
histBins = [25 50 100];
for k = 1:numel(histBins)
    nexttile;
    histogram(out_s1_s6, histBins(k));
    grid on;
    xlabel('Amplitude');
    ylabel('Count');
    title(sprintf('%d bins', histBins(k)));
end
saveas(gcf, fullfile(figureDir, 'p1a_histograms.png'));

figure('Name', 'Problem 1(b) - Window length comparison');
hold on;
windowLengths = [32 64 128 256 512];
Nfft = 2048;
for k = 1:numel(windowLengths)
    [P, F] = pspect(out_s1_s6, fs, Nfft, windowLengths(k));
    plot(F, 10 * log10(P + eps), 'LineWidth', 1.1);
end
grid on;
xlabel('Frequency (Hz)');
ylabel('Power spectrum (dB)');
title('Long-term average power spectrum: window-length comparison');
legend(compose('Nwin = %d', windowLengths), 'Location', 'best');
saveas(gcf, fullfile(figureDir, 'p1b_power_spectrum_windows.png'));

figure('Name', 'Problem 1(d) - Male vs female spectra');
[Pmale, Fmale] = pspect(out_male, fs, Nfft, 32);
[Pfemale, Ffemale] = pspect(out_female, fs, Nfft, 32);
plot(Fmale, 10 * log10(Pmale + eps), 'LineWidth', 1.2); hold on;
plot(Ffemale, 10 * log10(Pfemale + eps), 'LineWidth', 1.2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Power spectrum (dB)');
title('Male and female long-term average power spectra, Nwin = 32');
legend('Male: s2+s4+s5+s6', 'Female: s1+s3', 'Location', 'best');
saveas(gcf, fullfile(figureDir, 'p1d_male_female_spectra.png'));

fprintf('Problem 1 output saved in %s\n', outputDir);

function [y, fs] = concatenate_trimmed(fileList)
    y = [];
    fs = [];
    for ii = 1:numel(fileList)
        [x, currentFs] = audioread(fileList{ii});
        if size(x, 2) > 1
            x = mean(x, 2);
        end
        if isempty(fs)
            fs = currentFs;
        elseif fs ~= currentFs
            error('Sampling-rate mismatch in %s.', fileList{ii});
        end
        x = trim_edge_silence(x);
        y = [y; x(:)]; %#ok<AGROW>
    end
end

function y = trim_edge_silence(x)
    x = x(:);
    threshold = max(1e-5, 0.01 * max(abs(x)));
    active = find(abs(x) > threshold);
    if isempty(active)
        y = x;
    else
        y = x(active(1):active(end));
    end
end
