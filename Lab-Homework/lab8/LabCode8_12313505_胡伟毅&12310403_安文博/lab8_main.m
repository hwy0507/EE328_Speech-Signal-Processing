%% Lab 8: Algorithms for Estimating Speech Parameters
% This script completes the three tasks shown in the final pages of lab8(1).pdf.
% Updated version:
%   - Problem 2/3 explicitly use s6.wav
%   - More process figures are generated for report writing

clear; clc; close all;

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

gender = 'male';  % Change to 'female' if needed.

fileP1 = fullfile(rootDir, 'test_16k.wav');
filePitch = fullfile(rootDir, 's6.wav');
assert(exist(fileP1, 'file') == 2, 'Missing file: %s', fileP1);
assert(exist(filePitch, 'file') == 2, 'Missing file: %s', filePitch);

fprintf('Running Lab 8 script in: %s\n', rootDir);
fprintf('Pitch analysis file (Problem 2/3): %s\n', filePitch);

%% Problem 1: Linear / Median / Combination smoothing on short-time ZCR
[s1, fs1] = audioread(fileP1);
if size(s1, 2) > 1
    s1 = mean(s1, 2);
end

frameLen = round(0.010 * fs1);   % 10 ms
frameShift = round(0.005 * fs1); % 5 ms

[zcr, tZcr] = shortTimeZCR(s1, fs1, frameLen, frameShift);

bLP = designLowpassFIR(41, 0.10);  % keep low band up to 0.1*Fs
zcrLinear = filter(bLP, 1, zcr);
% Compensate approximately for linear-phase group delay
gd = floor((numel(bLP)-1)/2);
zcrLinear = [zcrLinear(gd+1:end); repmat(zcrLinear(end), gd, 1)];

zcrMedian = MedianSmoother(MedianSmoother(zcr, 7), 5);
zcrCombo = CombinationSmoother(zcr, 7, 5);

% Main contour figure
fig1 = figure('Name', 'Problem 1 - ZCR Smoothing', 'Color', 'w');
subplot(4,1,1);
plot(tZcr, zcr, 'k', 'LineWidth', 1.0); grid on;
title('Problem 1: Original Short-Time Zero-Crossing Count (10 ms frame)');
ylabel('ZCR Count');

subplot(4,1,2);
plot(tZcr, zcrLinear, 'b', 'LineWidth', 1.2); grid on;
title('Linear Smoother (FIR lowpass, L=41)');
ylabel('ZCR Count');

subplot(4,1,3);
plot(tZcr, zcrMedian, 'm', 'LineWidth', 1.2); grid on;
title('Median Smoother (7-point then 5-point)');
ylabel('ZCR Count');

subplot(4,1,4);
plot(tZcr, zcrCombo, 'r', 'LineWidth', 1.2); grid on;
title('Combination Smoother');
ylabel('ZCR Count');
xlabel('Time (s)');

saveas(fig1, fullfile(rootDir, 'problem1_zcr_smoothing.png'));

% Extra process figure: lowpass filter response + ZCR sequence spectrum
fig1p = figure('Name', 'Problem 1 - Process Figures', 'Color', 'w');
subplot(2,1,1);
[Hlp, flp] = freqz(bLP, 1, 2048, fs1);
plot(flp/1000, 20*log10(abs(Hlp)+eps), 'LineWidth', 1.2);
grid on;
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Problem 1 Process: Lowpass FIR Magnitude Response');

subplot(2,1,2);
fr = fs1 / frameShift; % frame-rate frequency axis
Nfft = 2^nextpow2(numel(zcr));
fAxis = (0:Nfft/2) / Nfft * fr;
Z0 = fft(zcr - mean(zcr), Nfft);
Z1 = fft(zcrLinear - mean(zcrLinear), Nfft);
Z2 = fft(zcrMedian - mean(zcrMedian), Nfft);
Z3 = fft(zcrCombo - mean(zcrCombo), Nfft);
plot(fAxis, 20*log10(abs(Z0(1:Nfft/2+1))+eps), 'k', 'LineWidth', 1.0); hold on;
plot(fAxis, 20*log10(abs(Z1(1:Nfft/2+1))+eps), 'b', 'LineWidth', 1.0);
plot(fAxis, 20*log10(abs(Z2(1:Nfft/2+1))+eps), 'm', 'LineWidth', 1.0);
plot(fAxis, 20*log10(abs(Z3(1:Nfft/2+1))+eps), 'r', 'LineWidth', 1.0); hold off;
grid on;
xlabel('Frame-Rate Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Problem 1 Process: ZCR Sequence Spectrum');
legend('Raw', 'Linear', 'Median', 'Combination', 'Location', 'best');

saveas(fig1p, fullfile(rootDir, 'problem1_process_figures.png'));

%% Problem 2: Autocorrelation-based pitch detector (use s6.wav)
[s2, fs2] = audioread(filePitch);
outAutoc = PitchDetector_Autocorrelation(s2, fs2, gender);

% Main summary figure
fig2 = figure('Name', 'Problem 2 - Autocorrelation Pitch Detector', 'Color', 'w');
for k = 1:2
    rr = outAutoc.result(k);

    subplot(2,2,k);
    plot(rr.time, rr.pitchThresholded, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0); hold on;
    plot(rr.time, rr.pitchSmoothed, 'r', 'LineWidth', 1.2); hold off;
    grid on;
    title(sprintf('Pitch Period (%s)', rr.name));
    xlabel('Time (s)');
    ylabel('Samples @10kHz');
    legend('Thresholded', '5-pt median', 'Location', 'best');

    subplot(2,2,k+2);
    plot(rr.time, rr.logConfidence, 'Color', [0.5 0.5 1.0], 'LineWidth', 1.0); hold on;
    plot(rr.time, rr.logConfidenceSmoothed, 'b', 'LineWidth', 1.2);
    yline(rr.threshold, '--r', 'Threshold');
    hold off;
    grid on;
    title(sprintf('Log Confidence (%s)', rr.name));
    xlabel('Time (s)');
    ylabel('log_{10}(corr max)');
    legend('Raw', '5-pt median', 'Threshold', 'Location', 'best');
end
saveas(fig2, fullfile(rootDir, 'problem2_autoc_pitch.png'));

% Process figure 1: bandpass filter + waveform comparison
fig2a = figure('Name', 'Problem 2 - Process Filter and Signal', 'Color', 'w');
subplot(2,1,1);
[Hbp, fbp] = freqz(outAutoc.filter, 1, 2048, outAutoc.fsOut);
plot(fbp, 20*log10(abs(Hbp)+eps), 'LineWidth', 1.2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Problem 2 Process: Bandpass FIR Magnitude Response');

subplot(2,1,2);
Ns = min(numel(outAutoc.resampled), round(0.6 * outAutoc.fsOut));
tsig = (0:Ns-1) / outAutoc.fsOut;
plot(tsig, outAutoc.resampled(1:Ns), 'k', 'LineWidth', 1.0); hold on;
plot(tsig, outAutoc.bandpass(1:Ns), 'b', 'LineWidth', 1.0); hold off;
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Problem 2 Process: Resampled vs Bandpass Waveform (First 0.6 s)');
legend('Resampled', 'Bandpass', 'Location', 'best');

saveas(fig2a, fullfile(rootDir, 'problem2_process_filter_signal.png'));

% Process figure 2: Step11 style (raw)
fig2b = figure('Name', 'Problem 2 - Step11 Raw', 'Color', 'w');
for k = 1:2
    rr = outAutoc.result(k);

    subplot(2,2,k);
    plot(rr.time, rr.logConfidence, 'b', 'LineWidth', 1.1); hold on;
    yline(rr.threshold, '--r', 'Threshold'); hold off;
    grid on;
    title(sprintf('Step11 Log Confidence (%s)', rr.name));
    xlabel('Time (s)'); ylabel('log_{10}(corr max)');

    subplot(2,2,k+2);
    plot(rr.time, rr.pitchThresholded, 'k', 'LineWidth', 1.1);
    grid on;
    title(sprintf('Step11 Pitch Contour (%s)', rr.name));
    xlabel('Time (s)'); ylabel('Samples @10kHz');
end
saveas(fig2b, fullfile(rootDir, 'problem2_step11_raw.png'));

% Process figure 3: Step13 style (5-point median)
fig2c = figure('Name', 'Problem 2 - Step13 Median Smoothed', 'Color', 'w');
for k = 1:2
    rr = outAutoc.result(k);

    subplot(2,2,k);
    plot(rr.time, rr.logConfidenceSmoothed, 'b', 'LineWidth', 1.1);
    grid on;
    title(sprintf('Step13 Smoothed Confidence (%s)', rr.name));
    xlabel('Time (s)'); ylabel('Smoothed log-confidence');

    subplot(2,2,k+2);
    plot(rr.time, rr.pitchSmoothed, 'r', 'LineWidth', 1.1);
    grid on;
    title(sprintf('Step13 Smoothed Pitch (%s)', rr.name));
    xlabel('Time (s)'); ylabel('Samples @10kHz');
end
saveas(fig2c, fullfile(rootDir, 'problem2_step13_median.png'));

save(fullfile(rootDir, 'out_autoc.mat'), 'outAutoc');

%% Problem 3: Cepstrum-based pitch detector (use s6.wav)
[s3, fs3] = audioread(filePitch);
outCep = PitchDetector_Cepstrum(s3, fs3, gender);

frameBeg = 48;
frameEnd = 63;

% (a1) Waterfall for original signal
fig3a = figure('Name', 'Problem 3 - Cepstrum Waterfall (Original)', 'Color', 'w');
plotCepstrumWaterfall(outCep.result(1), outCep.nlow, outCep.nhigh, frameBeg, frameEnd);
title(sprintf('Problem 3: Cepstrum Frames %d-%d (original)', frameBeg, frameEnd));
saveas(fig3a, fullfile(rootDir, 'problem3_cepstrum_waterfall.png'));

% (a2) Waterfall for bandpass signal
fig3a2 = figure('Name', 'Problem 3 - Cepstrum Waterfall (Bandpass)', 'Color', 'w');
plotCepstrumWaterfall(outCep.result(2), outCep.nlow, outCep.nhigh, frameBeg, frameEnd);
title(sprintf('Problem 3: Cepstrum Frames %d-%d (bandpass)', frameBeg, frameEnd));
saveas(fig3a2, fullfile(rootDir, 'problem3_cepstrum_waterfall_bandpass.png'));

% (b) Pitch contour + confidence for both original and bandpass signals
fig3b = figure('Name', 'Problem 3 - Cepstrum Pitch Contours', 'Color', 'w');
for k = 1:2
    rr = outCep.result(k);

    subplot(2,2,k);
    plot(rr.time, rr.pitchRaw, 'Color', [0.65 0.65 0.65], 'LineWidth', 1.0); hold on;
    plot(rr.time, rr.pitchSmoothed, 'k', 'LineWidth', 1.2); hold off;
    grid on;
    title(sprintf('Pitch Period (%s)', rr.name));
    xlabel('Time (s)');
    ylabel('Samples @10kHz');
    legend('Raw (reliable-region extended)', '5-pt median', 'Location', 'best');

    subplot(2,2,k+2);
    plot(rr.time, rr.confidenceRaw, 'Color', [0.5 0.5 1.0], 'LineWidth', 1.0); hold on;
    plot(rr.time, rr.confidenceSmoothed, 'b', 'LineWidth', 1.2); hold off;
    grid on;
    title(sprintf('Cepstral Confidence (%s)', rr.name));
    xlabel('Time (s)');
    ylabel('Primary Cepstral Peak');
    legend('Raw', '5-pt median', 'Location', 'best');
end
saveas(fig3b, fullfile(rootDir, 'problem3_cepstrum_pitch.png'));

% (c) Extra process figure: reliable-region extension details
fig3c = figure('Name', 'Problem 3 - Process Reliable Region', 'Color', 'w');
for k = 1:2
    rr = outCep.result(k);
    tt = rr.time;

    subplot(3,2,k);
    plot(tt, rr.ratio, 'b', 'LineWidth', 1.0); hold on;
    yline(outCep.pthr1, '--r', 'pthr1');
    plot(tt(rr.reliable), rr.ratio(rr.reliable), 'ko', 'MarkerSize', 3, 'LineWidth', 1.0);
    hold off;
    grid on;
    title(sprintf('Ratio p1/p2 (%s)', rr.name));
    xlabel('Time (s)'); ylabel('Ratio');
    legend('p1/p2', 'Threshold', 'Seed frames', 'Location', 'best');

    subplot(3,2,k+2);
    plot(tt, rr.primaryPitch, 'Color', [0 0.45 0.74], 'LineWidth', 1.0); hold on;
    plot(tt, rr.secondaryPitch, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0); hold off;
    grid on;
    title(sprintf('Primary / Secondary Pitch (%s)', rr.name));
    xlabel('Time (s)'); ylabel('Samples @10kHz');
    legend('Primary', 'Secondary', 'Location', 'best');

    subplot(3,2,k+4);
    yyaxis left;
    stairs(tt, double(rr.regionMask), 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
    ylim([-0.1 1.1]);
    ylabel('Region Mask');

    yyaxis right;
    plot(tt, rr.pitchRaw, 'k', 'LineWidth', 1.0); hold on;
    plot(tt, rr.pitchSmoothed, 'r', 'LineWidth', 1.1); hold off;
    ylabel('Pitch (samples @10kHz)');

    grid on;
    title(sprintf('Reliable Extension + Smoothing (%s)', rr.name));
    xlabel('Time (s)');
end
saveas(fig3c, fullfile(rootDir, 'problem3_process_reliable.png'));

save(fullfile(rootDir, 'out_cepstrum.mat'), 'outCep');

fprintf('All tasks completed.\n');
fprintf('Generated files:\n');
fprintf('  - problem1_zcr_smoothing.png\n');
fprintf('  - problem1_process_figures.png\n');
fprintf('  - problem2_autoc_pitch.png\n');
fprintf('  - problem2_process_filter_signal.png\n');
fprintf('  - problem2_step11_raw.png\n');
fprintf('  - problem2_step13_median.png\n');
fprintf('  - problem3_cepstrum_waterfall.png\n');
fprintf('  - problem3_cepstrum_waterfall_bandpass.png\n');
fprintf('  - problem3_cepstrum_pitch.png\n');
fprintf('  - problem3_process_reliable.png\n');
fprintf('  - out_autoc.mat\n');
fprintf('  - out_cepstrum.mat\n');

%% Local helper functions
function [zcr, t] = shortTimeZCR(x, fs, frameLen, frameShift)
    x = x(:);
    N = numel(x);
    numFrames = floor((N - frameLen) / frameShift) + 1;

    zcr = zeros(numFrames, 1);
    t = zeros(numFrames, 1);

    for i = 1:numFrames
        n0 = (i - 1) * frameShift + 1;
        frame = x(n0:n0 + frameLen - 1);

        sgn = sign(frame);
        sgn(sgn == 0) = 1;
        zcr(i) = 0.5 * sum(abs(diff(sgn)));  % crossing count in this frame
        t(i) = (n0 + frameLen/2 - 1) / fs;
    end
end

function b = designLowpassFIR(N, fcNorm)
% fcNorm is normalized to sampling rate Fs (0 < fcNorm < 0.5).
    M = N - 1;
    n = 0:M;
    a = M / 2;

    hIdeal = 2 * fcNorm * sinc(2 * fcNorm * (n - a));
    w = 0.54 - 0.46 * cos(2 * pi * n / M);
    b = hIdeal .* w;
    b = b / sum(b);
end

function plotCepstrumWaterfall(rr, nlow, nhigh, frameBeg, frameEnd)
    q = nlow:nhigh;
    nFrm = size(rr.cepstrumRange, 1);
    idxFrames = frameBeg:min(frameEnd, nFrm);

    hold on;
    if isempty(idxFrames)
        text(mean([nlow, nhigh]), 0, 'No frames available', 'HorizontalAlignment', 'center');
    else
        scale = max(abs(rr.cepstrumRange(:)));
        if scale == 0
            scale = 1;
        end
        offsetStep = 1.4 * scale;

        for ii = 1:numel(idxFrames)
            fr = idxFrames(ii);
            c = rr.cepstrumRange(fr, :);
            y = c + (numel(idxFrames) - ii) * offsetStep;

            plot(q, y, 'k', 'LineWidth', 1.0);

            i1 = rr.primaryPitch(fr) - nlow + 1;
            i2 = rr.secondaryPitch(fr) - nlow + 1;
            if i1 >= 1 && i1 <= numel(c)
                plot(rr.primaryPitch(fr), y(i1), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
            end
            if i2 >= 1 && i2 <= numel(c)
                plot(rr.secondaryPitch(fr), y(i2), 'o', 'Color', [0.5 0.5 0.5], ...
                     'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerSize', 5);
            end
        end
    end
    hold off;
    grid on;
    xlabel('Pitch Period (samples @10kHz)');
    ylabel('Stacked Cepstrum');
    legend('Cepstral frame', 'Primary peak', 'Secondary peak', 'Location', 'best');
end
