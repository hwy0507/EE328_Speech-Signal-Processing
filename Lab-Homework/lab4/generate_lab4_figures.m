%% Generate figures for Lab 4 report
clear; clc; close all;
outDir = fullfile(pwd, 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% Problem 1 figures
L = 101;
Fs = 1;
Nfft = 2^18;
winNames = {'Rectangular', 'Triangular', 'Hann', 'Hamming', 'Blackman'};
windows = {rectwin(L), triang(L), hann(L), hamming(L), blackman(L)};
colors = lines(numel(windows));

% Time response
n = 0:L-1;
fig = figure('Color', 'w', 'Position', [100, 100, 820, 420]);
hold on;
for k = 1:numel(windows)
    plot(n, windows{k}, 'LineWidth', 1.5, 'Color', colors(k, :));
end
grid on;
xlabel('Sample index n');
ylabel('Amplitude');
title(sprintf('Time Responses of Five Windows (L=%d)', L));
legend(winNames, 'Location', 'best');
exportgraphics(fig, fullfile(outDir, 'p1_time_windows.png'), 'Resolution', 300);
close(fig);

% Log-magnitude response: draw each window separately
% Layout: each row = one window, left = full band, right = narrow band.
fig = figure('Color', 'w', 'Position', [100, 60, 1300, 1500]);
tiledlayout(5, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

bw = zeros(numel(windows),1);
psl = zeros(numel(windows),1);

for k = 1:numel(windows)
    [f, magDb, bwHz, pslDb] = spectrum_metrics(windows{k}, Fs, Nfft);

    nexttile;
    plot(f, magDb, 'LineWidth', 1.2, 'Color', colors(k,:));
    xlim([0, Fs/2]); ylim([-140, 5]); grid on;
    title(sprintf('%s - Full Band', winNames{k}));
    ylabel('Magnitude (dB)');
    if k == numel(windows)
        xlabel('Frequency (cycles/sample)');
    end

    nexttile;
    plot(f, magDb, 'LineWidth', 1.2, 'Color', colors(k,:));
    xlim([0, 5*Fs/L]); ylim([-140, 5]); grid on;
    title(sprintf('%s - Narrow Band', winNames{k}));
    ylabel('Magnitude (dB)');
    if k == numel(windows)
        xlabel('Frequency (cycles/sample)');
    end

    bw(k) = bwHz;
    psl(k) = pslDb;
end
sgtitle(sprintf('Log-Magnitude Responses (Separated) | Narrow Band: 0 to %.4f = 5Fs/L', 5*Fs/L));
exportgraphics(fig, fullfile(outDir, 'p1_log_windows.png'), 'Resolution', 300);
close(fig);

T = table(winNames', bw, psl, 'VariableNames', {'Window','EffectiveBandwidth','PeakSidelobe_dB'});
writetable(T, fullfile(outDir, 'p1_metrics.csv'));

%% Problem 2 figure (s5.wav)
fileS5 = locate_audio_file('s5.wav');
if ~isempty(fileS5)
    [x, fs] = audioread(fileS5);
    if size(x,2) > 1, x = mean(x,2); end
    L = round(0.020*fs); if mod(L,2)==0, L=L+1; end
    R = round(0.010*fs);
    win = hamming(L);
    feats = short_time_name(x, fs, R, win);

    fig = figure('Color', 'w', 'Position', [100, 100, 860, 860]);
    t = (0:numel(x)-1)/fs;
    subplot(4,1,1); plot(t, x, 'k', 'LineWidth', 1); grid on; ylabel('Amp.'); title('Entire Waveform');
    subplot(4,1,2); plot(feats.time, feats.energy, 'b', 'LineWidth', 1.2); grid on; ylabel('Energy'); title('Short-Time Energy');
    subplot(4,1,3); plot(feats.time, feats.magnitude, 'm', 'LineWidth', 1.2); grid on; ylabel('Magnitude'); title('Short-Time Magnitude');
    subplot(4,1,4); stairs(feats.time, feats.zc, 'r', 'LineWidth', 1.2); grid on; ylabel('ZC Count'); xlabel('Time (s)'); title('Short-Time Zero-Crossing');
    sgtitle(sprintf('Problem 2 on s5.wav | L=%d (%.1f ms), R=%d (%.1f ms), Hamming', L, 1000*L/fs, R, 1000*R/fs));
    exportgraphics(fig, fullfile(outDir, 'p2_short_time_features.png'), 'Resolution', 300);
    close(fig);
end

%% Problem 3 figure (test_16k.wav)
fileT = locate_audio_file('test_16k.wav');
if ~isempty(fileT)
    [x, fs] = audioread(fileT);
    if size(x,2) > 1, x = mean(x,2); end

    Lset = [51,101,201,401];
    fig = figure('Color', 'w', 'Position', [100, 100, 860, 760]);
    ax1 = subplot(3,1,1); hold(ax1,'on'); grid(ax1,'on');
    ax2 = subplot(3,1,2); hold(ax2,'on'); grid(ax2,'on');
    ax3 = subplot(3,1,3); hold(ax3,'on'); grid(ax3,'on');

    for k = 1:numel(Lset)
        L = Lset(k);
        R = (L-1)/2;
        win = hamming(L);
        feats = short_time_name(x, fs, R, win);

        % Normalize to remove scale bias caused by different window lengths.
        eNorm = feats.energy ./ sum(win.^2);
        mNorm = feats.magnitude ./ sum(abs(win));
        zNorm = feats.zc ./ (L - 1); % zero-crossing rate per sample

        plot(ax1, feats.time, eNorm, 'LineWidth', 1.1, 'DisplayName', sprintf('L=%d',L));
        plot(ax2, feats.time, mNorm, 'LineWidth', 1.1, 'DisplayName', sprintf('L=%d',L));
        plot(ax3, feats.time, zNorm, 'LineWidth', 1.1, 'DisplayName', sprintf('L=%d',L));
    end

    title(ax1, 'Normalized Short-Time Energy'); ylabel(ax1, 'Energy (norm.)'); xlabel(ax1, 'Time (s)'); legend(ax1, 'Location', 'best');
    title(ax2, 'Normalized Short-Time Magnitude'); ylabel(ax2, 'Magnitude (norm.)'); xlabel(ax2, 'Time (s)'); legend(ax2, 'Location', 'best');
    title(ax3, 'Normalized Short-Time Zero-Crossing Rate'); ylabel(ax3, 'ZCR'); xlabel(ax3, 'Time (s)'); legend(ax3, 'Location', 'best');
    sgtitle('Problem 3 on test\_16k.wav | Hamming Window | Normalized Features');

    exportgraphics(fig, fullfile(outDir, 'p3_window_length_effect.png'), 'Resolution', 300);
    close(fig);
end

%% ===== Local functions =====
function [f, magDb, bwHz, pslDb] = spectrum_metrics(w, Fs, Nfft)
    W = abs(fft(w, Nfft));
    W = W(1:Nfft/2 + 1);
    W = W ./ max(W);
    f = (0:Nfft/2)' * (Fs / Nfft);
    magDb = 20 * log10(max(W, 1e-12));

    dW = diff(W);
    idxNull = find(dW(1:end-1) <= 0 & dW(2:end) >= 0, 1, 'first') + 1;
    if isempty(idxNull), idxNull = 2; end
    bwHz = 2 * f(idxNull);

    if idxNull + 1 <= numel(magDb)
        pslDb = max(magDb(idxNull+1:end));
    else
        pslDb = NaN;
    end
end

function feats = short_time_name(x, fs, R, window)
    x = x(:);
    window = window(:);
    L = numel(window);
    N = numel(x);
    numFrames = floor((N - L) / R) + 1;
    if numFrames < 1
        error('Signal is too short for selected L and R.');
    end

    energy = zeros(numFrames, 1);
    magnitude = zeros(numFrames, 1);
    zc = zeros(numFrames, 1);
    tFrame = zeros(numFrames, 1);

    for n = 1:numFrames
        idx0 = (n-1)*R + 1;
        frame = x(idx0:idx0+L-1) .* window;
        energy(n) = sum(frame.^2);
        magnitude(n) = sum(abs(frame));
        s = sign(frame); s(s==0)=1;
        zc(n) = 0.5 * sum(abs(diff(s)));
        tFrame(n) = (idx0 - 1 + (L - 1)/2) / fs;
    end

    feats.time = tFrame;
    feats.energy = energy;
    feats.magnitude = magnitude;
    feats.zc = zc;
end

function filePath = locate_audio_file(fileName)
    cands = {
        fullfile(pwd, fileName), ...
        fullfile('/Users/hwy/Desktop/个人/26春/语音信号处理/语音信号处理圣遗物/Lab4', fileName)
    };
    filePath = '';
    for i = 1:numel(cands)
        if exist(cands{i}, 'file') == 2
            filePath = cands{i};
            return;
        end
    end
end
