%% Problem 3: Effect of window duration on normalized short-time features
% Use test_16k.wav and compare L = [51, 101, 201, 401]
% Window type can be Hamming or rectangular. Here we use Hamming by default.

clear; clc; close all;

filePath = locate_audio_file('test_16k.wav');
if isempty(filePath)
    error('test_16k.wav not found.');
end

[x, fs] = audioread(filePath);
if size(x, 2) > 1
    x = mean(x, 2);
end

Lset = [51, 101, 201, 401];
winType = 'hamming';   % change to 'rectangular' if needed

figure('Color', 'w', 'Name', 'Problem 3 - Window Length Effect (Normalized)');
ax1 = subplot(3, 1, 1); hold(ax1, 'on'); grid(ax1, 'on');
ax2 = subplot(3, 1, 2); hold(ax2, 'on'); grid(ax2, 'on');
ax3 = subplot(3, 1, 3); hold(ax3, 'on'); grid(ax3, 'on');

for k = 1:numel(Lset)
    L = Lset(k);
    R = (L - 1) / 2;      % around 50% overlap for odd L

    switch lower(winType)
        case 'hamming'
            win = hamming(L);
            winLabel = 'Hamming';
        case 'rectangular'
            win = rectwin(L);
            winLabel = 'Rectangular';
        otherwise
            error('Unsupported window type: %s', winType);
    end

    feats = short_time_name(x, fs, R, win);
    eNorm = feats.energy ./ sum(win .^ 2);
    mNorm = feats.magnitude ./ sum(abs(win));
    zNorm = feats.zc ./ (L - 1);

    plot(ax1, feats.time, eNorm, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('L=%d', L));
    plot(ax2, feats.time, mNorm, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('L=%d', L));
    plot(ax3, feats.time, zNorm, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('L=%d', L));
end

xlabel(ax1, 'Time (s)');
ylabel(ax1, 'Energy (norm.)');
title(ax1, 'Normalized Short-Time Energy');
legend(ax1, 'Location', 'best');

xlabel(ax2, 'Time (s)');
ylabel(ax2, 'Magnitude (norm.)');
title(ax2, 'Normalized Short-Time Magnitude');
legend(ax2, 'Location', 'best');

xlabel(ax3, 'Time (s)');
ylabel(ax3, 'ZCR');
title(ax3, 'Normalized Short-Time Zero-Crossing Rate');
legend(ax3, 'Location', 'best');

sgtitle(sprintf('%s | %s Window | Normalized Features', filePath, winLabel), ...
    'Interpreter', 'none');

% Observations:
% 1) After normalization, feature amplitudes across different L become comparable.
% 2) As L increases, curves become smoother and rapid local variations are suppressed.
% 3) As L decreases, temporal resolution is better but the estimates fluctuate more.

%% Local function: short-time analysis
function feats = short_time_name(x, fs, R, window)
    x = x(:);
    window = window(:);

    L = numel(window);
    N = numel(x);
    numFrames = floor((N - L) / R) + 1;
    if numFrames < 1
        error('Signal is too short for the selected window length.');
    end

    energy = zeros(numFrames, 1);
    magnitude = zeros(numFrames, 1);
    zc = zeros(numFrames, 1);
    tFrame = zeros(numFrames, 1);

    for n = 1:numFrames
        idx0 = (n - 1) * R + 1;
        frame = x(idx0:idx0 + L - 1) .* window;

        energy(n) = sum(frame .^ 2);
        magnitude(n) = sum(abs(frame));

        s = sign(frame);
        s(s == 0) = 1;
        zc(n) = 0.5 * sum(abs(diff(s)));

        tFrame(n) = (idx0 - 1 + (L - 1) / 2) / fs;
    end

    feats.time = tFrame;
    feats.energy = energy;
    feats.magnitude = magnitude;
    feats.zc = zc;
end

%% Local function: search audio file
function filePath = locate_audio_file(fileName)
    candidates = {
        fullfile(pwd, fileName), ...
        fullfile('/Users/hwy/Desktop/个人/26春/语音信号处理/语音信号处理圣遗物/Lab4', fileName)
    };

    filePath = '';
    for k = 1:numel(candidates)
        if exist(candidates{k}, 'file') == 2
            filePath = candidates{k};
            return;
        end
    end
end
