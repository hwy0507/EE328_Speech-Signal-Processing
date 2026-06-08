%% Problem 2: Short-time speech analysis on one page
% Measurements to plot:
% 1) Entire speech waveform
% 2) Short-time energy
% 3) Short-time magnitude
% 4) Short-time zero-crossing rate/count

clear; clc; close all;

% Choose one or more files to test.
audioFiles = {'s5.wav', 'test_16k.wav'};

% Parameter choice (typical for speech):
% L = 20 ms, R = 10 ms (50% overlap), Hamming window.
for i = 1:numel(audioFiles)
    filePath = locate_audio_file(audioFiles{i});
    if isempty(filePath)
        warning('Audio file "%s" not found. Skipped.', audioFiles{i});
        continue;
    end

    [x, fs] = audioread(filePath);
    if size(x, 2) > 1
        x = mean(x, 2); % convert to mono if needed
    end

    L = round(0.020 * fs);   % 20 ms
    if mod(L, 2) == 0
        L = L + 1;           % use odd length for symmetry
    end
    R = round(0.010 * fs);   % 10 ms shift (50% overlap)
    win = hamming(L);

    feats = short_time_name(x, fs, R, win);
    plot_short_time_page(x, fs, feats, filePath, L, R, 'Hamming');
end

% Why this setup:
% - 20 ms captures quasi-stationary speech frames.
% - 50% overlap improves temporal tracking while keeping computation moderate.
% - Hamming suppresses sidelobes better than rectangular in frequency domain.

%% Local function: unified short-time analysis
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
        zc(n) = 0.5 * sum(abs(diff(s))); % zero-crossing count per frame

        tFrame(n) = (idx0 - 1 + (L - 1) / 2) / fs; % frame-center time
    end

    feats.time = tFrame;
    feats.energy = energy;
    feats.magnitude = magnitude;
    feats.zc = zc;
end

%% Local function: plotting
function plot_short_time_page(x, fs, feats, filePath, L, R, winName)
    t = (0:numel(x)-1) / fs;

    figure('Color', 'w', 'Name', ['Problem 2 - ', filePath]);

    subplot(4, 1, 1);
    plot(t, x, 'k', 'LineWidth', 1);
    grid on;
    ylabel('Amplitude');
    title('Entire Speech Waveform');

    subplot(4, 1, 2);
    plot(feats.time, feats.energy, 'b', 'LineWidth', 1.2);
    grid on;
    ylabel('Energy');
    title('Short-Time Energy');

    subplot(4, 1, 3);
    plot(feats.time, feats.magnitude, 'm', 'LineWidth', 1.2);
    grid on;
    ylabel('Magnitude');
    title('Short-Time Magnitude');

    subplot(4, 1, 4);
    stairs(feats.time, feats.zc, 'r', 'LineWidth', 1.2);
    grid on;
    ylabel('ZC Count');
    xlabel('Time (s)');
    title('Short-Time Zero-Crossing');

    sgtitle(sprintf('%s | L = %d (%.1f ms), R = %d (%.1f ms), %s Window', ...
        filePath, L, 1000 * L / fs, R, 1000 * R / fs, winName), ...
        'Interpreter', 'none');
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
