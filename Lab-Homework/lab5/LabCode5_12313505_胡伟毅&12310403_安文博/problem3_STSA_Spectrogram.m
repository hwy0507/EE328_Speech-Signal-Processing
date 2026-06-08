%% Problem 3: Wideband and narrowband spectrograms
% Handout test settings: grayscale and color as two separate plots.
STSA_Spectrogram("s5.wav", 0, [5 50], [1024 1024], "log", 60, "gray");
STSA_Spectrogram("s5.wav", 0, [5 50], [1024 1024], "log", 60, "color");
% Optional explorations (as suggested in the handout):
% STSA_Spectrogram("s5.wav", 4000, [5 50], [1024 1024], "log", 60, "color");
% STSA_Spectrogram("s5.wav", 16000, [5 50], [1024 1024], "log", 60, "color");
% STSA_Spectrogram("s5.wav", 0, [2 100], [1024 1024], "log", 60, "color");
% STSA_Spectrogram("s5.wav", 0, [5 50], [512 512], "log", 60, "color");
% STSA_Spectrogram("s5.wav", 0, [5 50], [2048 2048], "log", 60, "color");
% STSA_Spectrogram("s5.wav", 0, [5 50], [1024 1024], "linear", 60, "color");
% STSA_Spectrogram("s5.wav", 0, [5 50], [1024 1024], "log", 30, "color");
% STSA_Spectrogram("s5.wav", 0, [5 50], [1024 1024], "log", 120, "color");

function figPath = STSA_Spectrogram(filename, resamplerate, windowlengths, FFTlengths, magscale, range, color)
    [x, fs] = audioread(filename);
    if size(x, 2) > 1
        x = mean(x, 2);
    end
    x = x(:);

    if resamplerate > 0 && resamplerate ~= fs
        x = resample(x, resamplerate, fs);
        fs = resamplerate;
    end

    if numel(windowlengths) ~= 2 || numel(FFTlengths) ~= 2
        error("windowlengths and FFTlengths must both be 2-element vectors.");
    end

    Lwide = round(windowlengths(1) * 1e-3 * fs);
    Lnarrow = round(windowlengths(2) * 1e-3 * fs);
    FFTwide = round(FFTlengths(1));
    FFTnarrow = round(FFTlengths(2));

    if any([Lwide Lnarrow FFTwide FFTnarrow] < 2)
        error("Window lengths and FFT lengths must be >= 2 samples.");
    end

    wWide = hamming(Lwide);
    wNarrow = hamming(Lnarrow);

    ovWide = round(0.75 * Lwide);
    ovNarrow = round(0.75 * Lnarrow);

    % Bonus version: manual STFT (no built-in spectrogram routine).
    [Swide, fWide, tWide] = manualSTFT(x, fs, wWide, FFTwide, ovWide);
    [Snarrow, fNarrow, tNarrow] = manualSTFT(x, fs, wNarrow, FFTnarrow, ovNarrow);

    switch lower(string(magscale))
        case "log"
            Mwide = mag2db(abs(Swide) + eps);
            Mnarrow = mag2db(abs(Snarrow) + eps);
        case "linear"
            Mwide = abs(Swide);
            Mnarrow = abs(Snarrow);
        otherwise
            error("magscale must be 'log' or 'linear'.");
    end

    hFig = figure("Color", "w", "Name", sprintf("Problem3 | %s", filename));

    subplot(2,1,1);
    imagesc(tWide, fWide, Mwide);
    set(gca, "YDir", "normal");
    title(sprintf("Wideband spectrogram (L = %g ms, FFT = %d)", windowlengths(1), FFTwide));
    xlabel("Time (s)");
    ylabel("Frequency (Hz)");
    colorbar;
    applyRange(Mwide, range, magscale);

    subplot(2,1,2);
    imagesc(tNarrow, fNarrow, Mnarrow);
    set(gca, "YDir", "normal");
    title(sprintf("Narrowband spectrogram (L = %g ms, FFT = %d)", windowlengths(2), FFTnarrow));
    xlabel("Time (s)");
    ylabel("Frequency (Hz)");
    colorbar;
    applyRange(Mnarrow, range, magscale);

    switch lower(string(color))
        case "gray"
            colormap(gray);
        case "color"
            colormap(turbo);
        otherwise
            error("color must be 'gray' or 'color'.");
    end

    sgtitle(sprintf("STSA\\_Spectrogram | fs = %d Hz | mag = %s", fs, lower(string(magscale))));
    figPath = saveFigure(hFig, filename, fs, windowlengths, FFTlengths, magscale, range, color);
    disp("Saved figure: " + figPath);
end

function [S, f, t] = manualSTFT(x, fs, w, nfft, noverlap)
    x = x(:);
    w = w(:);
    L = numel(w);
    hop = L - round(noverlap);
    if hop < 1
        error("Invalid overlap: hop size must be >= 1.");
    end
    if nfft < L
        error("FFT length must be >= window length for this implementation.");
    end

    N = numel(x);
    if N <= L
        numFrames = 1;
    else
        numFrames = ceil((N - L) / hop) + 1;
    end
    totalLen = (numFrames - 1) * hop + L;
    xPad = zeros(totalLen, 1);
    xPad(1:N) = x;

    kMax = floor(nfft / 2);
    S = zeros(kMax + 1, numFrames);
    t = zeros(1, numFrames);

    for n = 1:numFrames
        idx0 = (n - 1) * hop + 1;
        frame = xPad(idx0:idx0 + L - 1) .* w;
        X = fft(frame, nfft);
        S(:, n) = X(1:kMax + 1);
        t(n) = (idx0 - 1 + (L - 1) / 2) / fs; % frame-center time
    end

    f = (0:kMax)' * fs / nfft;
end

function applyRange(M, dynRange, magscale)
    if isempty(dynRange) || dynRange <= 0
        return;
    end

    topVal = max(M(:));
    if lower(string(magscale)) == "linear"
        lowVal = max(topVal - dynRange, 0);
    else
        lowVal = topVal - dynRange;
    end
    caxis([lowVal topVal]);
end

function figPath = saveFigure(hFig, filename, fs, windowlengths, FFTlengths, magscale, range, color)
    figDir = fullfile(pwd, "figures");
    if ~exist(figDir, "dir")
        mkdir(figDir);
    end

    [~, base, ~] = fileparts(char(filename));
    winTag = regexprep(sprintf("%g_%g", windowlengths(1), windowlengths(2)), "[^0-9A-Za-z_]+", "p");
    fftTag = sprintf("%d_%d", round(FFTlengths(1)), round(FFTlengths(2)));
    magTag = lower(char(string(magscale)));
    colorTag = lower(char(string(color)));
    figName = sprintf("p3_%s_fs%d_win%s_fft%s_%s_r%d_%s.png", ...
        base, round(fs), winTag, fftTag, magTag, round(range), colorTag);
    figPath = fullfile(figDir, figName);
    exportgraphics(hFig, figPath, "Resolution", 180);
end
