%% Problem 2: Compare multiple short-time analyses
% Handout test setting.
STSA_MultiLengths(4, "s5.wav", 7000, [5 10 20 40]);

function STSA_MultiLengths(num, filename, startsmp, framelengths)
    [x, fs] = audioread(filename);
    if size(x, 2) > 1
        x = mean(x, 2);
    end
    x = x(:);

    framelengths = framelengths(:).';
    if num ~= numel(framelengths)
        error("num must match the number of entries in framelengths.");
    end

    startsmp = round(startsmp);
    if startsmp < 1 || startsmp > numel(x)
        error("startsmp must be in [1, length(x)].");
    end

    frameLens = round(framelengths * 1e-3 * fs);
    if any(frameLens < 1)
        error("All frame lengths must be positive.");
    end

    nfft = 2^nextpow2(max(512, 4 * max(frameLens)));
    f = (0:nfft/2)' * fs / nfft;
    cmap = lines(num);

    plotForWindowType("Hamming window", true);
    plotForWindowType("Rectangular window", false);

    function plotForWindowType(figLabel, useHamming)
        hFig = figure("Color", "w", "Name", sprintf("Problem2 | %s", figLabel), ...
            "Position", [100 100 1220 820]);

        subplot(2,2,1);
        t = (0:numel(x)-1)' / fs;
        plot(t, x, "b");
        grid on;
        xlabel("Time (s)");
        ylabel("Amplitude");
        title(sprintf("Speech waveform: %s", filename), "Interpreter", "none", "FontSize", 11);

        subplot(2,2,2);
        hold on;
        leg2 = strings(1, num);
        for k = 1:num
            N = frameLens(k);
            frame = extractFrame(x, startsmp, N);
            if useHamming
                w = hamming(N);
            else
                w = ones(N, 1);
            end
            segMs = (0:N-1)' / fs * 1e3;
            plot(segMs, frame, "Color", cmap(k,:), "LineWidth", 1.0);

            wScaled = w * max(max(abs(frame)), eps);
            plot(segMs, wScaled, "--", "Color", cmap(k,:), "LineWidth", 1.2);
            leg2(k) = sprintf("%g ms", framelengths(k));
        end
        grid on;
        xlabel("Local frame time (ms)");
        ylabel("Amplitude");
        title("Speech segments + window shapes", "FontSize", 11);
        legend(leg2, "Location", "best");

        subplot(2,2,3);
        hold on;
        leg3 = strings(1, num);
        for k = 1:num
            N = frameLens(k);
            frame = extractFrame(x, startsmp, N);
            if useHamming
                w = hamming(N);
            else
                w = ones(N, 1);
            end
            X = fft(frame .* w, nfft);
            mag = abs(X(1:nfft/2+1));
            plot(f, mag, "Color", cmap(k,:), "LineWidth", 1.1);
            leg3(k) = sprintf("%g ms", framelengths(k));
        end
        xlim([0 fs/2]);
        grid on;
        xlabel("Frequency (Hz)");
        ylabel("Magnitude");
        title("Magnitude spectra", "FontSize", 11);
        legend(leg3, "Location", "northeast");

        subplot(2,2,4);
        hold on;
        leg4 = strings(1, num);
        for k = 1:num
            N = frameLens(k);
            frame = extractFrame(x, startsmp, N);
            if useHamming
                w = hamming(N);
            else
                w = ones(N, 1);
            end
            X = fft(frame .* w, nfft);
            magdB = 20 * log10(abs(X(1:nfft/2+1)) + eps);
            plot(f, magdB, "Color", cmap(k,:), "LineWidth", 1.1);
            leg4(k) = sprintf("%g ms", framelengths(k));
        end
        xlim([0 fs/2]);
        grid on;
        xlabel("Frequency (Hz)");
        ylabel("Magnitude (dB)");
        title("Log magnitude spectra (dB)", "FontSize", 11);
        legend(leg4, "Location", "northeast");

        sgtitle(sprintf("STSA_MultiLengths - %s | file=%s | startsmp=%d", ...
            figLabel, filename, startsmp), "Interpreter", "none", "FontSize", 12);
        figPath = saveFigure(hFig, filename, startsmp, framelengths, figLabel);
        disp("Saved figure: " + figPath);
    end
end

function frame = extractFrame(x, startsmp, N)
    frame = zeros(N, 1);
    endIdx = min(startsmp + N - 1, numel(x));
    L = endIdx - startsmp + 1;
    frame(1:L) = x(startsmp:endIdx);
end

function figPath = saveFigure(hFig, filename, startsmp, framelengths, figLabel)
    figDir = fullfile(pwd, "figures");
    if ~exist(figDir, "dir")
        mkdir(figDir);
    end

    [~, base, ~] = fileparts(char(filename));
    lenTag = strjoin(compose("%g", framelengths), "_");
    labelTag = regexprep(lower(char(figLabel)), "[^a-z0-9]+", "");
    figName = sprintf("p2_%s_%s_start%d_L%s.png", base, labelTag, round(startsmp), lenTag);
    figPath = fullfile(figDir, figName);
    exportgraphics(hFig, figPath, "Resolution", 180);
end
