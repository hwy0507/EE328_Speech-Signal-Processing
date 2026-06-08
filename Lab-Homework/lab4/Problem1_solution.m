%% Problem 1: Compare five L-point windows in time and frequency
% Requirement summary:
% 1) L must be an odd integer.
% 2) Plot time responses of 5 windows on one figure.
% 3) Plot log-magnitude responses with zero-padding FFT.
% 4) Compare effective bandwidth and peak sidelobe ripple (dB).

clear; clc; close all;

L = 101;                    % try changing this value
Fs = 1;                     % normalized sampling frequency
Nfft = 2^18;                % large FFT size for zero-padding effect

if mod(L, 2) == 0 || L <= 1
    error('L must be an odd integer larger than 1.');
end

winNames = {'Rectangular', 'Triangular', 'Hann', 'Hamming', 'Blackman'};
windows = {
    rectwin(L), ...
    triang(L), ...
    hann(L), ...
    hamming(L), ...
    blackman(L)
};
colors = lines(numel(windows));

%% 1) Time responses on a common plot
n = 0:L-1;
figure('Color', 'w', 'Name', 'Problem 1 - Time Responses');
hold on;
for k = 1:numel(windows)
    plot(n, windows{k}, 'LineWidth', 1.5, 'Color', colors(k, :));
end
grid on;
xlabel('Sample index n');
ylabel('Amplitude');
title(sprintf('Time Responses of 5 Windows (L = %d)', L));
legend(winNames, 'Location', 'best');

%% 2) Log-magnitude responses (zero-padding FFT)
figure('Color', 'w', 'Name', 'Problem 1 - Log Magnitude Responses');
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');

mainLobeBW = zeros(numel(windows), 1);
peakSidelobe = zeros(numel(windows), 1);

for k = 1:numel(windows)
    [f, magDb, bwHz, pslDb] = spectrum_metrics(windows{k}, Fs, Nfft);
    plot(ax1, f, magDb, 'LineWidth', 1.2, 'Color', colors(k, :), ...
        'DisplayName', winNames{k});
    plot(ax2, f, magDb, 'LineWidth', 1.2, 'Color', colors(k, :), ...
        'DisplayName', winNames{k});

    mainLobeBW(k) = bwHz;
    peakSidelobe(k) = pslDb;
end

% Full band
xlim(ax1, [0, Fs/2]);
ylim(ax1, [-140, 5]);
grid(ax1, 'on');
title(ax1, 'Log-Magnitude Responses (Full Band)');
ylabel(ax1, 'Magnitude (dB)');
legend(ax1, 'Location', 'southwest');

% Narrow-band view for effective bandwidth comparison (hint: 0 to 5*Fs/L)
xlim(ax2, [0, 5*Fs/L]);
ylim(ax2, [-140, 5]);
grid(ax2, 'on');
title(ax2, sprintf('Narrow-Band View (0 to %.4f = 5*Fs/L)', 5*Fs/L));
xlabel(ax2, 'Frequency (cycles/sample)');
ylabel(ax2, 'Magnitude (dB)');
legend(ax2, 'Location', 'southwest');

%% 3) Numerical comparison table
ResultTable = table(winNames', mainLobeBW, peakSidelobe, ...
    'VariableNames', {'Window', 'EffectiveBandwidth_Hz', 'PeakSidelobe_dB'});

disp(ResultTable);

% Typical trend:
% - Rectangular: narrowest mainlobe but highest sidelobe level.
% - Blackman: widest mainlobe but lowest sidelobe level.

%% Local function
function [f, magDb, bwHz, pslDb] = spectrum_metrics(w, Fs, Nfft)
    W = abs(fft(w, Nfft));
    W = W(1:Nfft/2 + 1);
    W = W ./ max(W);

    f = (0:Nfft/2)' * (Fs / Nfft);
    magDb = 20 * log10(max(W, 1e-12));

    % Find the first local minimum after DC as the first null location.
    dW = diff(W);
    idxNull = find(dW(1:end-1) <= 0 & dW(2:end) >= 0, 1, 'first') + 1;
    if isempty(idxNull)
        idxNull = 2;
    end

    bwHz = 2 * f(idxNull); % two-sided mainlobe width around DC

    if idxNull + 1 <= numel(magDb)
        pslDb = max(magDb(idxNull + 1:end));
    else
        pslDb = NaN;
    end
end
