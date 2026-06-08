function [x, fs, xSeg] = load_play_plot(filename, fstart, M, N, doPlay, plotColor)
%LOAD_PLAY_PLOT Strict implementation for Problem 2.32 and 2.33.
%   [x, fs, xSeg] = load_play_plot(filename, fstart, M, N, doPlay, plotColor)
%
% 2.32:
%   - Plot samples from fstart to fstart+N-1
%   - M samples per line
%   - Up to 4 lines per page
% 2.33:
%   - Use strips_modified.m with the same fstart, M, N data segment
%
% Inputs:
%   filename  : .wav file path
%   fstart    : starting sample index (1-based)
%   M         : samples per line
%   N         : total number of samples to plot
%   doPlay    : play original speech if true (default true)
%   plotColor : line color (default 'k')

if nargin < 2 || isempty(fstart)
    fstart = 1;
end
if nargin < 3 || isempty(M)
    M = 250;
end
if nargin < 4 || isempty(N)
    N = 22000;
end
if nargin < 5 || isempty(doPlay)
    doPlay = true;
end
if nargin < 6 || isempty(plotColor)
    plotColor = 'k';
end

[x, fs] = audioread(filename);
if size(x, 2) > 1
    x = mean(x, 2);
end
x = x(:);

L = numel(x);
if fstart < 1 || fstart > L
    error('fstart (%d) is out of range. Valid range is [1, %d].', fstart, L);
end

fend = min(fstart + N - 1, L);
xSeg = x(fstart:fend);
Nactual = numel(xSeg);

fprintf('Loaded: %s\n', filename);
fprintf('Fs = %d Hz, total samples = %d, duration = %.3f s\n', fs, L, L/fs);
fprintf('Using fstart = %d, M = %d, N(requested) = %d, N(actual) = %d\n', ...
    fstart, M, N, Nactual);

if doPlay
    fprintf('Playing original speech...\n');
    soundsc(x, fs);
end

%% Problem 2.32: manual plotting, M samples/line, up to 4 lines/page
numLines = ceil(Nactual / M);
linesPerPage = 4;
numPages = ceil(numLines / linesPerPage);

for p = 1:numPages
    figure('Name', sprintf('Problem 2.32 - Page %d', p), 'Color', 'w');

    for r = 1:linesPerPage
        k = (p - 1) * linesPerPage + r;
        if k > numLines
            break;
        end

        idx1 = (k - 1) * M + 1;
        idx2 = min(k * M, Nactual);

        yLine = xSeg(idx1:idx2);
        sampleAxis = (fstart + idx1 - 1) : (fstart + idx2 - 1);

        subplot(linesPerPage, 1, r);
        plot(sampleAxis, yLine, plotColor, 'LineWidth', 1);
        grid on;
        xlim([sampleAxis(1), sampleAxis(end)]);
        ylabel('Amplitude');
        title(sprintf('Line %d: sample %d to %d', k, sampleAxis(1), sampleAxis(end)));
        try
            legend(sprintf('M=%d samples per line, page %d', M, p), 'Location', 'northeast');
        catch
            legend(sprintf('M=%d samples per line, page %d', M, p));
        end

        if r == linesPerPage || k == numLines
            xlabel('Sample index');
        end
    end

    if exist('sgtitle', 'file') == 2
        sgtitle(sprintf('Problem 2.32 (page %d/%d): fstart=%d, M=%d, N=%d', ...
            p, numPages, fstart, M, Nactual));
    end
end

%% Problem 2.33: strips_modified with the same fstart, M, N
if exist('strips_modified', 'file') ~= 2 && exist('strips_modified.m', 'file') ~= 2
    warning('strips_modified.m not found. Problem 2.33 figure skipped.');
    return;
end

figure('Name', 'Problem 2.33 - strips\_modified', 'Color', 'w');
stripDurationSec = M / fs;      % M samples per strip line
startLabelSec = (fstart - 1) / fs;
strips_modified(xSeg, stripDurationSec, startLabelSec, fs, 1, plotColor);
xlabel('Time (s) within each strip');
ylabel('Strip start time (s)');
title(sprintf('Problem 2.33: strips\_modified (fstart=%d, M=%d, N=%d)', ...
    fstart, M, Nactual));

% Report-friendly legend and caption:
try
    legend(sprintf('Each strip line = %d samples (%.0f ms)', M, stripDurationSec * 1000), ...
        'Location', 'northeast');
catch
    % Skip legend for older MATLAB versions.
end

annotation('textbox', [0.13, 0.01, 0.78, 0.08], ...
    'String', ['Caption: In Problem 2.33, each horizontal trace is one strip. ', ...
               'The y-axis indicates the start time (in seconds) of that strip in the original waveform.'], ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'FontSize', 9);

end
