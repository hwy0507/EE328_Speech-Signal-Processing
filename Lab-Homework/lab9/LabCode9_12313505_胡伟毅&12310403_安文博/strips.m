function strips(x, sd, fs)
%STRIPS Plot a signal in consecutive horizontal strips.
%   strips(x, sd, fs) plots vector x using sd seconds per line and sampling
%   frequency fs. For 2000 samples per line at 8000 Hz, use sd = 2000/fs.
%   Each strip is vertically scaled for display so small quantization errors
%   remain visible while the relative shape within each line is preserved.

if nargin ~= 3
    error('Usage: strips(x, sd, fs)');
end

x = x(:);
samplesPerLine = max(1, round(sd * fs));
numLines = ceil(length(x) / samplesPerLine);
timeAxis = (0:samplesPerLine - 1) / fs;
displayScale = 0.4 / max(abs(x) + eps);

holdState = ishold;
hold on;
for lineIndex = 1:numLines
    firstSample = (lineIndex - 1) * samplesPerLine + 1;
    lastSample = min(lineIndex * samplesPerLine, length(x));
    segment = x(firstSample:lastSample);
    offset = numLines - lineIndex;
    plot(timeAxis(1:length(segment)), segment * displayScale + offset, 'LineWidth', 0.9);
end
grid on;
xlabel('Time within strip (s)');
ylabel('Strip index, scaled amplitude offset');
yticks(0:numLines - 1);
yticklabels(string(numLines:-1:1));
ylim([-0.55, numLines - 0.45]);
if ~holdState
    hold off;
end
end
