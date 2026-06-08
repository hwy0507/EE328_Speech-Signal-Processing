function y = CombinationSmoother(x, Lmed, Llin)
%COMBINATIONSMOOTHER Robust smoother: smooth trend + smooth residual.
%   y = S{x} + S{x - S{x}}, where S is median+linear smoothing.

    if nargin < 2
        Lmed = 7;
    end
    if nargin < 3
        Llin = 5;
    end

    smoothPart = LinearSmoother(MedianSmoother(x, Lmed), Llin);
    residual = x - smoothPart;
    residualSmooth = LinearSmoother(MedianSmoother(residual, Lmed), Llin);

    y = smoothPart + residualSmooth;
end
