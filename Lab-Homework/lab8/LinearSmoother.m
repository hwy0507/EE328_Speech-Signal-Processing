function y = LinearSmoother(x, L)
%LINEARSMOOTHER Running mean smoother with odd window length L.

    if nargin < 2
        error('LinearSmoother requires two inputs: x and L.');
    end
    if ~isscalar(L) || L < 1 || mod(L, 2) == 0
        error('L must be a positive odd integer.');
    end

    wasRow = isrow(x);
    x = x(:);
    N = numel(x);
    h = (L - 1) / 2;

    y = zeros(N, 1);
    for n = 1:N
        lo = max(1, n - h);
        hi = min(N, n + h);
        y(n) = mean(x(lo:hi));
    end

    if wasRow
        y = y.';
    end
end
