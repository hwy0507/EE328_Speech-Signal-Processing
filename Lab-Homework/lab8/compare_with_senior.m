function metrics = compare_with_senior()
%COMPARE_WITH_SENIOR Quantitative comparison between our Lab8 code and seniors' code.

    rootDir = fileparts(mfilename('fullpath'));
    if isempty(rootDir)
        rootDir = pwd;
    end

    seniorDir = ['/Users/hwy/Desktop/个人/26春/语音信号处理/' ...
        '语音信号处理圣遗物/Lab8/Lab8Code'];

    fprintf('=== Lab8 Cross-Check (ours vs seniors) ===\n');
    fprintf('Root: %s\n', rootDir);
    fprintf('Senior code: %s\n\n', seniorDir);

    %% Problem 1: ZCR smoothing
    [x1, fs1] = audioread(fullfile(rootDir, 'test_16k.wav'));
    if size(x1,2) > 1
        x1 = mean(x1, 2);
    end

    frameLen = round(0.010 * fs1);
    frameShift = round(0.005 * fs1);
    zcr = shortTimeZCRLocal(x1, frameLen, frameShift);

    bLP = designLowpassFIRLocal(41, 0.10);
    zcrLinOurs = filter(bLP, 1, zcr);
    gd = floor((numel(bLP)-1)/2);
    zcrLinOurs = [zcrLinOurs(gd+1:end); repmat(zcrLinOurs(end), gd, 1)];
    zcrMedOurs = MedianSmoother(MedianSmoother(zcr, 7), 5);
    zcrComOurs = CombinationSmoother(zcr, 7, 5);

    zcrLinSenior = seniorLinearSmoother(zcr, seniorDir);
    zcrLinSenior = [zcrLinSenior(gd+1:end); repmat(zcrLinSenior(end), gd, 1)];
    zcrMedSenior = seniorMedianSmoother(seniorMedianSmoother(zcr, 7), 5);
    [zcrComSenior, delay] = seniorCombinationSmoother(zcr, 7, seniorDir);
    if numel(zcrComSenior) >= numel(zcr) + 2*delay
        zcrComSenior = zcrComSenior(2*delay+1 : 2*delay+numel(zcr));
    else
        zcrComSenior = zcrComSenior(1:min(numel(zcrComSenior), numel(zcr)));
    end

    metrics.problem1.linear = pairMetrics(zcrLinOurs, zcrLinSenior);
    metrics.problem1.median = pairMetrics(zcrMedOurs, zcrMedSenior);
    metrics.problem1.combination = pairMetrics(zcrComOurs, zcrComSenior);

    fprintf('[P1] linear   corr=%.4f, mae=%.4f\n', metrics.problem1.linear.corr, metrics.problem1.linear.mae);
    fprintf('[P1] median   corr=%.4f, mae=%.4f\n', metrics.problem1.median.corr, metrics.problem1.median.mae);
    fprintf('[P1] combo    corr=%.4f, mae=%.4f\n\n', metrics.problem1.combination.corr, metrics.problem1.combination.mae);

    %% Problem 2: Autocorrelation pitch
    [x2, fs2] = audioread(fullfile(rootDir, 's6.wav'));
    if size(x2,2) > 1
        x2 = mean(x2, 2);
    end

    outAutocOurs = PitchDetector_Autocorrelation(x2, fs2, 'male');
    [autoFullSenior, autoBPSenior] = seniorAutocPitch(x2, fs2, 'male', seniorDir);

    metrics.problem2.full_pitch = pairMetrics(outAutocOurs.result(1).pitchSmoothed, autoFullSenior.pitchSmoothed);
    metrics.problem2.bp_pitch = pairMetrics(outAutocOurs.result(2).pitchSmoothed, autoBPSenior.pitchSmoothed);

    fprintf('[P2] fullband pitch corr=%.4f, mae=%.4f, voiced(ours/senior)=%.4f/%.4f\n', ...
        metrics.problem2.full_pitch.corr, metrics.problem2.full_pitch.mae, ...
        metrics.problem2.full_pitch.voiced_ratio_a, metrics.problem2.full_pitch.voiced_ratio_b);
    fprintf('[P2] bandpass pitch corr=%.4f, mae=%.4f, voiced(ours/senior)=%.4f/%.4f\n\n', ...
        metrics.problem2.bp_pitch.corr, metrics.problem2.bp_pitch.mae, ...
        metrics.problem2.bp_pitch.voiced_ratio_a, metrics.problem2.bp_pitch.voiced_ratio_b);

    %% Problem 3: Cepstrum pitch
    outCepOurs = PitchDetector_Cepstrum(x2, fs2, 'male');
    [cepFullSenior, cepBPSenior] = seniorCepstrumPitch(x2, fs2, 'male', seniorDir);

    metrics.problem3.full_pitch = pairMetrics(outCepOurs.result(1).pitchSmoothed, cepFullSenior.pitchSmoothed);
    metrics.problem3.bp_pitch = pairMetrics(outCepOurs.result(2).pitchSmoothed, cepBPSenior.pitchSmoothed);

    fprintf('[P3] fullband pitch corr=%.4f, mae=%.4f, voiced(ours/senior)=%.4f/%.4f\n', ...
        metrics.problem3.full_pitch.corr, metrics.problem3.full_pitch.mae, ...
        metrics.problem3.full_pitch.voiced_ratio_a, metrics.problem3.full_pitch.voiced_ratio_b);
    fprintf('[P3] bandpass pitch corr=%.4f, mae=%.4f, voiced(ours/senior)=%.4f/%.4f\n\n', ...
        metrics.problem3.bp_pitch.corr, metrics.problem3.bp_pitch.mae, ...
        metrics.problem3.bp_pitch.voiced_ratio_a, metrics.problem3.bp_pitch.voiced_ratio_b);

    fprintf('=== End of comparison ===\n');
end

function zcr = shortTimeZCRLocal(x, frameLen, frameShift)
    x = x(:);
    N = numel(x);
    frameN = floor((N - frameLen) / frameShift) + 1;
    zcr = zeros(frameN, 1);
    for i = 1:frameN
        idx = (1:frameLen) + (i-1)*frameShift;
        frame = x(idx);
        sgn = sign(frame);
        sgn(sgn == 0) = 1;
        zcr(i) = 0.5 * sum(abs(diff(sgn)));
    end
end

function b = designLowpassFIRLocal(N, fcNorm)
    M = N - 1;
    n = 0:M;
    a = M / 2;
    hIdeal = 2 * fcNorm * sinc(2 * fcNorm * (n - a));
    w = 0.54 - 0.46 * cos(2 * pi * n / M);
    b = hIdeal .* w;
    b = b / sum(b);
end

function y = seniorLinearSmoother(x, seniorDir)
    D = load(fullfile(seniorDir, 'lp.mat'));
    y = filter(D.Hd1, x);
    y = y(:);
end

function y = seniorMedianSmoother(x, n)
    x = x(:).';
    y = zeros(1, length(x));
    for i = 1:n-1
        y(i) = median(x(1:i));
    end
    for i = n:length(x)
        y(i) = median(x(i-n+1:i));
    end
    y = y(:);
end

function [y, delay] = seniorCombinationSmoother(x, n, seniorDir)
    x = x(:).';
    delay = (41-1)/2;

    smooth_input = seniorLinearSmoother(seniorMedianSmoother(x, n), seniorDir);
    smooth_input = smooth_input(:).';

    smooth_input_pad = [smooth_input zeros(1,delay)];
    input_delay = [zeros(1,delay) x];
    diffv = abs(input_delay - smooth_input_pad);

    smooth_diff = seniorLinearSmoother(seniorMedianSmoother(diffv, n), seniorDir);
    smooth_diff = smooth_diff(:).';

    smooth_diff_pad = [smooth_diff zeros(1,delay)];
    smooth_input_delay = [zeros(1,delay) smooth_input_pad];
    y = smooth_diff_pad + smooth_input_delay;
    y = y(:);
end

function [fullOut, bpOut] = seniorAutocPitch(s, fs, gender, seniorDir)
    fsout = 10000;
    s = s(:);
    sRes = resample(s, fsout, fs);

    D = load(fullfile(seniorDir, 'bp.mat'));
    sBP = filter(D.Hd, sRes);

    fullOut = runOne(sRes, gender);
    bpOut = runOne(sBP, gender);

    function out = runOne(sig, genderLocal)
        N = length(sig);
        L = 400;
        R = 100;

        if genderLocal == 'male'
            pdhigh = round(fsout/75);
            pdlow = round(fsout/200);
        else
            pdhigh = round(fsout/150);
            pdlow = round(fsout/300);
        end

        frameN = round((N-pdhigh-L)/R);
        confident = zeros(1,frameN);
        p = zeros(1,frameN);

        for i = 1:frameN
            s1 = sig((1:L)+(i-1)*R);
            s2 = sig((1:L+pdhigh)+(i-1)*R);
            correlation_modified = xcorr(s1, s2);
            j = round(length(correlation_modified)/2);
            [confident(i), idx] = max(correlation_modified(j+pdlow : j+pdhigh));
            p(i) = pdlow + idx - 1;
        end

        confident_log = mag2db(confident)/20;
        threshold = 0.75 * max(confident_log);
        for ii = 1:frameN
            if (confident_log(ii) < threshold) || isnan(confident_log(ii)) || isinf(confident_log(ii))
                p(ii) = 0;
            end
        end

        out.pitchSmoothed = seniorMedianSmoother(p, 5);
        out.confSmoothed = seniorMedianSmoother(confident, 5);
    end
end

function [fullOut, bpOut] = seniorCepstrumPitch(s, fs, gender, seniorDir)
    fsout = 10000;
    s = s(:);
    sRes = resample(s, fsout, fs);

    D = load(fullfile(seniorDir, 'bp.mat'));
    sBP = filter(D.Hd, sRes);

    fullOut = runOne(sRes, gender);
    bpOut = runOne(sBP, gender);

    function out = runOne(sig, genderLocal)
        N = length(sig);
        nfft = 4000;
        L = 400;
        R = 100;
        pthrl = 4;

        if genderLocal == 'male'
            nlow = 40;
            nhigh = 167;
        else
            nlow = 28;
            nhigh = 67;
        end

        Nframe = round((N-L)/R);
        p1 = zeros(1, N);
        pd1 = zeros(1, N);
        p2 = zeros(1, N);
        pd2 = zeros(1, N);
        confident = false(1, N);

        i = 1;
        win = hamming(L, 'periodic');
        while (L + (i-1)*R <= N)
            frame = sig((1:L)+(i-1)*R) .* win;
            X = fft(frame, nfft);
            rcep = ifft(log(abs(X)));

            [p1(i), pd1(i)] = max(rcep(nlow:nhigh));
            pd1(i) = nlow + pd1(i) - 1;

            lo = max(1, pd1(i)-4);
            hi = min(length(rcep), pd1(i)+4);
            rcep(lo:hi) = 0;

            [p2(i), pd2(i)] = max(rcep(nlow:nhigh));
            pd2(i) = nlow + pd2(i) - 1;

            if (p1(i)/p2(i)) >= pthrl
                confident(i) = true;
            end
            i = i + 1;
        end

        p1 = p1(1:Nframe);
        pd1 = pd1(1:Nframe);
        p2 = p2(1:Nframe);
        pd2 = pd2(1:Nframe);
        confident = confident(1:Nframe);

        for ii = 1:Nframe
            if confident(ii)
                j = 1;
                while (ii+j <= length(pd1))
                    if (abs(pd1(ii+j)-pd1(ii+j-1)) <= 0.1*pd1(ii+j-1)) || ...
                       (abs(pd2(ii+j)-pd2(ii+j-1)) <= 0.1*pd2(ii+j-1))
                        confident(ii+j) = true;
                        j = j + 1;
                    else
                        break;
                    end
                end

                j = 1;
                while (ii-j > 0)
                    if (abs(pd1(ii-j)-pd1(ii-j+1)) <= 0.1*pd1(ii-j+1)) || ...
                       (abs(pd2(ii-j)-pd2(ii-j+1)) <= 0.1*pd2(ii-j+1))
                        confident(ii-j) = true;
                        j = j + 1;
                    else
                        break;
                    end
                end
            end
        end

        for ii = 1:Nframe
            if ~confident(ii)
                pd1(ii) = 0;
            end
        end

        out.pitchSmoothed = seniorMedianSmoother(pd1, 5);
        out.confSmoothed = seniorMedianSmoother(p1, 5);
    end
end

function m = pairMetrics(a, b)
    a = a(:);
    b = b(:);
    n = min(numel(a), numel(b));
    if n == 0
        m.corr = NaN;
        m.mae = NaN;
        m.rmse = NaN;
        m.n = 0;
        m.voiced_ratio_a = NaN;
        m.voiced_ratio_b = NaN;
        return;
    end

    a = a(1:n);
    b = b(1:n);

    m.n = n;
    if std(a) > 0 && std(b) > 0
        C = corrcoef(double(a), double(b));
        m.corr = C(1,2);
    else
        m.corr = NaN;
    end

    d = double(a) - double(b);
    m.mae = mean(abs(d));
    m.rmse = sqrt(mean(d.^2));
    m.voiced_ratio_a = mean(a ~= 0);
    m.voiced_ratio_b = mean(b ~= 0);

    idx = (a ~= 0) & (b ~= 0);
    if sum(idx) >= 3
        C2 = corrcoef(double(a(idx)), double(b(idx)));
        m.voiced_corr = C2(1,2);
        dv = double(a(idx)) - double(b(idx));
        m.voiced_mae = mean(abs(dv));
    else
        m.voiced_corr = NaN;
        m.voiced_mae = NaN;
    end
end
