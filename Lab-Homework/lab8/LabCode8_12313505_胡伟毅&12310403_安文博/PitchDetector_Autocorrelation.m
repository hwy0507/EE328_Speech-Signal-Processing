function out = PitchDetector_Autocorrelation(s, fs, gender)
%PITCHDETECTOR_AUTOCORRELATION Autocorrelation-based pitch detector.
%   out = PitchDetector_Autocorrelation(s, fs, gender)
%   Implements the workflow required by Lab 8 Problem 2.

    if nargin < 3 || isempty(gender)
        gender = 'male';
    end

    s = double(s);
    if size(s, 2) > 1
        s = mean(s, 2);
    end

    fsout = 10000;
    sRes = resampleLinear(s, fs, fsout);

    if strcmpi(gender, 'male')
        pdlow = ceil(fsout / 200);
        pdhigh = floor(fsout / 75);
    else
        pdlow = ceil(fsout / 300);
        pdhigh = floor(fsout / 150);
    end

    b_bp = designBandpassFIR(fsout, 301, 150, 900);
    sBP = filter(b_bp, 1, sRes);

    L = 400;
    R = 100;

    signalList = {sRes, sBP};
    signalNames = {'original', 'bandpass'};

    result = repmat(struct(), 1, numel(signalList));
    for k = 1:numel(signalList)
        [pitchRaw, confRaw] = autocPitchTrack(signalList{k}, L, R, pdlow, pdhigh);

        logConf = log10(max(confRaw, eps));
        if isempty(logConf)
            th = -inf;
        else
            th = 0.75 * max(logConf);
        end

        pitchMask = pitchRaw;
        pitchMask(logConf < th) = 0;

        pitchSmooth = MedianSmoother(pitchMask, 5);
        confSmooth = MedianSmoother(logConf, 5);

        t = ((0:numel(pitchRaw)-1) * R + L/2) / fsout;

        result(k).name = signalNames{k};
        result(k).pitchRaw = pitchRaw;
        result(k).pitchThresholded = pitchMask;
        result(k).pitchSmoothed = pitchSmooth;
        result(k).confidenceRaw = confRaw;
        result(k).logConfidence = logConf;
        result(k).logConfidenceSmoothed = confSmooth;
        result(k).threshold = th;
        result(k).time = t(:);
    end

    out = struct();
    out.gender = gender;
    out.fsIn = fs;
    out.fsOut = fsout;
    out.frameLength = L;
    out.frameShift = R;
    out.pdlow = pdlow;
    out.pdhigh = pdhigh;
    out.filter = b_bp(:);
    out.resampled = sRes(:);
    out.bandpass = sBP(:);
    out.result = result;
end

function [pitch, conf] = autocPitchTrack(sig, L, R, pdlow, pdhigh)
    sig = sig(:);
    N = numel(sig);
    numFrames = floor((N - (L + pdhigh)) / R) + 1;

    if numFrames < 1
        pitch = zeros(0, 1);
        conf = zeros(0, 1);
        return;
    end

    pitch = zeros(numFrames, 1);
    conf = zeros(numFrames, 1);

    win = hammingLocal(L);

    for i = 1:numFrames
        n0 = (i - 1) * R + 1;

        x1 = sig(n0:n0 + L - 1);
        x1 = x1 - mean(x1);
        x1 = x1 .* win;

        x2 = sig(n0:n0 + L + pdhigh - 1);
        x2 = x2 - mean(x2);

        c = zeros(pdhigh + 1, 1);
        for lag = 0:pdhigh
            c(lag + 1) = sum(x1 .* x2(1 + lag:L + lag));
        end

        [pk, idx] = max(c(pdlow + 1:pdhigh + 1));
        pitch(i) = pdlow + idx - 1;
        conf(i) = pk;
    end
end

function b = designBandpassFIR(fs, N, fLow, fHigh)
% Windowed-sinc FIR bandpass design (odd N recommended).

    M = N - 1;
    n = 0:M;
    a = M / 2;

    fc1 = fLow / fs;
    fc2 = fHigh / fs;

    hLow2 = 2 * fc2 * sinc(2 * fc2 * (n - a));
    hLow1 = 2 * fc1 * sinc(2 * fc1 * (n - a));
    hIdeal = hLow2 - hLow1;

    w = 0.54 - 0.46 * cos(2 * pi * n / M);
    b = hIdeal .* w;

    fMid = (fLow + fHigh) / 2 / fs;
    Hmid = sum(b .* exp(-1j * 2 * pi * fMid * (n - a)));
    b = real(b / abs(Hmid));
end

function y = resampleLinear(x, fsIn, fsOut)
    x = x(:);
    if fsIn == fsOut
        y = x;
        return;
    end

    tIn = (0:numel(x)-1).' / fsIn;
    tOut = (0:1/fsOut:tIn(end)).';
    y = interp1(tIn, x, tOut, 'linear', 'extrap');
end

function w = hammingLocal(L)
    n = (0:L-1).';
    w = 0.54 - 0.46 * cos(2 * pi * n / (L - 1));
end
