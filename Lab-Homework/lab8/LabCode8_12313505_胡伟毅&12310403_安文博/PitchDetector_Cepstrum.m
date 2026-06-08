function out = PitchDetector_Cepstrum(s, fs, gender)
%PITCHDETECTOR_CEPSTRUM Cepstrum-based pitch detector.
%   out = PitchDetector_Cepstrum(s, fs, gender)
%   Implements the workflow required by Lab 8 Problem 3.

    if nargin < 3 || isempty(gender)
        gender = 'male';
    end

    s = double(s);
    if size(s, 2) > 1
        s = mean(s, 2);
    end

    fsout = 10000;
    sRes = resampleLinear(s, fs, fsout);

    b_bp = designBandpassFIR(fsout, 301, 150, 900);
    sBP = filter(b_bp, 1, sRes);

    nfft = 4000;
    pthr1 = 4.0;
    delta = 0.10;

    if strcmpi(gender, 'male')
        nlow = 40;
        nhigh = 167;
    else
        nlow = 28;
        nhigh = 67;
    end

    L = 400;
    R = 100;

    signalList = {sRes, sBP};
    signalNames = {'original', 'bandpass'};

    result = repmat(struct(), 1, numel(signalList));
    for k = 1:numel(signalList)
        [pd1, p1, pd2, p2, cepsRange] = cepstralPeakTrack(signalList{k}, L, R, nfft, nlow, nhigh);

        ratio = p1 ./ max(p2, eps);
        reliable = ratio >= pthr1;
        % Fallback: if no high-confidence seed exists, keep the best frame
        % as a seed so the contour extension still works.
        if ~any(reliable) && ~isempty(ratio)
            [~, bestIdx] = max(ratio);
            reliable(bestIdx) = true;
        end

        [pitchRaw, regionMask] = extendReliableRegion(pd1, pd2, p1, reliable, delta);
        pitchSmooth = MedianSmoother(pitchRaw, 5);

        confRaw = p1;
        confSmooth = MedianSmoother(confRaw, 5);

        t = ((0:numel(pd1)-1) * R + L/2) / fsout;

        result(k).name = signalNames{k};
        result(k).primaryPitch = pd1;
        result(k).secondaryPitch = pd2;
        result(k).primaryPeak = p1;
        result(k).secondaryPeak = p2;
        result(k).ratio = ratio;
        result(k).reliable = reliable;
        result(k).regionMask = regionMask;
        result(k).pitchRaw = pitchRaw;
        result(k).pitchSmoothed = pitchSmooth;
        result(k).confidenceRaw = confRaw;
        result(k).confidenceSmoothed = confSmooth;
        result(k).cepstrumRange = cepsRange;
        result(k).time = t(:);
    end

    out = struct();
    out.gender = gender;
    out.fsIn = fs;
    out.fsOut = fsout;
    out.frameLength = L;
    out.frameShift = R;
    out.nfft = nfft;
    out.pthr1 = pthr1;
    out.nlow = nlow;
    out.nhigh = nhigh;
    out.delta = delta;
    out.filter = b_bp(:);
    out.resampled = sRes(:);
    out.bandpass = sBP(:);
    out.result = result;
end

function [pd1, p1, pd2, p2, cepsRange] = cepstralPeakTrack(sig, L, R, nfft, nlow, nhigh)
    sig = sig(:);
    N = numel(sig);
    numFrames = floor((N - L) / R) + 1;

    if numFrames < 1
        pd1 = zeros(0, 1);
        p1 = zeros(0, 1);
        pd2 = zeros(0, 1);
        p2 = zeros(0, 1);
        cepsRange = zeros(0, nhigh - nlow + 1);
        return;
    end

    win = hammingLocal(L);
    K = nhigh - nlow + 1;

    pd1 = zeros(numFrames, 1);
    p1 = zeros(numFrames, 1);
    pd2 = zeros(numFrames, 1);
    p2 = zeros(numFrames, 1);
    cepsRange = zeros(numFrames, K);

    for i = 1:numFrames
        n0 = (i - 1) * R + 1;
        frame = sig(n0:n0 + L - 1);
        frame = frame - mean(frame);
        frame = frame .* win;

        X = fft(frame, nfft);
        cep = real(ifft(log(abs(X) + eps)));

        cSeg = cep(nlow:nhigh);
        cepsRange(i, :) = cSeg(:).';

        [pk1, idx1] = max(cSeg);
        p1(i) = pk1;
        pd1(i) = nlow + idx1 - 1;

        cSeg2 = cSeg;
        z1 = max(1, idx1 - 4);
        z2 = min(K, idx1 + 4);
        cSeg2(z1:z2) = 0;

        [pk2, idx2] = max(cSeg2);
        p2(i) = pk2;
        pd2(i) = nlow + idx2 - 1;
    end
end

function [pitch, regionMask] = extendReliableRegion(pd1, pd2, p1, reliable, delta)
    N = numel(pd1);
    pitch = zeros(N, 1);
    regionMask = false(N, 1);

    seedIdx = find(reliable);
    if isempty(seedIdx)
        return;
    end

    [~, order] = sort(p1(seedIdx), 'descend');
    seedIdx = seedIdx(order);

    for i = 1:numel(seedIdx)
        s = seedIdx(i);
        if regionMask(s)
            continue;
        end

        regionMask(s) = true;
        pitch(s) = pd1(s);

        refPitch = pitch(s);
        j = s - 1;
        while j >= 1 && ~regionMask(j)
            [cand, ok] = pickCandidate(pd1(j), pd2(j), refPitch, delta);
            if ~ok
                break;
            end
            regionMask(j) = true;
            pitch(j) = cand;
            refPitch = cand;
            j = j - 1;
        end

        refPitch = pitch(s);
        j = s + 1;
        while j <= N && ~regionMask(j)
            [cand, ok] = pickCandidate(pd1(j), pd2(j), refPitch, delta);
            if ~ok
                break;
            end
            regionMask(j) = true;
            pitch(j) = cand;
            refPitch = cand;
            j = j + 1;
        end
    end
end

function [cand, ok] = pickCandidate(pdA, pdB, refPitch, delta)
    dA = abs(pdA - refPitch) / max(refPitch, eps);
    dB = abs(pdB - refPitch) / max(refPitch, eps);

    if dA <= delta || dB <= delta
        if dA <= dB
            cand = pdA;
        else
            cand = pdB;
        end
        ok = true;
    else
        cand = 0;
        ok = false;
    end
end

function b = designBandpassFIR(fs, N, fLow, fHigh)
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
