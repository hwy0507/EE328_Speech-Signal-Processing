import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, resample_poly

BASE = '/Users/hwy/Desktop/个人/26春/语音信号处理/lab5'
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def read_wav(path):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return fs, x


def nextpow2(n):
    return int(2 ** np.ceil(np.log2(n)))


def stsa_single_frame_figure(filename, startsmp, framelength_ms, outpath):
    fs, x = read_wav(os.path.join(BASE, filename))
    startsmp = int(round(startsmp))
    N = int(round(framelength_ms * 1e-3 * fs))

    frame = np.zeros(N)
    end = min(startsmp - 1 + N, len(x))
    valid = end - (startsmp - 1)
    frame[:valid] = x[startsmp - 1:end]

    w = np.hamming(N)
    xw = frame * w

    nfft = nextpow2(max(512, 4 * N))
    X = np.fft.rfft(xw, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(X)
    mag_db = 20 * np.log10(mag + 1e-12)

    t = np.arange(len(x)) / fs
    tf = (np.arange(startsmp - 1, startsmp - 1 + N)) / fs

    plt.figure(figsize=(12, 8), dpi=150)

    ax = plt.subplot(2, 2, 1)
    ax.plot(t, x, color='tab:blue', linewidth=0.9)
    ax.axvspan(tf[0], tf[-1], color='orange', alpha=0.25)
    ax.set_title('Entire speech waveform (selected frame highlighted)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)

    ax = plt.subplot(2, 2, 2)
    ax.plot(tf, xw, color='black', linewidth=1.0)
    ax.set_title(f'Windowed frame (Hamming), L={N} samples')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)

    ax = plt.subplot(2, 2, 3)
    ax.plot(f, mag, linewidth=1.0)
    ax.set_xlim(0, fs / 2)
    ax.set_title(f'|STFT| with zero padding (NFFT={nfft})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(alpha=0.3)

    ax = plt.subplot(2, 2, 4)
    ax.plot(f, mag_db, linewidth=1.0)
    ax.set_xlim(0, fs / 2)
    ax.set_title('Log magnitude of STFT')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Problem 1: STSA_SingleFrame | {filename} | startsmp={startsmp}, L={framelength_ms} ms')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath)
    plt.close()


def extract_frame(x, startsmp, N):
    frame = np.zeros(N)
    end = min(startsmp - 1 + N, len(x))
    valid = end - (startsmp - 1)
    frame[:valid] = x[startsmp - 1:end]
    return frame


def stsa_multilengths_figure(filename, startsmp, framelengths_ms, use_hamming, outpath):
    fs, x = read_wav(os.path.join(BASE, filename))
    startsmp = int(round(startsmp))
    framelengths_ms = np.array(framelengths_ms, dtype=float)
    frame_lens = np.round(framelengths_ms * 1e-3 * fs).astype(int)
    nfft = nextpow2(max(512, 4 * frame_lens.max()))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(frame_lens))]

    t = np.arange(len(x)) / fs

    plt.figure(figsize=(12, 8), dpi=150)

    ax = plt.subplot(2, 2, 1)
    ax.plot(t, x, color='tab:blue', linewidth=0.9)
    ax.set_title(f'Speech waveform: {filename}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)

    ax = plt.subplot(2, 2, 2)
    for i, N in enumerate(frame_lens):
        frame = extract_frame(x, startsmp, N)
        win = np.hamming(N) if use_hamming else np.ones(N)
        xw = frame * win
        tloc = np.arange(N) / fs * 1e3
        ax.plot(tloc, xw, color=colors[i], linewidth=1.0, label=f'{framelengths_ms[i]:g} ms')
    ax.set_title('Windowed speech segments')
    ax.set_xlabel('Local frame time (ms)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 3)
    for i, N in enumerate(frame_lens):
        frame = extract_frame(x, startsmp, N)
        win = np.hamming(N) if use_hamming else np.ones(N)
        X = np.fft.rfft(frame * win, n=nfft)
        mag = np.abs(X)
        ax.plot(f, mag, color=colors[i], linewidth=1.0, label=f'{framelengths_ms[i]:g} ms')
    ax.set_xlim(0, fs / 2)
    ax.set_title('Magnitude spectra')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 4)
    for i, N in enumerate(frame_lens):
        frame = extract_frame(x, startsmp, N)
        win = np.hamming(N) if use_hamming else np.ones(N)
        X = np.fft.rfft(frame * win, n=nfft)
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)
        ax.plot(f, mag_db, color=colors[i], linewidth=1.0, label=f'{framelengths_ms[i]:g} ms')
    ax.set_xlim(0, fs / 2)
    ax.set_title('Log magnitude spectra')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    wname = 'Hamming window' if use_hamming else 'Rectangular window'
    plt.suptitle(f'Problem 2: STSA_MultiLengths | {wname} | startsmp={startsmp}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath)
    plt.close()


def compute_spec(x, fs, win_ms, nfft):
    nwin = int(round(win_ms * 1e-3 * fs))
    nwin = max(2, nwin)
    noverlap = int(round(0.75 * nwin))
    f, t, S = spectrogram(x, fs=fs, window=('hamming'), nperseg=nwin, noverlap=noverlap, nfft=int(nfft), mode='complex')
    return t, f, S


def spectrogram_pair_figure(filename, resamplerate, windowlengths_ms, fftlengths, magscale, dyn_range, cmap_name, outpath):
    fs0, x = read_wav(os.path.join(BASE, filename))
    fs = fs0
    if resamplerate and int(resamplerate) != fs0:
        target = int(resamplerate)
        g = np.gcd(target, fs0)
        up = target // g
        down = fs0 // g
        x = resample_poly(x, up, down)
        fs = target

    t_w, f_w, S_w = compute_spec(x, fs, windowlengths_ms[0], fftlengths[0])
    t_n, f_n, S_n = compute_spec(x, fs, windowlengths_ms[1], fftlengths[1])

    if magscale.lower() == 'log':
        M_w = 20 * np.log10(np.abs(S_w) + 1e-12)
        M_n = 20 * np.log10(np.abs(S_n) + 1e-12)
    else:
        M_w = np.abs(S_w)
        M_n = np.abs(S_n)

    plt.figure(figsize=(10, 7), dpi=150)
    ax1 = plt.subplot(2, 1, 1)
    im1 = ax1.pcolormesh(t_w, f_w / 1000.0, M_w, shading='auto', cmap=cmap_name)
    ax1.set_title(f'Wideband (L={windowlengths_ms[0]} ms, FFT={fftlengths[0]})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (kHz)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.tick_params(labelsize=8)

    ax2 = plt.subplot(2, 1, 2)
    im2 = ax2.pcolormesh(t_n, f_n / 1000.0, M_n, shading='auto', cmap=cmap_name)
    ax2.set_title(f'Narrowband (L={windowlengths_ms[1]} ms, FFT={fftlengths[1]})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (kHz)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.ax.tick_params(labelsize=8)

    if dyn_range is not None and dyn_range > 0:
        if magscale.lower() == 'log':
            top1 = np.nanmax(M_w)
            top2 = np.nanmax(M_n)
            im1.set_clim(top1 - dyn_range, top1)
            im2.set_clim(top2 - dyn_range, top2)
        else:
            top1 = np.nanmax(M_w)
            top2 = np.nanmax(M_n)
            im1.set_clim(max(top1 - dyn_range, 0), top1)
            im2.set_clim(max(top2 - dyn_range, 0), top2)

    plt.suptitle(
        f'Problem 3: fs={fs} Hz, mag={magscale}, range={dyn_range}, cmap={cmap_name}',
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath)
    plt.close()


def main():
    stsa_single_frame_figure('s5.wav', 7000, 40, os.path.join(FIGDIR, 'p1_s5_singleframe.png'))
    stsa_single_frame_figure('vowel_iy_100hz.wav', 1000, 40, os.path.join(FIGDIR, 'p1_vowel_singleframe.png'))

    stsa_multilengths_figure('s5.wav', 7000, [5, 10, 20, 40], True, os.path.join(FIGDIR, 'p2_hamming_multilengths.png'))
    stsa_multilengths_figure('s5.wav', 7000, [5, 10, 20, 40], False, os.path.join(FIGDIR, 'p2_rect_multilengths.png'))

    spectrogram_pair_figure('s5.wav', 0, [5, 50], [1024, 1024], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_default_color.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [1024, 1024], 'log', 60, 'gray', os.path.join(FIGDIR, 'p3_default_gray.png'))
    spectrogram_pair_figure('s5.wav', 4000, [5, 50], [1024, 1024], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_resample_4000.png'))
    spectrogram_pair_figure('s5.wav', 16000, [5, 50], [1024, 1024], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_resample_16000.png'))
    spectrogram_pair_figure('s5.wav', 0, [2, 100], [1024, 1024], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_window_2_100.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [512, 512], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_fft_512.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [2048, 2048], 'log', 60, 'turbo', os.path.join(FIGDIR, 'p3_fft_2048.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [1024, 1024], 'linear', 60, 'turbo', os.path.join(FIGDIR, 'p3_linear_mag.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [1024, 1024], 'log', 30, 'turbo', os.path.join(FIGDIR, 'p3_range_30.png'))
    spectrogram_pair_figure('s5.wav', 0, [5, 50], [1024, 1024], 'log', 120, 'turbo', os.path.join(FIGDIR, 'p3_range_120.png'))


if __name__ == '__main__':
    main()
