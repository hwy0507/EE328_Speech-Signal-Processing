from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, filtfilt, get_window

EPS = 1e-8
FRAME_MS = 32
HOP_MS = 10
MIN_F0 = 70.0
MAX_F0 = 350.0


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, audio


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -0.999, 0.999)
    int_audio = np.int16(np.round(clipped * 32767.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, int_audio)


def frame_centers(length: int, frame_length: int, hop_length: int) -> np.ndarray:
    if length < frame_length:
        return np.array([0, max(length - 1, 0)], dtype=np.int32)
    starts = np.arange(0, length - frame_length + 1, hop_length, dtype=np.int32)
    centers = starts + frame_length // 2
    if centers[-1] != length - 1:
        centers = np.concatenate([centers, np.array([length - 1], dtype=np.int32)])
    if centers[0] != 0:
        centers = np.concatenate([np.array([0], dtype=np.int32), centers])
    return centers


def frame_rms_envelope(audio: np.ndarray, sample_rate: int, frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS) -> np.ndarray:
    frame_length = max(int(sample_rate * frame_ms / 1000), 1)
    hop_length = max(int(sample_rate * hop_ms / 1000), 1)
    if len(audio) < frame_length:
        rms = float(np.sqrt(np.mean(np.square(audio)) + EPS))
        return np.full(len(audio), rms, dtype=np.float32)

    starts = np.arange(0, len(audio) - frame_length + 1, hop_length, dtype=np.int32)
    values = [float(np.sqrt(np.mean(np.square(audio[start : start + frame_length])) + EPS)) for start in starts]
    centers = starts + frame_length // 2
    if centers[0] != 0:
        centers = np.concatenate([np.array([0], dtype=np.int32), centers])
        values = [values[0], *values]
    if centers[-1] != len(audio) - 1:
        centers = np.concatenate([centers, np.array([len(audio) - 1], dtype=np.int32)])
        values = [*values, values[-1]]
    return np.interp(np.arange(len(audio)), centers, np.array(values, dtype=np.float32)).astype(np.float32)


def compute_voiced_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = max(int(sample_rate * FRAME_MS / 1000), 1)
    hop_length = max(int(sample_rate * HOP_MS / 1000), 1)
    if len(audio) < frame_length:
        return np.ones(len(audio), dtype=np.float32)

    starts = np.arange(0, len(audio) - frame_length + 1, hop_length, dtype=np.int32)
    centers = starts + frame_length // 2
    window = get_window("hann", frame_length, fftbins=False)
    min_lag = max(int(sample_rate / MAX_F0), 1)
    max_lag = min(int(sample_rate / MIN_F0), frame_length - 1)

    energies = []
    zcrs = []
    corr_peaks = []
    for start in starts:
        frame = audio[start : start + frame_length]
        energies.append(float(np.mean(frame * frame)))
        zcrs.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))
        win_frame = frame * window
        corr = correlate(win_frame, win_frame, mode="full")
        corr = corr[len(corr) // 2 :]
        corr[:min_lag] = 0
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            corr_peaks.append(0.0)
            continue
        peak_idx = int(np.argmax(segment)) + min_lag
        corr_peaks.append(float(corr[peak_idx] / (corr[0] + EPS)))

    energies_np = np.asarray(energies, dtype=np.float32)
    zcrs_np = np.asarray(zcrs, dtype=np.float32)
    corr_np = np.asarray(corr_peaks, dtype=np.float32)
    energy_floor = max(float(np.percentile(energies_np, 35)), 1e-5)
    voiced_frames = ((energies_np > energy_floor) & (zcrs_np < 0.24) & (corr_np > 0.28)).astype(np.float32)

    if centers[0] != 0:
        centers = np.concatenate([np.array([0], dtype=np.int32), centers])
        voiced_frames = np.concatenate([voiced_frames[:1], voiced_frames])
    if centers[-1] != len(audio) - 1:
        centers = np.concatenate([centers, np.array([len(audio) - 1], dtype=np.int32)])
        voiced_frames = np.concatenate([voiced_frames, voiced_frames[-1:]])

    mask = np.interp(np.arange(len(audio)), centers, voiced_frames).astype(np.float32)
    smooth_len = max(int(sample_rate * 0.02), 3)
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    mask = np.convolve(mask, kernel, mode="same")
    return np.clip(mask, 0.0, 1.0)


def compute_detail_gate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    env = frame_rms_envelope(audio, sample_rate)
    noise_floor = max(float(np.percentile(env, 20)), 1e-4)
    speech_level = max(float(np.percentile(env, 80)), noise_floor + EPS)
    gate = np.clip((env - noise_floor) / (speech_level - noise_floor + EPS), 0.0, 1.0)
    smooth_len = max(int(sample_rate * 0.015), 3)
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    gate = np.convolve(gate, kernel, mode="same")
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    normalized = cutoff_hz / (sample_rate / 2.0)
    b, a = butter(2, normalized, btype="highpass")
    return filtfilt(b, a, audio).astype(np.float32)


def match_envelope(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> np.ndarray:
    ref_env = frame_rms_envelope(reference, sample_rate)
    cand_env = frame_rms_envelope(candidate, sample_rate)
    gain = ref_env / (cand_env + EPS)
    gain = np.clip(gain, 0.65, 1.45)
    return candidate * gain


def match_rms(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_rms = float(np.sqrt(np.mean(np.square(reference)) + EPS))
    cand_rms = float(np.sqrt(np.mean(np.square(candidate)) + EPS))
    scaled = candidate * (ref_rms / (cand_rms + EPS))
    return scaled.astype(np.float32)


def peak_limit(audio: np.ndarray, peak_dbfs: float = -1.5) -> np.ndarray:
    target_peak = float(10 ** (peak_dbfs / 20.0))
    peak = float(np.max(np.abs(audio)) + EPS)
    if peak > target_peak:
        audio = audio * (target_peak / peak)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def humanize_audio(source_audio: np.ndarray, candidate_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    length = min(len(source_audio), len(candidate_audio))
    source = source_audio[:length].astype(np.float32)
    candidate = candidate_audio[:length].astype(np.float32)

    mask = compute_voiced_mask(source, sample_rate)
    detail_gate = compute_detail_gate(source, sample_rate)
    candidate = match_envelope(source, candidate, sample_rate)
    consonant_detail = highpass_filter(source, sample_rate, 2600.0)
    air_detail = highpass_filter(source, sample_rate, 4200.0)

    voiced_source_weight = 0.03 * detail_gate
    voiced_candidate_weight = 1.0 - voiced_source_weight
    unvoiced_source_weight = 0.08 * detail_gate
    unvoiced_candidate_weight = 1.0 - unvoiced_source_weight

    voiced_mix = mask * (voiced_candidate_weight * candidate + voiced_source_weight * source)
    unvoiced_mix = (1.0 - mask) * (unvoiced_candidate_weight * candidate + unvoiced_source_weight * source)
    detail_mix = 0.035 * (1.0 - mask) * detail_gate * consonant_detail + 0.01 * mask * detail_gate * air_detail
    output = voiced_mix + unvoiced_mix + detail_mix
    output = match_rms(candidate, output)
    return peak_limit(output)


def humanize_candidate(source_path: Path, candidate_path: Path, output_path: Path | None = None) -> Path:
    source_rate, source_audio = load_audio(source_path)
    candidate_rate, candidate_audio = load_audio(candidate_path)
    if source_rate != candidate_rate:
        raise ValueError(f"Sample rate mismatch: {source_path} vs {candidate_path}")
    output_audio = humanize_audio(source_audio, candidate_audio, source_rate)
    target_path = output_path or candidate_path
    save_audio(target_path, output_audio, source_rate)
    return target_path
