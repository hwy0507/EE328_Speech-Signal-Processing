from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, get_window, istft, stft

EPS = 1e-8
FRAME_MS = 40
HOP_MS = 10
MIN_F0 = 70.0
MAX_F0 = 360.0
TONE_LABELS = ("level", "rising", "falling", "dipping", "neutral")


@dataclass(frozen=True)
class ToneSegment:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_f0_hz: float | None
    end_f0_hz: float | None
    slope_st: float
    label: str
    confidence: float


@dataclass(frozen=True)
class FrameAnalysis:
    sample_rate: int
    hop_length: int
    frame_length: int
    times_sec: np.ndarray
    rms: np.ndarray
    voiced: np.ndarray
    f0_hz: np.ndarray
    log_f0: np.ndarray
    content_posterior: np.ndarray
    content_entropy: np.ndarray
    tone_posterior: np.ndarray
    tone_error_st: np.ndarray
    tone_segments: tuple[ToneSegment, ...]


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, audio.astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -0.999, 0.999)
    wavfile.write(path, sample_rate, np.int16(np.round(clipped * 32767.0)))


def peak_limit(audio: np.ndarray, peak_dbfs: float = -1.2) -> np.ndarray:
    target_peak = float(10 ** (peak_dbfs / 20.0))
    peak = float(np.max(np.abs(audio)) + EPS)
    if peak > target_peak:
        audio = audio * (target_peak / peak)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def match_rms(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_rms = float(np.sqrt(np.mean(np.square(reference)) + EPS))
    cand_rms = float(np.sqrt(np.mean(np.square(candidate)) + EPS))
    return (candidate * (ref_rms / (cand_rms + EPS))).astype(np.float32)


def frame_starts(length: int, frame_length: int, hop_length: int) -> np.ndarray:
    if length <= frame_length:
        return np.array([0], dtype=np.int32)
    return np.arange(0, length - frame_length + 1, hop_length, dtype=np.int32)


def interpolate_frames(values: np.ndarray, analysis: FrameAnalysis, target_length: int) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(
            np.arange(target_length, dtype=np.float32) / analysis.sample_rate,
            analysis.times_sec,
            values,
            left=float(values[0]),
            right=float(values[-1]),
        ).astype(np.float32)

    columns = [
        np.interp(
            np.arange(target_length, dtype=np.float32) / analysis.sample_rate,
            analysis.times_sec,
            values[:, col],
            left=float(values[0, col]),
            right=float(values[-1, col]),
        )
        for col in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def frame_rms(audio: np.ndarray, starts: np.ndarray, frame_length: int) -> np.ndarray:
    values = []
    for start in starts:
        frame = audio[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        values.append(float(np.sqrt(np.mean(np.square(frame)) + EPS)))
    return np.asarray(values, dtype=np.float32)


def estimate_f0_track(
    audio: np.ndarray,
    sample_rate: int,
    starts: np.ndarray,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    window = get_window("hann", frame_length, fftbins=False)
    min_lag = max(int(sample_rate / MAX_F0), 1)
    max_lag = min(int(sample_rate / MIN_F0), frame_length - 1)
    f0_values: list[float] = []
    peak_values: list[float] = []

    for start in starts:
        frame = audio[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame = frame - float(np.mean(frame))
        win_frame = frame * window
        corr = correlate(win_frame, win_frame, mode="full")
        corr = corr[len(corr) // 2 :]
        zero_lag = float(corr[0])
        corr[:min_lag] = 0.0
        segment = corr[min_lag:max_lag]
        if segment.size == 0 or zero_lag <= EPS:
            f0_values.append(0.0)
            peak_values.append(0.0)
            continue
        lag = int(np.argmax(segment)) + min_lag
        normalized_peak = float(corr[lag] / (zero_lag + EPS))
        f0_values.append(float(sample_rate / lag) if normalized_peak >= 0.25 else 0.0)
        peak_values.append(normalized_peak)

    return np.asarray(f0_values, dtype=np.float32), np.asarray(peak_values, dtype=np.float32)


def build_voiced_mask(rms: np.ndarray, f0_hz: np.ndarray, corr_peak: np.ndarray) -> np.ndarray:
    energy_floor = max(float(np.percentile(rms, 30)), 1e-4)
    strong = (rms > energy_floor) & (f0_hz > 0.0) & (corr_peak > 0.28)
    voiced = strong.astype(np.float32)
    if len(voiced) >= 5:
        kernel = np.array([0.12, 0.2, 0.36, 0.2, 0.12], dtype=np.float32)
        voiced = np.convolve(voiced, kernel, mode="same")
    return np.clip(voiced, 0.0, 1.0).astype(np.float32)


def fill_log_f0(f0_hz: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    log_f0 = np.zeros_like(f0_hz, dtype=np.float32)
    valid = (f0_hz > 0.0) & (voiced > 0.25)
    if not np.any(valid):
        return log_f0
    frames = np.arange(len(f0_hz), dtype=np.float32)
    valid_frames = frames[valid]
    valid_log = np.log2(f0_hz[valid])
    log_f0 = np.interp(frames, valid_frames, valid_log, left=float(valid_log[0]), right=float(valid_log[-1]))
    return log_f0.astype(np.float32)


def smooth_1d(values: np.ndarray, passes: int = 2) -> np.ndarray:
    output = values.astype(np.float32).copy()
    if len(output) < 3:
        return output
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    for _ in range(passes):
        padded = np.pad(output, (1, 1), mode="edge")
        output = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    return output


def band_edges_hz(sample_rate: int, num_bands: int = 18) -> np.ndarray:
    upper = min(sample_rate / 2.0, 7600.0)
    low = 80.0
    mel_low = 2595.0 * math.log10(1.0 + low / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + upper / 700.0)
    mels = np.linspace(mel_low, mel_high, num_bands + 1)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    hz[0] = 0.0
    hz[-1] = upper
    return hz.astype(np.float32)


def content_posterior(audio: np.ndarray, sample_rate: int, starts: np.ndarray, frame_length: int) -> tuple[np.ndarray, np.ndarray]:
    window = get_window("hann", frame_length, fftbins=False)
    edges = band_edges_hz(sample_rate)
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    band_masks = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(freqs - (low + high) * 0.5)))
            mask[nearest] = True
        band_masks.append(mask)

    rows = []
    for start in starts:
        frame = audio[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        spectrum = np.abs(np.fft.rfft((frame - float(np.mean(frame))) * window)) + EPS
        energies = np.asarray([float(np.mean(spectrum[mask])) for mask in band_masks], dtype=np.float32)
        log_energy = np.log(energies + EPS)
        centered = log_energy - float(np.mean(log_energy))
        scaled = centered / (float(np.std(centered)) + 1.0)
        shifted = scaled - float(np.max(scaled))
        probs = np.exp(shifted)
        probs = probs / (float(np.sum(probs)) + EPS)
        rows.append(probs.astype(np.float32))

    posterior = np.stack(rows, axis=0)
    entropy = -np.sum(posterior * np.log(posterior + EPS), axis=1) / math.log(posterior.shape[1])
    return posterior.astype(np.float32), entropy.astype(np.float32)


def voiced_runs(voiced: np.ndarray, threshold: float = 0.45, min_frames: int = 4) -> list[tuple[int, int]]:
    active = voiced >= threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_frames:
                runs.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_frames:
        runs.append((start, len(active)))
    return runs


def tone_posterior_from_label(label: str, confidence: float) -> np.ndarray:
    base = np.full(len(TONE_LABELS), (1.0 - confidence) / max(len(TONE_LABELS) - 1, 1), dtype=np.float32)
    base[TONE_LABELS.index(label)] = confidence
    return base / (float(np.sum(base)) + EPS)


def classify_tone(log_f0: np.ndarray, start: int, end: int, hop_length: int, sample_rate: int) -> tuple[str, float, float, np.ndarray]:
    segment = log_f0[start:end].astype(np.float32)
    duration_sec = (end - start) * hop_length / sample_rate
    if len(segment) < 3 or duration_sec < 0.08:
        return "neutral", 0.62, 0.0, segment.copy()

    smoothed = smooth_1d(segment, passes=2)
    start_level = float(np.percentile(smoothed[: max(2, len(smoothed) // 4)], 50))
    end_level = float(np.percentile(smoothed[-max(2, len(smoothed) // 4) :], 50))
    slope_st = (end_level - start_level) * 12.0
    mid = float(np.percentile(smoothed[len(smoothed) // 3 : max(len(smoothed) // 3 + 1, 2 * len(smoothed) // 3)], 30))
    edge_min = min(start_level, end_level)
    dip_st = (edge_min - mid) * 12.0

    if dip_st > 1.7 and slope_st > -1.2:
        label = "dipping"
        confidence = min(0.9, 0.55 + dip_st / 8.0)
    elif slope_st > 2.0:
        label = "rising"
        confidence = min(0.9, 0.55 + abs(slope_st) / 12.0)
    elif slope_st < -2.0:
        label = "falling"
        confidence = min(0.9, 0.55 + abs(slope_st) / 12.0)
    elif duration_sec < 0.16:
        label = "neutral"
        confidence = 0.6
    else:
        label = "level"
        confidence = max(0.58, 0.82 - abs(slope_st) / 12.0)

    if label == "level":
        target = np.full_like(smoothed, float(np.median(smoothed)))
    elif label in {"rising", "falling"}:
        target = np.linspace(start_level, end_level, len(smoothed), dtype=np.float32)
    elif label == "dipping":
        left = np.linspace(start_level, min(start_level, end_level) - dip_st / 12.0 * 0.65, len(smoothed) // 2 + 1)
        right = np.linspace(left[-1], end_level, len(smoothed) - len(left) + 1)
        target = np.concatenate([left[:-1], right]).astype(np.float32)
    else:
        target = smoothed

    return label, float(confidence), float(slope_st), target.astype(np.float32)


def analyze_frames(audio: np.ndarray, sample_rate: int) -> FrameAnalysis:
    frame_length = max(int(sample_rate * FRAME_MS / 1000), 1)
    hop_length = max(int(sample_rate * HOP_MS / 1000), 1)
    starts = frame_starts(len(audio), frame_length, hop_length)
    centers = starts + frame_length // 2
    times = centers.astype(np.float32) / sample_rate
    rms = frame_rms(audio, starts, frame_length)
    f0_hz, corr_peak = estimate_f0_track(audio, sample_rate, starts, frame_length)
    voiced = build_voiced_mask(rms, f0_hz, corr_peak)
    log_f0 = fill_log_f0(f0_hz, voiced)
    posterior, entropy = content_posterior(audio, sample_rate, starts, frame_length)

    tone_posterior = np.tile(tone_posterior_from_label("neutral", 0.4), (len(starts), 1))
    tone_error_st = np.zeros(len(starts), dtype=np.float32)
    segments: list[ToneSegment] = []

    for start, end in voiced_runs(voiced):
        label, confidence, slope_st, target = classify_tone(log_f0, start, end, hop_length, sample_rate)
        tone_posterior[start:end] = tone_posterior_from_label(label, confidence)
        segment_log = log_f0[start:end]
        tone_error_st[start:end] = np.abs(segment_log - target[: end - start]) * 12.0
        start_f0 = float(2 ** segment_log[0]) if segment_log.size else None
        end_f0 = float(2 ** segment_log[-1]) if segment_log.size else None
        segments.append(
            ToneSegment(
                start_frame=start,
                end_frame=end,
                start_sec=float(times[start]),
                end_sec=float(times[min(end - 1, len(times) - 1)]),
                duration_sec=float((end - start) * hop_length / sample_rate),
                start_f0_hz=start_f0,
                end_f0_hz=end_f0,
                slope_st=slope_st,
                label=label,
                confidence=confidence,
            )
        )

    return FrameAnalysis(
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_length=frame_length,
        times_sec=times.astype(np.float32),
        rms=rms,
        voiced=voiced,
        f0_hz=f0_hz,
        log_f0=log_f0,
        content_posterior=posterior,
        content_entropy=entropy,
        tone_posterior=tone_posterior.astype(np.float32),
        tone_error_st=tone_error_st,
        tone_segments=tuple(segments),
    )


def coarse_spectral_envelope(log_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    edges = band_edges_hz(int(freqs[-1] * 2), num_bands=20)
    band_centers = []
    band_values = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        band_centers.append((low + high) * 0.5)
        band_values.append(np.mean(log_mag[mask, :], axis=0))
    centers = np.asarray(band_centers, dtype=np.float32)
    values = np.stack(band_values, axis=0)

    coarse = np.empty_like(log_mag, dtype=np.float32)
    for frame_idx in range(log_mag.shape[1]):
        coarse[:, frame_idx] = np.interp(freqs, centers, values[:, frame_idx], left=values[0, frame_idx], right=values[-1, frame_idx])

    time_kernel = np.array([0.06, 0.16, 0.56, 0.16, 0.06], dtype=np.float32)
    freq_kernel = np.array([0.12, 0.2, 0.36, 0.2, 0.12], dtype=np.float32)
    padded_time = np.pad(coarse, ((0, 0), (2, 2)), mode="edge")
    coarse = np.apply_along_axis(lambda row: np.convolve(row, time_kernel, mode="valid"), 1, padded_time)
    padded_freq = np.pad(coarse, ((2, 2), (0, 0)), mode="edge")
    coarse = np.apply_along_axis(lambda col: np.convolve(col, freq_kernel, mode="valid"), 0, padded_freq)
    return coarse.astype(np.float32)


def tone_energy_curve(source_analysis: FrameAnalysis, target_length: int) -> np.ndarray:
    curve = np.ones(len(source_analysis.times_sec), dtype=np.float32)
    for segment in source_analysis.tone_segments:
        start = segment.start_frame
        end = segment.end_frame
        if end <= start:
            continue
        count = end - start
        x = np.linspace(0.0, 1.0, count, dtype=np.float32)
        if segment.label == "rising":
            local = 0.985 + 0.035 * x
        elif segment.label == "falling":
            local = 1.02 - 0.035 * x
        elif segment.label == "dipping":
            local = 1.0 - 0.03 * np.sin(np.pi * x)
        elif segment.label == "neutral":
            local = np.full(count, 0.99, dtype=np.float32)
        else:
            local = np.ones(count, dtype=np.float32)
        curve[start:end] = local
    return interpolate_frames(curve, source_analysis, target_length)


def ppg_tone_naturalize_audio(
    source_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    length = min(len(source_audio), len(candidate_audio))
    source = source_audio[:length].astype(np.float32)
    candidate = candidate_audio[:length].astype(np.float32)

    source_analysis = analyze_frames(source, sample_rate)
    before_analysis = analyze_frames(candidate, sample_rate)

    nperseg = 512
    noverlap = 384
    freqs, _, spec = stft(candidate, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True)
    mag = np.abs(spec) + EPS
    phase = np.exp(1j * np.angle(spec))
    log_mag = np.log(mag).astype(np.float32)
    coarse_log_mag = coarse_spectral_envelope(log_mag, freqs.astype(np.float32))

    stft_times = np.linspace(0.0, length / sample_rate, spec.shape[1], dtype=np.float32)
    voiced_stft = np.interp(stft_times, before_analysis.times_sec, before_analysis.voiced, left=0.0, right=0.0)
    tone_error_stft = np.interp(stft_times, before_analysis.times_sec, before_analysis.tone_error_st, left=0.0, right=0.0)
    entropy_stft = np.interp(stft_times, before_analysis.times_sec, before_analysis.content_entropy, left=1.0, right=1.0)
    roughness = np.clip((tone_error_stft - 1.2) / 4.0, 0.0, 1.0)
    entropy_gate = np.clip((entropy_stft - 0.62) / 0.25, 0.0, 1.0)
    safe_strength = float(np.clip(strength, 0.4, 2.2))
    alpha = (0.04 + safe_strength * (0.03 + 0.13 * voiced_stft + 0.08 * roughness + 0.05 * entropy_gate)).astype(np.float32)
    alpha = np.clip(alpha, 0.04, 0.38)[None, :]

    natural_log_mag = (1.0 - alpha) * log_mag + alpha * coarse_log_mag
    natural_spec = np.exp(natural_log_mag) * phase
    _, spectral_audio = istft(natural_spec, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    spectral_audio = spectral_audio[:length].astype(np.float32)
    if len(spectral_audio) < length:
        spectral_audio = np.pad(spectral_audio, (0, length - len(spectral_audio)))

    source_env = interpolate_frames(source_analysis.rms, source_analysis, length)
    cand_env = interpolate_frames(before_analysis.rms, before_analysis, length)
    env_gain = np.clip(source_env / (cand_env + EPS), 0.72, 1.32)
    envelope_matched = spectral_audio * env_gain
    shaped = envelope_matched * tone_energy_curve(source_analysis, length)

    voiced_samples = interpolate_frames(before_analysis.voiced, before_analysis, length)
    tone_error_samples = interpolate_frames(before_analysis.tone_error_st, before_analysis, length)
    blend = 0.05 + safe_strength * (0.03 + 0.16 * voiced_samples + 0.04 * np.clip(tone_error_samples / 4.0, 0.0, 1.0))
    blend = np.clip(blend, 0.06, 0.42)
    output = (1.0 - blend) * candidate + blend * shaped
    output = match_rms(candidate, output)
    output = peak_limit(output)

    after_analysis = analyze_frames(output, sample_rate)
    metadata = {
        "method": "ppg_inspired_content_bottleneck_with_mandarin_tone_naturalization",
        "limits": [
            "This is a lightweight PPG-inspired postfilter, not a trained neural PPG extractor.",
            "It preserves FreeVC anonymized timbre and only applies conservative spectral-envelope and tone-contour smoothing.",
        ],
        "source": summarize_analysis(source_analysis),
        "before": summarize_analysis(before_analysis),
        "after": summarize_analysis(after_analysis),
        "processing": {
            "spectral_bottleneck_alpha_mean": float(np.mean(alpha)),
            "spectral_bottleneck_alpha_max": float(np.max(alpha)),
            "sample_blend_mean": float(np.mean(blend)),
            "sample_blend_max": float(np.max(blend)),
            "strength": safe_strength,
        },
    }
    return output.astype(np.float32), metadata


def summarize_analysis(analysis: FrameAnalysis) -> dict[str, Any]:
    valid_f0 = analysis.f0_hz[analysis.f0_hz > 0.0]
    tone_counts = {label: 0 for label in TONE_LABELS}
    for segment in analysis.tone_segments:
        tone_counts[segment.label] += 1
    if len(valid_f0) >= 2:
        step_st = np.abs(np.diff(np.log2(valid_f0))) * 12.0
        p95_step = float(np.percentile(step_st, 95))
        jump_ratio = float(np.mean(step_st > 2.0))
    else:
        p95_step = 0.0
        jump_ratio = 0.0

    return {
        "frame_count": int(len(analysis.times_sec)),
        "voiced_ratio": float(np.mean(analysis.voiced >= 0.45)),
        "median_f0_hz": float(np.median(valid_f0)) if len(valid_f0) else None,
        "p95_f0_step_st": p95_step,
        "f0_jump_ratio": jump_ratio,
        "content_entropy_mean": float(np.mean(analysis.content_entropy)),
        "tone_error_st_mean": float(np.mean(analysis.tone_error_st)),
        "tone_counts": tone_counts,
        "tone_posterior_mean": {
            label: float(value)
            for label, value in zip(TONE_LABELS, np.mean(analysis.tone_posterior, axis=0))
        },
        "tone_segments": [asdict(segment) for segment in analysis.tone_segments],
    }


def naturalize_file(
    source_path: Path,
    candidate_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    strength: float = 1.0,
) -> dict[str, Any]:
    source_rate, source_audio = load_audio(source_path)
    candidate_rate, candidate_audio = load_audio(candidate_path)
    if source_rate != candidate_rate:
        raise ValueError(f"Sample-rate mismatch: {source_path} ({source_rate}) vs {candidate_path} ({candidate_rate})")

    output_audio, metadata = ppg_tone_naturalize_audio(source_audio, candidate_audio, source_rate, strength=strength)
    metadata.update(
        {
            "source_path": str(source_path),
            "candidate_path": str(candidate_path),
            "output_path": str(output_path),
            "sample_rate": source_rate,
        }
    )
    save_audio(output_path, output_audio, source_rate)
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
