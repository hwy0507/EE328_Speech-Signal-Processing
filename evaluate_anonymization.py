from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, get_window, stft

EPS = 1e-8
FRAME_MS = 40
HOP_MS = 10
MIN_F0 = 70.0
MAX_F0 = 350.0


@dataclass(frozen=True)
class AudioStats:
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    median_f0_hz: float | None
    voiced_ratio: float
    p95_f0_step_st: float
    f0_jump_ratio: float


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, audio


def compute_audio_stats(sample_rate: int, audio: np.ndarray) -> AudioStats:
    peak = float(np.max(np.abs(audio)) + EPS)
    rms = float(np.sqrt(np.mean(np.square(audio)) + EPS))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    median_f0, voiced_ratio, p95_f0_step_st, f0_jump_ratio = estimate_pitch_stats(sample_rate, audio)
    return AudioStats(
        duration_sec=float(len(audio) / sample_rate),
        peak_dbfs=20.0 * math.log10(peak),
        rms_dbfs=20.0 * math.log10(rms),
        clipping_ratio=clipping_ratio,
        median_f0_hz=median_f0,
        voiced_ratio=voiced_ratio,
        p95_f0_step_st=p95_f0_step_st,
        f0_jump_ratio=f0_jump_ratio,
    )


def estimate_pitch_stats(sample_rate: int, audio: np.ndarray) -> tuple[float | None, float, float, float]:
    frame_length = int(sample_rate * FRAME_MS / 1000)
    hop_length = int(sample_rate * HOP_MS / 1000)
    if len(audio) < frame_length or frame_length <= 0 or hop_length <= 0:
        return None, 0.0, 0.0, 0.0

    min_lag = int(sample_rate / MAX_F0)
    max_lag = int(sample_rate / MIN_F0)
    window = get_window("hann", frame_length, fftbins=False)
    f0_values: list[float] = []
    voiced_frames = 0
    total_frames = 0

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start : start + frame_length]
        energy = float(np.mean(frame * frame))
        total_frames += 1
        if energy < 1e-4:
            continue
        frame = frame * window
        corr = correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        corr[:min_lag] = 0
        local_max_lag = min(max_lag, len(corr) - 1)
        if local_max_lag <= min_lag:
            continue
        segment = corr[min_lag:local_max_lag]
        if segment.size == 0:
            continue
        peak_idx = int(np.argmax(segment)) + min_lag
        denom = corr[0] + EPS
        normalized_peak = corr[peak_idx] / denom
        if normalized_peak < 0.28:
            continue
        voiced_frames += 1
        f0_values.append(sample_rate / peak_idx)

    voiced_ratio = voiced_frames / max(total_frames, 1)
    if not f0_values:
        return None, voiced_ratio, 0.0, 0.0

    f0_np = np.asarray(f0_values, dtype=np.float32)
    if len(f0_np) < 2:
        return float(np.median(f0_np)), voiced_ratio, 0.0, 0.0

    step_st = np.abs(np.diff(np.log2(f0_np))) * 12.0
    return float(np.median(f0_np)), voiced_ratio, float(np.percentile(step_st, 95)), float(np.mean(step_st > 2.0))


def align_audio(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(reference), len(candidate))
    return reference[:length], candidate[:length]


def log_spectral_distance(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    reference, candidate = align_audio(reference, candidate)
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=512, noverlap=384, padded=False, boundary=None)
    _, _, cand_spec = stft(candidate, fs=sample_rate, nperseg=512, noverlap=384, padded=False, boundary=None)
    min_frames = min(ref_spec.shape[1], cand_spec.shape[1])
    ref_mag = np.abs(ref_spec[:, :min_frames]) + EPS
    cand_mag = np.abs(cand_spec[:, :min_frames]) + EPS
    diff = 20 * np.log10(ref_mag) - 20 * np.log10(cand_mag)
    return float(np.mean(np.sqrt(np.mean(diff * diff, axis=0))))


def rms_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = max(int(sample_rate * FRAME_MS / 1000), 1)
    hop_length = max(int(sample_rate * HOP_MS / 1000), 1)
    if len(audio) < frame_length:
        return np.array([float(np.sqrt(np.mean(np.square(audio)) + EPS))], dtype=np.float32)
    values = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start : start + frame_length]
        values.append(float(np.sqrt(np.mean(np.square(frame)) + EPS)))
    return np.asarray(values, dtype=np.float32)


def envelope_correlation(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    ref_env = rms_envelope(reference, sample_rate)
    cand_env = rms_envelope(candidate, sample_rate)
    length = min(len(ref_env), len(cand_env))
    if length < 2:
        return 1.0
    ref_env = ref_env[:length]
    cand_env = cand_env[:length]
    ref_env = ref_env - np.mean(ref_env)
    cand_env = cand_env - np.mean(cand_env)
    denom = float(np.linalg.norm(ref_env) * np.linalg.norm(cand_env) + EPS)
    return float(np.dot(ref_env, cand_env) / denom)


def safe_plot_label(path: Path) -> str:
    label = path.stem.encode("ascii", "ignore").decode().strip("_")
    return label or path.parent.name.encode("ascii", "ignore").decode().strip("_") or "audio"


def plot_spectrogram_pair(reference_path: Path, candidate_path: Path, output_path: Path) -> None:
    ref_sr, reference = load_audio(reference_path)
    cand_sr, candidate = load_audio(candidate_path)
    if ref_sr != cand_sr:
        raise ValueError("Sample rates must match for comparison plots")

    _, _, ref_spec = stft(reference, fs=ref_sr, nperseg=512, noverlap=384, padded=False, boundary=None)
    _, _, cand_spec = stft(candidate, fs=cand_sr, nperseg=512, noverlap=384, padded=False, boundary=None)
    ref_db = 20 * np.log10(np.abs(ref_spec) + EPS)
    cand_db = 20 * np.log10(np.abs(cand_spec) + EPS)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].imshow(ref_db, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title(safe_plot_label(reference_path))
    axes[0].set_ylabel("Frequency bin")
    axes[1].imshow(cand_db, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title(safe_plot_label(candidate_path))
    axes[1].set_ylabel("Frequency bin")
    axes[1].set_xlabel("Frame")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def score_candidate(
    source_stats: AudioStats,
    candidate_stats: AudioStats,
    spectral_distance_db: float,
    envelope_corr: float,
) -> float:
    duration_penalty = abs(candidate_stats.duration_sec - source_stats.duration_sec) / max(source_stats.duration_sec, EPS)
    voiced_penalty = abs(candidate_stats.voiced_ratio - source_stats.voiced_ratio)
    rms_penalty = abs(candidate_stats.rms_dbfs - source_stats.rms_dbfs) / 12.0
    pitch_bonus = 0.0
    if source_stats.median_f0_hz and candidate_stats.median_f0_hz:
        pitch_bonus = min(abs(math.log2(candidate_stats.median_f0_hz / source_stats.median_f0_hz)) * 12.0 / 6.0, 1.5)
    clipping_penalty = candidate_stats.clipping_ratio * 20.0
    naturalness_bonus = max(envelope_corr, -1.0) * 1.5
    prosody_penalty = max(candidate_stats.p95_f0_step_st - 7.0, 0.0) * 0.45 + max(candidate_stats.f0_jump_ratio - 0.13, 0.0) * 14.0
    return spectral_distance_db + pitch_bonus + naturalness_bonus - 3.5 * duration_penalty - 2.5 * voiced_penalty - rms_penalty - clipping_penalty - prosody_penalty


def evaluate_candidate(source_path: Path, candidate_path: Path, plot_root: Path) -> dict[str, Any]:
    source_sr, source_audio = load_audio(source_path)
    candidate_sr, candidate_audio = load_audio(candidate_path)
    if source_sr != candidate_sr:
        raise ValueError(f"Sample rate mismatch: {source_path} vs {candidate_path}")

    source_stats = compute_audio_stats(source_sr, source_audio)
    candidate_stats = compute_audio_stats(candidate_sr, candidate_audio)
    spectral_distance_db = log_spectral_distance(source_audio, candidate_audio, source_sr)
    envelope_corr = envelope_correlation(source_audio, candidate_audio, source_sr)
    spectrogram_path = plot_root / source_path.stem / f"{candidate_path.stem}.png"
    plot_spectrogram_pair(source_path, candidate_path, spectrogram_path)
    score = score_candidate(source_stats, candidate_stats, spectral_distance_db, envelope_corr)

    return {
        "source_path": str(source_path),
        "candidate_path": str(candidate_path),
        "source_stats": source_stats.__dict__,
        "candidate_stats": candidate_stats.__dict__,
        "spectral_distance_db": spectral_distance_db,
        "envelope_corr": envelope_corr,
        "score": score,
        "spectrogram_plot": str(spectrogram_path),
    }


def evaluate_manifest(manifest_path: Path, plot_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError(f"Empty manifest: {manifest_path}")

    source_path_value = manifest[0].get("source_path")
    if source_path_value:
        source_path = Path(source_path_value)
    else:
        first_candidate = Path(manifest[0]["output_path"])
        source_name = first_candidate.parent.name
        source_file = source_name.replace("_denoised", "") + "_denoised.wav"
        source_path = Path(__file__).resolve().parent / "work" / "denoised" / source_file
    evaluations = []
    for item in manifest:
        candidate_path = Path(item["output_path"])
        result = evaluate_candidate(source_path, candidate_path, plot_root)
        result["profile"] = item["profile"]
        evaluations.append(result)

    evaluations.sort(key=lambda item: item["score"], reverse=True)
    return {
        "source_path": str(source_path),
        "best_candidate": evaluations[0],
        "all_candidates": evaluations,
    }


def evaluate_all(candidate_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plot_root = output_root / "plots"
    summary: dict[str, Any] = {}
    for manifest_path in sorted(candidate_root.glob("*/manifest.json")):
        result = evaluate_manifest(manifest_path, plot_root)
        source_stem = Path(result["source_path"]).stem
        summary[source_stem] = result
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anonymized candidate audio files.")
    parser.add_argument(
        "--candidate-root",
        default=str(Path(__file__).resolve().parent / "work" / "baseline_candidates"),
        help="Directory containing per-source candidate folders with manifest.json files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "work" / "evaluation"),
        help="Directory for evaluation JSON and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_all(
        candidate_root=Path(args.candidate_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
