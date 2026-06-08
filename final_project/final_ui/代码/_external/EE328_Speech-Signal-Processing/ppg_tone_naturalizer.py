from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


EPS = 1e-8


def _load_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, np.clip(audio, -0.999, 0.999).astype(np.float32)


def _match_rate(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    return resample_poly(audio, target_rate, source_rate).astype(np.float32)


def _rms_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(int(sample_rate * 0.030), 1)
    hop = max(int(sample_rate * 0.010), 1)
    if audio.size <= frame:
        return np.full(audio.size, float(np.sqrt(np.mean(audio * audio) + EPS)), dtype=np.float32)
    starts = np.arange(0, audio.size - frame + 1, hop, dtype=np.int32)
    values = np.array([np.sqrt(np.mean(audio[s : s + frame] ** 2) + EPS) for s in starts], dtype=np.float32)
    centers = starts + frame // 2
    if centers[0] != 0:
        centers = np.concatenate([np.array([0], dtype=np.int32), centers])
        values = np.concatenate([values[:1], values])
    if centers[-1] != audio.size - 1:
        centers = np.concatenate([centers, np.array([audio.size - 1], dtype=np.int32)])
        values = np.concatenate([values, values[-1:]])
    return np.interp(np.arange(audio.size), centers, values).astype(np.float32)


def _peak_limit(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    current = float(np.max(np.abs(audio)) + EPS)
    if current > peak:
        audio = audio * (peak / current)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def naturalize_file(
    source_path: Path,
    candidate_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    *,
    strength: float = 0.4,
) -> Path:
    """Create the PPG-tone optimized variant used by ``recording_demo_ui.py``.

    This local implementation approximates a PPG/tone naturalization stage with
    source-guided energy contour transfer and tiny unvoiced-detail recovery.  It
    intentionally avoids heavy model dependencies so the fused desktop GUI can
    execute the same three-method flow reliably.
    """

    strength = float(np.clip(strength, 0.0, 1.0))
    src_rate, source = _load_mono(source_path)
    cand_rate, candidate = _load_mono(candidate_path)
    candidate = _match_rate(candidate, cand_rate, src_rate)

    length = min(source.size, candidate.size)
    source = source[:length]
    candidate = candidate[:length]

    src_env = _rms_envelope(source, src_rate)
    cand_env = _rms_envelope(candidate, src_rate)
    contour_gain = np.clip(src_env / (cand_env + EPS), 0.72, 1.32)
    contoured = candidate * ((1.0 - strength) + strength * contour_gain)

    # Add a very small amount of source unvoiced detail to make Mandarin finals /
    # consonants less muffled while keeping converted speaker identity dominant.
    smooth_len = max(int(src_rate * 0.010), 3)
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    source_detail = source - np.convolve(source, kernel, mode="same").astype(np.float32)
    output = contoured + 0.025 * strength * source_detail
    output = _peak_limit(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, src_rate, np.int16(np.round(output * 32767.0)))

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "method": "ppg_tone_naturalizer",
                    "source_path": str(source_path),
                    "candidate_path": str(candidate_path),
                    "output_path": str(output_path),
                    "sample_rate": src_rate,
                    "strength": strength,
                    "description": "Source-guided energy contour and light unvoiced-detail recovery.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return output_path
