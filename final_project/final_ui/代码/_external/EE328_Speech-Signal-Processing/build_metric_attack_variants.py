from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


EPS = 1e-8


@dataclass(frozen=True)
class AttackProfile:
    """Lightweight post-processing profile used by the recording demo.

    The name is kept compatible with ``recording_demo_ui.py``.  In the fused GUI
    this stage is not an adversarial attack; it is a metric-aware clarity guard:
    keep the FreeVC speaker change, but clean low-frequency rumble, gently recover
    consonant detail, and normalize level for fair listening/evaluation.
    """

    name: str
    description: str
    highpass_hz: float = 65.0
    presence_gain: float = 0.08
    detail_mix: float = 0.035
    target_peak_dbfs: float = -1.5


ATTACK_PROFILES: tuple[AttackProfile, ...] = (
    AttackProfile(
        name="clarity_guard",
        description="Metric+clarity guard: de-rumble, preserve speech detail, and peak-normalize.",
    ),
)


def _to_float_audio(data: np.ndarray) -> np.ndarray:
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def _peak_limit(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    target_peak = float(10 ** (peak_dbfs / 20.0))
    peak = float(np.max(np.abs(audio)) + EPS)
    if peak > target_peak:
        audio = audio * (target_peak / peak)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def _safe_highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if audio.size < 32 or cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return audio.astype(np.float32)
    b, a = butter(2, cutoff_hz / (sample_rate / 2.0), btype="highpass")
    return filtfilt(b, a, audio).astype(np.float32)


def apply_attack_profile(input_path: Path, output_path: Path, profile: AttackProfile) -> Path:
    """Apply the clarity profile and write a GUI-ready WAV file."""

    sample_rate, data = wavfile.read(input_path)
    audio = _to_float_audio(data)
    cleaned = _safe_highpass(audio, sample_rate, profile.highpass_hz)

    # Recover a small amount of consonant/high-frequency detail without making
    # the result sound synthetic.  This improves intelligibility while leaving the
    # speaker-converted timbre dominant.
    smooth_len = max(int(sample_rate * 0.012), 3)
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    low_band = np.convolve(cleaned, kernel, mode="same").astype(np.float32)
    detail = cleaned - low_band
    enhanced = cleaned + profile.presence_gain * low_band + profile.detail_mix * detail
    enhanced = _peak_limit(enhanced, profile.target_peak_dbfs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, np.int16(np.round(enhanced * 32767.0)))
    return output_path
