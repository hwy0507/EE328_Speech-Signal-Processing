from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from audio_preprocess import TARGET_SAMPLE_RATE
from naturalness_postprocess import humanize_candidate

DEFAULT_VC_PYTHON = Path(os.environ.get("VC_PYTHON", sys.executable))
DEFAULT_TARGET_POOL_CONFIG = Path(__file__).resolve().parent / "vc_target_pool.json"


@dataclass(frozen=True)
class VCCandidateResult:
    source_path: Path
    target_reference: Path
    output_path: Path
    backend: str
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    status_json: Path


@dataclass(frozen=True)
class TargetPoolSpec:
    name: str
    strategy: str
    reference_paths: tuple[Path, ...]
    clip_seconds: float
    silence_ms: int
    normalize_peak_dbfs: float


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def analyze_levels(path: Path) -> tuple[float, float, float]:
    sample_rate, data = wavfile.read(path)
    del sample_rate
    audio = data.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
    full_scale = float(np.iinfo(data.dtype).max)
    clipping_ratio = float(np.mean(np.abs(audio) >= full_scale))
    peak_dbfs = 20.0 * np.log10(max(peak / full_scale, 1e-8))
    rms_dbfs = 20.0 * np.log10(max(rms / full_scale, 1e-8))
    return peak_dbfs, rms_dbfs, clipping_ratio


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def safe_name(text: str) -> str:
    return text.replace(" ", "_")


def canonical_speaker_stem(path: Path) -> str:
    """Return a normalized stem used to avoid converting a speaker to itself.

    The GUI pipeline feeds denoised files such as ``s6_denoised.wav`` while the
    target pool contains references such as ``s6.wav``.  Comparing raw stems
    would miss that these are the same speaker/sample, so strip common pipeline
    suffixes before comparing.
    """
    stem = safe_stem(path).lower()
    for suffix in ("_denoised", "_normalized", "_norm"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def is_same_source_reference(source_path: Path, target_reference: Path) -> bool:
    return canonical_speaker_stem(source_path) == canonical_speaker_stem(target_reference)


def resample_output_to_match(source_path: Path, output_path: Path) -> None:
    target_sr, _ = wavfile.read(source_path)
    source_sr, data = wavfile.read(output_path)
    if source_sr == target_sr:
        return
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    resampled = resample_poly(audio, target_sr, source_sr, axis=0)
    resampled = np.clip(resampled, -0.999, 0.999)
    wavfile.write(output_path, target_sr, np.int16(np.round(resampled * 32767.0)))


def resolve_config_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def load_audio(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate).astype(np.float32)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    clipped = np.clip(audio, -0.999, 0.999)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.int16(np.round(clipped * 32767.0)))


def rms_level(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-8)


def spectral_centroid(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    if audio.size == 0:
        return 0.0
    windowed = audio.astype(np.float32)
    if windowed.size > sample_rate * 8:
        windowed = windowed[: sample_rate * 8]
    spectrum = np.abs(np.fft.rfft(windowed * np.hanning(windowed.size).astype(np.float32)))
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8))


def target_aware_dsp_fallback(source_path: Path, target_reference: Path, output_path: Path) -> None:
    """Create a deterministic fallback candidate when the optional FreeVC/TTS backend is unavailable.

    This is not a pretrained voice conversion model. It uses the target reference only to steer
    spectral tilt and loudness, so the full anonymization pipeline can still produce evaluable
    male-target anonymization variants in offline classroom environments.
    """
    source_audio = load_audio(source_path)
    target_audio = load_audio(target_reference)
    if source_audio.size == 0:
        raise ValueError(f"Empty source audio: {source_path}")

    source_centroid = spectral_centroid(source_audio)
    target_centroid = spectral_centroid(target_audio)
    centroid_ratio = target_centroid / max(source_centroid, 1.0)
    brightness = float(np.clip(math.log2(max(centroid_ratio, 1e-4)), -1.0, 1.0))

    smooth_window = max(int(TARGET_SAMPLE_RATE * 0.018), 3)
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    low_band = np.convolve(source_audio, kernel, mode="same").astype(np.float32)
    high_band = (source_audio - low_band).astype(np.float32)

    low_gain = 1.0 - 0.22 * brightness
    high_gain = 1.0 + 0.38 * brightness
    transformed = low_band * low_gain + high_band * high_gain

    # Add a very small deterministic modulation to reduce direct speaker similarity while
    # preserving intelligibility. The modulation is phase-stable and does not require randomness.
    timeline = np.arange(transformed.size, dtype=np.float32) / float(TARGET_SAMPLE_RATE)
    modulation = 1.0 + 0.018 * np.sin(2.0 * np.pi * (3.1 + abs(brightness) * 1.7) * timeline)
    transformed = transformed * modulation.astype(np.float32)

    target_rms = rms_level(target_audio)
    source_rms = rms_level(source_audio)
    desired_rms = float(np.clip(0.65 * source_rms + 0.35 * target_rms, 0.015, 0.22))
    transformed = transformed * (desired_rms / rms_level(transformed))
    peak = float(np.max(np.abs(transformed)) + 1e-8)
    if peak > 0.96:
        transformed = transformed * (0.96 / peak)
    save_audio(output_path, transformed.astype(np.float32), TARGET_SAMPLE_RATE)


def normalize_peak(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-8)
    target_peak = float(10 ** (peak_dbfs / 20.0))
    return np.clip(audio * (target_peak / peak), -0.999, 0.999).astype(np.float32)


def extract_active_segment(audio: np.ndarray, sample_rate: int, clip_seconds: float) -> np.ndarray:
    desired = max(int(sample_rate * clip_seconds), 1)
    if len(audio) <= desired:
        return audio.astype(np.float32)

    smooth = max(int(sample_rate * 0.03), 1)
    activity = np.convolve(np.abs(audio), np.ones(smooth, dtype=np.float32) / smooth, mode="same")
    window_scores = np.convolve(activity, np.ones(desired, dtype=np.float32), mode="valid")
    start = int(np.argmax(window_scores))
    return audio[start : start + desired].astype(np.float32)


def load_target_pool_config(config_path: Path) -> TargetPoolSpec:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    reference_paths = tuple(resolve_config_path(item, config_path.parent) for item in payload["reference_paths"])
    return TargetPoolSpec(
        name=payload.get("name", config_path.stem),
        strategy=payload.get("strategy", "montage_pool"),
        reference_paths=reference_paths,
        clip_seconds=float(payload.get("clip_seconds", 1.0)),
        silence_ms=int(payload.get("silence_ms", 80)),
        normalize_peak_dbfs=float(payload.get("normalize_peak_dbfs", -3.0)),
    )


def build_montage_pool(pool_spec: TargetPoolSpec, output_root: Path) -> Path:
    if pool_spec.strategy != "montage_pool":
        raise ValueError(f"Unsupported target pool strategy: {pool_spec.strategy}")

    silence_samples = max(int(TARGET_SAMPLE_RATE * pool_spec.silence_ms / 1000), 0)
    silence = np.zeros(silence_samples, dtype=np.float32)
    clips: list[np.ndarray] = []

    for reference_path in pool_spec.reference_paths:
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing target pool reference: {reference_path}")
        audio = load_audio(reference_path)
        clip = extract_active_segment(audio, TARGET_SAMPLE_RATE, pool_spec.clip_seconds)
        clip = normalize_peak(clip, pool_spec.normalize_peak_dbfs)
        if clip.size == 0:
            continue
        clips.append(clip)

    if not clips:
        raise ValueError("Target pool config did not yield any usable reference clips")

    montage_parts: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        montage_parts.append(clip)
        if silence_samples > 0 and index < len(clips) - 1:
            montage_parts.append(silence)
    pooled_audio = normalize_peak(np.concatenate(montage_parts), pool_spec.normalize_peak_dbfs)

    pooled_target_path = output_root / f"{safe_name(pool_spec.name)}.wav"
    save_audio(pooled_target_path, pooled_audio)
    return pooled_target_path


def build_one_candidate(
    source_path: Path,
    target_reference: Path,
    output_path: Path,
    status_json: Path,
    python_executable: Path,
) -> VCCandidateResult | None:
    vc_script = Path(__file__).resolve().parent / "vc_anonymizer.py"
    command = [
        str(python_executable),
        str(vc_script),
        "--source",
        str(source_path),
        "--target",
        str(target_reference),
        "--out",
        str(output_path),
        "--status-json",
        str(status_json),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    backend = "freevc"
    if process.returncode != 0:
        backend = "freevc_fallback"
        target_aware_dsp_fallback(source_path, target_reference, output_path)
        fallback_status = {
            "source": str(source_path),
            "target": str(target_reference),
            "output": str(output_path),
            "backend_available": False,
            "fallback_used": True,
            "fallback_backend": backend,
            "message": "FreeVC/TTS backend failed or is unavailable; generated target-aware DSP fallback candidate.",
            "vc_returncode": process.returncode,
            "vc_stdout": process.stdout,
            "vc_stderr": process.stderr,
        }
        status_json.parent.mkdir(parents=True, exist_ok=True)
        status_json.write_text(json.dumps(fallback_status, ensure_ascii=False, indent=2), encoding="utf-8")
    resample_output_to_match(source_path, output_path)
    humanize_candidate(source_path, output_path)
    peak_dbfs, rms_dbfs, clipping_ratio = analyze_levels(output_path)
    return VCCandidateResult(
        source_path=source_path,
        target_reference=target_reference,
        output_path=output_path,
        backend=backend,
        duration_sec=duration_seconds(output_path),
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        status_json=status_json,
    )


def build_candidate_record(result: VCCandidateResult, profile_name: str, pool_spec: TargetPoolSpec) -> dict:
    return {
        "source_path": str(result.source_path),
        "target_reference": str(result.target_reference),
        "output_path": str(result.output_path),
        "duration_sec": result.duration_sec,
        "peak_dbfs": result.peak_dbfs,
        "rms_dbfs": result.rms_dbfs,
        "clipping_ratio": result.clipping_ratio,
        "status_json": str(result.status_json),
        "profile": {
            "backend": result.backend,
            "name": profile_name,
            "target_reference": str(result.target_reference),
            "target_strategy": pool_spec.strategy,
            "target_pool_name": pool_spec.name,
            "target_reference_count": len(pool_spec.reference_paths),
            "target_reference_paths": [str(path) for path in pool_spec.reference_paths],
            "postprocess": "humanize_candidate",
        },
    }


def build_vc_candidates(
    source_wavs: list[Path],
    output_root: Path,
    python_executable: Path = DEFAULT_VC_PYTHON,
    target_pool_config: Path = DEFAULT_TARGET_POOL_CONFIG,
) -> dict[str, list[dict]]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}

    pool_spec = load_target_pool_config(target_pool_config)
    pooled_target: Path | None = None
    if pool_spec.strategy == "montage_pool":
        pooled_target = build_montage_pool(pool_spec, output_root.parent / "target_pools")

    for source_path in source_wavs:
        source_dir = output_root / safe_stem(source_path)
        source_dir.mkdir(parents=True, exist_ok=True)
        items: list[dict] = []

        if pool_spec.strategy == "montage_pool":
            assert pooled_target is not None
            profile_name = f"freevc_{safe_name(pool_spec.name)}"
            output_path = source_dir / f"{safe_stem(source_path)}_{profile_name}.wav"
            status_json = source_dir / f"{safe_stem(source_path)}_{profile_name}.json"
            result = build_one_candidate(
                source_path=source_path,
                target_reference=pooled_target,
                output_path=output_path,
                status_json=status_json,
                python_executable=python_executable,
            )
            if result is not None:
                items.append(build_candidate_record(result, profile_name, pool_spec))
        elif pool_spec.strategy == "single_ref_group":
            for target_reference in pool_spec.reference_paths:
                if is_same_source_reference(source_path, target_reference):
                    continue
                profile_name = f"freevc_{safe_name(pool_spec.name)}_{safe_stem(target_reference)}"
                output_path = source_dir / f"{safe_stem(source_path)}_{profile_name}.wav"
                status_json = source_dir / f"{safe_stem(source_path)}_{profile_name}.json"
                result = build_one_candidate(
                    source_path=source_path,
                    target_reference=target_reference,
                    output_path=output_path,
                    status_json=status_json,
                    python_executable=python_executable,
                )
                if result is not None:
                    items.append(build_candidate_record(result, profile_name, pool_spec))
            if not items:
                raise ValueError(
                    f"Target pool '{pool_spec.name}' has no usable references after excluding the input speaker "
                    f"for source {source_path}. Add more non-self reference wavs to {target_pool_config}."
                )
        else:
            raise ValueError(f"Unsupported target pool strategy: {pool_spec.strategy}")

        manifest_path = source_dir / "manifest.json"
        manifest_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[str(source_path)] = items
    return summary
