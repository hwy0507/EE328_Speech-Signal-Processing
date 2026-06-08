from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile

from naturalness_postprocess import humanize_candidate
from vc_candidate_builder import analyze_levels, build_one_candidate, duration_seconds, safe_stem


EPS = 1e-8


@dataclass(frozen=True)
class TargetReference:
    name: str
    path: Path


@dataclass(frozen=True)
class CandidateScore:
    target_name: str
    variant_name: str
    target_reference: Path
    output_path: Path
    standard_similarity_score: float
    naturalness_proxy: float
    selection_score: float
    duration_ratio: float
    transform_penalty: float


@dataclass(frozen=True)
class PrivacyTargetSelection:
    selected_target: TargetReference
    selected_variant: str
    selected_output: Path
    candidates: list[CandidateScore]
    target_pool_size: int
    evaluated_target_count: int


def standard_similarity_from_cosine(cosine: float) -> float:
    """Match the project standard: score=(cosine+1)/2, range roughly [0, 1]."""

    return (float(cosine) + 1.0) / 2.0


def _load_pool_payload(config_path: Path) -> list[str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw = payload.get("reference_paths") or payload.get("targets") or payload.get("paths") or []
    else:
        raw = payload
    paths: list[str] = []
    for item in raw:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            value = item.get("path") or item.get("audio_path") or item.get("reference")
            if value:
                paths.append(str(value))
    return paths


def load_target_references(config_path: Path, fallback_paths: Iterable[Path] | None = None) -> list[TargetReference]:
    config_path = Path(config_path).expanduser().resolve()
    candidates: list[Path] = []
    if config_path.exists():
        for raw_path in _load_pool_payload(config_path):
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            candidates.append(path)
    if fallback_paths:
        candidates.extend(Path(p).expanduser().resolve() for p in fallback_paths)

    seen: set[Path] = set()
    refs: list[TargetReference] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        refs.append(TargetReference(name=safe_stem(resolved), path=resolved))
    if not refs:
        raise FileNotFoundError(f"No usable target references found from {config_path}")
    return refs


def _load_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, np.clip(audio, -0.999, 0.999).astype(np.float32)


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    segment = audio[: min(audio.size, sample_rate * 8)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size).astype(np.float32)))
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + EPS))


def _cosine_proxy(source_path: Path, output_path: Path) -> float:
    src_rate, src_audio = _load_mono(source_path)
    out_rate, out_audio = _load_mono(output_path)
    del out_rate
    src_centroid = _spectral_centroid(src_audio, src_rate)
    out_centroid = _spectral_centroid(out_audio, src_rate)
    centroid_gap = abs(math.log((out_centroid + 1.0) / (src_centroid + 1.0)))
    duration_gap = abs(duration_seconds(output_path) / max(duration_seconds(source_path), EPS) - 1.0)
    return float(np.clip(1.0 - 0.55 * centroid_gap - 0.25 * duration_gap, -1.0, 1.0))


def _score_candidate(source_path: Path, target: TargetReference, output_path: Path, variant_name: str) -> CandidateScore:
    cosine = _cosine_proxy(source_path, output_path)
    standard_score = standard_similarity_from_cosine(cosine) * 100.0
    duration_ratio = duration_seconds(output_path) / max(duration_seconds(source_path), EPS)
    peak_dbfs, rms_dbfs, clipping_ratio = analyze_levels(output_path)
    duration_match = float(np.clip(1.0 - abs(duration_ratio - 1.0) / 0.20, 0.0, 1.0))
    level_quality = float(np.clip((rms_dbfs + 42.0) / 24.0, 0.0, 1.0))
    clipping_quality = float(np.clip(1.0 - clipping_ratio * 50.0, 0.0, 1.0))
    peak_quality = float(np.clip(1.0 - max(peak_dbfs + 0.2, 0.0) / 6.0, 0.0, 1.0))
    naturalness = float(np.clip(0.45 * duration_match + 0.25 * level_quality + 0.20 * clipping_quality + 0.10 * peak_quality, 0.0, 1.0))
    transform_penalty = float(np.clip(abs(duration_ratio - 1.0) * 0.08 + clipping_ratio * 2.0, 0.0, 1.0))
    privacy = float(np.clip((70.0 - standard_score) / 35.0, 0.0, 1.0))
    selection_score = 0.56 * privacy + 0.44 * naturalness - transform_penalty
    return CandidateScore(
        target_name=target.name,
        variant_name=variant_name,
        target_reference=target.path,
        output_path=output_path,
        standard_similarity_score=standard_score,
        naturalness_proxy=naturalness,
        selection_score=selection_score,
        duration_ratio=duration_ratio,
        transform_penalty=transform_penalty,
    )


def optimize_privacy_target(
    source_path: Path,
    target_references: list[TargetReference],
    output_root: Path,
    *,
    max_targets: int = 9,
    vc_python: Path,
) -> PrivacyTargetSelection:
    source_path = Path(source_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    refs = target_references[: max(1, int(max_targets))]
    scores: list[CandidateScore] = []
    for target in refs:
        candidate_dir = output_root / safe_stem(target.path)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        raw_output = candidate_dir / f"{safe_stem(source_path)}_to_{safe_stem(target.path)}_freevc.wav"
        status_json = candidate_dir / "freevc_status.json"
        result = build_one_candidate(source_path, target.path, raw_output, status_json, Path(vc_python))
        if result is None or not raw_output.exists():
            continue
        humanized_output = candidate_dir / f"{safe_stem(source_path)}_to_{safe_stem(target.path)}_optimized.wav"
        humanize_candidate(source_path, raw_output, humanized_output)
        scores.append(_score_candidate(source_path, target, humanized_output, "optimized"))

    if not scores:
        raise RuntimeError("No FreeVC target candidates were generated; cannot run privacy target optimization.")

    scores.sort(key=lambda item: item.selection_score, reverse=True)
    best = scores[0]
    selected_target = next(ref for ref in refs if ref.name == best.target_name)
    summary_path = output_root / "privacy_target_optimizer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selected_target": selected_target.name,
                "selected_output": str(best.output_path),
                "evaluated_target_count": len(scores),
                "target_pool_size": len(target_references),
                "candidates": [score.__dict__ | {"target_reference": str(score.target_reference), "output_path": str(score.output_path)} for score in scores],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return PrivacyTargetSelection(
        selected_target=selected_target,
        selected_variant=best.variant_name,
        selected_output=best.output_path,
        candidates=scores,
        target_pool_size=len(target_references),
        evaluated_target_count=len(scores),
    )
