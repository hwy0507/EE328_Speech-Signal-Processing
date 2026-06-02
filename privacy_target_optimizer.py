from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceprivacy import (
    DEFAULT_ASV_PYTHON,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_SPEAKER_SAVEDIR_ROOT,
    cosine_score,
    run_asv_embeddings,
)
from vc_candidate_builder import DEFAULT_VC_PYTHON, analyze_levels, build_one_candidate, duration_seconds


@dataclass(frozen=True)
class TargetReference:
    name: str
    path: Path
    origin: str


@dataclass(frozen=True)
class ProsodyVariant:
    name: str
    ffmpeg_filter: str | None
    transform_penalty: float
    description: str


@dataclass(frozen=True)
class CandidateScore:
    target_name: str
    target_reference: str
    variant_name: str
    output_path: str
    source_cosine_similarity: float
    standard_similarity: float
    standard_similarity_score: float
    duration_ratio: float
    clipping_ratio: float
    transform_penalty: float
    naturalness_proxy: float
    selection_score: float


@dataclass(frozen=True)
class PrivacyOptimizedSelection:
    selected_output: Path
    selected_target: TargetReference
    selected_variant: str
    candidates: list[CandidateScore]
    target_pool_size: int
    evaluated_target_count: int


PROSODY_VARIANTS = (
    ProsodyVariant(
        name="plain",
        ffmpeg_filter=None,
        transform_penalty=0.0,
        description="FreeVC output without extra prosody perturbation.",
    ),
    ProsodyVariant(
        name="tempo_096",
        ffmpeg_filter="atempo=0.96",
        transform_penalty=0.020,
        description="Slightly slower tempo while preserving pitch.",
    ),
    ProsodyVariant(
        name="tempo_104",
        ffmpeg_filter="atempo=1.04",
        transform_penalty=0.020,
        description="Slightly faster tempo while preserving pitch.",
    ),
    ProsodyVariant(
        name="pitch_096",
        ffmpeg_filter="asetrate=15360,aresample=16000,atempo=1.0416667",
        transform_penalty=0.040,
        description="Conservative pitch-down perturbation with duration compensation.",
    ),
    ProsodyVariant(
        name="pitch_104",
        ffmpeg_filter="asetrate=16640,aresample=16000,atempo=0.9615385",
        transform_penalty=0.040,
        description="Conservative pitch-up perturbation with duration compensation.",
    ),
)


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "target"


def standard_similarity_from_cosine(cosine: float) -> float:
    """Map cosine similarity to the 0-1 standard score described in standard.docx."""
    return float((cosine + 1.0) / 2.0)


def resolve_config_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_target_references(config_path: Path, fallback_paths: list[Path] | None = None) -> list[TargetReference]:
    config_path = config_path.expanduser().resolve()
    targets: list[TargetReference] = []
    seen: set[str] = set()
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        for raw_path in payload.get("reference_paths", []):
            path = resolve_config_path(raw_path, config_path.parent)
            if path.exists() and str(path) not in seen:
                targets.append(TargetReference(name=safe_name(path.stem), path=path, origin=payload.get("name", config_path.stem)))
                seen.add(str(path))

    for path in fallback_paths or []:
        resolved = path.expanduser().resolve()
        if resolved.exists() and str(resolved) not in seen:
            targets.append(TargetReference(name=safe_name(resolved.stem), path=resolved, origin="fallback"))
            seen.add(str(resolved))

    if not targets:
        raise FileNotFoundError(
            f"No usable target references found in {config_path}. "
            "Run prepare_external_male_targets.py or provide a fallback target."
        )
    return targets


def apply_audio_filter(input_path: Path, output_path: Path, ffmpeg_filter: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-af",
        f"{ffmpeg_filter},alimiter=limit=0.97",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def duration_ratio(path: Path, source_duration: float) -> float:
    return float(duration_seconds(path) / max(source_duration, 1e-6))


def quality_proxy(path: Path, source_duration: float, variant: ProsodyVariant) -> tuple[float, float, float]:
    ratio = duration_ratio(path, source_duration)
    _, _, clipping = analyze_levels(path)
    duration_penalty = min(abs(ratio - 1.0) / 0.18, 1.0)
    clipping_penalty = min(clipping * 1000.0, 1.0)
    proxy = 1.0 - 0.45 * duration_penalty - 0.35 * variant.transform_penalty - 0.20 * clipping_penalty
    return max(0.0, min(proxy, 1.0)), ratio, float(clipping)


def preselect_targets_by_embedding(
    source_path: Path,
    targets: list[TargetReference],
    max_targets: int,
    asv_python: Path = DEFAULT_ASV_PYTHON,
    speaker_model: Path = DEFAULT_SPEAKER_MODEL,
    speaker_savedir_root: Path = DEFAULT_SPEAKER_SAVEDIR_ROOT,
) -> list[TargetReference]:
    if len(targets) <= max_targets:
        return targets
    paths = [source_path, *[target.path for target in targets]]
    embeddings = run_asv_embeddings(paths, asv_python, speaker_model, speaker_savedir_root)
    source_embedding = embeddings[str(source_path)]
    ranked = sorted(
        targets,
        key=lambda target: cosine_score(source_embedding, embeddings[str(target.path)]),
    )
    return ranked[:max_targets]


def optimize_privacy_target(
    source_path: Path,
    target_references: list[TargetReference],
    output_root: Path,
    *,
    max_targets: int = 5,
    vc_python: Path = DEFAULT_VC_PYTHON,
    asv_python: Path = DEFAULT_ASV_PYTHON,
    speaker_model: Path = DEFAULT_SPEAKER_MODEL,
    speaker_savedir_root: Path = DEFAULT_SPEAKER_SAVEDIR_ROOT,
) -> PrivacyOptimizedSelection:
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = source_path.expanduser().resolve()
    evaluated_targets = preselect_targets_by_embedding(
        source_path,
        target_references,
        max_targets=max_targets,
        asv_python=asv_python,
        speaker_model=speaker_model,
        speaker_savedir_root=speaker_savedir_root,
    )

    source_duration = duration_seconds(source_path)
    candidate_outputs: list[tuple[TargetReference, ProsodyVariant, Path]] = []
    raw_root = output_root / "raw_freevc_pool"
    variant_root = output_root / "prosody_variants"

    for target in evaluated_targets:
        stem = safe_name(target.name)
        raw_output = raw_root / f"{safe_name(source_path.stem)}_{stem}_freevc.wav"
        status_json = raw_root / f"{safe_name(source_path.stem)}_{stem}_freevc.json"
        result = build_one_candidate(
            source_path=source_path,
            target_reference=target.path,
            output_path=raw_output,
            status_json=status_json,
            python_executable=vc_python,
        )
        if result is None:
            continue

        for variant in PROSODY_VARIANTS:
            if variant.ffmpeg_filter is None:
                candidate_outputs.append((target, variant, raw_output))
                continue
            variant_output = variant_root / f"{safe_name(source_path.stem)}_{stem}_{variant.name}.wav"
            apply_audio_filter(raw_output, variant_output, variant.ffmpeg_filter)
            candidate_outputs.append((target, variant, variant_output))

    if not candidate_outputs:
        raise RuntimeError("No FreeVC candidates could be generated for the target pool.")

    unique_paths: dict[str, Path] = {str(source_path): source_path}
    for _, _, path in candidate_outputs:
        unique_paths[str(path)] = path
    embeddings = run_asv_embeddings(
        list(unique_paths.values()),
        asv_python=asv_python,
        speaker_model=speaker_model,
        speaker_savedir_root=speaker_savedir_root,
    )
    source_embedding = embeddings[str(source_path)]

    scores: list[CandidateScore] = []
    for target, variant, path in candidate_outputs:
        cosine = cosine_score(source_embedding, embeddings[str(path)])
        standard_similarity = standard_similarity_from_cosine(cosine)
        proxy, ratio, clipping = quality_proxy(path, source_duration, variant)
        duration_penalty = min(abs(ratio - 1.0) / 0.18, 1.0)
        selection_score = (
            standard_similarity
            + 0.10 * variant.transform_penalty
            + 0.06 * duration_penalty
            + 0.05 * (1.0 - proxy)
            + 0.05 * min(clipping * 1000.0, 1.0)
        )
        if not math.isfinite(selection_score):
            continue
        scores.append(
            CandidateScore(
                target_name=target.name,
                target_reference=str(target.path),
                variant_name=variant.name,
                output_path=str(path),
                source_cosine_similarity=cosine,
                standard_similarity=standard_similarity,
                standard_similarity_score=standard_similarity * 100.0,
                duration_ratio=ratio,
                clipping_ratio=clipping,
                transform_penalty=variant.transform_penalty,
                naturalness_proxy=proxy,
                selection_score=float(selection_score),
            )
        )

    if not scores:
        raise RuntimeError("No scorable privacy candidates were generated.")
    selected_score = min(scores, key=lambda score: score.selection_score)
    selected_target = next(target for target in evaluated_targets if target.name == selected_score.target_name)
    selection = PrivacyOptimizedSelection(
        selected_output=Path(selected_score.output_path),
        selected_target=selected_target,
        selected_variant=selected_score.variant_name,
        candidates=sorted(scores, key=lambda score: score.selection_score),
        target_pool_size=len(target_references),
        evaluated_target_count=len(evaluated_targets),
    )
    (output_root / "target_selection.json").write_text(
        json.dumps(
            {
                "selected": asdict(selected_score),
                "target_pool_size": selection.target_pool_size,
                "evaluated_target_count": selection.evaluated_target_count,
                "candidates": [asdict(score) for score in selection.candidates],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selection
