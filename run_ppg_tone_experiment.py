from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ppg_tone_naturalizer import naturalize_file


DEFAULT_VARIANT_MAP = {
    "balanced_phone_clean_male": "ppg_tone_male",
    "balanced_phone_clean_female": "ppg_tone_female",
}

AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PPG-inspired Mandarin tone naturalized anonymization variants "
            "from existing male/female FreeVC metric-attack selections."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root.",
    )
    parser.add_argument(
        "--base-root",
        default="work_metric_attack/final/recommended",
        help="Directory containing existing recommended selection folders.",
    )
    parser.add_argument(
        "--output-root",
        default="work_ppg_tone",
        help="Directory where PPG-tone outputs and metadata will be written.",
    )
    parser.add_argument(
        "--base-variants",
        nargs="+",
        default=list(DEFAULT_VARIANT_MAP.keys()),
        help="Existing recommended variants to use as anonymized bases.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="Postfilter strength. 0.4 is the tuned conservative default for preserving ASV/ASR attack metrics.",
    )
    return parser.parse_args()


def resolve_source_audio(project_root: Path, source_name: str) -> Path:
    denoised = project_root / "work_smooth_verify" / "denoised" / f"{source_name}.wav"
    if denoised.exists():
        return denoised.resolve()

    stem = source_name.removesuffix("_denoised")
    matches = [
        path
        for path in project_root.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve source audio for {source_name!r}: {matches}")
    return matches[0].resolve()


def output_variant_name(base_variant: str) -> str:
    if base_variant in DEFAULT_VARIANT_MAP:
        return DEFAULT_VARIANT_MAP[base_variant]
    return f"ppg_tone_{base_variant}"


def update_selection_item(
    item: dict[str, Any],
    base_variant: str,
    new_variant: str,
    candidate_path: Path,
    output_path: Path,
    metadata_path: Path,
    strength: float,
) -> dict[str, Any]:
    updated = copy.deepcopy(item)
    updated["variant"] = "ppg_tone_naturalizer"
    updated["base_variant"] = base_variant
    updated["selected_candidate"] = str(candidate_path)
    updated["final_output"] = str(output_path)
    updated["ppg_tone_metadata"] = str(metadata_path)
    updated["original_final_output"] = str(candidate_path)

    profile = updated.setdefault("profile", {})
    prior_postprocess = profile.get("postprocess")
    if prior_postprocess:
        profile["postprocess"] = f"{prior_postprocess}+ppg_tone_naturalizer"
    else:
        profile["postprocess"] = "ppg_tone_naturalizer"
    profile["ppg_tone_base_variant"] = base_variant
    profile["ppg_tone_output_variant"] = new_variant
    profile["ppg_tone_method"] = "PPG-inspired content bottleneck + Mandarin tone contour naturalization"
    profile["ppg_tone_strength"] = strength
    return updated


def build_variant(project_root: Path, base_root: Path, output_root: Path, base_variant: str, strength: float) -> dict[str, Any]:
    new_variant = output_variant_name(base_variant)
    selection_path = base_root / base_variant / f"{base_variant}_selections.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing base selection file: {selection_path}")

    items = json.loads(selection_path.read_text(encoding="utf-8"))
    output_dir = output_root / "final" / "recommended" / new_variant
    metadata_dir = output_root / "analysis" / new_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    new_items: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []

    for item in items:
        source_name = item["source_name"]
        source_path = resolve_source_audio(project_root, source_name)
        candidate_path = Path(item["final_output"]).expanduser().resolve()
        output_path = (output_dir / f"{source_name}_{new_variant}.wav").resolve()
        metadata_path = (metadata_dir / f"{source_name}.json").resolve()

        metadata = naturalize_file(source_path, candidate_path, output_path, metadata_path, strength=strength)
        new_items.append(
            update_selection_item(
                item=item,
                base_variant=base_variant,
                new_variant=new_variant,
                candidate_path=candidate_path,
                output_path=output_path,
                metadata_path=metadata_path,
                strength=strength,
            )
        )
        analysis_rows.append(
            {
                "source_name": source_name,
                "source_path": str(source_path),
                "base_candidate": str(candidate_path),
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "before": metadata["before"],
                "after": metadata["after"],
                "processing": metadata["processing"],
            }
        )

    new_selection_path = output_dir / f"{new_variant}_selections.json"
    new_selection_path.write_text(json.dumps(new_items, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "base_variant": base_variant,
        "output_variant": new_variant,
        "selection_path": str(new_selection_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "items": analysis_rows,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    base_root = (project_root / args.base_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "PPG-inspired content posterior bottleneck with Mandarin tone naturalization",
        "note": (
            "This branch uses a lightweight posterior-style spectral bottleneck and tone-contour "
            "postfilter on top of existing FreeVC anonymized outputs. It does not require a "
            "downloaded neural PPG model."
        ),
        "base_root": str(base_root),
        "output_root": str(output_root),
        "variants": [],
        "strength": args.strength,
    }

    for base_variant in args.base_variants:
        summary["variants"].append(build_variant(project_root, base_root, output_root, base_variant, args.strength))

    summary_path = output_root / "ppg_tone_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
