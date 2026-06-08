from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from audio_preprocess import DENOISE_PRESETS, preprocess_many
from baseline_anonymizer import anonymize_many
from evaluate_anonymization import evaluate_all
from vc_candidate_builder import build_vc_candidates, load_target_pool_config

DEFAULT_INPUTS = (
    "test.wav",
    "绿色.m4a",
)
DEFAULT_VC_TARGET_CONFIGS = (
    "vc_target_pool_male.json",
)


def copy_backend_best(summary: dict, final_root: Path, backend_name: str) -> list[dict]:
    final_root.mkdir(parents=True, exist_ok=True)
    selections = []
    for source_name, result in summary.items():
        best = result["best_candidate"]
        candidate_path = Path(best["candidate_path"])
        destination = final_root / f"{source_name}_{backend_name}_best.wav"
        shutil.copy2(candidate_path, destination)
        record = {
            "source_name": source_name,
            "backend": backend_name,
            "selected_candidate": str(candidate_path),
            "final_output": str(destination),
            "score": best["score"],
            "profile": best["profile"],
        }
        selections.append(record)
    selections_path = final_root / f"{backend_name}_selections.json"
    selections_path.write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    return selections


def copy_variant_preferred_results(vc_summary: dict, final_root: Path, variant_name: str) -> list[dict]:
    final_root.mkdir(parents=True, exist_ok=True)
    selections = []
    for source_name, result in vc_summary.items():
        best = result["best_candidate"]
        candidate_path = Path(best["candidate_path"])
        destination = final_root / f"{source_name}_{variant_name}.wav"
        shutil.copy2(candidate_path, destination)
        selections.append(
            {
                "source_name": source_name,
                "variant": variant_name,
                "selected_candidate": str(candidate_path),
                "final_output": str(destination),
                "score": best["score"],
                "profile": best["profile"],
            }
        )
    selections_path = final_root / f"{variant_name}_selections.json"
    selections_path.write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    return selections


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the speech anonymization pipeline with baseline and VC backends.")
    parser.add_argument(
        "--project-root",
        default=str(project_root),
        help="Project directory containing the raw input audio files.",
    )
    parser.add_argument(
        "--work-root",
        default=str(project_root / "work"),
        help="Directory for all intermediate and final artifacts.",
    )
    parser.add_argument(
        "--vc-target-config",
        action="append",
        dest="vc_target_configs",
        help="JSON config describing one pooled VC target reference set. Repeat to generate multiple variants.",
    )
    parser.add_argument(
        "--denoise-preset",
        choices=sorted(DENOISE_PRESETS),
        default="standard",
        help="Preprocess denoise preset.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional explicit input files. Defaults to the two course-project files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    raw_vc_configs = args.vc_target_configs or list(DEFAULT_VC_TARGET_CONFIGS)
    vc_target_configs = [
        (project_root / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        for item in raw_vc_configs
    ]
    input_paths = [
        (project_root / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        for item in (args.inputs or DEFAULT_INPUTS)
    ]

    preprocess_results = preprocess_many(input_paths, work_root, denoise_preset=args.denoise_preset)
    denoised_inputs = [result.denoised_wav for result in preprocess_results]

    baseline_candidate_summary = anonymize_many(denoised_inputs, work_root / "baseline_candidates")
    baseline_evaluation_summary = evaluate_all(work_root / "baseline_candidates", work_root / "evaluation_baseline")
    baseline_selections = copy_backend_best(baseline_evaluation_summary, work_root / "final" / "baseline", "baseline")

    vc_variants: dict[str, dict] = {}
    for config_path in vc_target_configs:
        pool_spec = load_target_pool_config(config_path)
        variant_name = pool_spec.name
        candidate_root = work_root / "vc_candidates" / variant_name
        evaluation_root = work_root / "evaluation_vc" / variant_name
        final_root = work_root / "final" / variant_name

        vc_candidate_summary = build_vc_candidates(
            denoised_inputs,
            candidate_root,
            target_pool_config=config_path,
        )
        vc_evaluation_summary = evaluate_all(candidate_root, evaluation_root)
        vc_selections = copy_backend_best(vc_evaluation_summary, final_root, variant_name)
        preferred_selections = copy_variant_preferred_results(
            vc_evaluation_summary,
            work_root / "final" / "preferred_variants",
            variant_name,
        )

        vc_variants[variant_name] = {
            "config_path": str(config_path),
            "candidate_summary": vc_candidate_summary,
            "evaluation_summary": vc_evaluation_summary,
            "selections": vc_selections,
            "preferred_selections": preferred_selections,
        }

    report = {
        "inputs": [str(path) for path in input_paths],
        "vc_target_configs": [str(path) for path in vc_target_configs],
        "denoise_preset": args.denoise_preset,
        "preprocess_outputs": [
            {
                "source": str(result.source),
                "normalized_wav": str(result.normalized_wav),
                "denoised_wav": str(result.denoised_wav),
                "metadata_json": str(result.metadata_json),
            }
            for result in preprocess_results
        ],
        "baseline_candidate_summary": baseline_candidate_summary,
        "baseline_evaluation_summary": baseline_evaluation_summary,
        "baseline_selections": baseline_selections,
        "vc_variants": vc_variants,
    }
    report_path = work_root / "pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
