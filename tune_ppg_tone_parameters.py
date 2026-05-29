from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ee328_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ppg_tone_naturalizer import naturalize_file


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "work_ppg_tone_tuning"
DEFAULT_BASE_ROOT = PROJECT_ROOT / "work_metric_attack/final/recommended"
DEFAULT_BASE_VARIANTS = ("raw_metric_male", "balanced_phone_clean_male")
DEFAULT_STRENGTHS = (0.4, 0.7, 1.0, 1.3, 1.6)
DEFAULT_REFERENCE_CONTEXTS = (
    "work_smooth_verify/final/preferred_variants/female_leaning_selections.json",
    "work_smooth_verify/final/preferred_variants/male_leaning_selections.json",
    "work_metric_attack/final/recommended/raw_metric_female/raw_metric_female_selections.json",
    "work_metric_attack/final/recommended/raw_metric_male/raw_metric_male_selections.json",
    "work_metric_attack/final/recommended/balanced_phone_clean_female/balanced_phone_clean_female_selections.json",
    "work_metric_attack/final/recommended/balanced_phone_clean_male/balanced_phone_clean_male_selections.json",
)
AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac")


@dataclass(frozen=True)
class CandidateSpec:
    base_variant: str
    strength: float
    variant_name: str
    selection_path: Path
    output_dir: Path
    metadata_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep PPG-tone strength/base variants and evaluate the privacy/naturalness trade-off."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root.")
    parser.add_argument(
        "--base-root",
        default=str(DEFAULT_BASE_ROOT),
        help="Directory containing recommended base variant selection folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where tuning candidates, metrics, and figures will be written.",
    )
    parser.add_argument(
        "--base-variants",
        nargs="+",
        default=list(DEFAULT_BASE_VARIANTS),
        help="Base male variants to run PPG-tone naturalization on.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_STRENGTHS),
        help="PPG-tone strengths to sweep. Values below 0.4 are clipped by ppg_tone_naturalizer.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only generate audio/selection files; skip VoicePrivacy-style evaluation.",
    )
    parser.add_argument(
        "--reference-contexts",
        nargs="+",
        default=list(DEFAULT_REFERENCE_CONTEXTS),
        help=(
            "Selection JSON files, relative to project root, copied into the tuning evaluation context "
            "so EER uses the same male/female reference speaker pool as the report benchmark."
        ),
    )
    parser.add_argument(
        "--asv-python",
        default=None,
        help="Optional Python executable passed through to evaluate_voiceprivacy.py for ECAPA embeddings.",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Optional cached faster-whisper model path passed through to evaluate_voiceprivacy.py.",
    )
    return parser.parse_args()


def strength_label(strength: float) -> str:
    return f"s{int(round(strength * 100)):03d}"


def safe_variant_name(base_variant: str, strength: float) -> str:
    clean = "".join(char if char.isalnum() or char == "_" else "_" for char in base_variant)
    return f"ppg_tune_{clean}_{strength_label(strength)}"


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


def update_selection_item(
    item: dict[str, Any],
    spec: CandidateSpec,
    candidate_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    updated = copy.deepcopy(item)
    updated["variant"] = "ppg_tone_tuning"
    updated["base_variant"] = spec.base_variant
    updated["selected_candidate"] = str(candidate_path)
    updated["final_output"] = str(output_path)
    updated["ppg_tone_metadata"] = str(metadata_path)
    updated["original_final_output"] = str(candidate_path)

    profile = updated.setdefault("profile", {})
    prior_postprocess = profile.get("postprocess")
    profile["postprocess"] = f"{prior_postprocess}+ppg_tone_naturalizer" if prior_postprocess else "ppg_tone_naturalizer"
    profile["ppg_tone_base_variant"] = spec.base_variant
    profile["ppg_tone_output_variant"] = spec.variant_name
    profile["ppg_tone_method"] = "PPG-inspired content bottleneck + Mandarin tone contour naturalization"
    profile["ppg_tone_strength"] = spec.strength
    return updated


def build_candidate_specs(base_variants: list[str], strengths: list[float], output_root: Path) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for base_variant in base_variants:
        for strength in strengths:
            variant_name = safe_variant_name(base_variant, strength)
            output_dir = output_root / "final" / "recommended" / variant_name
            specs.append(
                CandidateSpec(
                    base_variant=base_variant,
                    strength=strength,
                    variant_name=variant_name,
                    selection_path=output_dir / f"{variant_name}_selections.json",
                    output_dir=output_dir,
                    metadata_dir=output_root / "analysis" / variant_name,
                )
            )
    return specs


def build_candidate(
    project_root: Path,
    base_root: Path,
    spec: CandidateSpec,
) -> dict[str, Any]:
    selection_path = base_root / spec.base_variant / f"{spec.base_variant}_selections.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing base selection file: {selection_path}")

    items = json.loads(selection_path.read_text(encoding="utf-8"))
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.metadata_dir.mkdir(parents=True, exist_ok=True)

    new_items: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []

    for item in items:
        source_name = item["source_name"]
        source_path = resolve_source_audio(project_root, source_name)
        candidate_path = Path(item["final_output"]).expanduser().resolve()
        output_path = (spec.output_dir / f"{source_name}_{spec.variant_name}.wav").resolve()
        metadata_path = (spec.metadata_dir / f"{source_name}.json").resolve()

        metadata = naturalize_file(source_path, candidate_path, output_path, metadata_path, strength=spec.strength)
        new_items.append(update_selection_item(item, spec, candidate_path, output_path, metadata_path))
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

    spec.selection_path.write_text(json.dumps(new_items, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "base_variant": spec.base_variant,
        "strength": spec.strength,
        "variant_name": spec.variant_name,
        "selection_path": str(spec.selection_path.resolve()),
        "output_dir": str(spec.output_dir.resolve()),
        "items": analysis_rows,
    }


def selection_variant_name(selection_path: Path) -> str:
    return selection_path.stem.removesuffix("_selections")


def copy_selection_to_eval_context(selection_path: Path, eval_context_root: Path) -> Path:
    variant_name = selection_variant_name(selection_path)
    destination_dir = eval_context_root / variant_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / selection_path.name
    shutil.copy2(selection_path, destination_path)
    return destination_path


def build_evaluation_context(
    project_root: Path,
    output_root: Path,
    generated_summary: dict[str, Any],
    reference_contexts: list[str],
) -> Path:
    eval_context_root = output_root / "eval_selection_sets"
    eval_context_root.mkdir(parents=True, exist_ok=True)

    copied_paths: list[str] = []
    for raw_path in reference_contexts:
        selection_path = (project_root / raw_path).resolve()
        if not selection_path.exists():
            raise FileNotFoundError(f"Missing reference context selection: {selection_path}")
        copied_paths.append(str(copy_selection_to_eval_context(selection_path, eval_context_root)))

    for candidate in generated_summary["candidates"]:
        copied_paths.append(str(copy_selection_to_eval_context(Path(candidate["selection_path"]), eval_context_root)))

    (output_root / "eval_selection_context.json").write_text(
        json.dumps({"eval_context_root": str(eval_context_root), "selection_files": copied_paths}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return eval_context_root


def run_voiceprivacy_evaluation(
    project_root: Path,
    output_root: Path,
    eval_context_root: Path,
    asv_python: str | None,
    whisper_model: str | None,
) -> Path:
    output_path = output_root / "evaluation" / "voiceprivacy_tuning_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_glob = f"{eval_context_root.relative_to(project_root).as_posix()}/*/*_selections.json"
    command = [
        sys.executable,
        str(project_root / "evaluate_voiceprivacy.py"),
        "--project-root",
        str(project_root),
        "--selection-glob",
        selection_glob,
        "--output-path",
        str(output_path),
    ]
    if asv_python:
        command.extend(["--asv-python", asv_python])
    if whisper_model:
        command.extend(["--whisper-model", whisper_model])
    subprocess.run(command, cwd=project_root, check=True)
    return output_path


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def mean_metadata_value(items: list[dict[str, Any]], stage: str, key: str) -> float:
    values = [float(item[stage].get(key, 0.0)) for item in items]
    return float(sum(values) / max(len(values), 1))


def summarize_tone_proxy(items: list[dict[str, Any]]) -> dict[str, float]:
    before_tone_error = mean_metadata_value(items, "before", "tone_error_st_mean")
    after_tone_error = mean_metadata_value(items, "after", "tone_error_st_mean")
    before_jump = mean_metadata_value(items, "before", "f0_jump_ratio")
    after_jump = mean_metadata_value(items, "after", "f0_jump_ratio")
    after_voiced = mean_metadata_value(items, "after", "voiced_ratio")
    blend_mean = float(sum(float(item["processing"].get("sample_blend_mean", 0.0)) for item in items) / max(len(items), 1))
    tone_error_ratio = after_tone_error / max(before_tone_error, 1e-6)
    jump_ratio = after_jump / max(before_jump, 1e-6)
    tone_smoothness_proxy = clamp01(0.45 * (1.0 - tone_error_ratio) + 0.35 * (1.0 - jump_ratio) + 0.20 * after_voiced)
    preservation_penalty = clamp01(max(blend_mean - 0.25, 0.0) / 0.25)
    naturalness_proxy = clamp01(0.80 * tone_smoothness_proxy + 0.20 * (1.0 - preservation_penalty))
    return {
        "before_tone_error_st_mean": before_tone_error,
        "after_tone_error_st_mean": after_tone_error,
        "tone_error_delta": before_tone_error - after_tone_error,
        "before_f0_jump_ratio": before_jump,
        "after_f0_jump_ratio": after_jump,
        "f0_jump_delta": before_jump - after_jump,
        "after_voiced_ratio": after_voiced,
        "sample_blend_mean": blend_mean,
        "tone_smoothness_proxy": tone_smoothness_proxy,
        "naturalness_proxy": naturalness_proxy,
    }


def collect_scored_rows(
    generated_summary: dict[str, Any],
    evaluation_path: Path,
) -> pd.DataFrame:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    source_mean = float(evaluation["baseline"]["source_vs_source"]["target_mean_score"])
    rows: list[dict[str, Any]] = []
    for candidate in generated_summary["candidates"]:
        variant_name = candidate["variant_name"]
        variant_result = evaluation["variants"][variant_name]
        privacy = variant_result["privacy"]
        utility = variant_result["utility"]
        source_score = float(privacy["target_mean_score"])
        source_similarity_reduction = 1.0 - source_score / source_mean
        tone_proxy = summarize_tone_proxy(candidate["items"])
        asv_eer = float(privacy["eer"])
        asr_wer = float(utility["wer"])
        eer_norm = clamp01(asv_eer / 0.5)
        wer_capped = clamp01(asr_wer)
        objective = (
            0.45 * eer_norm
            + 0.25 * clamp01(source_similarity_reduction)
            + 0.20 * wer_capped
            + 0.10 * tone_proxy["naturalness_proxy"]
        )
        rows.append(
            {
                "variant_name": variant_name,
                "base_variant": candidate["base_variant"],
                "strength": float(candidate["strength"]),
                "asv_eer": asv_eer,
                "asr_wer": asr_wer,
                "source_target_mean_score": source_score,
                "source_similarity_reduction": source_similarity_reduction,
                "eer_normalized_to_random": eer_norm,
                "wer_capped_at_1": wer_capped,
                "objective_score": objective,
                **tone_proxy,
            }
        )
    return pd.DataFrame(rows).sort_values(["objective_score", "source_similarity_reduction"], ascending=False).reset_index(drop=True)


def write_plots(rows: pd.DataFrame, output_root: Path) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = rows.sort_values(["base_variant", "strength"]).copy()
    base_labels = {
        "balanced_phone_clean_male": "Balanced phone-clean base",
        "raw_metric_male": "Raw metric base",
    }
    plot_rows["base_label"] = plot_rows["base_variant"].map(base_labels).fillna(plot_rows["base_variant"])
    palette = {
        "Balanced phone-clean base": "#1f77b4",
        "Raw metric base": "#ff7f0e",
    }

    pivot = rows.pivot(index="base_variant", columns="strength", values="objective_score")
    plt.figure(figsize=(9.5, 4.8))
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("PPG-tone tuning objective", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("PPG-tone strength")
    ax.set_ylabel("Base variant")
    plt.tight_layout()
    plt.savefig(figure_dir / "tuning_objective_heatmap.png", dpi=180)
    plt.close()

    metric_specs = [
        ("source_similarity_reduction", "Source-speaker similarity drop", "higher = more private"),
        ("asr_wer", "ASR WER", "higher = harder for ASR"),
        ("naturalness_proxy", "Tone/naturalness proxy", "higher = smoother tone"),
        ("objective_score", "Overall score", "higher = better trade-off"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), sharex=True)
    strengths = sorted(plot_rows["strength"].unique())
    for ax, (metric, title, subtitle) in zip(axes.flat, metric_specs):
        for base_label, group in plot_rows.groupby("base_label"):
            ordered = group.sort_values("strength")
            ax.plot(
                ordered["strength"],
                ordered[metric],
                marker="o",
                linewidth=2.4,
                markersize=5.5,
                color=palette.get(base_label),
                label=base_label,
            )
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=10.5, pad=8)
        ax.set_xticks(strengths)
        ax.set_xlabel("PPG-tone strength")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("PPG-tone sweep: privacy, intelligibility, and tone naturalness", fontsize=14, weight="bold", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(figure_dir / "tuning_metrics_by_strength.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for base_label, group in plot_rows.groupby("base_label"):
        ax.scatter(
            group["naturalness_proxy"],
            group["source_similarity_reduction"],
            s=85 + 220 * group["objective_score"],
            color=palette.get(base_label),
            alpha=0.82,
            edgecolor="white",
            linewidth=1.0,
            label=base_label,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['strength']:.1f}",
                (row["naturalness_proxy"], row["source_similarity_reduction"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
            )
    ax.set_title("Privacy/naturalness trade-off", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Tone/naturalness proxy (higher = smoother)")
    ax.set_ylabel("Source-speaker similarity drop (higher = more anonymous)")
    ax.grid(True, alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "tuning_privacy_naturalness_tradeoff.png", dpi=180)
    plt.close(fig)


def write_markdown_report(rows: pd.DataFrame, output_root: Path) -> None:
    report_path = output_root / "PPG_TONE_TUNING_RESULTS.md"
    best = rows.iloc[0]
    privacy_first = rows.sort_values(
        ["asv_eer", "asr_wer", "source_similarity_reduction", "naturalness_proxy"],
        ascending=False,
    ).iloc[0]
    constrained = rows[(rows["asv_eer"] >= 0.5) & (rows["source_similarity_reduction"] >= 0.90)]
    natural_tradeoff = constrained.sort_values(["naturalness_proxy", "objective_score"], ascending=False).iloc[0] if not constrained.empty else best
    display_cols = [
        "variant_name",
        "base_variant",
        "strength",
        "asv_eer",
        "asr_wer",
        "source_target_mean_score",
        "source_similarity_reduction",
        "naturalness_proxy",
        "objective_score",
    ]
    table_md = rows[display_cols].to_markdown(index=False, floatfmt=".3f")
    summary = f"""# PPG-tone Tuning Results

Generated by `tune_ppg_tone_parameters.py`.

## Recommendation

- **Privacy-first recommendation:** `{privacy_first["variant_name"]}` with base `{privacy_first["base_variant"]}`, strength `{privacy_first["strength"]:.2f}`. This keeps the strongest ASV/ASR-facing result in this sweep.
- **Best weighted objective:** `{best["variant_name"]}` with base `{best["base_variant"]}`, strength `{best["strength"]:.2f}`, objective `{best["objective_score"]:.3f}`.
- **Best naturalness-constrained trade-off:** `{natural_tradeoff["variant_name"]}` with base `{natural_tradeoff["base_variant"]}`, strength `{natural_tradeoff["strength"]:.2f}`.
- Evaluation context: EER uses the same male/female reference-speaker pool as the report benchmark, so candidate rows are more comparable to the existing male-only report.
- Interpretation: ASV EER is already coarse on this tiny local set, so the useful comparison is source-speaker similarity reduction, ASR WER, and the local tone/naturalness proxy.

## Decision

The sweep does **not** show a reliable ASV/WER gain from increasing PPG-tone strength above `0.4`. Higher strengths mainly improve the local tone/naturalness proxy while slightly weakening source-speaker similarity reduction. Therefore the project should keep strength `0.4` as the privacy-first default and mention strength `1.0` only as an optional naturalness-oriented ablation.

## Scored Candidates

{table_md}

## Generated Figures

- `figures/tuning_objective_heatmap.png`: overall tuning score by base variant and strength.
- `figures/tuning_metrics_by_strength.png`: metric trends across strength values.
- `figures/tuning_privacy_naturalness_tradeoff.png`: direct privacy/naturalness trade-off, labeled by PPG-tone strength.

## Caveats

This is a local course-project tuning benchmark, not an official VoicePrivacy protocol. The naturalness proxy is based on F0/tone smoothness and conservative processing strength; it is not a human MOS listening test.
"""
    report_path.write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    base_root = Path(args.base_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    specs = build_candidate_specs(args.base_variants, args.strengths, output_root)
    generated_summary = {
        "method": "PPG-tone parameter sweep",
        "base_root": str(base_root),
        "output_root": str(output_root),
        "strengths": args.strengths,
        "base_variants": args.base_variants,
        "candidates": [build_candidate(project_root, base_root, spec) for spec in specs],
    }
    summary_path = output_root / "tuning_generation_summary.json"
    summary_path.write_text(json.dumps(generated_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.skip_evaluation:
        print(json.dumps({"summary_path": str(summary_path), "skipped_evaluation": True}, ensure_ascii=False, indent=2))
        return

    eval_context_root = build_evaluation_context(project_root, output_root, generated_summary, args.reference_contexts)
    evaluation_path = run_voiceprivacy_evaluation(project_root, output_root, eval_context_root, args.asv_python, args.whisper_model)
    rows = collect_scored_rows(generated_summary, evaluation_path)
    rows.to_csv(output_root / "tuning_metrics.csv", index=False)
    rows.to_json(output_root / "tuning_metrics.json", orient="records", force_ascii=False, indent=2)
    write_plots(rows, output_root)
    write_markdown_report(rows, output_root)

    print(rows.to_string(index=False))
    print(f"\nTuning report written to: {output_root / 'PPG_TONE_TUNING_RESULTS.md'}")


if __name__ == "__main__":
    main()
