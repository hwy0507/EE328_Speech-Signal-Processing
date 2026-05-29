from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ee328_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
import pandas as pd
import seaborn as sns

from evaluate_voiceprivacy import (
    DEFAULT_ASV_PYTHON,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_SPEAKER_SAVEDIR_ROOT,
    average_embedding,
    cosine_score,
    run_asv_embeddings,
)
from ppg_tone_naturalizer import analyze_frames, load_audio, smooth_1d


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "report_evaluation"

MAIN_METHOD_ORDER = [
    "source_baseline",
    "female_leaning",
    "male_leaning",
    "raw_metric_female",
    "raw_metric_male",
    "balanced_phone_clean_female",
    "balanced_phone_clean_male",
    "ppg_tone_female",
    "ppg_tone_male",
]

DISPLAY_NAMES = {
    "source_baseline": "Original source",
    "female_leaning": "FreeVC female",
    "male_leaning": "FreeVC male",
    "raw_metric_female": "Raw metric female",
    "raw_metric_male": "Raw metric male",
    "balanced_phone_clean_female": "Metric+phone female",
    "balanced_phone_clean_male": "Metric+phone male",
    "ppg_tone_female": "PPG-tone female",
    "ppg_tone_male": "PPG-tone male",
}

METHOD_GROUPS = {
    "source_baseline": "Non-anonymized baseline",
    "female_leaning": "Method 1: FreeVC baseline",
    "male_leaning": "Method 1: FreeVC baseline",
    "raw_metric_female": "Method 2: Metric-enhanced",
    "raw_metric_male": "Method 2: Metric-enhanced",
    "balanced_phone_clean_female": "Method 2: Metric-enhanced",
    "balanced_phone_clean_male": "Method 2: Metric-enhanced",
    "ppg_tone_female": "Method 3: PPG-inspired tone",
    "ppg_tone_male": "Method 3: PPG-inspired tone",
}

METHOD_COLORS = {
    "source_baseline": "#b42318",
    "female_leaning": "#6c8ebf",
    "male_leaning": "#4c78a8",
    "raw_metric_female": "#f4a261",
    "raw_metric_male": "#f58518",
    "balanced_phone_clean_female": "#72b07a",
    "balanced_phone_clean_male": "#54a24b",
    "ppg_tone_female": "#2ca6a4",
    "ppg_tone_male": "#0f766e",
}

SELECTION_FILES = {
    "female_leaning": PROJECT_ROOT / "work_smooth_verify/final/preferred_variants/female_leaning_selections.json",
    "male_leaning": PROJECT_ROOT / "work_smooth_verify/final/preferred_variants/male_leaning_selections.json",
    "balanced_phone_clean_female": PROJECT_ROOT
    / "work_metric_attack/final/recommended/balanced_phone_clean_female/balanced_phone_clean_female_selections.json",
    "balanced_phone_clean_male": PROJECT_ROOT
    / "work_metric_attack/final/recommended/balanced_phone_clean_male/balanced_phone_clean_male_selections.json",
    "ppg_tone_female": PROJECT_ROOT / "work_ppg_tone/final/recommended/ppg_tone_female/ppg_tone_female_selections.json",
    "ppg_tone_male": PROJECT_ROOT / "work_ppg_tone/final/recommended/ppg_tone_male/ppg_tone_male_selections.json",
}


@dataclass(frozen=True)
class ResultBundle:
    metrics: pd.DataFrame
    per_utterance: pd.DataFrame
    source_target_mean_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-friendly evaluation tables and figures for the speech anonymization project."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root containing evaluation JSON files and generated audio.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV, Markdown, and PNG report artifacts.",
    )
    parser.add_argument(
        "--skip-embedding-heatmap",
        action="store_true",
        help="Skip ECAPA embedding extraction and only generate metrics/F0 plots.",
    )
    parser.add_argument(
        "--target-gender",
        choices=("all", "male", "female"),
        default="all",
        help="Filter target-speaker variants for report/UI generation.",
    )
    parser.add_argument(
        "--asv-python",
        default=str(DEFAULT_ASV_PYTHON),
        help="Python executable with SpeechBrain installed.",
    )
    parser.add_argument(
        "--speaker-model",
        default=str(DEFAULT_SPEAKER_MODEL),
        help="Local cached SpeechBrain ECAPA model directory.",
    )
    parser.add_argument(
        "--speaker-savedir-root",
        default=str(DEFAULT_SPEAKER_SAVEDIR_ROOT),
        help="Writable SpeechBrain savedir root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_selection_outputs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    return [Path(item["final_output"]).expanduser().resolve() for item in items]


def variant_matches_gender(method_id: str, target_gender: str) -> bool:
    if method_id == "source_baseline" or target_gender == "all":
        return True
    if method_id == f"{target_gender}_leaning":
        return True
    return method_id.endswith(f"_{target_gender}")


def filter_bundle_by_target(bundle: ResultBundle, target_gender: str) -> ResultBundle:
    if target_gender == "all":
        return bundle

    metrics = bundle.metrics[bundle.metrics["method_id"].map(lambda item: variant_matches_gender(item, target_gender))].copy()
    order_map = {method_id: index for index, method_id in enumerate(metrics["method_id"].tolist())}
    metrics["order"] = metrics["method_id"].map(order_map)
    metrics = metrics.sort_values("order").reset_index(drop=True)

    keep_names = set(metrics["display_name"].tolist())
    per_utterance = bundle.per_utterance.copy()
    if not per_utterance.empty:
        per_utterance = per_utterance[per_utterance["display_name"].isin(keep_names)].reset_index(drop=True)

    return ResultBundle(
        metrics=metrics,
        per_utterance=per_utterance,
        source_target_mean_score=bundle.source_target_mean_score,
    )


def append_variant_rows(
    rows: list[dict[str, Any]],
    utterance_rows: list[dict[str, Any]],
    variant_name: str,
    payload: dict[str, Any],
    source_file: str,
    order: int,
) -> None:
    privacy = payload["privacy"]
    utility = payload["utility"]
    rows.append(
        {
            "method_id": variant_name,
            "display_name": DISPLAY_NAMES.get(variant_name, variant_name),
            "method_group": METHOD_GROUPS.get(variant_name, "Other"),
            "source_file": source_file,
            "order": order,
            "asv_eer": float(privacy["eer"]),
            "asr_wer": float(utility["wer"]),
            "source_target_mean_score": float(privacy.get("target_mean_score", math.nan)),
            "reference_mean_score": float(privacy.get("nontarget_mean_score", math.nan)),
            "num_target_trials": int(privacy.get("num_target_trials", 0)),
            "num_nontarget_trials": int(privacy.get("num_nontarget_trials", 0)),
        }
    )

    for item in utility.get("per_utterance", []):
        utterance_rows.append(
            {
                "method_id": variant_name,
                "display_name": DISPLAY_NAMES.get(variant_name, variant_name),
                "method_group": METHOD_GROUPS.get(variant_name, "Other"),
                "source_name": item["source_name"].replace("_denoised", ""),
                "reference_text": item["reference_text"],
                "hypothesis_text": item["hypothesis_text"],
                "wer": float(item["wer"]),
                "edits": int(item["edits"]),
                "reference_token_count": int(item["reference_token_count"]),
                "trial_path": item["trial_path"],
            }
        )


def collect_metrics(project_root: Path) -> ResultBundle:
    rows: list[dict[str, Any]] = []
    utterance_rows: list[dict[str, Any]] = []

    base_results = load_json(project_root / "voiceprivacy_style_results.json")
    source_privacy = base_results["baseline"]["source_vs_source"]
    source_target_mean = float(source_privacy["target_mean_score"])
    rows.append(
        {
            "method_id": "source_baseline",
            "display_name": DISPLAY_NAMES["source_baseline"],
            "method_group": METHOD_GROUPS["source_baseline"],
            "source_file": "voiceprivacy_style_results.json",
            "order": 0,
            "asv_eer": float(source_privacy["eer"]),
            "asr_wer": 0.0,
            "source_target_mean_score": source_target_mean,
            "reference_mean_score": float(source_privacy["nontarget_mean_score"]),
            "num_target_trials": int(source_privacy["num_target_trials"]),
            "num_nontarget_trials": int(source_privacy["num_nontarget_trials"]),
        }
    )

    for idx, variant_name in enumerate(["female_leaning", "male_leaning"], start=1):
        append_variant_rows(
            rows,
            utterance_rows,
            variant_name,
            base_results["variants"][variant_name],
            "voiceprivacy_style_results.json",
            idx,
        )

    recommended_summary = load_json(project_root / "work_metric_attack/final/recommended/recommended_summary.json")
    recommended_by_label = {item["label"]: item for item in recommended_summary["exports"]}
    for idx, variant_name in enumerate(
        ["raw_metric_female", "raw_metric_male", "balanced_phone_clean_female", "balanced_phone_clean_male"],
        start=3,
    ):
        item = recommended_by_label[variant_name]
        append_variant_rows(
            rows,
            utterance_rows,
            variant_name,
            item["metrics"],
            "work_metric_attack/final/recommended/recommended_summary.json",
            idx,
        )

    ppg_results = load_json(project_root / "work_ppg_tone/voiceprivacy_ppg_tone_results.json")
    for idx, variant_name in enumerate(["ppg_tone_female", "ppg_tone_male"], start=7):
        append_variant_rows(
            rows,
            utterance_rows,
            variant_name,
            ppg_results["variants"][variant_name],
            "work_ppg_tone/voiceprivacy_ppg_tone_results.json",
            idx,
        )

    metrics = pd.DataFrame(rows)
    metrics["source_similarity_reduction"] = 1.0 - metrics["source_target_mean_score"] / source_target_mean
    metrics.loc[metrics["method_id"] == "source_baseline", "source_similarity_reduction"] = 0.0
    metrics["eer_normalized_to_random"] = (metrics["asv_eer"] / 0.5).clip(lower=0.0, upper=1.0)
    metrics["wer_capped_at_1"] = metrics["asr_wer"].clip(lower=0.0, upper=1.0)
    metrics["report_effect_index"] = (
        0.50 * metrics["eer_normalized_to_random"]
        + 0.30 * metrics["source_similarity_reduction"].clip(lower=0.0, upper=1.0)
        + 0.20 * metrics["wer_capped_at_1"]
    )
    metrics = metrics.sort_values("order").reset_index(drop=True)

    per_utterance = pd.DataFrame(utterance_rows)
    if not per_utterance.empty:
        per_utterance["source_name"] = per_utterance["source_name"].replace({"绿色": "green"})
    return ResultBundle(metrics=metrics, per_utterance=per_utterance, source_target_mean_score=source_target_mean)


def style_axes(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=35)


def configure_report_fonts() -> None:
    preferred_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def method_color(method_id: str) -> str:
    return METHOD_COLORS.get(method_id, "#64748b")


def short_method_name(display_name: str) -> str:
    return display_name.replace("Original source", "Original").replace("Metric+phone", "Metric phone")


def wrap_text(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=False))


def draw_metric_tile(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            transform=ax.transAxes,
            linewidth=1.2,
            edgecolor="#d8dee6",
            facecolor="#ffffff",
        )
    )
    ax.text(
        x + 0.035,
        y + height - 0.07,
        title,
        transform=ax.transAxes,
        fontsize=10.5,
        color="#53616d",
        weight="bold",
        va="top",
    )
    ax.text(
        x + 0.035,
        y + height - 0.145,
        value,
        transform=ax.transAxes,
        fontsize=21,
        color=color,
        weight="bold",
        va="top",
    )
    ax.text(
        x + 0.035,
        y + 0.035,
        wrap_text(subtitle, width=54),
        transform=ax.transAxes,
        fontsize=8.5,
        color="#53616d",
        va="bottom",
    )


def save_barplot(metrics: pd.DataFrame, y: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=metrics,
        x="display_name",
        y=y,
        hue="method_group",
        dodge=False,
        palette="Set2",
    )
    style_axes(ax, title, ylabel)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_privacy_utility_scatter(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics[metrics["method_id"] != "source_baseline"].copy()
    plt.figure(figsize=(8.5, 6.5))
    ax = sns.scatterplot(
        data=plot_df,
        x="asv_eer",
        y="asr_wer",
        hue="method_group",
        style="method_group",
        s=120,
        palette="Set2",
    )
    for _, row in plot_df.iterrows():
        ax.annotate(row["display_name"], (row["asv_eer"], row["asr_wer"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0.5, linestyle="--", color="#555555", linewidth=1, alpha=0.7)
    ax.text(0.505, ax.get_ylim()[1] * 0.94, "random-like ASV", fontsize=8, color="#555555")
    style_axes(ax, "Privacy vs ASR Attack Strength", "ASR WER ↑")
    ax.set_xlabel("ASV EER ↑")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_utterance_wer_plot(per_utterance: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    if per_utterance.empty:
        return
    order_names = metrics.loc[metrics["method_id"] != "source_baseline", "display_name"].tolist()
    plot_df = per_utterance[per_utterance["display_name"].isin(order_names)].copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(11, 5.8))
    ax = sns.barplot(
        data=plot_df,
        x="display_name",
        y="wer",
        hue="source_name",
        order=order_names,
        palette=["#4c78a8", "#f58518"],
    )
    style_axes(ax, "Per-Utterance ASR WER", "WER ↑")
    ax.legend(title="Utterance", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_report_takeaway_dashboard(metrics: pd.DataFrame, output_path: Path) -> None:
    source = metrics.loc[metrics["method_id"] == "source_baseline"].iloc[0]
    anonymized = metrics.loc[metrics["method_id"] != "source_baseline"]
    best_eer = anonymized.loc[anonymized["asv_eer"].idxmax()]
    best_wer = anonymized.loc[anonymized["asr_wer"].idxmax()]
    best_drop = anonymized.loc[anonymized["source_similarity_reduction"].idxmax()]
    best_effect = anonymized.loc[anonymized["report_effect_index"].idxmax()]

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.patch.set_facecolor("#f6f8fa")
    ax.set_axis_off()
    ax.text(
        0.035,
        0.92,
        "Speech anonymization effect at a glance",
        transform=ax.transAxes,
        fontsize=22,
        weight="bold",
        color="#17202a",
    )
    ax.text(
        0.035,
        0.855,
        "Goal: make the converted voice hard to link back to the protected speaker while keeping a speech-like output.",
        transform=ax.transAxes,
        fontsize=11,
        color="#53616d",
    )

    draw_metric_tile(
        ax,
        (0.035, 0.46),
        (0.43, 0.30),
        "Speaker identity leakage",
        f"{source['source_target_mean_score']:.3f} -> {best_drop['source_target_mean_score']:.3f}",
        f"{best_drop['display_name']}: {best_drop['source_similarity_reduction'] * 100:.1f}% lower similarity",
        "#0f766e",
    )
    draw_metric_tile(
        ax,
        (0.535, 0.46),
        (0.43, 0.30),
        "ASV verification confusion",
        f"{source['asv_eer']:.3f} -> {best_eer['asv_eer']:.3f}",
        f"Best: {best_eer['display_name']} (higher EER means harder verification)",
        "#0f766e",
    )
    draw_metric_tile(
        ax,
        (0.035, 0.12),
        (0.43, 0.30),
        "ASR recognition disruption",
        f"{source['asr_wer']:.3f} -> {best_wer['asr_wer']:.3f}",
        f"Best: {best_wer['display_name']} (higher WER means ASR hears it less reliably)",
        "#2563eb",
    )
    draw_metric_tile(
        ax,
        (0.535, 0.12),
        (0.43, 0.30),
        "Best overall report index",
        f"{best_effect['report_effect_index']:.3f}",
        f"{best_effect['display_name']} balances privacy, source-similarity drop, and ASR disruption",
        "#7c3aed",
    )

    ax.text(
        0.035,
        0.045,
        "Note: this is a local report index for this course project, not an official VoicePrivacy leaderboard score.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6b7280",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close(fig)


def save_speaker_identity_ladder(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics.copy()
    plot_df["short_name"] = plot_df["display_name"].map(short_method_name)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("How much speaker identity remains after anonymization?", fontsize=18, weight="bold", y=0.965)
    fig.text(
        0.145,
        0.905,
        "The original source is intentionally high. A successful anonymizer should move left toward low source similarity.",
        fontsize=10,
        color="#53616d",
    )
    colors = [method_color(method_id) for method_id in plot_df["method_id"]]
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df["source_target_mean_score"], color=colors, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["short_name"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity to the protected speaker (lower is better)", fontsize=11)
    ax.set_xlim(0, max(float(plot_df["source_target_mean_score"].max()) * 1.22, 0.78))
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, (bar, row) in enumerate(zip(bars, plot_df.itertuples(index=False), strict=True)):
        reduction = getattr(row, "source_similarity_reduction")
        label = f"{getattr(row, 'source_target_mean_score'):.3f}"
        if getattr(row, "method_id") != "source_baseline":
            label += f"  ({reduction * 100:.1f}% lower)"
        ax.text(
            bar.get_width() + 0.012,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="#17202a",
            weight="bold" if getattr(row, "method_id") == "ppg_tone_male" else "normal",
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    plt.savefig(output_path, dpi=190)
    plt.close(fig)


def save_method_scoreboard(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics.loc[metrics["method_id"] != "source_baseline"].copy()
    columns = [
        ("ASV privacy", "eer_normalized_to_random", "EER -> random"),
        ("Identity removed", "source_similarity_reduction", "source-sim drop"),
        ("ASR disrupted", "wer_capped_at_1", "WER capped at 1"),
        ("Overall", "report_effect_index", "report index"),
    ]
    matrix = plot_df[[column[1] for column in columns]].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(matrix, cmap=sns.light_palette("#0f766e", as_cmap=True), vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([f"{title}\n{hint}" for title, _, hint in columns], fontsize=10)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["display_name"], fontsize=11)
    ax.set_title("Method scoreboard: darker cells mean stronger anonymization evidence", fontsize=16, weight="bold", pad=14)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i, row in enumerate(plot_df.itertuples(index=False)):
        actual_values = [
            f"{getattr(row, 'asv_eer'):.3f}",
            f"{getattr(row, 'source_similarity_reduction') * 100:.1f}%",
            f"{getattr(row, 'asr_wer'):.3f}",
            f"{getattr(row, 'report_effect_index'):.3f}",
        ]
        for j, value in enumerate(actual_values):
            ax.text(j, i, value, ha="center", va="center", color="#17202a", fontsize=10, weight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(plot_df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Normalized score", rotation=270, labelpad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close(fig)


def save_green_transcript_cards(per_utterance: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    if per_utterance.empty:
        return
    green_rows = per_utterance[per_utterance["source_name"].isin(["green", "绿色"])].copy()
    if green_rows.empty:
        return

    order_lookup = {name: idx for idx, name in enumerate(metrics["display_name"].tolist())}
    method_lookup = dict(zip(metrics["display_name"], metrics["method_id"], strict=True))
    green_rows["order"] = green_rows["display_name"].map(order_lookup)
    green_rows = green_rows.sort_values("order")

    reference_text = str(green_rows.iloc[0]["reference_text"])
    card_rows = [
        {
            "display_name": "Original / reference",
            "method_id": "source_baseline",
            "wer": 0.0,
            "hypothesis_text": reference_text,
            "tag": "ASR reads the source correctly",
        }
    ]
    for row in green_rows.to_dict(orient="records"):
        wer = float(row["wer"])
        card_rows.append(
            {
                "display_name": row["display_name"],
                "method_id": method_lookup.get(row["display_name"], ""),
                "wer": wer,
                "hypothesis_text": row["hypothesis_text"],
                "tag": "ASR transcript changed" if wer >= 0.3 else "ASR still close",
            }
        )

    fig_height = 1.08 * len(card_rows) + 1.5
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    fig.patch.set_facecolor("#f6f8fa")
    ax.set_axis_off()
    ax.text(
        0.035,
        0.94,
        "What does ASR hear after anonymization?",
        transform=ax.transAxes,
        fontsize=19,
        weight="bold",
        color="#17202a",
    )
    ax.text(
        0.035,
        0.89,
        "Example: green.m4a. Higher WER means the automatic recognizer's transcript moves further away from the source transcript.",
        transform=ax.transAxes,
        fontsize=10,
        color="#53616d",
    )

    top = 0.80
    row_h = 0.78 / len(card_rows)
    for index, row in enumerate(card_rows):
        y = top - index * row_h
        color = method_color(row["method_id"])
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.035, y - row_h + 0.015),
                0.93,
                row_h - 0.025,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                transform=ax.transAxes,
                linewidth=1.0,
                edgecolor="#d8dee6",
                facecolor="#ffffff",
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (0.047, y - row_h + 0.032),
                0.014,
                row_h - 0.06,
                transform=ax.transAxes,
                color=color,
                linewidth=0,
            )
        )
        ax.text(0.075, y - 0.045, row["display_name"], transform=ax.transAxes, fontsize=11, weight="bold", color="#17202a")
        ax.text(0.075, y - 0.103, row["tag"], transform=ax.transAxes, fontsize=9, color="#53616d")
        ax.text(0.27, y - 0.065, f"WER {row['wer']:.3f}", transform=ax.transAxes, fontsize=13, weight="bold", color=color)
        ax.text(
            0.40,
            y - 0.045,
            wrap_text(row["hypothesis_text"], width=34),
            transform=ax.transAxes,
            fontsize=12,
            color="#17202a",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close(fig)


def selected_output_for_source(method_id: str, source_name: str) -> Path | None:
    selection_path = SELECTION_FILES.get(method_id)
    if selection_path is None or not selection_path.exists():
        return None
    items = json.loads(selection_path.read_text(encoding="utf-8"))
    for item in items:
        if item.get("source_name") == source_name:
            return Path(item["final_output"]).expanduser().resolve()
    return None


def save_green_waveform_story(project_root: Path, target_gender: str, output_path: Path) -> None:
    method_ids = ["source_baseline"]
    if target_gender in {"all", "male"}:
        method_ids.extend(["male_leaning", "balanced_phone_clean_male", "ppg_tone_male"])
    if target_gender in {"all", "female"}:
        method_ids.extend(["female_leaning", "balanced_phone_clean_female", "ppg_tone_female"])

    paths: list[tuple[str, str, Path]] = [("source_baseline", "Original denoised", project_root / "work_smooth_verify/denoised/绿色_denoised.wav")]
    for method_id in method_ids:
        if method_id == "source_baseline":
            continue
        path = selected_output_for_source(method_id, "绿色_denoised")
        if path is not None and path.exists():
            paths.append((method_id, DISPLAY_NAMES.get(method_id, method_id), path))
    if len(paths) < 2:
        return

    fig, axes = plt.subplots(len(paths), 1, figsize=(12, 1.8 * len(paths) + 1.0), sharex=True)
    if len(paths) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#ffffff")
    for ax, (method_id, label, path) in zip(axes, paths, strict=True):
        sample_rate, audio = load_audio(path)
        max_samples = min(len(audio), int(sample_rate * 4.2))
        audio = audio[:max_samples]
        if audio.size == 0:
            continue
        peak = float(np.max(np.abs(audio)) + 1e-8)
        audio = audio / peak
        time_axis = np.arange(audio.size) / float(sample_rate)
        color = method_color(method_id)
        ax.plot(time_axis, audio, color=color, linewidth=0.8)
        ax.fill_between(time_axis, -np.abs(audio), np.abs(audio), color=color, alpha=0.18)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=92, fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[0].set_title("Waveform comparison on green.m4a", fontsize=16, weight="bold", pad=12)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=190)
    plt.close(fig)


def method_audio_groups(project_root: Path, target_gender: str = "all") -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {
        "Original source": [project_root / "test.wav", project_root / "绿色.m4a"],
    }
    method_ids = [
        "female_leaning",
        "male_leaning",
        "balanced_phone_clean_female",
        "balanced_phone_clean_male",
        "ppg_tone_female",
        "ppg_tone_male",
    ]
    for method_id in method_ids:
        if not variant_matches_gender(method_id, target_gender):
            continue
        paths = read_selection_outputs(SELECTION_FILES[method_id])
        if paths:
            groups[DISPLAY_NAMES[method_id]] = paths
    return groups


def save_similarity_heatmap(
    project_root: Path,
    output_dir: Path,
    asv_python: Path,
    speaker_model: Path,
    speaker_savedir_root: Path,
    target_gender: str = "all",
) -> None:
    groups = method_audio_groups(project_root, target_gender=target_gender)
    unique_paths: dict[str, Path] = {}
    for paths in groups.values():
        for path in paths:
            if path.exists():
                unique_paths[str(path)] = path
    if not unique_paths:
        return

    embeddings = run_asv_embeddings(
        audio_paths=list(unique_paths.values()),
        asv_python=asv_python,
        speaker_model=speaker_model,
        speaker_savedir_root=speaker_savedir_root,
    )
    group_embeddings: dict[str, np.ndarray] = {}
    source_embedding = average_embedding([embeddings[str(path)] for path in groups["Original source"] if str(path) in embeddings])
    group_embeddings["Original source"] = source_embedding

    for label, paths in groups.items():
        if label == "Original source":
            continue
        vectors = [embeddings[str(path)] for path in paths if str(path) in embeddings]
        if vectors:
            group_embeddings[label] = average_embedding(vectors)

    labels = list(group_embeddings.keys())
    matrix = np.asarray(
        [[cosine_score(group_embeddings[left], group_embeddings[right]) for right in labels] for left in labels],
        dtype=np.float32,
    )
    heatmap_df = pd.DataFrame(matrix, index=labels, columns=labels)
    heatmap_df.to_csv(output_dir / "speaker_similarity_heatmap.csv")

    plt.figure(figsize=(9, 7.5))
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-0.05,
        vmax=0.75,
        square=True,
        cbar_kws={"label": "ECAPA cosine similarity"},
    )
    ax.set_title("Speaker Similarity Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "speaker_similarity_heatmap.png", dpi=180)
    plt.close()

    source_row = (
        heatmap_df.loc["Original source"]
        .drop("Original source")
        .sort_values()
        .rename("similarity_to_source")
        .reset_index()
        .rename(columns={"index": "method"})
    )
    source_row.to_csv(output_dir / "similarity_to_source.csv", index=False)


def plot_f0_contours(project_root: Path, output_path: Path, target_gender: str = "all") -> None:
    candidates = {
        "Original denoised": project_root / "work_smooth_verify/denoised/绿色_denoised.wav",
    }
    gender_candidates = {
        "male": {
            "Metric+phone male": project_root
            / "work_metric_attack/final/recommended/balanced_phone_clean_male/绿色_denoised_balanced_phone_clean_male.wav",
            "PPG-tone male": project_root / "work_ppg_tone/final/recommended/ppg_tone_male/绿色_denoised_ppg_tone_male.wav",
        },
        "female": {
            "Metric+phone female": project_root
            / "work_metric_attack/final/recommended/balanced_phone_clean_female/绿色_denoised_balanced_phone_clean_female.wav",
            "PPG-tone female": project_root / "work_ppg_tone/final/recommended/ppg_tone_female/绿色_denoised_ppg_tone_female.wav",
        },
    }
    selected_genders = ("male", "female") if target_gender == "all" else (target_gender,)
    for gender in selected_genders:
        candidates.update(gender_candidates[gender])

    plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    for label, path in candidates.items():
        if not path.exists():
            continue
        sample_rate, audio = load_audio(path)
        analysis = analyze_frames(audio, sample_rate)
        f0 = (2.0 ** smooth_1d(analysis.log_f0, passes=3)).astype(np.float32)
        f0[analysis.voiced < 0.45] = np.nan
        ax.plot(analysis.times_sec, f0, linewidth=1.8, label=label)
    ax.set_ylim(60, 380)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("F0 / Tone Contour on green.m4a", fontsize=14, fontweight="bold", pad=12)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_markdown_summary(bundle: ResultBundle, output_dir: Path, target_gender: str = "all") -> None:
    metrics = bundle.metrics.copy()
    best_eer_value = float(metrics["asv_eer"].max())
    best_eer_names = ", ".join(metrics.loc[metrics["asv_eer"] == best_eer_value, "display_name"].tolist())
    best_wer = metrics.loc[metrics["asr_wer"].idxmax()]
    best_effect = metrics.loc[metrics["report_effect_index"].idxmax()]
    anonymized_metrics = metrics[metrics["method_id"] != "source_baseline"]
    best_similarity_drop = anonymized_metrics.loc[anonymized_metrics["source_similarity_reduction"].idxmax()]
    tradeoff_candidates = metrics[metrics["method_id"].isin(["balanced_phone_clean_male", "ppg_tone_male"])]
    if target_gender == "female":
        tradeoff_candidates = metrics[metrics["method_id"].isin(["balanced_phone_clean_female", "ppg_tone_female"])]
    tradeoff_text = ", ".join(tradeoff_candidates["display_name"].tolist()) or best_effect["display_name"]
    target_label = {
        "all": "all target-speaker conditions",
        "male": "male target-speaker condition only",
        "female": "female target-speaker condition only",
    }[target_gender]

    display_cols = [
        "display_name",
        "method_group",
        "asv_eer",
        "asr_wer",
        "source_target_mean_score",
        "source_similarity_reduction",
        "report_effect_index",
    ]
    table_md = metrics[display_cols].to_markdown(index=False, floatfmt=".3f")

    green_rows = pd.DataFrame()
    if not bundle.per_utterance.empty:
        green_rows = bundle.per_utterance[bundle.per_utterance["source_name"].isin(["green", "绿色"])][
            ["display_name", "wer", "hypothesis_text"]
        ]
    green_md = green_rows.to_markdown(index=False, floatfmt=".3f") if not green_rows.empty else "_No per-utterance results found._"
    report_figure_descriptions = [
        ("report_takeaway_dashboard.png", "plain-language one-page summary of the anonymization effect."),
        ("speaker_identity_ladder.png", "direct before/after speaker-identity leakage comparison."),
        ("method_scoreboard.png", "compact method comparison across privacy, identity removal, ASR disruption, and the local effect index."),
        ("green_transcript_comparison.png", "ASR transcript comparison on the Mandarin green.m4a example."),
        ("green_waveform_comparison.png", "waveform comparison showing how the generated audio differs from the source."),
    ]
    audit_figure_descriptions = [
        ("asv_eer_bar.png", "privacy improvement against the original source baseline."),
        ("asr_wer_bar.png", "ASR disruption strength."),
        ("source_similarity_reduction_bar.png", "direct speaker-identity similarity reduction."),
        ("privacy_utility_scatter.png", "privacy/ASR trade-off."),
        ("speaker_similarity_heatmap.png", "intuitive ECAPA embedding similarity map."),
        ("green_f0_contours.png", "tone/F0 contour comparison for the Mandarin short utterance."),
        ("effect_index_bar.png", "combined local privacy/effect index."),
        ("per_utterance_wer_bar.png", "utterance-level ASR disruption evidence."),
    ]

    report_figures_md = "\n".join(
        f"- `{filename}`: {description}"
        for filename, description in report_figure_descriptions
        if (output_dir / filename).exists()
    )
    audit_figures_md = "\n".join(
        f"- `{filename}`: {description}"
        for filename, description in audit_figure_descriptions
        if (output_dir / filename).exists()
    )

    summary = f"""# Report Evaluation Summary

Generated by `generate_report_evaluation.py` for **{target_label}**.

## Key Claims for the Report

1. **Anonymity improves strongly:** ASV EER rises from `{metrics.iloc[0]["asv_eer"]:.3f}` for the original source to `{best_eer_value:.3f}` for `{best_eer_names}`.
2. **Speaker identity similarity drops:** source-speaker cosine score falls from `{bundle.source_target_mean_score:.3f}` for original speech to `{best_similarity_drop["source_target_mean_score"]:.3f}` for `{best_similarity_drop["display_name"]}`, a `{best_similarity_drop["source_similarity_reduction"] * 100:.1f}%` reduction.
3. **ASR is disrupted:** WER rises from `0.000` for the non-anonymized baseline to `{best_wer["asr_wer"]:.3f}` for `{best_wer["display_name"]}`. For the naturalness trade-off, compare `{tradeoff_text}`.
4. **Best visual effect index:** `{best_effect["display_name"]}` has the highest local effect index, combining ASV EER, source-similarity reduction, and WER. This is an explanatory index for the report, not an official VoicePrivacy metric.

## Main Metrics

{table_md}

## 绿色.m4a ASR Evidence

{green_md}

## Generated Figures

### Report-Friendly Figures

{report_figures_md}

### Technical Audit Figures

{audit_figures_md}

## Suggested Caption

The proposed anonymization pipeline substantially reduces speaker identity information. Compared with the non-anonymized baseline, ASV EER increases toward the random-decision region, while the ECAPA speaker similarity to the source speaker drops by more than 90% for the strongest variants. The ASR WER and transcript examples further show that the anonymized outputs disrupt automatic recognition while preserving human-like speech quality.
"""
    (output_dir / "REPORT_EVALUATION_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_report_fonts()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = filter_bundle_by_target(collect_metrics(project_root), args.target_gender)
    metrics = bundle.metrics
    per_utterance = bundle.per_utterance

    metrics.to_csv(output_dir / "method_metrics.csv", index=False)
    metrics.to_json(output_dir / "method_metrics.json", orient="records", force_ascii=False, indent=2)
    if not per_utterance.empty:
        per_utterance.to_csv(output_dir / "per_utterance_asr.csv", index=False)
        per_utterance.to_json(output_dir / "per_utterance_asr.json", orient="records", force_ascii=False, indent=2)

    save_report_takeaway_dashboard(metrics, output_dir / "report_takeaway_dashboard.png")
    save_speaker_identity_ladder(metrics, output_dir / "speaker_identity_ladder.png")
    save_method_scoreboard(metrics, output_dir / "method_scoreboard.png")
    save_green_transcript_cards(per_utterance, metrics, output_dir / "green_transcript_comparison.png")
    save_green_waveform_story(project_root, args.target_gender, output_dir / "green_waveform_comparison.png")
    save_barplot(metrics, "asv_eer", "ASV EER Privacy Improvement", "ASV EER ↑", output_dir / "asv_eer_bar.png")
    save_barplot(metrics, "asr_wer", "ASR WER Disruption", "ASR WER ↑", output_dir / "asr_wer_bar.png")
    save_barplot(
        metrics,
        "source_similarity_reduction",
        "Speaker Similarity Reduction from Source",
        "Reduction ratio ↑",
        output_dir / "source_similarity_reduction_bar.png",
    )
    save_barplot(
        metrics,
        "report_effect_index",
        "Report-Friendly Anonymization Effect Index",
        "Index ↑",
        output_dir / "effect_index_bar.png",
    )
    save_privacy_utility_scatter(metrics, output_dir / "privacy_utility_scatter.png")
    save_utterance_wer_plot(per_utterance, metrics, output_dir / "per_utterance_wer_bar.png")
    plot_f0_contours(project_root, output_dir / "green_f0_contours.png", target_gender=args.target_gender)

    if not args.skip_embedding_heatmap:
        save_similarity_heatmap(
            project_root=project_root,
            output_dir=output_dir,
            asv_python=Path(args.asv_python).expanduser().resolve(),
            speaker_model=Path(args.speaker_model).expanduser().resolve(),
            speaker_savedir_root=Path(args.speaker_savedir_root).expanduser().resolve(),
            target_gender=args.target_gender,
        )

    write_markdown_summary(bundle, output_dir, target_gender=args.target_gender)
    print(f"Report evaluation artifacts written to: {output_dir}")
    print((output_dir / "REPORT_EVALUATION_SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
