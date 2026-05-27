from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = DEFAULT_PROJECT_ROOT / "work_metric_attack" / "voiceprivacy_metric_attack_results.json"
DEFAULT_SELECTION_ROOT = DEFAULT_PROJECT_ROOT / "work_metric_attack" / "final" / "preferred_variants"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "work_metric_attack" / "final" / "recommended"

EXPORTS = {
    "balanced_phone_clean_male": "phone_clean_m_s6__m_s6",
    "balanced_phone_clean_female": "phone_clean_f_s4__f_s4",
    "raw_metric_male": "raw_m_s6__m_s6",
    "raw_metric_female": "raw_f_s4__f_s4",
    "mixed_metric_reference": "phone_clean_m_s6__f_s4",
    "max_metric_vowel_mask_reference": "vowel_mask_m_s6__f_s4",
}


def load_selection(selection_root: Path, variant_name: str) -> list[dict[str, Any]]:
    selection_path = selection_root / f"{variant_name}_selections.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection file: {selection_path}")
    return json.loads(selection_path.read_text(encoding="utf-8"))


def export_variant(
    label: str,
    variant_name: str,
    selection_root: Path,
    output_root: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    items = load_selection(selection_root, variant_name)
    variant_root = output_root / label
    variant_root.mkdir(parents=True, exist_ok=True)

    exported_items = []
    for item in items:
        source_name = item["source_name"]
        source_path = Path(item["final_output"]).expanduser().resolve()
        destination = variant_root / f"{source_name}_{label}.wav"
        shutil.copy2(source_path, destination)
        exported = dict(item)
        exported["original_final_output"] = item["final_output"]
        exported["final_output"] = str(destination)
        exported_items.append(exported)

    selection_export_path = variant_root / f"{label}_selections.json"
    selection_export_path.write_text(json.dumps(exported_items, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "label": label,
        "variant_name": variant_name,
        "output_dir": str(variant_root),
        "selection_json": str(selection_export_path),
        "metrics": metrics["variants"][variant_name],
        "files": [item["final_output"] for item in exported_items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export recommended metric-attack variants.")
    parser.add_argument(
        "--results-path",
        default=str(DEFAULT_RESULTS_PATH),
        help="VoicePrivacy-style result JSON produced for metric attack variants.",
    )
    parser.add_argument(
        "--selection-root",
        default=str(DEFAULT_SELECTION_ROOT),
        help="Directory containing variant selection JSON files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where recommended wav files should be copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path).expanduser().resolve()
    selection_root = Path(args.selection_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metrics = json.loads(results_path.read_text(encoding="utf-8"))

    summary = {
        "results_path": str(results_path),
        "selection_root": str(selection_root),
        "output_root": str(output_root),
        "exports": [
            export_variant(label, variant_name, selection_root, output_root, metrics)
            for label, variant_name in EXPORTS.items()
        ],
    }
    summary_path = output_root / "recommended_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
