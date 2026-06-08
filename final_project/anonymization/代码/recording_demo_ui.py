from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

from audio_preprocess import preprocess_file
from build_metric_attack_variants import ATTACK_PROFILES, apply_attack_profile
from evaluate_voiceprivacy import (
    DEFAULT_ASV_PYTHON,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_SPEAKER_SAVEDIR_ROOT,
    DEFAULT_WHISPER_MODEL,
    cosine_score,
    edit_distance,
    normalize_transcript,
    run_asv_embeddings,
    transcribe_audio,
)
from ppg_tone_naturalizer import naturalize_file
from privacy_target_optimizer import (
    load_target_references,
    optimize_privacy_target,
    standard_similarity_from_cosine,
)
from vc_candidate_builder import DEFAULT_VC_PYTHON

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_WORK_ROOT = PROJECT_ROOT / "work_recording_demo"
DEFAULT_TARGET_POOL_CONFIG = PROJECT_ROOT / "vc_target_pool_male_external.json"
REPORT_EVALUATION_DIR = REPO_ROOT / "文档" / "report_evaluation_male"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FALLBACK_MALE_TARGET = REPO_ROOT.parent / "常规lab/lab9/s6.wav"
REPORT_FIGURES = (
    ("report_takeaway_dashboard.png", "Project effect summary"),
    ("speaker_identity_ladder.png", "Speaker identity leakage"),
    ("method_scoreboard.png", "Method scoreboard"),
    ("identity_privacy_index_bar.png", "Identity privacy ranking"),
    ("green_transcript_comparison.png", "ASR transcript example"),
    ("green_waveform_comparison.png", "Waveform comparison"),
)


@dataclass(frozen=True)
class DemoConfig:
    project_root: Path
    work_root: Path
    vc_python: Path
    host: str
    port: int
    target_pool_config: Path
    max_targets: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a browser-based record-and-anonymize demo UI.")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT), help="Directory for recorded demo artifacts.")
    parser.add_argument("--vc-python", default=str(DEFAULT_VC_PYTHON), help="Python executable for FreeVC inference.")
    parser.add_argument(
        "--target-pool-config",
        default=str(DEFAULT_TARGET_POOL_CONFIG),
        help="JSON config containing male target references. External targets are skipped until prepared.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=9,
        help="Maximum target references to evaluate with FreeVC for each recording.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local Flask server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local Flask server.")
    return parser.parse_args()


def safe_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"


def ui_clarity_profile():
    for profile in ATTACK_PROFILES:
        if profile.name == "clarity_guard":
            return profile
    raise RuntimeError("Missing clarity_guard attack profile")


def profile_payload(
    target_name: str,
    target_reference: Path,
    postprocess: str,
    target_reference_count: int,
    target_pool_name: str,
) -> dict[str, Any]:
    return {
        "backend": "freevc",
        "name": f"freevc_demo_{target_name}",
        "target_reference": str(target_reference),
        "target_strategy": "privacy_optimized_single_ref",
        "target_pool_name": target_pool_name,
        "target_reference_count": target_reference_count,
        "target_reference_paths": [str(target_reference)],
        "postprocess": postprocess,
    }


def build_public_url(session_id: str, path: Path, config: DemoConfig) -> str:
    relative = path.relative_to(config.work_root)
    return f"/outputs/{session_id}/{relative.as_posix().split('/', 1)[1]}"


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


BALANCE_FORMULA = (
    "Balance = 0.50*Privacy + 0.50*Fidelity; "
    "Privacy = clamp((70 - standard_score)/20); "
    "Fidelity = 0.60*naturalness + 0.25*duration_match + 0.15*processing_simplicity."
)


def candidate_balance_components(score: Any) -> dict[str, float]:
    """Candidate-level privacy/fidelity proxy used for UI explanation."""
    privacy_score = clamp01((70.0 - float(score.standard_similarity_score)) / 20.0)
    duration_match = clamp01(1.0 - abs(float(score.duration_ratio) - 1.0) / 0.18)
    processing_simplicity = clamp01(1.0 - float(score.transform_penalty) / 0.04)
    fidelity_score = clamp01(
        0.60 * float(score.naturalness_proxy)
        + 0.25 * duration_match
        + 0.15 * processing_simplicity
    )
    return {
        "privacy_score": privacy_score,
        "fidelity_score": fidelity_score,
        "balance_score": clamp01(0.50 * privacy_score + 0.50 * fidelity_score),
        "duration_match": duration_match,
        "processing_simplicity": processing_simplicity,
    }


def balance_ranked_scores(selection: Any) -> list[Any]:
    return sorted(
        selection.candidates,
        key=lambda score: (
            candidate_balance_components(score)["balance_score"],
            candidate_balance_components(score)["privacy_score"],
            float(score.naturalness_proxy),
        ),
        reverse=True,
    )


def top_candidate_summaries(selection: Any, session_dir: Path, config: DemoConfig, limit: int = 3) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, score in enumerate(balance_ranked_scores(selection)[:limit]):
        components = candidate_balance_components(score)
        output_path = Path(score.output_path).expanduser().resolve()
        summaries.append(
            {
                "rank": index + 1,
                "target": f"male:{score.target_name}/{score.variant_name}",
                "audio_path": str(output_path),
                "audio_url": build_public_url(session_dir.name, output_path, config),
                "standard_similarity_score": score.standard_similarity_score,
                "naturalness_proxy": score.naturalness_proxy,
                "selection_score": score.selection_score,
                "privacy_score": components["privacy_score"],
                "fidelity_score": components["fidelity_score"],
                "balance_score": components["balance_score"],
                "duration_ratio": score.duration_ratio,
                "duration_match": components["duration_match"],
                "processing_simplicity": components["processing_simplicity"],
                "used_by_default": index == 0,
            }
        )
    return summaries


def evaluate_recording_session(denoised_path: Path, results: list[dict[str, Any]], session_dir: Path) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    output_paths = [Path(item["audio_path"]).expanduser().resolve() for item in results]
    denoised_resolved = denoised_path.expanduser().resolve()
    all_paths = [denoised_resolved, *output_paths]
    embeddings = run_asv_embeddings(
        all_paths,
        DEFAULT_ASV_PYTHON,
        DEFAULT_SPEAKER_MODEL,
        DEFAULT_SPEAKER_SAVEDIR_ROOT,
    )
    source_embedding = embeddings[str(denoised_resolved)]

    whisper_model = WhisperModel(str(DEFAULT_WHISPER_MODEL), device="cpu", compute_type="int8")
    reference_text = transcribe_audio(whisper_model, denoised_resolved, "zh")
    reference_tokens = normalize_transcript(reference_text)
    token_count = max(len(reference_tokens), 1)

    rows: list[dict[str, Any]] = []
    for item, output_path in zip(results, output_paths):
        output_embedding = embeddings[str(output_path)]
        similarity = cosine_score(source_embedding, output_embedding)
        standard_similarity = standard_similarity_from_cosine(similarity)
        similarity_drop = clamp01(1.0 - similarity)
        hypothesis_text = transcribe_audio(whisper_model, output_path, "zh")
        edits = edit_distance(reference_tokens, normalize_transcript(hypothesis_text))
        wer = edits / token_count
        content_preservation = 1.0 - clamp01(wer)
        timbre_privacy_index = 0.70 * similarity_drop + 0.30 * content_preservation
        rows.append(
            {
                "label": item["label"],
                "method": item["method"],
                "source_similarity": similarity,
                "standard_similarity": standard_similarity,
                "standard_similarity_score": standard_similarity * 100.0,
                "source_similarity_reduction": similarity_drop,
                "asr_wer": wer,
                "content_preservation": content_preservation,
                "asr_edits": edits,
                "reference_token_count": token_count,
                "reference_text": reference_text,
                "hypothesis_text": hypothesis_text,
                "timbre_privacy_index": timbre_privacy_index,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (row["timbre_privacy_index"], row["source_similarity_reduction"], row["content_preservation"]),
        reverse=True,
    )
    rank_by_label = {row["label"]: index + 1 for index, row in enumerate(ranked)}
    for row in rows:
        row["privacy_rank"] = rank_by_label[row["label"]]

    payload = {
        "available": True,
        "protocol": "single-recording local evaluation",
        "note": "Single recordings cannot produce a reliable ASV EER. This view scores timbre anonymization with source-to-output speaker similarity, while ASR WER is used as a content-preservation check where lower is better.",
        "source_audio": str(denoised_resolved),
        "reference_text": reference_text,
        "metrics": rows,
    }
    (session_dir / "recording_evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def process_recording(input_path: Path, session_dir: Path, config: DemoConfig) -> dict[str, Any]:
    preprocess = preprocess_file(input_path, session_dir / "preprocess", denoise_preset="standard")
    denoised_path = preprocess.denoised_wav
    profile = ui_clarity_profile()

    results: list[dict[str, Any]] = []
    target_references = load_target_references(config.target_pool_config, fallback_paths=[FALLBACK_MALE_TARGET])
    selection = optimize_privacy_target(
        denoised_path,
        target_references,
        session_dir / "target_optimization",
        max_targets=config.max_targets,
        vc_python=config.vc_python,
    )
    selected_score = balance_ranked_scores(selection)[0]
    selected_target = next(target for target in target_references if target.name == selected_score.target_name)
    target_name = f"male:{selected_target.name}/{selected_score.variant_name}"
    target_reference = selected_target.path
    target_pool_name = "external_male_privacy_pool"
    raw_output = Path(selected_score.output_path).expanduser().resolve()

    clarity_output = session_dir / "metric_clarity" / "demo_male_metric_clarity.wav"
    apply_attack_profile(raw_output, clarity_output, profile)
    ppg_output = session_dir / "ppg_tone" / "demo_male_ppg_tone.wav"
    ppg_metadata = session_dir / "ppg_tone" / "demo_male_ppg_tone.json"
    naturalize_file(denoised_path, clarity_output, ppg_output, ppg_metadata, strength=0.4)

    results.extend(
        [
            {
                "label": "FreeVC optimized male",
                "method": "freevc_baseline",
                "target": target_name,
                "audio_url": build_public_url(session_dir.name, raw_output, config),
                "audio_path": str(raw_output),
                "selection": {
                    "target": target_name,
                    "standard_similarity_score": selected_score.standard_similarity_score,
                    "naturalness_proxy": selected_score.naturalness_proxy,
                    "evaluated_target_count": selection.evaluated_target_count,
                    "target_pool_size": selection.target_pool_size,
                },
                "profile": profile_payload(
                    target_name,
                    target_reference,
                    "humanize_candidate + privacy_target_optimizer",
                    selection.target_pool_size,
                    target_pool_name,
                ),
            },
            {
                "label": "Metric+clarity optimized male",
                "method": "metric_clarity",
                "target": target_name,
                "audio_url": build_public_url(session_dir.name, clarity_output, config),
                "audio_path": str(clarity_output),
                "selection": {
                    "target": target_name,
                    "standard_similarity_score": selected_score.standard_similarity_score,
                    "naturalness_proxy": selected_score.naturalness_proxy,
                    "evaluated_target_count": selection.evaluated_target_count,
                    "target_pool_size": selection.target_pool_size,
                },
                "profile": profile_payload(
                    target_name,
                    target_reference,
                    "humanize_candidate + privacy_target_optimizer + clarity_guard",
                    selection.target_pool_size,
                    target_pool_name,
                ),
            },
            {
                "label": "PPG-tone optimized male",
                "method": "ppg_tone",
                "target": target_name,
                "audio_url": build_public_url(session_dir.name, ppg_output, config),
                "audio_path": str(ppg_output),
                "metadata_path": str(ppg_metadata),
                "selection": {
                    "target": target_name,
                    "standard_similarity_score": selected_score.standard_similarity_score,
                    "naturalness_proxy": selected_score.naturalness_proxy,
                    "evaluated_target_count": selection.evaluated_target_count,
                    "target_pool_size": selection.target_pool_size,
                },
                "profile": profile_payload(
                    target_name,
                    target_reference,
                    "humanize_candidate + privacy_target_optimizer + clarity_guard + ppg_tone_naturalizer",
                    selection.target_pool_size,
                    target_pool_name,
                ),
            },
        ]
    )
    (session_dir / "selected_target_summary.json").write_text(
        json.dumps(
            {
                "selected_target": target_name,
                "selected_reference": str(target_reference),
                "selected_raw_output": str(raw_output),
                "selected_standard_similarity_score": selected_score.standard_similarity_score,
                "selected_naturalness_proxy": selected_score.naturalness_proxy,
                "selected_balance_score": candidate_balance_components(selected_score)["balance_score"],
                "target_pool_size": selection.target_pool_size,
                "evaluated_target_count": selection.evaluated_target_count,
                "top_candidate_formula": BALANCE_FORMULA,
                "top_candidate_summaries": top_candidate_summaries(selection, session_dir, config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "session_id": session_dir.name,
        "input_path": str(input_path),
        "denoised_path": str(denoised_path),
        "denoised_url": build_public_url(session_dir.name, denoised_path, config),
        "results": results,
        "selection_candidates": top_candidate_summaries(selection, session_dir, config),
        "balance_formula": BALANCE_FORMULA,
        "notes": [
            "This is a record-then-process demo, not low-latency streaming.",
            "The UI evaluates a male target pool and selects the candidate with the highest privacy/fidelity balance score.",
            "Each recording runs three male-target methods from the selected candidate: FreeVC optimized baseline, Metric+clarity, and PPG-tone.",
            "The standard similarity score follows standard.docx: score=(cosine+1)/2*100; lower is better for anonymization.",
        ],
    }
    try:
        summary["recording_evaluation"] = evaluate_recording_session(denoised_path, results, session_dir)
    except Exception as exc:  # noqa: BLE001 - keep generated audio usable even if local scoring fails.
        summary["recording_evaluation"] = {
            "available": False,
            "error": str(exc),
            "note": "Audio was generated, but per-recording scoring failed.",
        }
    (session_dir / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_report_evaluation() -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "generate_report_evaluation.py"),
        "--target-gender",
        "male",
        "--output-dir",
        str(REPORT_EVALUATION_DIR),
        "--skip-embedding-heatmap",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)


def load_report_evaluation_payload() -> dict[str, Any]:
    metrics_path = REPORT_EVALUATION_DIR / "method_metrics.json"
    per_utterance_path = REPORT_EVALUATION_DIR / "per_utterance_asr.json"
    summary_path = REPORT_EVALUATION_DIR / "REPORT_EVALUATION_SUMMARY.md"
    if not metrics_path.exists():
        return {
            "available": False,
            "message": "Male-only evaluation has not been generated yet.",
            "output_dir": str(REPORT_EVALUATION_DIR),
        }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_utterance = []
    if per_utterance_path.exists():
        per_utterance = json.loads(per_utterance_path.read_text(encoding="utf-8"))

    figures = []
    for filename, label in REPORT_FIGURES:
        figure_path = REPORT_EVALUATION_DIR / filename
        if figure_path.exists():
            version = int(figure_path.stat().st_mtime)
            figures.append(
                {
                    "label": label,
                    "filename": filename,
                    "url": f"/report-evaluation/{filename}?v={version}",
                }
            )

    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    generated_at = int(metrics_path.stat().st_mtime)
    return {
        "available": True,
        "target_gender": "male",
        "output_dir": str(REPORT_EVALUATION_DIR),
        "generated_at": generated_at,
        "metrics": metrics,
        "per_utterance": per_utterance,
        "figures": figures,
        "summary_markdown": summary_text,
    }


def create_app(config: DemoConfig) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.post("/api/anonymize")
    def anonymize():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file received."}), 400

        session_dir = config.work_root / safe_session_id()
        session_dir.mkdir(parents=True, exist_ok=True)
        uploaded = request.files["audio"]
        suffix = Path(uploaded.filename or "recording.webm").suffix or ".webm"
        input_path = session_dir / f"recording{suffix}"
        uploaded.save(input_path)

        try:
            summary = process_recording(input_path, session_dir, config)
        except Exception as exc:  # noqa: BLE001 - surface a clear UI error for the local demo.
            error_payload = {
                "session_id": session_dir.name,
                "error": str(exc),
                "input_path": str(input_path),
            }
            (session_dir / "error.json").write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return jsonify(error_payload), 500

        return jsonify(summary)

    @app.get("/outputs/<session_id>/<path:filename>")
    def outputs(session_id: str, filename: str):
        base = (config.work_root / session_id).resolve()
        target = (base / filename).resolve()
        if not target.is_file() or base not in target.parents:
            return jsonify({"error": "File not found"}), 404
        return send_file(target)

    @app.get("/report-evaluation/<path:filename>")
    def report_evaluation_file(filename: str):
        base = REPORT_EVALUATION_DIR.resolve()
        target = (base / filename).resolve()
        if not target.is_file() or base not in target.parents:
            return jsonify({"error": "File not found"}), 404
        return send_file(target)

    @app.get("/api/report-evaluation")
    def get_report_evaluation():
        return jsonify(load_report_evaluation_payload())

    @app.post("/api/report-evaluation")
    def refresh_report_evaluation():
        try:
            run_report_evaluation()
        except subprocess.CalledProcessError as exc:
            return (
                jsonify(
                    {
                        "available": False,
                        "error": "Report evaluation failed.",
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    }
                ),
                500,
            )
        return jsonify(load_report_evaluation_payload())

    @app.get("/api/config")
    def api_config():
        target_references = load_target_references(config.target_pool_config, fallback_paths=[FALLBACK_MALE_TARGET])
        return jsonify(
            {
                "work_root": str(config.work_root),
                "vc_python": str(config.vc_python),
                "target_pool_config": str(config.target_pool_config),
                "target_pool_size": len(target_references),
                "max_targets": config.max_targets,
                "targets": [str(reference.path) for reference in target_references],
            }
        )

    return app


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Speech Anonymization Recorder</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #182026;
      --muted: #5b6770;
      --line: #d8dde3;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --good: #16794c;
      --soft: #eef5f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 26px;
      line-height: 1.2;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 16px 0;
    }
    button {
      appearance: none;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 650;
      cursor: pointer;
      min-height: 42px;
    }
    button:disabled {
      opacity: 0.48;
      cursor: not-allowed;
    }
    .primary { background: var(--accent); color: #fff; }
    .primary:hover:not(:disabled) { background: var(--accent-dark); }
    .secondary { background: #fff; border-color: var(--line); color: var(--text); }
    .danger { background: var(--danger); color: #fff; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 6px;
      background: #eef2f3;
      color: var(--muted);
      font-size: 14px;
    }
    .status.recording { color: var(--danger); background: #fff1f0; }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 99px;
      background: currentColor;
    }
    audio {
      width: 100%;
      margin-top: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .result h3 {
      margin: 0 0 4px;
      font-size: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
      min-height: 90px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric .value {
      color: var(--good);
      font-size: 24px;
      font-weight: 750;
      line-height: 1.1;
    }
    .metric .sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .figure-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      margin-top: 14px;
    }
    .figure {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .figure h3 {
      margin: 0 0 8px;
      font-size: 15px;
    }
    .figure img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #edf0f2;
      border-radius: 6px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: #fff;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }
    th {
      background: #f3f5f6;
      color: var(--muted);
      font-weight: 700;
    }
    tr:last-child td { border-bottom: 0; }
    .eval-status {
      color: var(--muted);
      font-size: 14px;
    }
    .path {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .error {
      color: var(--danger);
      white-space: pre-wrap;
    }
    .notice {
      border: 1px solid #cbd8d6;
      background: var(--soft);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--muted);
      line-height: 1.5;
      margin: 12px 0;
    }
    details {
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }
    pre {
      background: #101820;
      color: #e8eef2;
      padding: 12px;
      border-radius: 8px;
      overflow: auto;
      font-size: 12px;
    }
    @media (max-width: 760px) {
      header { display: block; }
      .grid, .metric-grid, .figure-grid { grid-template-columns: 1fr; }
      main { padding: 20px 14px 32px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Speech Anonymization Recorder</h1>
        <p>录音后使用男声目标池全池搜索，生成 FreeVC baseline、Metric+clarity 和 PPG-tone 三种匿名化版本。</p>
      </div>
      <div class="status" id="status"><span class="dot"></span><span id="statusText">Ready</span></div>
    </header>

    <section class="panel">
      <h2>录音</h2>
      <p>建议录 3-10 秒。当前默认会评估 9 个男声参考，处理通常需要约 1-2 分钟。</p>
      <div class="controls">
        <button class="primary" id="startBtn">开始录音</button>
        <button class="danger" id="stopBtn" disabled>结束录音</button>
        <button class="secondary" id="processBtn" disabled>匿名化处理</button>
      </div>
      <audio id="preview" controls hidden></audio>
      <div id="error" class="error"></div>
    </section>

    <section class="panel">
      <h2>本次录音输出与即时评估</h2>
      <p>这里展示刚刚采集的录音、三种匿名化输出，以及针对本次录音计算的声纹相似度和 ASR 转写变化。</p>
      <div id="results" class="grid"></div>
      <div id="summary"></div>
    </section>

    <section class="panel">
      <h2>固定 benchmark 参考</h2>
      <p>这里是 `test.wav` 和 `绿色.m4a` 的报告基准测试，不代表刚刚网页录音的结果。</p>
      <details>
        <summary>展开固定测试集图表</summary>
        <div class="controls">
          <button class="secondary" id="evalBtn">加载 / 刷新固定 benchmark</button>
          <span class="eval-status" id="evalStatus">Not loaded</span>
        </div>
        <div id="evaluation"></div>
      </details>
    </section>
  </main>

  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const processBtn = document.getElementById("processBtn");
    const preview = document.getElementById("preview");
    const statusBox = document.getElementById("status");
    const statusText = document.getElementById("statusText");
    const results = document.getElementById("results");
    const summary = document.getElementById("summary");
    const errorBox = document.getElementById("error");
    const evalBtn = document.getElementById("evalBtn");
    const evalStatus = document.getElementById("evalStatus");
    const evaluation = document.getElementById("evaluation");

    let mediaRecorder = null;
    let chunks = [];
    let recordedBlob = null;

    function setStatus(text, recording = false) {
      statusText.textContent = text;
      statusBox.classList.toggle("recording", recording);
    }

    function showError(message) {
      errorBox.textContent = message || "";
    }

    startBtn.addEventListener("click", async () => {
      showError("");
      results.innerHTML = "";
      summary.innerHTML = "";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];
        recordedBlob = null;
        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) chunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
          recordedBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
          preview.src = URL.createObjectURL(recordedBlob);
          preview.hidden = false;
          processBtn.disabled = false;
          stream.getTracks().forEach(track => track.stop());
          setStatus("Recorded. Ready to anonymize.");
        };
        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        processBtn.disabled = true;
        setStatus("Recording", true);
      } catch (error) {
        showError("无法访问麦克风：" + error.message);
        setStatus("Microphone error");
      }
    });

    stopBtn.addEventListener("click", () => {
      if (!mediaRecorder) return;
      mediaRecorder.stop();
      startBtn.disabled = false;
      stopBtn.disabled = true;
    });

    processBtn.addEventListener("click", async () => {
      if (!recordedBlob) return;
      showError("");
      results.innerHTML = "";
      summary.innerHTML = "";
      processBtn.disabled = true;
      startBtn.disabled = true;
      setStatus("Processing target pool");

      const formData = new FormData();
      const ext = recordedBlob.type.includes("wav") ? "wav" : "webm";
      formData.append("audio", recordedBlob, `recording.${ext}`);
      try {
        const response = await fetch("/api/anonymize", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || JSON.stringify(payload));
        }
        renderResults(payload);
        setStatus("Done");
      } catch (error) {
        showError("处理失败：\\n" + error.message);
        setStatus("Failed");
      } finally {
        processBtn.disabled = false;
        startBtn.disabled = false;
      }
    });

    evalBtn.addEventListener("click", async () => {
      await loadEvaluation(true);
    });

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatNumber(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(3) : "-";
    }

    function formatPercent(value) {
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
    }

    function bestBy(rows, key) {
      return rows.reduce((best, row) => Number(row[key]) > Number(best[key]) ? row : best, rows[0]);
    }

    function minBy(rows, key) {
      return rows.reduce((best, row) => Number(row[key]) < Number(best[key]) ? row : best, rows[0]);
    }

    function renderResults(payload) {
      results.innerHTML = "";
      for (const item of payload.results || []) {
        const card = document.createElement("div");
        card.className = "result";
        card.innerHTML = `
          <h3>${escapeHtml(item.label)}</h3>
          <audio controls src="${escapeHtml(item.audio_url)}"></audio>
        `;
        results.appendChild(card);
      }
      summary.innerHTML = `
        <h3>Session</h3>
        ${renderTargetSelection(payload.selection_candidates || [], payload.balance_formula || "")}
        ${renderRecordingEvaluation(payload.recording_evaluation)}
        <pre>${escapeHtml(JSON.stringify({
          session_id: payload.session_id,
          denoised_path: payload.denoised_path,
          balance_formula: payload.balance_formula,
          notes: payload.notes
        }, null, 2))}</pre>
      `;
    }

    function renderTargetSelection(candidates, formula) {
      if (!candidates.length) return "";
      const formulaLines = escapeHtml(formula).replaceAll("; ", "<br>");
      const cards = candidates.map(row => `
        <div class="result">
          <h3>#${Number(row.rank)} ${escapeHtml(row.target)}</h3>
          <audio controls src="${escapeHtml(row.audio_url)}"></audio>
          <p>${row.used_by_default ? "当前自动使用的 base target" : "候选 base target"}</p>
          <p>Balance ${formatNumber(row.balance_score)} = 0.50 privacy ${formatNumber(row.privacy_score)} + 0.50 fidelity ${formatNumber(row.fidelity_score)}</p>
          <p>Standard ${formatNumber(row.standard_similarity_score)} / 100, naturalness ${formatNumber(row.naturalness_proxy)}, duration ${formatNumber(row.duration_ratio)}</p>
        </div>
      `).join("");
      return `
        <div class="notice">
          <strong>最高的三个候选音色</strong><br>
          系统按 Balance 分数排序，并自动选择 #1 作为本次匿名化 base target。评分标准如下：<br>
          ${formulaLines}
        </div>
        <div class="grid">${cards}</div>
      `;
    }

    function renderRecordingEvaluation(evaluationPayload) {
      if (!evaluationPayload || !evaluationPayload.available) {
        const message = evaluationPayload?.error || "本次录音评估暂不可用，但音频已经生成。";
        return `<div class="notice">${escapeHtml(message)}</div>`;
      }

      const rows = evaluationPayload.metrics || [];
      if (!rows.length) {
        return `<div class="notice">本次录音没有可展示的即时评估结果。</div>`;
      }

      const lowestStandard = minBy(rows, "standard_similarity_score");
      const bestDrop = bestBy(rows, "source_similarity_reduction");
      const bestContent = minBy(rows, "asr_wer");
      const bestEffect = bestBy(rows, "timbre_privacy_index");
      const cards = [
        ["Lowest standard score", formatNumber(lowestStandard.standard_similarity_score), lowestStandard.label],
        ["Best timbre removal", formatPercent(bestDrop.source_similarity_reduction), bestDrop.label],
        ["Lowest ASR WER", formatNumber(bestContent.asr_wer), bestContent.label],
        ["Best timbre index", formatNumber(bestEffect.timbre_privacy_index), bestEffect.label],
        ["Reference transcript", "本次录音", evaluationPayload.reference_text || "-"],
      ].map(([label, value, sub]) => `
        <div class="metric">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
          <div class="sub">${escapeHtml(sub)}</div>
        </div>
      `).join("");

      const tableRows = rows
        .slice()
        .sort((left, right) => Number(left.privacy_rank) - Number(right.privacy_rank))
        .map(row => `
          <tr>
            <td>${Number(row.privacy_rank)}</td>
            <td>${escapeHtml(row.label)}</td>
            <td>${formatNumber(row.source_similarity)}</td>
            <td>${formatNumber(row.standard_similarity_score)}</td>
            <td>${formatPercent(row.source_similarity_reduction)}</td>
            <td>${formatNumber(row.asr_wer)}</td>
            <td>${formatPercent(row.content_preservation)}</td>
            <td>${escapeHtml(row.hypothesis_text)}</td>
            <td>${formatNumber(row.timbre_privacy_index)}</td>
          </tr>
        `).join("");

      return `
        <div class="notice">${escapeHtml(evaluationPayload.note || "")}</div>
        <div class="metric-grid">${cards}</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Method</th>
                <th>Source similarity</th>
                <th>Standard score ↓</th>
                <th>Similarity drop</th>
                <th>ASR WER ↓</th>
                <th>Content kept ↑</th>
                <th>ASR hypothesis</th>
                <th>Timbre index</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      `;
    }

    async function loadEvaluation(refresh) {
      evalBtn.disabled = true;
      evalStatus.textContent = refresh ? "Running evaluation..." : "Loading...";
      try {
        const response = await fetch("/api/report-evaluation", { method: refresh ? "POST" : "GET" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || payload.stderr || JSON.stringify(payload));
        }
        renderEvaluation(payload);
        if (payload.available) {
          evalStatus.textContent = `Updated ${new Date(payload.generated_at * 1000).toLocaleString()}`;
        } else {
          evalStatus.textContent = payload.message || "No evaluation yet";
        }
      } catch (error) {
        evalStatus.textContent = "Failed";
        evaluation.innerHTML = `<div class="error">评估失败：\\n${escapeHtml(error.message)}</div>`;
      } finally {
        evalBtn.disabled = false;
      }
    }

    function renderEvaluation(payload) {
      if (!payload.available) {
        evaluation.innerHTML = `<p>${escapeHtml(payload.message || "No evaluation available.")}</p>`;
        return;
      }

      const rows = payload.metrics || [];
      const anonymizedRows = rows.filter(row => row.method_id !== "source_baseline");
      if (!rows.length || !anonymizedRows.length) {
        evaluation.innerHTML = "<p>评估结果为空。</p>";
        return;
      }

      const bestEer = bestBy(anonymizedRows, "asv_eer");
      const bestWer = bestBy(anonymizedRows, "asr_wer");
      const bestDrop = bestBy(anonymizedRows, "source_similarity_reduction");
      const bestIdentity = bestBy(anonymizedRows, "identity_privacy_index");
      const bestEffect = bestBy(anonymizedRows, "report_effect_index");
      const greenRows = (payload.per_utterance || []).filter(item => item.source_name === "green" || item.source_name === "绿色");

      const cards = [
        ["Best ASV EER", formatNumber(bestEer.asv_eer), bestEer.display_name],
        ["Highest benchmark WER", formatNumber(bestWer.asr_wer), bestWer.display_name],
        ["Max source-sim drop", formatPercent(bestDrop.source_similarity_reduction), bestDrop.display_name],
        ["Best identity index", formatNumber(bestIdentity.identity_privacy_index), bestIdentity.display_name],
        ["Best effect index", formatNumber(bestEffect.report_effect_index), bestEffect.display_name],
      ].map(([label, value, sub]) => `
        <div class="metric">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
          <div class="sub">${escapeHtml(sub)}</div>
        </div>
      `).join("");

      const tableRows = rows.map(row => `
        <tr>
          <td>${escapeHtml(row.display_name)}</td>
          <td>${escapeHtml(row.method_group)}</td>
          <td>${formatNumber(row.asv_eer)}</td>
          <td>${formatNumber(row.asr_wer)}</td>
          <td>${formatNumber(row.source_target_mean_score)}</td>
          <td>${formatPercent(row.source_similarity_reduction)}</td>
          <td>${formatNumber(row.identity_privacy_index)}</td>
          <td>${row.privacy_rank ? Number(row.privacy_rank) : "-"}</td>
          <td>${formatNumber(row.report_effect_index)}</td>
        </tr>
      `).join("");

      const greenTable = greenRows.length ? `
        <h3>绿色.m4a ASR Evidence</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Method</th><th>WER</th><th>Whisper hypothesis</th></tr></thead>
            <tbody>
              ${greenRows.map(row => `
                <tr>
                  <td>${escapeHtml(row.display_name)}</td>
                  <td>${formatNumber(row.wer)}</td>
                  <td>${escapeHtml(row.hypothesis_text)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      ` : "";

      const figures = (payload.figures || []).map(figure => `
        <div class="figure">
          <h3>${escapeHtml(figure.label)}</h3>
          <img src="${escapeHtml(figure.url)}" alt="${escapeHtml(figure.label)}">
        </div>
      `).join("");

      evaluation.innerHTML = `
        <div class="metric-grid">${cards}</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>Group</th>
                <th>ASV EER</th>
                <th>ASR WER</th>
                <th>Source score</th>
                <th>Similarity drop</th>
                <th>Identity index</th>
                <th>Privacy rank</th>
                <th>Effect index</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
        ${greenTable}
        <div class="figure-grid">${figures}</div>
        <div class="path">${escapeHtml(payload.output_dir)}</div>
      `;
    }

    window.addEventListener("load", () => {
      evaluation.innerHTML = `<div class="notice">固定 benchmark 需要时再加载；上方处理结果会显示本次录音的即时评估。</div>`;
    });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    config = DemoConfig(
        project_root=PROJECT_ROOT,
        work_root=Path(args.work_root).expanduser().resolve(),
        vc_python=Path(args.vc_python).expanduser().resolve(),
        host=args.host,
        port=args.port,
        target_pool_config=Path(args.target_pool_config).expanduser().resolve(),
        max_targets=args.max_targets,
    )
    config.work_root.mkdir(parents=True, exist_ok=True)
    app = create_app(config)
    print(json.dumps({key: str(value) for key, value in asdict(config).items()}, ensure_ascii=False, indent=2))
    app.run(host=config.host, port=config.port, debug=False)


if __name__ == "__main__":
    main()
