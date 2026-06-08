from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SELECTION_GLOB = "work_smooth_verify/final/preferred_variants/*_selections.json"
DEFAULT_OUTPUT_PATH = DEFAULT_PROJECT_ROOT / "voiceprivacy_style_results.json"
DEFAULT_ASV_PYTHON = Path("/opt/anaconda3/envs/speech-anon310/bin/python")
DEFAULT_SPEAKER_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
)
DEFAULT_SPEAKER_SAVEDIR_ROOT = Path(
    os.environ.get(
        "VOICEPRIVACY_SPEAKER_SAVEDIR_ROOT",
        str(DEFAULT_PROJECT_ROOT / "work_speaker_savedir"),
    )
)
DEFAULT_WHISPER_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)
DEFAULT_FFMPEG_BINARY = os.environ.get("FFMPEG_BINARY", "ffmpeg")
AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac")

ASV_WORKER = r"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import FetchConfig, LocalStrategy

warnings.filterwarnings('ignore')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

request_path = Path(sys.argv[1])
request = json.loads(request_path.read_text(encoding='utf-8'))
model_dir = Path(request['speaker_model'])
ffmpeg_binary = request.get('ffmpeg_binary') or os.environ.get('FFMPEG_BINARY') or 'ffmpeg'
classifier = EncoderClassifier.from_hparams(
    source=str(model_dir),
    savedir=str(model_dir),
    local_strategy=LocalStrategy.COPY_SKIP_CACHE,
    fetch_config=FetchConfig(allow_network=False),
    run_opts={'device': 'cpu'},
)


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    raw = subprocess.check_output(
        [
            ffmpeg_binary,
            '-v',
            'error',
            '-i',
            path,
            '-f',
            'f32le',
            '-acodec',
            'pcm_f32le',
            '-ac',
            '1',
            '-ar',
            str(sample_rate),
            'pipe:1',
        ]
    )
    audio = np.frombuffer(raw, dtype=np.float32)
    return torch.from_numpy(audio.copy())


embeddings = {}
for raw_path in request['paths']:
    signal = load_audio(raw_path).unsqueeze(0)
    lengths = torch.tensor([1.0])
    embedding = classifier.encode_batch(signal, lengths).squeeze().cpu()
    embedding = F.normalize(embedding, dim=0)
    embeddings[raw_path] = embedding.tolist()

print(json.dumps({'embeddings': embeddings}, ensure_ascii=False))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anonymized outputs with VoicePrivacy-style metrics.")
    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
        help="Project root containing source audio and work directories.",
    )
    parser.add_argument(
        "--selection-glob",
        default=DEFAULT_SELECTION_GLOB,
        help="Glob pattern under project root for variant selection JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the aggregated VoicePrivacy-style evaluation JSON.",
    )
    parser.add_argument(
        "--asv-python",
        default=str(DEFAULT_ASV_PYTHON),
        help="Python executable with speechbrain installed for ASV embedding extraction.",
    )
    parser.add_argument(
        "--speaker-model",
        default=str(DEFAULT_SPEAKER_MODEL),
        help="Local path to the cached SpeechBrain ECAPA speaker model.",
    )
    parser.add_argument(
        "--whisper-model",
        default=str(DEFAULT_WHISPER_MODEL),
        help="Local path to the cached faster-whisper model.",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="ASR language code for faster-whisper.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        default=DEFAULT_FFMPEG_BINARY,
        help="ffmpeg executable for ASV worker audio decoding. Can be absolute path or available on PATH.",
    )
    return parser.parse_args()



def load_selection_sets(project_root: Path, selection_glob: str) -> dict[str, list[dict[str, Any]]]:
    variant_items: dict[str, list[dict[str, Any]]] = {}
    for selection_path in sorted(project_root.glob(selection_glob)):
        variant_name = selection_path.stem.removesuffix("_selections")
        variant_items[variant_name] = json.loads(selection_path.read_text(encoding="utf-8"))
    if not variant_items:
        raise FileNotFoundError(f"No selection files matched: {selection_glob}")
    return variant_items



def resolve_source_audio(project_root: Path, source_name: str) -> Path:
    stem = source_name.removesuffix("_denoised")
    matches = [path for path in project_root.iterdir() if path.is_file() and path.stem == stem and path.suffix.lower() in AUDIO_SUFFIXES]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve source audio for {source_name!r}: {matches}")
    return matches[0]



def build_variant_records(project_root: Path, selection_sets: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], list[Path], list[Path]]:
    variant_records: dict[str, list[dict[str, Any]]] = {}
    source_paths: dict[str, Path] = {}
    reference_paths: dict[str, Path] = {}

    for variant_name, items in selection_sets.items():
        resolved_items: list[dict[str, Any]] = []
        for item in items:
            source_path = resolve_source_audio(project_root, item["source_name"])
            final_output = Path(item["final_output"]).expanduser().resolve()
            resolved_items.append(
                {
                    "source_name": item["source_name"],
                    "source_path": source_path,
                    "final_output": final_output,
                    "profile": item.get("profile", {}),
                }
            )
            source_paths[str(source_path)] = source_path
            for raw_reference in item.get("profile", {}).get("target_reference_paths", []):
                reference_path = Path(raw_reference).expanduser().resolve()
                reference_paths[str(reference_path)] = reference_path
        variant_records[variant_name] = resolved_items

    return variant_records, list(source_paths.values()), list(reference_paths.values())



def resolve_ffmpeg_binary(project_root: Path, requested: str) -> str:
    """Resolve ffmpeg on Windows-friendly fused-project layouts.

    The original script assumed `ffmpeg` is globally available. In the fused
    project the binary is often vendored as `<repo>/ffmpeg/bin/ffmpeg.exe`, while
    this script lives under `<repo>/_external/EE328_Speech-Signal-Processing`.
    Keep the global-name fallback, but prefer explicit env/CLI paths and the
    local vendored binary when present.
    """
    raw = (requested or "ffmpeg").strip()
    if raw and raw.lower() != "ffmpeg":
        expanded = Path(raw).expanduser()
        if expanded.exists():
            return str(expanded.resolve())
        return raw

    local_candidates = [
        project_root / "ffmpeg" / "bin" / "ffmpeg.exe",
        project_root.parent / "ffmpeg" / "bin" / "ffmpeg.exe",
        project_root.parent.parent / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw or "ffmpeg"



def _looks_like_ffmpeg_binary(value: Any) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    name = Path(raw).name.lower()
    return raw.lower() == "ffmpeg" or name in {"ffmpeg", "ffmpeg.exe"}



def run_asv_embeddings(
    audio_paths: list[Path],
    asv_python: Path,
    speaker_model: Path,
    speaker_savedir_root: Path | str | None = None,
    ffmpeg_binary: str | None = None,
) -> dict[str, np.ndarray]:
    """Run SpeechBrain speaker embeddings with backward-compatible arguments.

    Older scripts in this fused project call this function as
    ``run_asv_embeddings(paths, py, model, ffmpeg_binary)``.  The newer
    recording demo imports ``DEFAULT_SPEAKER_SAVEDIR_ROOT`` and calls it as
    ``run_asv_embeddings(paths, py, model, speaker_savedir_root)``.  Keep both
    call styles valid: if the fourth positional argument looks like an ffmpeg
    executable, treat it as ffmpeg; otherwise use the local/project ffmpeg.
    """
    if ffmpeg_binary is None and _looks_like_ffmpeg_binary(speaker_savedir_root):
        ffmpeg_binary = str(speaker_savedir_root)
        speaker_savedir_root = None
    resolved_ffmpeg_binary = resolve_ffmpeg_binary(DEFAULT_PROJECT_ROOT, ffmpeg_binary or DEFAULT_FFMPEG_BINARY)
    if not asv_python.exists():
        raise FileNotFoundError(
            f"ASV Python executable not found: {asv_python}. "
            "Pass --asv-python as an absolute path to the environment that has speechbrain installed."
        )
    if not speaker_model.exists():
        raise FileNotFoundError(
            f"SpeechBrain speaker model cache not found: {speaker_model}. "
            "Set VOICEPRIVACY_SPEAKER_MODEL/SPEAKER_MODEL to a local speechbrain spkrec-ecapa-voxceleb snapshot, "
            "or download/cache the model before running offline evaluation."
        )
    request = {
        "paths": [str(path) for path in audio_paths],
        "speaker_model": str(speaker_model),
        "speaker_savedir_root": str(speaker_savedir_root or DEFAULT_SPEAKER_SAVEDIR_ROOT),
        "ffmpeg_binary": resolved_ffmpeg_binary,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False)
        request_path = Path(handle.name)
    try:
        try:
            output = subprocess.check_output(
                [str(asv_python), "-c", ASV_WORKER, str(request_path)],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to start ASV Python executable: {asv_python}") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.output or ""
            if len(detail) > 4000:
                detail = detail[-4000:]
            raise RuntimeError(
                "ASV embedding worker failed. This usually means speechbrain/torch/ffmpeg is missing, "
                f"or the local speaker model path is invalid. Command Python: {asv_python}. "
                f"ffmpeg: {ffmpeg_binary}. Output:\n{detail}"
            ) from exc
    finally:
        request_path.unlink(missing_ok=True)
    # SpeechBrain/HuggingFace may print cache/offline warnings to stdout before
    # the worker's JSON payload.  Parse the last JSON-looking line instead of
    # assuming stdout contains JSON only.
    payload_line = ""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload_line = stripped
            break
    if not payload_line:
        detail = output[-4000:] if len(output) > 4000 else output
        raise RuntimeError(f"ASV embedding worker did not emit JSON. Output:\n{detail}")
    payload = json.loads(payload_line)
    return {path: np.asarray(vector, dtype=np.float32) for path, vector in payload["embeddings"].items()}



def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8))



def average_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0)
    mean_vector = stacked.mean(axis=0)
    return mean_vector / (np.linalg.norm(mean_vector) + 1e-8)



def compute_eer(target_scores: list[float], nontarget_scores: list[float]) -> dict[str, float]:
    thresholds = sorted(set(target_scores + nontarget_scores), reverse=True)
    best: tuple[float, float, float, float, float] | None = None
    for threshold in thresholds:
        far = sum(score >= threshold for score in nontarget_scores) / len(nontarget_scores)
        frr = sum(score < threshold for score in target_scores) / len(target_scores)
        diff = abs(far - frr)
        if best is None or diff < best[0]:
            best = (diff, (far + frr) / 2.0, threshold, far, frr)
    assert best is not None
    _, eer, threshold, far, frr = best
    return {
        "eer": float(eer),
        "threshold": float(threshold),
        "false_accept_rate": float(far),
        "false_reject_rate": float(frr),
    }



def normalize_transcript(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [
        ch
        for ch in normalized
        if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    ]



def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    dp = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        previous = dp[0]
        dp[0] = row
        for col, hyp_token in enumerate(hypothesis, start=1):
            saved = dp[col]
            if ref_token == hyp_token:
                dp[col] = previous
            else:
                dp[col] = min(previous, dp[col], dp[col - 1]) + 1
            previous = saved
    return dp[-1]



def transcribe_audio(model: Any, path: Path, language: str) -> str:
    segments, _ = model.transcribe(str(path), language=language)
    return "".join(segment.text for segment in segments).strip()



def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    selection_sets = load_selection_sets(project_root, args.selection_glob)
    variant_records, source_paths, reference_paths = build_variant_records(project_root, selection_sets)

    unique_audio_paths: dict[str, Path] = {}
    for path in source_paths + reference_paths:
        unique_audio_paths[str(path)] = path
    for items in variant_records.values():
        for item in items:
            unique_audio_paths[str(item["final_output"])] = item["final_output"]

    embeddings = run_asv_embeddings(
        audio_paths=list(unique_audio_paths.values()),
        asv_python=Path(args.asv_python).expanduser().resolve(),
        speaker_model=Path(args.speaker_model).expanduser().resolve(),
        ffmpeg_binary=resolve_ffmpeg_binary(project_root, args.ffmpeg_binary),
    )
    source_enrollment = average_embedding([embeddings[str(path)] for path in source_paths])

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster_whisper is not installed in the Python environment running evaluate_voiceprivacy.py. "
            "Install faster-whisper there, or point VOICEPRIVACY_PYTHON to an environment that has it."
        ) from exc

    whisper_model_path = Path(args.whisper_model).expanduser().resolve()
    if not whisper_model_path.exists():
        raise FileNotFoundError(
            f"faster-whisper local model cache not found: {whisper_model_path}. "
            "Set VOICEPRIVACY_WHISPER_MODEL/WHISPER_MODEL to a local faster-whisper snapshot, "
            "or download/cache the model before running offline evaluation."
        )
    try:
        whisper_model = WhisperModel(str(whisper_model_path), device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(f"Failed to load faster-whisper model from {whisper_model_path}: {exc}") from exc
    transcript_cache: dict[str, str] = {}
    for source_path in source_paths:
        transcript_cache[str(source_path)] = transcribe_audio(whisper_model, source_path, args.language)

    results: dict[str, Any] = {
        "protocol": {
            "privacy_metric": "ASV EER",
            "utility_metric": "ASR WER",
            "asv_backend": "speechbrain/spkrec-ecapa-voxceleb (cached local model)",
            "asr_backend": "Systran/faster-whisper-small (cached local model)",
            "trial_setup": "Source-speaker enrollment is the mean embedding of all source utterances; each anonymized output is a trial; lab reference speakers are non-target trials.",
            "transcript_setup": "WER uses source-audio Whisper transcripts as pseudo ground truth and character-aware tokenization for Chinese.",
        },
        "caveats": [
            "This follows the same top-level VoicePrivacy evaluation axes (ASV-EER for privacy and ASR-WER for utility) but is not an official leaderboard submission protocol.",
            "The evaluation set in this project is tiny, with one protected speaker and two source utterances, so the reported numbers should be treated as local benchmark signals rather than challenge-comparable figures.",
        ],
        "source_audio": [str(path) for path in source_paths],
        "reference_speakers": [str(path) for path in reference_paths],
        "baseline": {},
        "variants": {},
    }

    baseline_target_scores = [cosine_score(source_enrollment, embeddings[str(path)]) for path in source_paths]
    baseline_nontarget_scores = [
        cosine_score(embeddings[str(reference_path)], embeddings[str(source_path)])
        for reference_path in reference_paths
        for source_path in source_paths
    ]
    results["baseline"]["source_vs_source"] = {
        **compute_eer(baseline_target_scores, baseline_nontarget_scores),
        "target_mean_score": float(np.mean(baseline_target_scores)),
        "nontarget_mean_score": float(np.mean(baseline_nontarget_scores)),
        "num_target_trials": len(baseline_target_scores),
        "num_nontarget_trials": len(baseline_nontarget_scores),
    }

    for variant_name, items in variant_records.items():
        target_scores = [cosine_score(source_enrollment, embeddings[str(item["final_output"])] ) for item in items]
        nontarget_scores = [
            cosine_score(embeddings[str(reference_path)], embeddings[str(item["final_output"])] )
            for reference_path in reference_paths
            for item in items
        ]
        privacy_result = {
            **compute_eer(target_scores, nontarget_scores),
            "target_mean_score": float(np.mean(target_scores)),
            "nontarget_mean_score": float(np.mean(nontarget_scores)),
            "num_target_trials": len(target_scores),
            "num_nontarget_trials": len(nontarget_scores),
        }

        per_utterance = []
        total_edits = 0
        total_tokens = 0
        for item in items:
            source_path = item["source_path"]
            hypothesis_text = transcribe_audio(whisper_model, item["final_output"], args.language)
            transcript_cache[str(item["final_output"])] = hypothesis_text
            reference_text = transcript_cache[str(source_path)]
            reference_tokens = normalize_transcript(reference_text)
            hypothesis_tokens = normalize_transcript(hypothesis_text)
            edits = edit_distance(reference_tokens, hypothesis_tokens)
            token_count = max(len(reference_tokens), 1)
            wer = edits / token_count
            total_edits += edits
            total_tokens += token_count
            per_utterance.append(
                {
                    "source_name": item["source_name"],
                    "source_path": str(source_path),
                    "trial_path": str(item["final_output"]),
                    "reference_text": reference_text,
                    "hypothesis_text": hypothesis_text,
                    "edits": edits,
                    "reference_token_count": token_count,
                    "wer": wer,
                }
            )

        results["variants"][variant_name] = {
            "privacy": privacy_result,
            "utility": {
                "wer": total_edits / max(total_tokens, 1),
                "total_edits": total_edits,
                "total_reference_tokens": total_tokens,
                "per_utterance": per_utterance,
            },
        }

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
