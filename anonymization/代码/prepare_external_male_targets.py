from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from torchaudio.datasets import CMUARCTIC


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_ROOT = Path("/private/tmp/speech_anonymization_datasets")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "external_voice_targets/male"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "vc_target_pool_male_external.json"
DEFAULT_SPEAKERS = ("bdl", "rms", "jmk", "ksp", "awb", "rxr")
TARGET_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public male reference voices for the anonymization target pool.")
    parser.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT), help="Where torchaudio downloads CMU ARCTIC.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where compact target WAVs are written.")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH), help="Target-pool JSON to write.")
    parser.add_argument("--speakers", nargs="+", default=list(DEFAULT_SPEAKERS), help="CMU ARCTIC speaker IDs.")
    parser.add_argument("--utterances-per-speaker", type=int, default=5, help="How many utterances to concatenate per speaker.")
    parser.add_argument("--target-seconds", type=float, default=9.0, help="Approximate maximum duration per reference WAV.")
    parser.add_argument("--silence-ms", type=int, default=120, help="Silence inserted between selected utterances.")
    return parser.parse_args()


def to_numpy_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.dtype.kind in {"i", "u"}:
        raise AssertionError("audio dtype should be converted before calling to_numpy_mono")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    return np.clip(audio / peak * 0.85, -0.95, 0.95).astype(np.float32)


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    if data.dtype.kind in {"i", "u"}:
        full_scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        audio = data.astype(np.float32) / float(full_scale)
    else:
        audio = data.astype(np.float32)
    return audio, sample_rate


def parse_transcript_line(line: str) -> tuple[str, str]:
    line = line.strip()
    if not line:
        return "", ""
    # Format: ( arctic_a0001 "text" )
    first_quote = line.find('"')
    last_quote = line.rfind('"')
    head = line[:first_quote].replace("(", "").strip()
    utt_id = head.split()[0] if head else ""
    text = line[first_quote + 1 : last_quote] if first_quote >= 0 and last_quote > first_quote else ""
    return utt_id, text


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, TARGET_SAMPLE_RATE, np.int16(np.round(np.clip(audio, -0.999, 0.999) * 32767.0)))


def select_indices(dataset_len: int, count: int) -> list[int]:
    if count >= dataset_len:
        return list(range(dataset_len))
    # Use spaced-out utterances to avoid overfitting to one sentence style.
    return np.linspace(0, dataset_len - 1, num=count, dtype=int).tolist()


def build_reference_for_speaker(
    speaker: str,
    download_root: Path,
    output_dir: Path,
    utterances_per_speaker: int,
    target_seconds: float,
    silence_ms: int,
) -> dict:
    # Use torchaudio only for robust download/extraction. The current local
    # torchaudio build requires torchcodec for loading, so WAVs are read below
    # with scipy instead.
    CMUARCTIC(root=str(download_root), url=speaker, download=True)
    speaker_root = download_root / "ARCTIC" / f"cmu_us_{speaker}_arctic"
    transcript_path = speaker_root / "etc/txt.done.data"
    lines = [line for line in transcript_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    silence = np.zeros(int(TARGET_SAMPLE_RATE * silence_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    transcripts: list[str] = []
    total_samples = 0
    max_samples = int(target_seconds * TARGET_SAMPLE_RATE)
    for index in select_indices(len(lines), utterances_per_speaker * 3):
        utterance_id, transcript = parse_transcript_line(lines[index])
        if not utterance_id:
            continue
        wav_path = speaker_root / "wav" / f"{utterance_id}.wav"
        raw_audio, sample_rate = load_wav_mono(wav_path)
        audio = to_numpy_mono(raw_audio, sample_rate)
        if len(audio) < int(0.8 * TARGET_SAMPLE_RATE):
            continue
        if total_samples + len(audio) > max_samples and parts:
            break
        parts.append(audio)
        parts.append(silence)
        transcripts.append(f"{utterance_id}: {transcript}")
        total_samples += len(audio) + len(silence)
        if len(transcripts) >= utterances_per_speaker:
            break

    if not parts:
        raise RuntimeError(f"No usable utterances found for CMU ARCTIC speaker {speaker}.")
    merged = np.concatenate(parts)
    peak = float(np.max(np.abs(merged)) + 1e-8)
    merged = np.clip(merged / peak * 0.90, -0.95, 0.95)
    output_path = output_dir / f"cmu_arctic_{speaker}_male_ref.wav"
    write_wav(output_path, merged)
    return {
        "speaker": speaker,
        "dataset": "CMU ARCTIC",
        "source": f"torchaudio.datasets.CMUARCTIC(url={speaker!r})",
        "output_path": str(output_path),
        "duration_sec": len(merged) / TARGET_SAMPLE_RATE,
        "utterances": transcripts,
    }


def main() -> None:
    args = parse_args()
    download_root = Path(args.download_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "CMU ARCTIC",
        "note": "Public speech synthesis corpus prepared as anonymous male target references. Generated audio files are local artifacts and are not intended to be committed.",
        "speakers": [],
    }
    reference_paths: list[str] = []
    for speaker in args.speakers:
        record = build_reference_for_speaker(
            speaker=speaker,
            download_root=download_root,
            output_dir=output_dir,
            utterances_per_speaker=args.utterances_per_speaker,
            target_seconds=args.target_seconds,
            silence_ms=args.silence_ms,
        )
        manifest["speakers"].append(record)
        reference_paths.append(str(Path(record["output_path"]).relative_to(PROJECT_ROOT)))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "name": "external_male_privacy_pool",
        "strategy": "single_ref_group",
        "reference_paths": [
            *reference_paths,
            "../../常规lab/lab9/s2.wav",
            "../../常规lab/lab9/s5.wav",
            "../../常规lab/lab9/s6.wav",
        ],
        "clip_seconds": 1.0,
        "silence_ms": 80,
        "normalize_peak_dbfs": -3.0,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config_path": str(config_path), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
