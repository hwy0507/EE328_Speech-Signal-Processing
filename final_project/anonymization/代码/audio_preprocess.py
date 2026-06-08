from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC_ARGS = ["-ar", str(TARGET_SAMPLE_RATE), "-ac", str(TARGET_CHANNELS), "-c:a", "pcm_s16le"]
NORMALIZE_FILTER = "highpass=f=55,lowpass=f=7600"
DENOISE_PRESETS = {
    "standard": "highpass=f=55,afftdn=nf=-26:tn=1,lowpass=f=7600,loudnorm=I=-18:TP=-1.5:LRA=11",
    "strong": "highpass=f=58,afftdn=nf=-30:tn=1,lowpass=f=7400,loudnorm=I=-18.5:TP=-1.5:LRA=10",
}


@dataclass(frozen=True)
class PreprocessResult:
    source: Path
    normalized_wav: Path
    denoised_wav: Path
    metadata_json: Path


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_audio(path: Path) -> dict:
    result = run_command([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=filename,format_name,duration,bit_rate",
        "-show_entries",
        "stream=index,codec_name,codec_type,sample_rate,channels,sample_fmt",
        "-of",
        "json",
        str(path),
    ])
    return json.loads(result.stdout)


def ffmpeg_convert(input_path: Path, output_path: Path, audio_filter: str | None = None) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
    ]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(TARGET_CODEC_ARGS)
    command.append(str(output_path))
    run_command(command)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype.kind in {"i", "u"}:
        audio /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return sample_rate, audio


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -0.999, 0.999)
    wavfile.write(path, sample_rate, np.int16(np.round(clipped * 32767.0)))


def suppress_low_activity_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = max(int(sample_rate * 0.032), 1)
    hop_length = max(int(sample_rate * 0.010), 1)
    if len(audio) < frame_length:
        return audio.astype(np.float32)

    starts = np.arange(0, len(audio) - frame_length + 1, hop_length, dtype=np.int32)
    energies = np.array([
        float(np.sqrt(np.mean(np.square(audio[start : start + frame_length])) + 1e-8))
        for start in starts
    ], dtype=np.float32)
    centers = starts + frame_length // 2
    if centers[0] != 0:
        centers = np.concatenate([np.array([0], dtype=np.int32), centers])
        energies = np.concatenate([energies[:1], energies])
    if centers[-1] != len(audio) - 1:
        centers = np.concatenate([centers, np.array([len(audio) - 1], dtype=np.int32)])
        energies = np.concatenate([energies, energies[-1:]])

    env = np.interp(np.arange(len(audio)), centers, energies).astype(np.float32)
    noise_floor = max(float(np.percentile(env, 20)), 1e-4)
    speech_level = max(float(np.percentile(env, 85)), noise_floor + 1e-8)
    activity = np.clip((env - noise_floor) / (speech_level - noise_floor + 1e-8), 0.0, 1.0)

    low_mask = activity < 0.1
    gate = np.ones(len(audio), dtype=np.float32)
    floor_gain = 0.55
    min_silence = max(int(sample_rate * 0.24), 1)
    fade = max(int(sample_rate * 0.08), 1)

    index = 0
    while index < len(low_mask):
        if not low_mask[index]:
            index += 1
            continue
        start = index
        while index < len(low_mask) and low_mask[index]:
            index += 1
        end = index
        if end - start < min_silence:
            continue

        inner_start = min(start + fade, end)
        inner_end = max(end - fade, start)
        if inner_start > start:
            gate[start:inner_start] = np.minimum(
                gate[start:inner_start],
                np.linspace(1.0, floor_gain, inner_start - start, endpoint=False, dtype=np.float32),
            )
        if inner_end > inner_start:
            gate[inner_start:inner_end] = np.minimum(gate[inner_start:inner_end], floor_gain)
        if end > inner_end:
            gate[inner_end:end] = np.minimum(
                gate[inner_end:end],
                np.linspace(floor_gain, 1.0, end - inner_end, endpoint=False, dtype=np.float32),
            )

    smooth = max(int(sample_rate * 0.04), 3)
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    gate = np.convolve(gate, kernel, mode="same")
    return (audio * gate).astype(np.float32)


def build_denoise_filter(preset: str) -> str:
    try:
        return DENOISE_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported denoise preset: {preset}") from exc


def preprocess_file(input_path: Path, output_root: Path, denoise_preset: str = "standard") -> PreprocessResult:
    normalized_dir = output_root / "normalized"
    denoised_dir = output_root / "denoised"
    metadata_dir = output_root / "metadata"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    denoised_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_stem(input_path)
    normalized_wav = normalized_dir / f"{stem}.wav"
    denoised_wav = denoised_dir / f"{stem}_denoised.wav"
    metadata_json = metadata_dir / f"{stem}.json"

    denoise_filter = build_denoise_filter(denoise_preset)
    source_probe = probe_audio(input_path)
    ffmpeg_convert(input_path, normalized_wav, NORMALIZE_FILTER)
    ffmpeg_convert(normalized_wav, denoised_wav, denoise_filter)
    sample_rate, denoised_audio = load_audio(denoised_wav)
    denoised_audio = suppress_low_activity_noise(denoised_audio, sample_rate)
    save_audio(denoised_wav, denoised_audio, sample_rate)
    normalized_probe = probe_audio(normalized_wav)
    denoised_probe = probe_audio(denoised_wav)

    metadata_json.write_text(
        json.dumps(
            {
                "source": str(input_path),
                "normalized_wav": str(normalized_wav),
                "denoised_wav": str(denoised_wav),
                "denoise_preset": denoise_preset,
                "normalize_filter": NORMALIZE_FILTER,
                "denoise_filter": denoise_filter,
                "source_probe": source_probe,
                "normalized_probe": normalized_probe,
                "denoised_probe": denoised_probe,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return PreprocessResult(
        source=input_path,
        normalized_wav=normalized_wav,
        denoised_wav=denoised_wav,
        metadata_json=metadata_json,
    )


def preprocess_many(input_paths: Iterable[Path], output_root: Path, denoise_preset: str = "standard") -> list[PreprocessResult]:
    return [preprocess_file(path, output_root, denoise_preset=denoise_preset) for path in input_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and denoise speech files for anonymization.")
    parser.add_argument("inputs", nargs="+", help="Input audio files.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "work"),
        help="Directory for normalized, denoised, and metadata outputs.",
    )
    parser.add_argument(
        "--denoise-preset",
        choices=sorted(DENOISE_PRESETS),
        default="standard",
        help="Denoise strength preset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(item).expanduser().resolve() for item in args.inputs]
    output_root = Path(args.output_root).expanduser().resolve()
    results = preprocess_many(input_paths, output_root, denoise_preset=args.denoise_preset)
    print(json.dumps([
        {
            "source": str(result.source),
            "normalized_wav": str(result.normalized_wav),
            "denoised_wav": str(result.denoised_wav),
            "metadata_json": str(result.metadata_json),
        }
        for result in results
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
