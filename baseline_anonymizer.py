from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from naturalness_postprocess import humanize_candidate

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    semitones: float
    tempo_scale: float
    equalizer_chain: tuple[str, ...]


PROFILES = (
    CandidateProfile(
        name="bright_up",
        semitones=4.2,
        tempo_scale=1.0,
        equalizer_chain=(
            "equalizer=f=140:t=q:w=0.9:g=-4",
            "equalizer=f=420:t=q:w=1.1:g=-2",
            "equalizer=f=1900:t=q:w=1.1:g=3",
            "equalizer=f=3400:t=q:w=0.8:g=4",
            "equalizer=f=5600:t=q:w=0.6:g=2",
        ),
    ),
    CandidateProfile(
        name="forward_up",
        semitones=2.8,
        tempo_scale=0.99,
        equalizer_chain=(
            "equalizer=f=120:t=q:w=0.8:g=-3",
            "equalizer=f=260:t=q:w=1.0:g=-2",
            "equalizer=f=1500:t=q:w=1.2:g=2.5",
            "equalizer=f=2800:t=q:w=1.0:g=3.2",
            "equalizer=f=4800:t=q:w=0.8:g=1.5",
        ),
    ),
    CandidateProfile(
        name="warm_down",
        semitones=-3.2,
        tempo_scale=1.01,
        equalizer_chain=(
            "equalizer=f=110:t=q:w=0.8:g=2",
            "equalizer=f=220:t=q:w=1.0:g=2.5",
            "equalizer=f=600:t=q:w=1.2:g=1.5",
            "equalizer=f=2400:t=q:w=1.0:g=-2.5",
            "equalizer=f=4300:t=q:w=0.7:g=-2",
        ),
    ),
    CandidateProfile(
        name="neutral_down",
        semitones=-1.8,
        tempo_scale=1.0,
        equalizer_chain=(
            "equalizer=f=150:t=q:w=0.9:g=-1.5",
            "equalizer=f=320:t=q:w=1.0:g=1.5",
            "equalizer=f=900:t=q:w=1.1:g=1.2",
            "equalizer=f=2400:t=q:w=1.0:g=-1.8",
            "equalizer=f=5100:t=q:w=0.8:g=-1.2",
        ),
    ),
)


@dataclass(frozen=True)
class CandidateResult:
    profile: CandidateProfile
    output_path: Path
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def pitch_ratio_from_semitones(semitones: float) -> float:
    return 2 ** (semitones / 12.0)


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def analyze_levels(path: Path) -> tuple[float, float, float]:
    sample_rate, data = wavfile.read(path)
    del sample_rate
    audio = data.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
    full_scale = float(np.iinfo(data.dtype).max)
    clipping_ratio = float(np.mean(np.abs(audio) >= full_scale))
    peak_dbfs = 20.0 * math.log10(max(peak / full_scale, 1e-8))
    rms_dbfs = 20.0 * math.log10(max(rms / full_scale, 1e-8))
    return peak_dbfs, rms_dbfs, clipping_ratio


def build_filter(profile: CandidateProfile) -> str:
    ratio = pitch_ratio_from_semitones(profile.semitones)
    stretch = (1.0 / ratio) * profile.tempo_scale
    filter_parts = [
        f"asetrate={int(TARGET_SAMPLE_RATE * ratio)}",
        f"aresample={TARGET_SAMPLE_RATE}",
        f"atempo={stretch:.6f}",
        *profile.equalizer_chain,
        "deesser=i=0.4:m=0.5:f=0.5:s=o",
        "dynaudnorm=f=100:g=25:p=0.9",
        "loudnorm=I=-18:TP=-1.5:LRA=9",
    ]
    return ",".join(filter_parts)


def anonymize_file(input_wav: Path, output_root: Path) -> list[CandidateResult]:
    candidate_dir = output_root / safe_stem(input_wav)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    results: list[CandidateResult] = []

    for profile in PROFILES:
        output_path = candidate_dir / f"{safe_stem(input_wav)}_{profile.name}.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            build_filter(profile),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        run_command(command)
        humanize_candidate(input_wav, output_path)
        peak_dbfs, rms_dbfs, clipping_ratio = analyze_levels(output_path)
        results.append(
            CandidateResult(
                profile=profile,
                output_path=output_path,
                duration_sec=duration_seconds(output_path),
                peak_dbfs=peak_dbfs,
                rms_dbfs=rms_dbfs,
                clipping_ratio=clipping_ratio,
            )
        )

    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    **asdict(result),
                    "source_path": str(input_wav),
                    "profile": {"backend": "baseline", **asdict(result.profile)},
                    "output_path": str(result.output_path),
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def anonymize_many(input_wavs: list[Path], output_root: Path) -> dict[str, list[dict]]:
    summary: dict[str, list[dict]] = {}
    for input_wav in input_wavs:
        summary[str(input_wav)] = [
            {
                **asdict(result),
                "source_path": str(input_wav),
                "profile": {"backend": "baseline", **asdict(result.profile)},
                "output_path": str(result.output_path),
            }
            for result in anonymize_file(input_wav, output_root)
        ]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple DSP anonymization candidates.")
    parser.add_argument("inputs", nargs="+", help="Preprocessed WAV files.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "work" / "baseline_candidates"),
        help="Directory for anonymized candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(item).expanduser().resolve() for item in args.inputs]
    output_root = Path(args.output_root).expanduser().resolve()
    summary = anonymize_many(inputs, output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
