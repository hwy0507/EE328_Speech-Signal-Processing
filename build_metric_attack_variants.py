from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_SAMPLE_RATE = 16_000
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_WORK_ROOT = DEFAULT_PROJECT_ROOT / "work_smooth_verify"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "work_metric_attack"
VARIANT_SOURCE_ORDER = ("test_denoised", "绿色_denoised")


@dataclass(frozen=True)
class AttackProfile:
    name: str
    description: str
    filter_parts: tuple[str, ...]
    pitch_semitones: float = 0.0
    noise_amplitude: float = 0.0


ATTACK_PROFILES = (
    AttackProfile(
        name="clarity_guard",
        description="Wideband clarity-preserving channel for live demo; keeps consonants and avoids swallowed words.",
        filter_parts=(
            "highpass=f=85",
            "lowpass=f=6200",
            "equalizer=f=1200:t=q:w=1.0:g=-1.0",
            "equalizer=f=3600:t=q:w=0.9:g=1.4",
            "acompressor=threshold=0.060:ratio=1.7:attack=10:release=110",
        ),
    ),
    AttackProfile(
        name="gentle_channel",
        description="Light channel coloration with a small pitch move; intended to keep the voice clean and human.",
        pitch_semitones=0.8,
        filter_parts=(
            "highpass=f=95",
            "lowpass=f=5200",
            "equalizer=f=1450:t=q:w=1.0:g=-2.5",
            "equalizer=f=3100:t=q:w=0.8:g=1.8",
            "acompressor=threshold=0.055:ratio=2.0:attack=8:release=90",
        ),
    ),
    AttackProfile(
        name="phone_clean",
        description="Narrowband telephone-like channel without added noise.",
        filter_parts=(
            "highpass=f=155",
            "lowpass=f=3550",
            "equalizer=f=650:t=q:w=0.9:g=2.0",
            "equalizer=f=1750:t=q:w=1.1:g=-3.2",
            "equalizer=f=2850:t=q:w=0.9:g=2.5",
            "acompressor=threshold=0.05:ratio=2.8:attack=6:release=80",
        ),
    ),
    AttackProfile(
        name="phone_noise",
        description="Telephone channel with low pink noise, like a real noisy recording path.",
        filter_parts=(
            "highpass=f=165",
            "lowpass=f=3400",
            "equalizer=f=730:t=q:w=1.0:g=2.0",
            "equalizer=f=1850:t=q:w=1.0:g=-4.5",
            "equalizer=f=2850:t=q:w=0.8:g=2.2",
            "acompressor=threshold=0.052:ratio=3.0:attack=6:release=85",
        ),
        noise_amplitude=0.010,
    ),
    AttackProfile(
        name="pitch_phone_up",
        description="Small upward pitch shift plus telephone channel and light noise.",
        pitch_semitones=1.45,
        filter_parts=(
            "highpass=f=150",
            "lowpass=f=3400",
            "equalizer=f=900:t=q:w=1.0:g=1.6",
            "equalizer=f=2050:t=q:w=1.2:g=-3.8",
            "acompressor=threshold=0.052:ratio=2.7:attack=6:release=85",
        ),
        noise_amplitude=0.008,
    ),
    AttackProfile(
        name="vowel_mask",
        description="Formant-region masking by band rejection; keeps speech voice-like but weakens ASR cues.",
        filter_parts=(
            "highpass=f=190",
            "lowpass=f=3050",
            "bandreject=f=850:w=260",
            "bandreject=f=2180:w=420",
            "equalizer=f=2650:t=q:w=0.9:g=2.0",
            "acompressor=threshold=0.055:ratio=3.0:attack=6:release=85",
        ),
        noise_amplitude=0.011,
    ),
    AttackProfile(
        name="asr_rough",
        description="Hard but still human-sounding poor-channel variant for maximizing ASR/ASV failure.",
        pitch_semitones=-1.1,
        filter_parts=(
            "highpass=f=245",
            "lowpass=f=2600",
            "bandreject=f=1180:w=420",
            "bandreject=f=2050:w=470",
            "acompressor=threshold=0.05:ratio=3.4:attack=5:release=95",
        ),
        noise_amplitude=0.017,
    ),
)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def duration_seconds(path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def safe_ascii(text: str) -> str:
    replacements = {
        "female_leaning": "f",
        "male_leaning": "m",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cleaned = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"_", "-"}):
            cleaned.append(char)
        elif char in {" ", "."}:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def candidate_id(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    pool_name = profile.get("target_pool_name", "vc")
    reference_stem = Path(profile.get("target_reference", profile.get("name", "ref"))).stem
    return safe_ascii(f"{pool_name}_{reference_stem}")


def source_sort_key(source_name: str) -> tuple[int, str]:
    try:
        return (VARIANT_SOURCE_ORDER.index(source_name), source_name)
    except ValueError:
        return (len(VARIANT_SOURCE_ORDER), source_name)


def load_candidates(source_work_root: Path) -> dict[str, list[dict[str, Any]]]:
    candidates_by_source: dict[str, list[dict[str, Any]]] = {}
    for summary_path in sorted((source_work_root / "evaluation_vc").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for source_name, result in summary.items():
            candidates_by_source.setdefault(source_name, [])
            for candidate in result["all_candidates"]:
                candidate = dict(candidate)
                candidate["candidate_id"] = candidate_id(candidate)
                candidates_by_source[source_name].append(candidate)

    if not candidates_by_source:
        raise FileNotFoundError(f"No evaluated VC candidates found under {source_work_root}")
    for source_name in list(candidates_by_source):
        candidates_by_source[source_name].sort(key=lambda item: (item["candidate_id"], -float(item["score"])))
    return dict(sorted(candidates_by_source.items(), key=lambda item: source_sort_key(item[0])))


def build_filter_chain(profile: AttackProfile) -> str:
    parts: list[str] = []
    if abs(profile.pitch_semitones) > 1e-6:
        ratio = 2 ** (profile.pitch_semitones / 12.0)
        shifted_rate = max(int(round(TARGET_SAMPLE_RATE * ratio)), 1)
        parts.extend(
            [
                f"asetrate={shifted_rate}",
                f"aresample={TARGET_SAMPLE_RATE}",
                f"atempo={1.0 / ratio:.6f}",
            ]
        )
    parts.extend(profile.filter_parts)
    return ",".join(parts)


def apply_attack_profile(input_path: Path, output_path: Path, profile: AttackProfile) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice_chain = build_filter_chain(profile)
    final_chain = "loudnorm=I=-18:TP=-1.5:LRA=9,aresample=16000"
    command = ["ffmpeg", "-y", "-i", str(input_path)]
    if profile.noise_amplitude > 0:
        duration = duration_seconds(input_path)
        command.extend(
            [
                "-filter_complex",
                (
                    f"[0:a]{voice_chain}[voice];"
                    f"anoisesrc=color=pink:amplitude={profile.noise_amplitude}:d={duration:.3f}[noise];"
                    f"[voice][noise]amix=inputs=2:duration=first:dropout_transition=0,{final_chain}[out]"
                ),
                "-map",
                "[out]",
            ]
        )
    else:
        command.extend(["-af", f"{voice_chain},{final_chain}"])
    command.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)])
    run_command(command)


def build_selection_item(
    source_name: str,
    candidate: dict[str, Any],
    final_output: Path,
    attack_profile: AttackProfile | None,
) -> dict[str, Any]:
    profile = dict(candidate.get("profile", {}))
    if attack_profile is not None:
        profile["metric_attack_profile"] = {
            "name": attack_profile.name,
            "description": attack_profile.description,
            "pitch_semitones": attack_profile.pitch_semitones,
            "noise_amplitude": attack_profile.noise_amplitude,
            "filter_parts": list(attack_profile.filter_parts),
        }
        profile["postprocess"] = f"{profile.get('postprocess', 'unknown')} + {attack_profile.name}"
    return {
        "source_name": source_name,
        "variant": "metric_attack",
        "selected_candidate": candidate["candidate_path"],
        "final_output": str(final_output),
        "score": candidate["score"],
        "metric_candidate_id": candidate["candidate_id"],
        "profile": profile,
    }


def write_selection(selection_root: Path, variant_name: str, items: list[dict[str, Any]]) -> None:
    selection_root.mkdir(parents=True, exist_ok=True)
    selection_path = selection_root / f"{safe_ascii(variant_name)}_selections.json"
    selection_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_candidate_combinations(candidates_by_source: dict[str, list[dict[str, Any]]]):
    source_names = list(candidates_by_source)
    for combo in itertools.product(*(candidates_by_source[source_name] for source_name in source_names)):
        yield source_names, combo


def build_variants(
    source_work_root: Path,
    output_root: Path,
    include_raw_combos: bool,
    clean_output: bool,
) -> dict[str, Any]:
    candidates_by_source = load_candidates(source_work_root)
    selection_root = output_root / "final" / "preferred_variants"
    postprocessed_root = output_root / "postprocessed"

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    selection_root.mkdir(parents=True, exist_ok=True)

    raw_variant_count = 0
    if include_raw_combos:
        for source_names, combo in iter_candidate_combinations(candidates_by_source):
            ids = [candidate["candidate_id"] for candidate in combo]
            items = [
                build_selection_item(source_name, candidate, Path(candidate["candidate_path"]), None)
                for source_name, candidate in zip(source_names, combo, strict=True)
            ]
            write_selection(selection_root, f"raw_{'__'.join(ids)}", items)
            raw_variant_count += 1

    generated_outputs: dict[tuple[str, str, str], Path] = {}
    for profile in ATTACK_PROFILES:
        for source_name, candidates in candidates_by_source.items():
            for candidate in candidates:
                input_path = Path(candidate["candidate_path"])
                output_path = (
                    postprocessed_root
                    / profile.name
                    / source_name
                    / f"{safe_ascii(input_path.stem)}_{profile.name}.wav"
                )
                apply_attack_profile(input_path, output_path, profile)
                generated_outputs[(profile.name, source_name, candidate["candidate_id"])] = output_path

    attack_variant_count = 0
    for profile in ATTACK_PROFILES:
        for source_names, combo in iter_candidate_combinations(candidates_by_source):
            ids = [candidate["candidate_id"] for candidate in combo]
            items = []
            for source_name, candidate in zip(source_names, combo, strict=True):
                final_output = generated_outputs[(profile.name, source_name, candidate["candidate_id"])]
                items.append(build_selection_item(source_name, candidate, final_output, profile))
            write_selection(selection_root, f"{profile.name}_{'__'.join(ids)}", items)
            attack_variant_count += 1

    return {
        "source_work_root": str(source_work_root),
        "output_root": str(output_root),
        "sources": list(candidates_by_source),
        "candidates_per_source": {source: len(items) for source, items in candidates_by_source.items()},
        "attack_profiles": [profile.name for profile in ATTACK_PROFILES],
        "raw_variant_count": raw_variant_count,
        "attack_variant_count": attack_variant_count,
        "selection_root": str(selection_root),
        "postprocessed_root": str(postprocessed_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metric-focused anonymization variants from existing FreeVC candidates."
    )
    parser.add_argument(
        "--source-work-root",
        default=str(DEFAULT_SOURCE_WORK_ROOT),
        help="Existing work directory containing evaluated VC candidates.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for generated metric-attack variants.",
    )
    parser.add_argument(
        "--no-raw-combos",
        action="store_true",
        help="Only write postprocessed attack variants, not raw candidate combinations.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before generating variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_variants(
        source_work_root=Path(args.source_work_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        include_raw_combos=not args.no_raw_combos,
        clean_output=args.clean_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
