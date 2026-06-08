from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"


def build_status(source: Path, target: Path, output: Path, *, available: bool, message: str) -> dict:
    return {
        "source": str(source),
        "target": str(target),
        "output": str(output),
        "backend_available": available,
        "message": message,
        "model": DEFAULT_MODEL,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional pretrained VC anonymization backend.")
    parser.add_argument("--source", required=True, help="Source speech WAV.")
    parser.add_argument("--target", required=True, help="Target anonymous reference WAV.")
    parser.add_argument("--out", required=True, help="Output WAV path.")
    parser.add_argument(
        "--status-json",
        default="",
        help="Optional JSON file to record backend availability and execution status.",
    )
    return parser.parse_args()


def maybe_write_status(path: str, payload: dict) -> None:
    if not path:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()

    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        status = build_status(
            source,
            target,
            output,
            available=False,
            message=(
                "Pretrained VC backend unavailable in the current sandbox. "
                "The required `TTS` package is not installed, and package/model downloads are currently blocked. "
                f"Original import error: {exc}"
            ),
        )
        maybe_write_status(args.status_json, status)
        print(json.dumps(status, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    try:
        tts = TTS(model_name=DEFAULT_MODEL)
        output.parent.mkdir(parents=True, exist_ok=True)
        tts.voice_conversion_to_file(
            source_wav=str(source),
            target_wav=str(target),
            file_path=str(output),
        )
    except Exception as exc:
        status = build_status(
            source,
            target,
            output,
            available=True,
            message=f"Backend import succeeded but conversion failed: {exc}",
        )
        maybe_write_status(args.status_json, status)
        print(json.dumps(status, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3

    status = build_status(
        source,
        target,
        output,
        available=True,
        message="Conversion completed successfully.",
    )
    maybe_write_status(args.status_json, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
