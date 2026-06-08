from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recording_demo_ui import (
    DEFAULT_TARGET_POOL_CONFIG,
    DEFAULT_WORK_ROOT,
    DEFAULT_VC_PYTHON,
    PROJECT_ROOT,
    DemoConfig,
    process_recording,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Web UI privacy optimizer on one local audio file.")
    parser.add_argument("input", help="Input recording path.")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT), help="Output work directory.")
    parser.add_argument("--target-pool-config", default=str(DEFAULT_TARGET_POOL_CONFIG), help="Target-pool config.")
    parser.add_argument("--max-targets", type=int, default=5, help="Maximum target references evaluated with FreeVC.")
    parser.add_argument("--vc-python", default=str(DEFAULT_VC_PYTHON), help="Python executable for FreeVC.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root).expanduser().resolve()
    session_dir = work_root / ("batch_" + time.strftime("%Y%m%d_%H%M%S"))
    config = DemoConfig(
        project_root=PROJECT_ROOT,
        work_root=work_root,
        vc_python=Path(args.vc_python).expanduser().resolve(),
        host="127.0.0.1",
        port=0,
        target_pool_config=Path(args.target_pool_config).expanduser().resolve(),
        max_targets=args.max_targets,
    )
    work_root.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    summary = process_recording(Path(args.input).expanduser().resolve(), session_dir, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
