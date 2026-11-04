#!/usr/bin/env python3
"""Pretty-print newly committed High-to-Low handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="The same output_root passed to high_system.launch.py",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    return parser.parse_args()


def handoff_files(root: Path) -> list[Path]:
    values: list[Path] = []
    for path in root.glob("run-*/task-*/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema_version") in {
            "flzat.high_handoff.v1", "flzat.high_handoff.v2"
        }:
            values.append(path)
    return sorted(values, key=lambda item: item.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0.0:
        raise SystemExit("--poll-seconds must be positive")
    root = args.output_root.expanduser().resolve()
    seen = set(handoff_files(root))
    print(f"Watching completed handoffs under {root}", flush=True)
    try:
        while True:
            for path in handoff_files(root):
                if path in seen:
                    continue
                seen.add(path)
                value = json.loads(path.read_text(encoding="utf-8"))
                print("\n========== HIGH → LOW HANDOFF ==========", flush=True)
                print(f"Path: {path}", flush=True)
                print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)
                print("========================================\n", flush=True)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
