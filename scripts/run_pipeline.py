#!/usr/bin/env python
"""Run the SmishKaBa pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    ("preprocessing", ["python", "-m", "src.preprocessing"]),
    ("train", ["python", "-m", "src.train"]),
    ("evaluate", ["python", "-m", "src.evaluate"]),
    ("explain", ["python", "-m", "src.explain"]),
    ("statistics", ["python", "-m", "src.statistics"]),
    ("research_report", ["python", "-m", "src.research_report"]),
]


def run_step(name: str, command: list[str]) -> None:
    """Run one pipeline step and stop on failure."""
    print(f"\n=== Running {name} ===")
    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise SystemExit(f"Pipeline failed at step: {name}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the full SmishKaBa pipeline."
    )

    for step_name, _ in PIPELINE_STEPS:
        parser.add_argument(
            f"--skip-{step_name.replace('_', '-')}",
            dest=f"skip_{step_name}",
            action="store_true",
            help=f"Skip the {step_name} step.",
        )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    for step_name, command in PIPELINE_STEPS:
        if getattr(args, f"skip_{step_name}", False):
            print(f"Skipping {step_name}")
            continue

        run_step(step_name, command)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
