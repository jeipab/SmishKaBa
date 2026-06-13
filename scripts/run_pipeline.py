#!/usr/bin/env python
"""Run the SmishKaBa pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    ("preprocessing", ["-m", "src.preprocessing"]),
    ("train", ["-m", "src.train"]),
    ("evaluate", ["-m", "src.evaluate"]),
    ("explain", ["-m", "src.explain"]),
    ("statistics", ["-m", "src.statistics"]),
    ("research_report", ["-m", "src.research_report"]),
]

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "shap": "shap",
    "matplotlib": "matplotlib",
}


def check_dependencies() -> None:
    """Fail fast when core packages are missing in the active environment."""
    missing = []

    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required packages in the current Python environment: "
            f"{packages}\n\n"
            "Activate your virtual environment, then run:\n"
            "  python -m pip install -r requirements.txt"
        )


def run_step(name: str, command_args: list[str]) -> None:
    """Run one pipeline step and stop on failure."""
    print(f"\n=== Running {name} ===")
    command = [sys.executable, *command_args]
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
    check_dependencies()

    for step_name, command_args in PIPELINE_STEPS:
        if getattr(args, f"skip_{step_name}", False):
            print(f"Skipping {step_name}")
            continue

        run_step(step_name, command_args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
