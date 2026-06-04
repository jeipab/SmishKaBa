# src/utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for a file path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary as JSON."""
    file_path = ensure_parent_dir(path)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def series_to_int_dict(series: pd.Series) -> dict[str, int]:
    """Convert value counts to JSON-safe dict."""
    return {str(key): int(value) for key, value in series.items()}


def require_file(path: str | Path, label: str = "File") -> Path:
    """Raise error if file does not exist."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")

    return file_path


def print_section(title: str) -> None:
    """Print simple console section title."""
    print(f"\n{title}")
    print("-" * len(title))