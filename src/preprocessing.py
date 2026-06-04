# src/preprocessing.py

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import json

# Default input/output paths
DEFAULT_INPUT_PATH = "data/raw/sms_dataset.csv"
DEFAULT_OUTPUT_PATH = "data/processed/cleaned_sms_dataset.csv"

# Expected dataset columns based on the uploaded CSV
REQUIRED_COLUMNS = ["LABEL", "TEXT", "URL", "EMAIL", "PHONE"]

# Fixed label encoding for consistency across training/evaluation
LABEL_TO_ID = {
    "ham": 0,
    "spam": 1,
    "smishing": 2,
}

VALID_LABELS = set(LABEL_TO_ID.keys())


# Basic patterns for extracting message-level indicators
URL_PATTERN = re.compile(
    r"""
    \b(?:https?://|www\.)[^\s]+
    |
    \b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s]*)?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)"
)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier validation."""
    df = df.copy()
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def normalize_text(value: object) -> str:
    """Normalize raw SMS text without removing useful content."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Clean common encoding/noise issues
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufffd", " ")
    text = text.replace('""', '"')

    # Normalize spacing
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().strip('"').strip()


def count_valid_phones(text: str) -> int:
    """Count likely phone numbers based on digit length."""
    count = 0

    for match in PHONE_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", match.group())

        if 7 <= len(digits) <= 15:
            count += 1

    return count


def replace_phone_tokens(text: str) -> str:
    """Replace likely phone numbers with a stable token."""
    def replacer(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())

        if 7 <= len(digits) <= 15:
            return " phonetoken "

        return match.group()

    return PHONE_PATTERN.sub(replacer, text)


def clean_text_for_model(value: object) -> str:
    """Create cleaned text to be used later for TF-IDF."""
    text = normalize_text(value).lower()

    # Replace sensitive/variable patterns with stable tokens
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = replace_phone_tokens(text)

    # Keep letters, numbers, apostrophes, and spaces
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def standardize_label(value: object) -> str:
    """Convert label values into ham, spam, or smishing."""
    label = normalize_text(value).lower()
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")

    label_map = {
        "ham": "ham",
        "normal": "ham",
        "legitimate": "ham",
        "legit": "ham",
        "safe": "ham",

        "spam": "spam",
        "promo": "spam",
        "promotion": "spam",
        "advertisement": "spam",
        "marketing": "spam",

        "smishing": "smishing",
        "smish": "smishing",
        "phishing": "smishing",
        "phish": "smishing",
        "scam": "smishing",
        "fraud": "smishing",
    }

    if label not in label_map:
        raise ValueError(f"Unknown label found: {value}")

    return label_map[label]


def yes_no_to_binary(value: object) -> int:
    """Convert Yes/No indicator values into 1/0."""
    if pd.isna(value):
        return 0

    value = str(value).strip().lower()

    if value in {"yes", "y", "true", "1", "present"}:
        return 1

    if value in {"no", "n", "false", "0", "none", ""}:
        return 0

    return 0


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the uploaded dataset has the required columns."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load CSV or Excel dataset."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            "utf-8",
            b"",
            0,
            1,
            "Could not read CSV using supported encodings.",
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Only CSV and Excel files are supported.")


def safe_standardize_label(value: object) -> str | None:
    """Standardize labels without stopping the whole script."""
    try:
        return standardize_label(value)
    except ValueError:
        return None


def add_removal_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Append a removal reason to matching rows."""
    df.loc[mask, "removal_reason"] = df.loc[mask, "removal_reason"].apply(
        lambda current: f"{current}; {reason}" if current else reason
    )


def series_to_int_dict(series: pd.Series) -> dict:
    """Convert value counts to JSON-safe dictionary."""
    return {str(key): int(value) for key, value in series.items()}

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Main preprocessing pipeline with audit outputs."""
    raw_df = load_dataset(input_path)
    raw_df = normalize_column_names(raw_df)

    validate_columns(raw_df)

    df = pd.DataFrame()

    # Keep source references for audit/debugging
    df["source_index"] = raw_df.index
    df["csv_line"] = raw_df.index + 2
    df["source_LABEL"] = raw_df["LABEL"]
    df["source_TEXT"] = raw_df["TEXT"]
    df["source_URL"] = raw_df["URL"]
    df["source_EMAIL"] = raw_df["EMAIL"]
    df["source_PHONE"] = raw_df["PHONE"]

    # Core text and label fields
    df["message"] = raw_df["TEXT"].apply(normalize_text)
    df["clean_text"] = raw_df["TEXT"].apply(clean_text_for_model)
    df["label"] = raw_df["LABEL"].apply(safe_standardize_label)
    df["label_id"] = df["label"].map(LABEL_TO_ID)

    # Convert existing Yes/No indicators
    source_url = raw_df["URL"].apply(yes_no_to_binary)
    source_email = raw_df["EMAIL"].apply(yes_no_to_binary)
    source_phone = raw_df["PHONE"].apply(yes_no_to_binary)

    # Generate backup counts directly from text
    detected_url_count = df["message"].apply(
        lambda text: len(URL_PATTERN.findall(text))
    ).astype(int)

    detected_email_count = df["message"].apply(
        lambda text: len(EMAIL_PATTERN.findall(text))
    ).astype(int)

    detected_phone_count = df["message"].apply(count_valid_phones).astype(int)

    # Use source indicators, but also catch indicators found in text
    df["URL"] = np.maximum(source_url, (detected_url_count > 0).astype(int))
    df["EMAIL"] = np.maximum(source_email, (detected_email_count > 0).astype(int))
    df["PHONE"] = np.maximum(source_phone, (detected_phone_count > 0).astype(int))

    # Keep counts as optional structured features
    df["url_count"] = np.maximum(detected_url_count, df["URL"])
    df["email_count"] = np.maximum(detected_email_count, df["EMAIL"])
    df["phone_count"] = np.maximum(detected_phone_count, df["PHONE"])

    # Track why rows are removed
    df["removal_reason"] = ""

    add_removal_reason(
        df,
        df["message"].str.len() == 0,
        "empty_message",
    )

    add_removal_reason(
        df,
        df["clean_text"].str.len() == 0,
        "empty_clean_text",
    )

    add_removal_reason(
        df,
        df["label"].isna() | ~df["label"].isin(VALID_LABELS),
        "invalid_label",
    )

    # Only check conflicts among otherwise usable rows
    usable_mask = df["removal_reason"].eq("")

    conflicting_messages = (
        df[usable_mask]
        .groupby("message")["label"]
        .nunique()
        .loc[lambda x: x > 1]
        .index
    )

    add_removal_reason(
        df,
        usable_mask & df["message"].isin(conflicting_messages),
        "same_message_conflicting_labels",
    )

    # Separate kept and removed rows
    removed_df = df[~df["removal_reason"].eq("")].copy()
    cleaned_df = df[df["removal_reason"].eq("")].copy().reset_index(drop=True)

    cleaned_df["label_id"] = cleaned_df["label"].map(LABEL_TO_ID).astype(int)

    # Report possible duplicates but keep them
    duplicate_mask = cleaned_df.duplicated(subset=["clean_text"], keep=False)
    duplicate_report = cleaned_df[duplicate_mask].copy()
    duplicate_report["duplicate_group_size"] = 0

    if not duplicate_report.empty:
        duplicate_report["duplicate_group_size"] = (
            duplicate_report.groupby("clean_text")["clean_text"].transform("size")
        )

        duplicate_report = duplicate_report.sort_values(
            by=["clean_text", "label", "csv_line"]
        )

    output_columns = [
        "message",
        "clean_text",
        "label",
        "label_id",
        "URL",
        "EMAIL",
        "PHONE",
        "url_count",
        "email_count",
        "phone_count",
    ]

    output_df = cleaned_df[output_columns].copy()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    audit_dir = output_file.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_file
    removed_path = audit_dir / "preprocessing_removed_rows.csv"
    duplicate_path = audit_dir / "preprocessing_duplicate_clean_text_report.csv"
    summary_path = audit_dir / "preprocessing_summary.json"

    output_df.to_csv(cleaned_path, index=False, encoding="utf-8")

    removed_columns = [
        "csv_line",
        "source_LABEL",
        "source_TEXT",
        "source_URL",
        "source_EMAIL",
        "source_PHONE",
        "message",
        "clean_text",
        "label",
        "removal_reason",
    ]

    removed_df[removed_columns].to_csv(
        removed_path,
        index=False,
        encoding="utf-8",
    )

    duplicate_columns = [
        "csv_line",
        "source_LABEL",
        "source_TEXT",
        "message",
        "clean_text",
        "label",
        "URL",
        "EMAIL",
        "PHONE",
        "duplicate_group_size",
    ]

    duplicate_report[duplicate_columns].to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "input_rows": int(len(raw_df)),
        "rows_saved": int(len(output_df)),
        "rows_removed": int(len(removed_df)),
        "removed_by_reason": series_to_int_dict(
            removed_df["removal_reason"].value_counts().sort_index()
        ),
        "possible_duplicate_clean_text_rows_kept": int(duplicate_mask.sum()),
        "class_distribution_after_preprocessing": series_to_int_dict(
            output_df["label"].value_counts().sort_index()
        ),
        "indicator_totals": series_to_int_dict(
            output_df[["URL", "EMAIL", "PHONE"]].sum()
        ),
        "files": {
            "cleaned_dataset": str(cleaned_path),
            "removed_rows": str(removed_path),
            "duplicate_report": str(duplicate_path),
            "summary": str(summary_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("Preprocessing complete.")
    print(f"Input: {input_path}")
    print(f"Output: {cleaned_path}")
    print(f"Rows saved: {len(output_df)}")
    print(f"Rows removed: {len(removed_df)}")

    print("\nClass distribution:")
    print(output_df["label"].value_counts().sort_index())

    print("\nIndicator totals:")
    print(output_df[["URL", "EMAIL", "PHONE"]].sum())

    print("\nAudit files:")
    print(f"Removed rows: {removed_path}")
    print(f"Duplicate report: {duplicate_path}")
    print(f"Summary: {summary_path}")

    return output_df


def build_arg_parser() -> argparse.ArgumentParser:
    """Set command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw SMS dataset for SmishKaBa."
    )

    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to raw dataset. Default: {DEFAULT_INPUT_PATH}",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to cleaned dataset. Default: {DEFAULT_OUTPUT_PATH}",
    )

    return parser


def main() -> None:
    """Run preprocessing from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()