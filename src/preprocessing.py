# src/preprocessing.py

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    LABEL_TO_ID,
    PREPROCESSING_DUPLICATE_REPORT_PATH,
    PREPROCESSING_REMOVED_ROWS_PATH,
    PREPROCESSING_SUMMARY_PATH,
    RAW_DATA_PATH,
)
from src.utils import save_json, series_to_int_dict, to_relative_path


REQUIRED_COLUMNS = ["LABEL", "TEXT", "URL", "EMAIL", "PHONE"]
VALID_LABELS = set(LABEL_TO_ID.keys())

LABEL_ALIASES = {
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

TRUE_VALUES = {"yes", "y", "true", "1", "present"}
FALSE_VALUES = {"no", "n", "false", "0", "none", ""}

OUTPUT_COLUMNS = [
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

REMOVED_COLUMNS = [
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

DUPLICATE_COLUMNS = [
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
    """Standardize column names."""
    df = df.copy()
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def normalize_text(value: object) -> str:
    """Normalize raw SMS text."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufffd", " ")
    text = text.replace('""', '"')
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().strip('"').strip()


def standardize_label(value: object) -> str:
    """Map labels to ham, spam, or smishing."""
    label = normalize_text(value).lower()
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")

    if label not in LABEL_ALIASES:
        raise ValueError(f"Unknown label found: {value}")

    return LABEL_ALIASES[label]


def safe_standardize_label(value: object) -> str | None:
    """Return None for invalid labels."""
    try:
        return standardize_label(value)
    except ValueError:
        return None


def yes_no_to_binary(value: object) -> int:
    """Convert Yes/No-like values to 1/0."""
    if pd.isna(value):
        return 0

    value = str(value).strip().lower()

    if value in TRUE_VALUES:
        return 1

    if value in FALSE_VALUES:
        return 0

    return 0


def count_urls(text: str) -> int:
    """Count URLs while avoiding email domains."""
    text = EMAIL_PATTERN.sub(" ", normalize_text(text))
    return len(URL_PATTERN.findall(text))


def count_emails(text: str) -> int:
    """Count email addresses."""
    return len(EMAIL_PATTERN.findall(normalize_text(text)))


def count_valid_phones(text: str) -> int:
    """Count likely phone numbers."""
    count = 0

    for match in PHONE_PATTERN.finditer(normalize_text(text)):
        digits = re.sub(r"\D", "", match.group())

        if 7 <= len(digits) <= 15:
            count += 1

    return count


def replace_phone_tokens(text: str) -> str:
    """Replace likely phone numbers with a token."""
    def replacer(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        return " phonetoken " if 7 <= len(digits) <= 15 else match.group()

    return PHONE_PATTERN.sub(replacer, text)


def clean_text_for_model(value: object) -> str:
    """Create cleaned text for TF-IDF."""
    text = normalize_text(value).lower()
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = replace_phone_tokens(text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def validate_columns(df: pd.DataFrame) -> None:
    """Check required columns."""
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
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
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


def add_removal_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Append removal reason to matching rows."""
    df.loc[mask, "removal_reason"] = df.loc[mask, "removal_reason"].apply(
        lambda current: f"{current}; {reason}" if current else reason
    )


def build_working_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build cleaned fields and feature-ready columns."""
    df = pd.DataFrame()

    # Source fields for audit tracking
    df["source_index"] = raw_df.index
    df["csv_line"] = raw_df.index + 2
    df["source_LABEL"] = raw_df["LABEL"]
    df["source_TEXT"] = raw_df["TEXT"]
    df["source_URL"] = raw_df["URL"]
    df["source_EMAIL"] = raw_df["EMAIL"]
    df["source_PHONE"] = raw_df["PHONE"]

    # Core fields
    df["message"] = raw_df["TEXT"].apply(normalize_text)
    df["clean_text"] = raw_df["TEXT"].apply(clean_text_for_model)
    df["label"] = raw_df["LABEL"].apply(safe_standardize_label)
    df["label_id"] = df["label"].map(LABEL_TO_ID)

    # Original indicators
    source_url = raw_df["URL"].apply(yes_no_to_binary)
    source_email = raw_df["EMAIL"].apply(yes_no_to_binary)
    source_phone = raw_df["PHONE"].apply(yes_no_to_binary)

    # Detected indicators from text
    detected_url_count = df["message"].apply(count_urls).astype(int)
    detected_email_count = df["message"].apply(count_emails).astype(int)
    detected_phone_count = df["message"].apply(count_valid_phones).astype(int)

    # Final binary indicators
    df["URL"] = np.maximum(source_url, (detected_url_count > 0).astype(int))
    df["EMAIL"] = np.maximum(source_email, (detected_email_count > 0).astype(int))
    df["PHONE"] = np.maximum(source_phone, (detected_phone_count > 0).astype(int))

    # Count features
    df["url_count"] = np.maximum(detected_url_count, df["URL"])
    df["email_count"] = np.maximum(detected_email_count, df["EMAIL"])
    df["phone_count"] = np.maximum(detected_phone_count, df["PHONE"])

    return df


def mark_removed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Tag unusable or conflicting rows."""
    df = df.copy()
    df["removal_reason"] = ""

    add_removal_reason(df, df["message"].str.len() == 0, "empty_message")
    add_removal_reason(df, df["clean_text"].str.len() == 0, "empty_clean_text")
    add_removal_reason(df, df["label"].isna() | ~df["label"].isin(VALID_LABELS), "invalid_label")

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

    return df


def build_duplicate_report(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Report duplicate clean_text rows but keep them."""
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

    return duplicate_report, int(duplicate_mask.sum())


def save_outputs(
    output_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    summary: dict,
    cleaned_path: Path,
    removed_path: Path,
    duplicate_path: Path,
    summary_path: Path,
) -> None:
    """Save cleaned data and audit files."""
    output_df.to_csv(cleaned_path, index=False, encoding="utf-8")
    removed_df[REMOVED_COLUMNS].to_csv(removed_path, index=False, encoding="utf-8")
    duplicate_report[DUPLICATE_COLUMNS].to_csv(duplicate_path, index=False, encoding="utf-8")
    save_json(summary, summary_path)


def print_summary(
    input_path: str,
    cleaned_path: Path,
    output_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    removed_path: Path,
    duplicate_path: Path,
    summary_path: Path,
) -> None:
    """Print compact preprocessing report."""
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


def preprocess_dataset(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Run preprocessing and audit export."""
    raw_df = normalize_column_names(load_dataset(input_path))
    validate_columns(raw_df)

    df = build_working_dataframe(raw_df)
    df = mark_removed_rows(df)

    removed_df = df[~df["removal_reason"].eq("")].copy()
    cleaned_df = df[df["removal_reason"].eq("")].copy().reset_index(drop=True)
    cleaned_df["label_id"] = cleaned_df["label"].map(LABEL_TO_ID).astype(int)

    duplicate_report, duplicate_count = build_duplicate_report(cleaned_df)
    output_df = cleaned_df[OUTPUT_COLUMNS].copy()

    cleaned_path = Path(output_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    removed_path = PREPROCESSING_REMOVED_ROWS_PATH
    duplicate_path = PREPROCESSING_DUPLICATE_REPORT_PATH
    summary_path = PREPROCESSING_SUMMARY_PATH

    removed_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_rows": int(len(raw_df)),
        "rows_saved": int(len(output_df)),
        "rows_removed": int(len(removed_df)),
        "removed_by_reason": series_to_int_dict(
            removed_df["removal_reason"].value_counts().sort_index()
        ),
        "possible_duplicate_clean_text_rows_kept": duplicate_count,
        "class_distribution_after_preprocessing": series_to_int_dict(
            output_df["label"].value_counts().sort_index()
        ),
        "indicator_totals": series_to_int_dict(
            output_df[["URL", "EMAIL", "PHONE"]].sum()
        ),
        "files": {
            "cleaned_dataset": to_relative_path(cleaned_path),
            "removed_rows": to_relative_path(removed_path),
            "duplicate_report": to_relative_path(duplicate_path),
            "summary": to_relative_path(summary_path),
        },
    }

    save_outputs(
        output_df,
        removed_df,
        duplicate_report,
        summary,
        cleaned_path,
        removed_path,
        duplicate_path,
        summary_path,
    )

    print_summary(
        input_path,
        cleaned_path,
        output_df,
        removed_df,
        removed_path,
        duplicate_path,
        summary_path,
    )

    return output_df


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw SMS dataset for SmishKaBa."
    )

    parser.add_argument(
        "-i",
        "--input",
        default=str(RAW_DATA_PATH),
        help=f"Path to raw dataset. Default: {RAW_DATA_PATH}",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=str(CLEANED_DATA_PATH),
        help=f"Path to cleaned dataset. Default: {CLEANED_DATA_PATH}",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    preprocess_dataset(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()