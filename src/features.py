# src/features.py

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_ID_COLUMN,
    TEXT_COLUMN,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)


REQUIRED_FEATURE_COLUMNS = FEATURE_COLUMNS
REQUIRED_TRAINING_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_feature_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """Check required columns."""
    required = REQUIRED_TRAINING_COLUMNS if require_target else REQUIRED_FEATURE_COLUMNS
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {required}"
        )


def prepare_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model input features."""
    validate_feature_columns(df, require_target=False)

    features = df[REQUIRED_FEATURE_COLUMNS].copy()

    # Text column for TF-IDF
    features[TEXT_COLUMN] = features[TEXT_COLUMN].fillna("").astype(str)

    # Numeric URL/EMAIL/PHONE features
    for column in NUMERIC_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).fillna(0)

    return features


def prepare_target(df: pd.DataFrame, use_label_id: bool = False) -> pd.Series:
    """Prepare target labels."""
    target_column = TARGET_ID_COLUMN if use_label_id else TARGET_COLUMN

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    return df[target_column]


def build_feature_transformer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
) -> ColumnTransformer:
    """Build TF-IDF + numeric feature transformer."""
    tfidf = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    return ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, TEXT_COLUMN),
            ("numeric", "passthrough", NUMERIC_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )


def get_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Get all fitted feature names."""
    if not hasattr(transformer, "get_feature_names_out"):
        raise ValueError("Transformer does not expose feature names.")

    return transformer.get_feature_names_out().tolist()


def get_text_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Get fitted TF-IDF feature names."""
    tfidf = transformer.named_transformers_.get("tfidf")

    if tfidf is None or not hasattr(tfidf, "get_feature_names_out"):
        raise ValueError("TF-IDF transformer is not fitted or unavailable.")

    return tfidf.get_feature_names_out().tolist()


def get_numeric_feature_names() -> list[str]:
    """Get structured feature names."""
    return NUMERIC_FEATURE_COLUMNS.copy()


def check_no_missing_features(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> None:
    """Check selected feature columns for missing values."""
    selected_columns = list(columns) if columns is not None else REQUIRED_FEATURE_COLUMNS
    missing_counts = df[selected_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(f"Missing feature values found:\n{missing_counts}")