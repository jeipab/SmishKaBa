# src/features.py

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_COLUMN = "clean_text"

BINARY_FEATURE_COLUMNS = ["URL", "EMAIL", "PHONE"]
COUNT_FEATURE_COLUMNS = ["url_count", "email_count", "phone_count"]
NUMERIC_FEATURE_COLUMNS = BINARY_FEATURE_COLUMNS + COUNT_FEATURE_COLUMNS

TARGET_COLUMN = "label"
TARGET_ID_COLUMN = "label_id"

REQUIRED_FEATURE_COLUMNS = [TEXT_COLUMN] + NUMERIC_FEATURE_COLUMNS
REQUIRED_TRAINING_COLUMNS = REQUIRED_FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_feature_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """Check if required columns exist."""
    required_columns = REQUIRED_TRAINING_COLUMNS if require_target else REQUIRED_FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )


def prepare_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model input features."""
    validate_feature_columns(df, require_target=False)

    features = df[REQUIRED_FEATURE_COLUMNS].copy()

    # Text input for TF-IDF
    features[TEXT_COLUMN] = features[TEXT_COLUMN].fillna("").astype(str)

    # Structured numeric features
    for column in NUMERIC_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        ).fillna(0)

    return features


def prepare_target(df: pd.DataFrame, use_label_id: bool = False) -> pd.Series:
    """Prepare target labels."""
    target_column = TARGET_ID_COLUMN if use_label_id else TARGET_COLUMN

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    return df[target_column]


def build_feature_transformer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
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

    # Combine text vectorization with URL/EMAIL/PHONE features
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