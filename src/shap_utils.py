# src/shap_utils.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.config import NUMERIC_FEATURE_COLUMNS, RANDOM_STATE


STRUCTURED_FEATURES = set(NUMERIC_FEATURE_COLUMNS)


def to_dense(matrix) -> np.ndarray:
    """Convert sparse matrix to dense."""
    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def clean_feature_name(feature_name: str) -> str:
    """Remove transformer prefix from feature name."""
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def get_feature_type(feature_name: str) -> str:
    """Identify whether a feature is structured or TF-IDF."""
    cleaned = clean_feature_name(feature_name)
    return "structured" if cleaned in STRUCTURED_FEATURES else "tfidf"


def normalize_shap_values(
    shap_values,
    class_labels: list[str],
    n_samples: int,
    n_features: int,
) -> dict[str, np.ndarray]:
    """Normalize SHAP output shape for multiclass models."""
    n_classes = len(class_labels)

    if isinstance(shap_values, list):
        return {
            class_labels[index]: np.asarray(values)
            for index, values in enumerate(shap_values)
        }

    values = np.asarray(shap_values)

    if values.ndim == 2:
        if n_classes != 1:
            raise ValueError("Unexpected 2D SHAP output for multiclass model.")

        return {class_labels[0]: values}

    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    if values.shape == (n_samples, n_features, n_classes):
        values = np.transpose(values, (2, 0, 1))
    elif values.shape != (n_classes, n_samples, n_features):
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    return {
        class_labels[index]: values[index]
        for index in range(n_classes)
    }


def get_row_shap_values(shap_values, class_index: int) -> np.ndarray:
    """Extract one-row SHAP values for a selected class."""
    if isinstance(shap_values, list):
        return shap_values[class_index][0]

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3 and shap_array.shape[0] == 1:
        return shap_array[0, :, class_index]

    if shap_array.ndim == 3:
        return shap_array[class_index, 0, :]

    return shap_array[0]


def sample_dataframe_rows(
    X: pd.DataFrame,
    y: pd.Series,
    size: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample rows for faster SHAP processing."""
    if len(X) <= size:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    sampled_indices = X.sample(n=size, random_state=random_state).index

    return (
        X.loc[sampled_indices].reset_index(drop=True),
        y.loc[sampled_indices].reset_index(drop=True),
    )
