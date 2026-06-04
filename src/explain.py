# src/explain.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

from src.features import get_feature_names, prepare_feature_dataframe, prepare_target, validate_feature_columns


DEFAULT_TRAIN_PATH = "data/processed/splits/train_split.csv"
DEFAULT_TEST_PATH = "data/processed/splits/test_split.csv"
DEFAULT_MODEL_PATH = "artifacts/mlr_pipeline.joblib"
DEFAULT_OUTPUT_DIR = "results/shap_outputs"

RANDOM_STATE = 42
BACKGROUND_SIZE = 200
EXPLAIN_SIZE = 500
LOCAL_ROWS = 20
LOCAL_TOP_N = 10
TARGET_CLASS = "smishing"

STRUCTURED_FEATURES = {"URL", "EMAIL", "PHONE", "url_count", "email_count", "phone_count"}


def load_split(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a saved train/test split."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    df = pd.read_csv(file_path)
    validate_feature_columns(df, require_target=True)

    return prepare_feature_dataframe(df), prepare_target(df, use_label_id=False)


def load_mlr_pipeline(model_path: str) -> Pipeline:
    """Load the trained MLR pipeline."""
    file_path = Path(model_path)

    if not file_path.exists():
        raise FileNotFoundError(f"MLR pipeline not found: {file_path}")

    pipeline = joblib.load(file_path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded model is not a scikit-learn Pipeline.")

    if "features" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain 'features' and 'model' steps.")

    return pipeline


def sample_rows(
    X: pd.DataFrame,
    y: pd.Series,
    size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample rows for faster SHAP processing."""
    if len(X) <= size:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    sampled_indices = X.sample(n=size, random_state=RANDOM_STATE).index

    return (
        X.loc[sampled_indices].reset_index(drop=True),
        y.loc[sampled_indices].reset_index(drop=True),
    )


def to_dense(matrix) -> np.ndarray:
    """Convert sparse matrix to dense array for SHAP."""
    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def clean_feature_name(feature_name: str) -> str:
    """Remove transformer prefixes."""
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def get_feature_type(feature_name: str) -> str:
    """Identify TF-IDF or structured feature."""
    cleaned = clean_feature_name(feature_name)

    if cleaned in STRUCTURED_FEATURES:
        return "structured"

    return "tfidf"


def normalize_shap_values(
    shap_values,
    class_labels: list[str],
    n_samples: int,
    n_features: int,
) -> dict[str, np.ndarray]:
    """Normalize SHAP output across SHAP versions."""
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


def build_global_importance(
    shap_by_class: dict[str, np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP values per class."""
    records = []

    for class_label, class_values in shap_by_class.items():
        mean_abs_values = np.abs(class_values).mean(axis=0)

        for feature_name, mean_abs_shap in zip(feature_names, mean_abs_values):
            records.append(
                {
                    "class": class_label,
                    "feature_name": clean_feature_name(feature_name),
                    "raw_feature_name": feature_name,
                    "feature_type": get_feature_type(feature_name),
                    "mean_abs_shap": float(mean_abs_shap),
                }
            )

    return pd.DataFrame(records).sort_values(
        by=["class", "mean_abs_shap"],
        ascending=[True, False],
    )


def build_local_explanations(
    pipeline: Pipeline,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    transformed_values: np.ndarray,
    shap_by_class: dict[str, np.ndarray],
    feature_names: list[str],
    target_class: str,
    local_rows: int,
    top_n: int,
) -> pd.DataFrame:
    """Build local SHAP explanations for target-class examples."""
    model = pipeline.named_steps["model"]

    predictions = pipeline.predict(X_explain)
    probabilities = pipeline.predict_proba(X_explain)

    if target_class not in model.classes_:
        raise ValueError(f"Target class not found in model classes: {target_class}")

    target_index = list(model.classes_).index(target_class)
    target_probabilities = probabilities[:, target_index]

    target_pred_indices = np.where(predictions == target_class)[0]

    if len(target_pred_indices) == 0:
        selected_indices = np.argsort(target_probabilities)[::-1][:local_rows]
    else:
        sorted_indices = target_pred_indices[
            np.argsort(target_probabilities[target_pred_indices])[::-1]
        ]
        selected_indices = sorted_indices[:local_rows]

    target_shap_values = shap_by_class[target_class]
    records = []

    for sample_rank, sample_index in enumerate(selected_indices, start=1):
        row_shap = target_shap_values[sample_index]
        top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]

        for feature_rank, feature_index in enumerate(top_indices, start=1):
            feature_name = feature_names[feature_index]

            records.append(
                {
                    "sample_rank": sample_rank,
                    "feature_rank": feature_rank,
                    "sample_index": int(sample_index),
                    "true_label": str(y_explain.iloc[sample_index]),
                    "predicted_label": str(predictions[sample_index]),
                    "target_class": target_class,
                    "target_probability": float(target_probabilities[sample_index]),
                    "clean_text": X_explain.iloc[sample_index]["clean_text"],
                    "feature_name": clean_feature_name(feature_name),
                    "raw_feature_name": feature_name,
                    "feature_type": get_feature_type(feature_name),
                    "feature_value": float(transformed_values[sample_index, feature_index]),
                    "shap_value": float(row_shap[feature_index]),
                }
            )

    return pd.DataFrame(records)


def save_target_plot(top_target_df: pd.DataFrame, output_path: Path) -> None:
    """Save top target-class SHAP bar plot."""
    plot_df = top_target_df.head(20).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature_name"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top SHAP Features for {TARGET_CLASS}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_shap_explanation(
    train_path: str,
    test_path: str,
    model_path: str,
    output_dir: str,
    background_size: int,
    explain_size: int,
    target_class: str,
    local_rows: int,
    local_top_n: int,
) -> None:
    """Run SHAP explanation workflow."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(train_path)
    X_test, y_test = load_split(test_path)
    pipeline = load_mlr_pipeline(model_path)

    X_background, _ = sample_rows(X_train, y_train, background_size)
    X_explain, y_explain = sample_rows(X_test, y_test, explain_size)

    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    background_matrix = to_dense(transformer.transform(X_background))
    explain_matrix = to_dense(transformer.transform(X_explain))

    feature_names = get_feature_names(transformer)
    class_labels = [str(label) for label in model.classes_]

    explainer = shap.LinearExplainer(model, background_matrix)
    raw_shap_values = explainer.shap_values(explain_matrix)

    shap_by_class = normalize_shap_values(
        shap_values=raw_shap_values,
        class_labels=class_labels,
        n_samples=explain_matrix.shape[0],
        n_features=explain_matrix.shape[1],
    )

    global_df = build_global_importance(shap_by_class, feature_names)

    target_global_df = global_df[global_df["class"] == target_class].copy()
    target_top_df = target_global_df.sort_values("mean_abs_shap", ascending=False)

    target_structured_df = target_top_df[
        target_top_df["feature_type"] == "structured"
    ].copy()

    local_df = build_local_explanations(
        pipeline=pipeline,
        X_explain=X_explain,
        y_explain=y_explain,
        transformed_values=explain_matrix,
        shap_by_class=shap_by_class,
        feature_names=feature_names,
        target_class=target_class,
        local_rows=local_rows,
        top_n=local_top_n,
    )

    global_path = output_path / "global_mean_abs_shap.csv"
    target_top_path = output_path / f"top_{target_class}_features.csv"
    structured_path = output_path / f"{target_class}_structured_feature_contributions.csv"
    local_path = output_path / f"local_{target_class}_explanations.csv"
    plot_path = output_path / f"top_{target_class}_features.png"
    summary_path = output_path / "shap_summary.json"

    global_df.to_csv(global_path, index=False, encoding="utf-8")
    target_top_df.to_csv(target_top_path, index=False, encoding="utf-8")
    target_structured_df.to_csv(structured_path, index=False, encoding="utf-8")
    local_df.to_csv(local_path, index=False, encoding="utf-8")
    save_target_plot(target_top_df, plot_path)

    summary = {
        "model_path": model_path,
        "train_path": train_path,
        "test_path": test_path,
        "target_class": target_class,
        "background_rows": int(len(X_background)),
        "explained_rows": int(len(X_explain)),
        "feature_count": int(len(feature_names)),
        "class_labels": class_labels,
        "files": {
            "global_mean_abs_shap": str(global_path),
            "target_top_features": str(target_top_path),
            "target_structured_contributions": str(structured_path),
            "local_explanations": str(local_path),
            "target_plot": str(plot_path),
            "summary": str(summary_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("SHAP explanation complete.")
    print(f"Target class: {target_class}")
    print(f"Explained rows: {len(X_explain)}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Outputs saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the MLR model."
    )

    parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train split. Default: {DEFAULT_TRAIN_PATH}",
    )

    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help=f"Path to test split. Default: {DEFAULT_TEST_PATH}",
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to MLR pipeline. Default: {DEFAULT_MODEL_PATH}",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for SHAP outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=BACKGROUND_SIZE,
        help=f"Background sample size. Default: {BACKGROUND_SIZE}",
    )

    parser.add_argument(
        "--explain-size",
        type=int,
        default=EXPLAIN_SIZE,
        help=f"Rows to explain. Default: {EXPLAIN_SIZE}",
    )

    parser.add_argument(
        "--target-class",
        default=TARGET_CLASS,
        help=f"Target class for focused outputs. Default: {TARGET_CLASS}",
    )

    parser.add_argument(
        "--local-rows",
        type=int,
        default=LOCAL_ROWS,
        help=f"Number of local examples. Default: {LOCAL_ROWS}",
    )

    parser.add_argument(
        "--local-top-n",
        type=int,
        default=LOCAL_TOP_N,
        help=f"Top features per local example. Default: {LOCAL_TOP_N}",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    run_shap_explanation(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        background_size=args.background_size,
        explain_size=args.explain_size,
        target_class=args.target_class,
        local_rows=args.local_rows,
        local_top_n=args.local_top_n,
    )


if __name__ == "__main__":
    main()