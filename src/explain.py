# src/explain.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import (
    MLR_PIPELINE_PATH,
    SHAP_BACKGROUND_SIZE,
    SHAP_EXPLAIN_SIZE,
    SHAP_LOCAL_ROWS,
    SHAP_LOCAL_TOP_N,
    SHAP_OUTPUTS_DIR,
    SHAP_TARGET_CLASS,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
)
from src.features import (
    get_feature_names,
    prepare_feature_dataframe,
    prepare_target,
    validate_feature_columns,
)
from src.shap_utils import (
    clean_feature_name,
    get_feature_type,
    normalize_shap_values,
    sample_dataframe_rows,
    to_dense,
)
from src.utils import ensure_dir, require_file, save_json, to_relative_path


def load_split(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load saved split."""
    file_path = require_file(path, "Split file")
    df = pd.read_csv(file_path)
    validate_feature_columns(df, require_target=True)

    return prepare_feature_dataframe(df), prepare_target(df, use_label_id=False)


def load_mlr_pipeline(model_path: str | Path) -> Pipeline:
    """Load trained MLR pipeline."""
    file_path = require_file(model_path, "MLR pipeline")
    pipeline = joblib.load(file_path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded model is not a scikit-learn Pipeline.")

    if "features" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain 'features' and 'model' steps.")

    return pipeline


def build_global_importance(
    shap_by_class: dict[str, np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP per class."""
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
    """Build local explanations for target-class examples."""
    model = pipeline.named_steps["model"]

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("MLR pipeline must support predict_proba for local output.")

    predictions = pipeline.predict(X_explain)
    probabilities = pipeline.predict_proba(X_explain)

    if target_class not in model.classes_:
        raise ValueError(f"Target class not found: {target_class}")

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


def save_target_plot(
    top_target_df: pd.DataFrame,
    output_path: Path,
    target_class: str,
) -> None:
    """Save target-class SHAP bar plot."""
    plot_df = top_target_df.head(20).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature_name"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top SHAP Features for {target_class}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_output_paths(output_dir: str | Path, target_class: str) -> dict[str, Path]:
    """Prepare SHAP output paths."""
    output_path = ensure_dir(output_dir)

    return {
        "global": output_path / "global_mean_abs_shap.csv",
        "target_top": output_path / f"top_{target_class}_features.csv",
        "structured": output_path / f"{target_class}_structured_feature_contributions.csv",
        "local": output_path / f"local_{target_class}_explanations.csv",
        "plot": output_path / f"top_{target_class}_features.png",
        "summary": output_path / "shap_summary.json",
    }


def run_shap_explanation(
    train_path: str | Path,
    test_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    background_size: int,
    explain_size: int,
    target_class: str,
    local_rows: int,
    local_top_n: int,
) -> None:
    """Run SHAP explanation workflow."""
    X_train, y_train = load_split(train_path)
    X_test, y_test = load_split(test_path)
    pipeline = load_mlr_pipeline(model_path)

    X_background, _ = sample_dataframe_rows(X_train, y_train, background_size)
    X_explain, y_explain = sample_dataframe_rows(X_test, y_test, explain_size)

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

    paths = build_output_paths(output_dir, target_class)

    global_df.to_csv(paths["global"], index=False, encoding="utf-8")
    target_top_df.to_csv(paths["target_top"], index=False, encoding="utf-8")
    target_structured_df.to_csv(paths["structured"], index=False, encoding="utf-8")
    local_df.to_csv(paths["local"], index=False, encoding="utf-8")
    save_target_plot(target_top_df, paths["plot"], target_class)

    summary = {
        "model_path": to_relative_path(model_path),
        "train_path": to_relative_path(train_path),
        "test_path": to_relative_path(test_path),
        "target_class": target_class,
        "background_rows": int(len(X_background)),
        "explained_rows": int(len(X_explain)),
        "feature_count": int(len(feature_names)),
        "class_labels": class_labels,
        "files": {
            "global_mean_abs_shap": to_relative_path(paths["global"]),
            "target_top_features": to_relative_path(paths["target_top"]),
            "target_structured_contributions": to_relative_path(paths["structured"]),
            "local_explanations": to_relative_path(paths["local"]),
            "target_plot": to_relative_path(paths["plot"]),
            "summary": to_relative_path(paths["summary"]),
        },
    }

    save_json(summary, paths["summary"])

    print("SHAP explanation complete.")
    print(f"Target class: {target_class}")
    print(f"Explained rows: {len(X_explain)}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Outputs saved to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the MLR model."
    )

    parser.add_argument(
        "--train-path",
        default=str(TRAIN_SPLIT_PATH),
        help=f"Path to train split. Default: {TRAIN_SPLIT_PATH}",
    )

    parser.add_argument(
        "--test-path",
        default=str(TEST_SPLIT_PATH),
        help=f"Path to test split. Default: {TEST_SPLIT_PATH}",
    )

    parser.add_argument(
        "--model-path",
        default=str(MLR_PIPELINE_PATH),
        help=f"Path to MLR pipeline. Default: {MLR_PIPELINE_PATH}",
    )

    parser.add_argument(
        "--output-dir",
        default=str(SHAP_OUTPUTS_DIR),
        help=f"Directory for SHAP outputs. Default: {SHAP_OUTPUTS_DIR}",
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=SHAP_BACKGROUND_SIZE,
        help=f"Background sample size. Default: {SHAP_BACKGROUND_SIZE}",
    )

    parser.add_argument(
        "--explain-size",
        type=int,
        default=SHAP_EXPLAIN_SIZE,
        help=f"Rows to explain. Default: {SHAP_EXPLAIN_SIZE}",
    )

    parser.add_argument(
        "--target-class",
        default=SHAP_TARGET_CLASS,
        help=f"Target class. Default: {SHAP_TARGET_CLASS}",
    )

    parser.add_argument(
        "--local-rows",
        type=int,
        default=SHAP_LOCAL_ROWS,
        help=f"Number of local examples. Default: {SHAP_LOCAL_ROWS}",
    )

    parser.add_argument(
        "--local-top-n",
        type=int,
        default=SHAP_LOCAL_TOP_N,
        help=f"Top features per local example. Default: {SHAP_LOCAL_TOP_N}",
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