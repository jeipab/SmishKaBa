# src/train.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    ARTIFACTS_DIR,
    CLEANED_DATA_PATH,
    DEFAULT_SPLIT_STRATEGY,
    MLR_PIPELINE_PATH,
    NB_PIPELINE_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    SPLIT_DIR,
    SVM_PIPELINE_PATH,
    TEST_SIZE,
    TRAINING_SUMMARY_PATH,
)
from src.features import (
    build_feature_transformer,
    prepare_feature_dataframe,
    prepare_target,
    validate_feature_columns,
)
from src.utils import ensure_dir, require_file, save_json, series_to_int_dict, to_relative_path


MODEL_CONFIGS = {
    "nb": {
        "path": NB_PIPELINE_PATH,
        "model": MultinomialNB(alpha=1.0),
    },
    "svm": {
        "path": SVM_PIPELINE_PATH,
        "model": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
    },
    "mlr": {
        "path": MLR_PIPELINE_PATH,
        "model": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    },
}


def load_cleaned_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load preprocessed dataset."""
    path = require_file(input_path, "Cleaned dataset")
    df = pd.read_csv(path)
    validate_feature_columns(df, require_target=True)
    return df


def build_pipeline(model) -> Pipeline:
    """Create feature + model pipeline."""
    return Pipeline(
        steps=[
            ("features", build_feature_transformer()),
            ("model", model),
        ]
    )


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run standard stratified split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def group_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Keep duplicate clean_text groups in the same split."""
    n_splits = round(1 / test_size)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def split_dataset(
    df: pd.DataFrame,
    split_strategy: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare X/y and apply selected split strategy."""
    X = prepare_feature_dataframe(df)
    y = prepare_target(df, use_label_id=False)

    if split_strategy == "group":
        return group_stratified_split(
            X=X,
            y=y,
            groups=df["clean_text"],
            test_size=test_size,
        )

    if split_strategy == "stratified":
        return stratified_split(X=X, y=y, test_size=test_size)

    raise ValueError("split_strategy must be either 'group' or 'stratified'.")


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts_dir: str | Path,
) -> dict[str, str]:
    """Train and save all models."""
    ensure_dir(artifacts_dir)
    saved_models = {}

    for model_name, config in MODEL_CONFIGS.items():
        model_path = Path(config["path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline = build_pipeline(clone(config["model"]))
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, model_path)
        saved_models[model_name] = to_relative_path(model_path)

        print(f"Saved {model_name.upper()}: {model_path}")

    return saved_models


def save_split_files(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split_dir: str | Path,
) -> tuple[Path, Path]:
    """Save train/test splits."""
    split_path = ensure_dir(split_dir)

    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["label"] = y_train.values
    test_df["label"] = y_test.values

    train_path = split_path / "train_split.csv"
    test_path = split_path / "test_split.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    return train_path, test_path


def count_overlapping_groups(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> int:
    """Count clean_text groups shared by train/test."""
    train_groups = set(X_train["clean_text"])
    test_groups = set(X_test["clean_text"])
    return len(train_groups.intersection(test_groups))


def build_training_summary(
    input_path: str | Path,
    split_strategy: str,
    test_size: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    saved_models: dict[str, str],
    train_split_path: Path,
    test_split_path: Path,
    summary_path: str | Path,
) -> dict:
    """Build training metadata."""
    return {
        "input_dataset": to_relative_path(input_path),
        "split_strategy": split_strategy,
        "test_size": float(test_size),
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "overlapping_clean_text_groups": count_overlapping_groups(X_train, X_test),
        "train_class_distribution": series_to_int_dict(
            y_train.value_counts().sort_index()
        ),
        "test_class_distribution": series_to_int_dict(
            y_test.value_counts().sort_index()
        ),
        "saved_models": saved_models,
        "files": {
            "train_split": to_relative_path(train_split_path),
            "test_split": to_relative_path(test_split_path),
            "training_summary": to_relative_path(summary_path),
        },
    }


def print_training_summary(summary: dict) -> None:
    """Print compact training report."""
    print("\nTraining complete.")
    print(f"Split strategy: {summary['split_strategy']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"Overlapping clean_text groups: {summary['overlapping_clean_text_groups']}")

    print("\nTrain distribution:")
    print(summary["train_class_distribution"])

    print("\nTest distribution:")
    print(summary["test_class_distribution"])

    print("\nSaved models:")
    for model_name, model_path in summary["saved_models"].items():
        print(f"{model_name.upper()}: {model_path}")

    print(f"\nSummary: {summary['files']['training_summary']}")


def train_from_cleaned_dataset(
    input_path: str | Path,
    artifacts_dir: str | Path,
    split_dir: str | Path,
    results_dir: str | Path,
    split_strategy: str,
    test_size: float,
) -> None:
    """Run splitting, training, and export."""
    df = load_cleaned_dataset(input_path)

    X_train, X_test, y_train, y_test = split_dataset(
        df=df,
        split_strategy=split_strategy,
        test_size=test_size,
    )

    saved_models = train_models(
        X_train=X_train,
        y_train=y_train,
        artifacts_dir=artifacts_dir,
    )

    train_split_path, test_split_path = save_split_files(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        split_dir=split_dir,
    )

    ensure_dir(results_dir)
    summary_path = TRAINING_SUMMARY_PATH

    summary = build_training_summary(
        input_path=input_path,
        split_strategy=split_strategy,
        test_size=test_size,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        saved_models=saved_models,
        train_split_path=train_split_path,
        test_split_path=test_split_path,
        summary_path=summary_path,
    )

    save_json(summary, summary_path)
    print_training_summary(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SmishKaBa NB, SVM, and MLR models."
    )

    parser.add_argument(
        "-i",
        "--input",
        default=str(CLEANED_DATA_PATH),
        help=f"Path to cleaned dataset. Default: {CLEANED_DATA_PATH}",
    )

    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"Directory for saved models. Default: {ARTIFACTS_DIR}",
    )

    parser.add_argument(
        "--split-dir",
        default=str(SPLIT_DIR),
        help=f"Directory for train/test splits. Default: {SPLIT_DIR}",
    )

    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Directory for training summary. Default: {RESULTS_DIR}",
    )

    parser.add_argument(
        "--split-strategy",
        choices=["group", "stratified"],
        default=DEFAULT_SPLIT_STRATEGY,
        help="Use group to keep duplicate clean_text rows together.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Test size. Default: {TEST_SIZE}",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    train_from_cleaned_dataset(
        input_path=args.input,
        artifacts_dir=args.artifacts_dir,
        split_dir=args.split_dir,
        results_dir=args.results_dir,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()