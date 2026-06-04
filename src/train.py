# src/train.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.features import (
    build_feature_transformer,
    prepare_feature_dataframe,
    prepare_target,
    validate_feature_columns,
)


DEFAULT_INPUT_PATH = "data/processed/cleaned_sms_dataset.csv"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SPLIT_DIR = "data/processed/splits"
DEFAULT_RESULTS_DIR = "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_CONFIGS = {
    "nb": {
        "filename": "nb_pipeline.joblib",
        "model": MultinomialNB(alpha=1.0),
    },
    "svm": {
        "filename": "svm_pipeline.joblib",
        "model": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
    },
    "mlr": {
        "filename": "mlr_pipeline.joblib",
        "model": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    },
}


def load_cleaned_dataset(input_path: str) -> pd.DataFrame:
    """Load preprocessed dataset."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")

    df = pd.read_csv(path)
    validate_feature_columns(df, require_target=True)

    return df


def get_class_distribution(y: pd.Series) -> dict[str, int]:
    """Return JSON-safe class counts."""
    return {str(label): int(count) for label, count in y.value_counts().sort_index().items()}


def group_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data while keeping duplicate clean_text groups together."""
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


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standard stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(model) -> Pipeline:
    """Create full feature + model pipeline."""
    feature_transformer = build_feature_transformer()

    return Pipeline(
        steps=[
            ("features", feature_transformer),
            ("model", model),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Train and save all model pipelines."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    for model_name, config in MODEL_CONFIGS.items():
        pipeline = build_pipeline(clone(config["model"]))
        pipeline.fit(X_train, y_train)

        model_path = artifacts_dir / config["filename"]
        joblib.dump(pipeline, model_path)

        saved_models[model_name] = str(model_path)
        print(f"Saved {model_name.upper()} model: {model_path}")

    return saved_models


def save_split_files(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split_dir: Path,
) -> tuple[Path, Path]:
    """Save train/test splits for evaluation."""
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["label"] = y_train.values
    test_df["label"] = y_test.values

    train_path = split_dir / "train_split.csv"
    test_path = split_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    return train_path, test_path


def save_training_summary(
    summary_path: Path,
    summary: dict,
) -> None:
    """Save training metadata."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


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
    input_path: str,
    artifacts_dir: str,
    split_dir: str,
    results_dir: str,
    split_strategy: str = "group",
    test_size: float = TEST_SIZE,
) -> None:
    """Run dataset split and model training."""
    df = load_cleaned_dataset(input_path)

    X = prepare_feature_dataframe(df)
    y = prepare_target(df, use_label_id=False)

    if split_strategy == "group":
        X_train, X_test, y_train, y_test = group_stratified_split(
            X=X,
            y=y,
            groups=df["clean_text"],
            test_size=test_size,
        )
    elif split_strategy == "stratified":
        X_train, X_test, y_train, y_test = stratified_split(
            X=X,
            y=y,
            test_size=test_size,
        )
    else:
        raise ValueError("split_strategy must be either 'group' or 'stratified'.")

    artifacts_path = Path(artifacts_dir)
    split_path = Path(split_dir)
    results_path = Path(results_dir)

    saved_models = train_models(X_train, y_train, artifacts_path)
    train_split_path, test_split_path = save_split_files(
        X_train,
        X_test,
        y_train,
        y_test,
        split_path,
    )

    train_groups = set(X_train["clean_text"])
    test_groups = set(X_test["clean_text"])
    overlapping_groups = train_groups.intersection(test_groups)

    summary_path = results_path / "training_summary.json"

    summary = {
        "input_dataset": input_path,
        "split_strategy": split_strategy,
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "overlapping_clean_text_groups": int(len(overlapping_groups)),
        "train_class_distribution": get_class_distribution(y_train),
        "test_class_distribution": get_class_distribution(y_test),
        "saved_models": saved_models,
        "files": {
            "train_split": str(train_split_path),
            "test_split": str(test_split_path),
            "training_summary": str(summary_path),
        },
    }

    save_training_summary(summary_path, summary)
    print_training_summary(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SmishKaBa NB, SVM, and MLR models."
    )

    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to cleaned dataset. Default: {DEFAULT_INPUT_PATH}",
    )

    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory for saved models. Default: {DEFAULT_ARTIFACTS_DIR}",
    )

    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help=f"Directory for train/test splits. Default: {DEFAULT_SPLIT_DIR}",
    )

    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for training summary. Default: {DEFAULT_RESULTS_DIR}",
    )

    parser.add_argument(
        "--split-strategy",
        choices=["group", "stratified"],
        default="group",
        help="Use 'group' to keep duplicate clean_text rows in the same split.",
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