# src/evaluate.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from src.features import prepare_feature_dataframe, prepare_target, validate_feature_columns


DEFAULT_TEST_PATH = "data/processed/splits/test_split.csv"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_RESULTS_DIR = "results"

MODEL_FILES = {
    "nb": "nb_pipeline.joblib",
    "svm": "svm_pipeline.joblib",
    "mlr": "mlr_pipeline.joblib",
}

LABELS = ["ham", "spam", "smishing"]


def load_test_data(test_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load saved test split."""
    path = Path(test_path)

    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")

    df = pd.read_csv(path)
    validate_feature_columns(df, require_target=True)

    X_test = prepare_feature_dataframe(df)
    y_test = prepare_target(df, use_label_id=False)

    return X_test, y_test


def load_models(artifacts_dir: str) -> dict[str, object]:
    """Load trained model pipelines."""
    artifacts_path = Path(artifacts_dir)
    models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = artifacts_path / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        models[model_name] = joblib.load(model_path)

    return models


def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: Path,
    matrices_dir: Path,
) -> dict:
    """Evaluate one model and save detailed outputs."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = reports_dir / f"{model_name}_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8")

    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)

    matrix_csv_path = matrices_dir / f"{model_name}_confusion_matrix.csv"
    matrix_png_path = matrices_dir / f"{model_name}_confusion_matrix.png"

    matrix_df.to_csv(matrix_csv_path, encoding="utf-8")

    # Save confusion matrix figure
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot(values_format="d")
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(matrix_png_path, dpi=300)
    plt.close()

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "ham_precision": float(report_dict["ham"]["precision"]),
        "ham_recall": float(report_dict["ham"]["recall"]),
        "ham_f1": float(report_dict["ham"]["f1-score"]),
        "spam_precision": float(report_dict["spam"]["precision"]),
        "spam_recall": float(report_dict["spam"]["recall"]),
        "spam_f1": float(report_dict["spam"]["f1-score"]),
        "smishing_precision": float(report_dict["smishing"]["precision"]),
        "smishing_recall": float(report_dict["smishing"]["recall"]),
        "smishing_f1": float(report_dict["smishing"]["f1-score"]),
        "classification_report": str(report_path),
        "confusion_matrix_csv": str(matrix_csv_path),
        "confusion_matrix_png": str(matrix_png_path),
    }


def save_evaluation_summary(
    summary: dict,
    comparison_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Save comparison table and JSON summary."""
    comparison_path = results_dir / "model_comparison.csv"
    summary_path = results_dir / "evaluation_summary.json"

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")

    summary["files"] = {
        "model_comparison": str(comparison_path),
        "evaluation_summary": str(summary_path),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def print_evaluation_summary(comparison_df: pd.DataFrame, results_dir: Path) -> None:
    """Print compact evaluation report."""
    print("Evaluation complete.")

    print("\nModel comparison:")
    print(
        comparison_df[
            [
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ]
        ].round(4)
    )

    print(f"\nResults saved to: {results_dir}")


def evaluate_saved_models(
    test_path: str,
    artifacts_dir: str,
    results_dir: str,
) -> None:
    """Run evaluation for all saved models."""
    results_path = Path(results_dir)
    reports_dir = results_path / "classification_reports"
    matrices_dir = results_path / "confusion_matrices"

    reports_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data(test_path)
    models = load_models(artifacts_dir)

    records = []

    for model_name, model in models.items():
        print(f"Evaluating {model_name.upper()}...")
        record = evaluate_model(
            model_name=model_name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            reports_dir=reports_dir,
            matrices_dir=matrices_dir,
        )
        records.append(record)

    comparison_df = pd.DataFrame(records)

    summary = {
        "test_file": test_path,
        "test_rows": int(len(y_test)),
        "class_distribution": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "models_evaluated": list(models.keys()),
    }

    save_evaluation_summary(summary, comparison_df, results_path)
    print_evaluation_summary(comparison_df, results_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved SmishKaBa models."
    )

    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help=f"Path to test split. Default: {DEFAULT_TEST_PATH}",
    )

    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing saved models. Default: {DEFAULT_ARTIFACTS_DIR}",
    )

    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for evaluation outputs. Default: {DEFAULT_RESULTS_DIR}",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    evaluate_saved_models(
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()