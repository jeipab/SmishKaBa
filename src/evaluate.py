# src/evaluate.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    ARTIFACTS_DIR,
    CLASSIFICATION_REPORTS_DIR,
    CONFUSION_MATRICES_DIR,
    EVALUATION_SUMMARY_PATH,
    LABELS,
    MLR_PIPELINE_PATH,
    MODEL_COMPARISON_PATH,
    NB_PIPELINE_PATH,
    RESULTS_DIR,
    SVM_PIPELINE_PATH,
    TEST_SPLIT_PATH,
)
from src.features import prepare_feature_dataframe, prepare_target, validate_feature_columns
from src.utils import ensure_dir, require_file, save_json, series_to_int_dict


MODEL_PATHS = {
    "nb": NB_PIPELINE_PATH,
    "svm": SVM_PIPELINE_PATH,
    "mlr": MLR_PIPELINE_PATH,
}


def load_test_data(test_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load saved test split."""
    path = require_file(test_path, "Test split")
    df = pd.read_csv(path)
    validate_feature_columns(df, require_target=True)

    X_test = prepare_feature_dataframe(df)
    y_test = prepare_target(df, use_label_id=False)

    return X_test, y_test


def load_models(artifacts_dir: str | Path) -> dict[str, object]:
    """Load trained pipelines."""
    models = {}

    for model_name, model_path in MODEL_PATHS.items():
        path = Path(model_path)

        # Allow custom artifact directory from CLI
        if Path(artifacts_dir) != ARTIFACTS_DIR:
            path = Path(artifacts_dir) / path.name

        require_file(path, f"{model_name.upper()} model")
        models[model_name] = joblib.load(path)

    return models


def save_confusion_matrix(
    model_name: str,
    y_test: pd.Series,
    y_pred: pd.Series,
    matrices_dir: Path,
) -> tuple[Path, Path]:
    """Save confusion matrix as CSV and PNG."""
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)

    csv_path = matrices_dir / f"{model_name}_confusion_matrix.csv"
    png_path = matrices_dir / f"{model_name}_confusion_matrix.png"

    matrix_df.to_csv(csv_path, encoding="utf-8")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot(values_format="d")
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    return csv_path, png_path


def save_classification_report(
    model_name: str,
    report_dict: dict,
    reports_dir: Path,
) -> Path:
    """Save classification report as CSV."""
    report_path = reports_dir / f"{model_name}_classification_report.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_path, encoding="utf-8")
    return report_path


def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: Path,
    matrices_dir: Path,
) -> dict:
    """Evaluate one model."""
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

    report_path = save_classification_report(model_name, report_dict, reports_dir)
    matrix_csv_path, matrix_png_path = save_confusion_matrix(
        model_name,
        y_test,
        y_pred,
        matrices_dir,
    )

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


def save_evaluation_outputs(comparison_df: pd.DataFrame, summary: dict) -> None:
    """Save comparison table and summary."""
    ensure_dir(RESULTS_DIR)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False, encoding="utf-8")

    summary["files"] = {
        "model_comparison": str(MODEL_COMPARISON_PATH),
        "evaluation_summary": str(EVALUATION_SUMMARY_PATH),
    }

    save_json(summary, EVALUATION_SUMMARY_PATH)


def print_evaluation_summary(comparison_df: pd.DataFrame) -> None:
    """Print compact evaluation report."""
    print("Evaluation complete.")

    print("\nModel comparison:")
    print(
        comparison_df[
            ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
        ].round(4)
    )

    print(f"\nComparison: {MODEL_COMPARISON_PATH}")
    print(f"Summary: {EVALUATION_SUMMARY_PATH}")


def evaluate_saved_models(
    test_path: str | Path,
    artifacts_dir: str | Path,
    results_dir: str | Path,
) -> None:
    """Evaluate all saved models."""
    ensure_dir(results_dir)
    reports_dir = ensure_dir(CLASSIFICATION_REPORTS_DIR)
    matrices_dir = ensure_dir(CONFUSION_MATRICES_DIR)

    X_test, y_test = load_test_data(test_path)
    models = load_models(artifacts_dir)

    records = []

    for model_name, model in models.items():
        print(f"Evaluating {model_name.upper()}...")
        records.append(
            evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                reports_dir=reports_dir,
                matrices_dir=matrices_dir,
            )
        )

    comparison_df = pd.DataFrame(records)

    summary = {
        "test_file": str(test_path),
        "test_rows": int(len(y_test)),
        "class_distribution": series_to_int_dict(
            y_test.value_counts().sort_index()
        ),
        "models_evaluated": list(models.keys()),
    }

    save_evaluation_outputs(comparison_df, summary)
    print_evaluation_summary(comparison_df)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved SmishKaBa models."
    )

    parser.add_argument(
        "--test-path",
        default=str(TEST_SPLIT_PATH),
        help=f"Path to test split. Default: {TEST_SPLIT_PATH}",
    )

    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"Directory containing models. Default: {ARTIFACTS_DIR}",
    )

    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Directory for outputs. Default: {RESULTS_DIR}",
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