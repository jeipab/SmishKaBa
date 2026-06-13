# src/statistics.py

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import (
    ARTIFACTS_DIR,
    HYPOTHESIS_SUMMARY_PATH,
    MODEL_PAIR_COMPARISONS_PATH,
    STATISTICAL_TESTS_DIR,
    TEST_SPLIT_PATH,
)
from src.evaluate import load_models, load_test_data
from src.utils import ensure_dir, save_json, to_relative_path


MODEL_PAIRS = list(combinations(["nb", "svm", "mlr"], 2))
SIGNIFICANCE_ALPHA = 0.05


def mcnemar_test(model_a_correct: np.ndarray, model_b_correct: np.ndarray) -> dict:
    """Exact McNemar test on paired correct/incorrect outcomes."""
    a_correct = model_a_correct.astype(bool)
    b_correct = model_b_correct.astype(bool)

    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only

    if discordant == 0:
        return {
            "model_a_only_correct": a_only,
            "model_b_only_correct": b_only,
            "discordant_pairs": 0,
            "p_value": 1.0,
            "significant_at_0_05": False,
        }

    p_value = float(
        binomtest(
            k=min(a_only, b_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

    if a_only > b_only:
        better_model = "model_a"
    elif b_only > a_only:
        better_model = "model_b"
    else:
        better_model = "tie"

    return {
        "model_a_only_correct": a_only,
        "model_b_only_correct": b_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "significant_at_0_05": p_value < SIGNIFICANCE_ALPHA,
        "better_model": better_model,
    }


def compare_model_pair(
    model_a: str,
    model_b: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict:
    """Compare two models with one McNemar test on overall correctness."""
    pred_a = predictions[model_a]
    pred_b = predictions[model_b]

    test_result = mcnemar_test(
        model_a_correct=(pred_a == y_true),
        model_b_correct=(pred_b == y_true),
    )

    if test_result["better_model"] == "model_a":
        winner = model_a.upper()
    elif test_result["better_model"] == "model_b":
        winner = model_b.upper()
    else:
        winner = "tie"

    if not test_result["significant_at_0_05"]:
        result = "no significant difference"
    elif winner == "tie":
        result = "significant difference, no clear winner"
    else:
        result = f"{winner} performs better"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "comparison": f"{model_a.upper()} vs {model_b.upper()}",
        "test": "mcnemar_overall_accuracy",
        "sample_size": int(len(y_true)),
        "model_a_only_correct": test_result["model_a_only_correct"],
        "model_b_only_correct": test_result["model_b_only_correct"],
        "discordant_pairs": test_result["discordant_pairs"],
        "p_value": test_result["p_value"],
        "significant_at_0_05": test_result["significant_at_0_05"],
        "result": result,
    }


def build_h01_summary(comparisons: list[dict]) -> dict:
    """Summarize H01 from pairwise McNemar results."""
    significant = [row for row in comparisons if row["significant_at_0_05"]]
    h01_rejected = len(significant) > 0

    if not h01_rejected:
        summary_statement = (
            "H01 is not rejected at alpha=0.05: McNemar tests found no significant "
            "difference in overall classification performance between NB, SVM, and MLR."
        )
    else:
        pairs = ", ".join(row["comparison"] for row in significant)
        summary_statement = (
            "H01 is rejected at alpha=0.05: at least one pairwise McNemar test found a "
            f"significant difference in overall classification performance ({pairs})."
        )

    return {
        "null_hypothesis": (
            "H01: There is no significant difference in class-specific performance "
            "of NB, SVM, and MLR for ham, spam, and smishing."
        ),
        "method": (
            "Pairwise McNemar tests on overall correct/incorrect predictions "
            "for the same test set."
        ),
        "alpha": SIGNIFICANCE_ALPHA,
        "h01_rejected": h01_rejected,
        "significant_pair_count": len(significant),
        "summary_statement": summary_statement,
        "pairwise_comparisons": comparisons,
    }


def run_statistical_tests(
    test_path: str | Path,
    artifacts_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run simple pairwise McNemar tests for NB, SVM, and MLR."""
    ensure_dir(output_dir)

    X_test, y_test = load_test_data(test_path)
    models = load_models(artifacts_dir)
    y_true = y_test.to_numpy()

    predictions = {
        model_name: model.predict(X_test)
        for model_name, model in models.items()
    }

    comparisons = [
        compare_model_pair(model_a, model_b, y_true, predictions)
        for model_a, model_b in MODEL_PAIRS
    ]

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(MODEL_PAIR_COMPARISONS_PATH, index=False, encoding="utf-8")

    summary = build_h01_summary(comparisons)
    summary["test_file"] = to_relative_path(test_path)
    summary["test_rows"] = int(len(y_test))
    summary["files"] = {
        "model_pair_comparisons": to_relative_path(MODEL_PAIR_COMPARISONS_PATH),
        "hypothesis_test_summary": to_relative_path(HYPOTHESIS_SUMMARY_PATH),
    }

    save_json(summary, HYPOTHESIS_SUMMARY_PATH)
    return summary


def print_statistics_summary(summary: dict) -> None:
    """Print compact console summary."""
    print("Statistical testing complete.")
    print(summary["summary_statement"])
    print(f"H01 rejected: {summary['h01_rejected']}")
    print(f"\nComparisons: {MODEL_PAIR_COMPARISONS_PATH}")
    print(f"Summary: {HYPOTHESIS_SUMMARY_PATH}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run pairwise McNemar tests for NB, SVM, and MLR."
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
        "--output-dir",
        default=str(STATISTICAL_TESTS_DIR),
        help=f"Directory for statistical outputs. Default: {STATISTICAL_TESTS_DIR}",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    summary = run_statistical_tests(
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
    )

    print_statistics_summary(summary)


if __name__ == "__main__":
    main()
