# src/research_report.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    EVALUATION_SUMMARY_PATH,
    HYPOTHESIS_SUMMARY_PATH,
    LABELS,
    MODEL_COMPARISON_PATH,
    MODEL_PAIR_COMPARISONS_PATH,
    RESEARCH_OUTPUTS_DIR,
    RESULTS_DIR,
    SHAP_OUTPUTS_DIR,
    SHAP_TARGET_CLASS,
)
from src.utils import ensure_dir, load_json, require_file, save_json


RQ1_MD_PATH = RESEARCH_OUTPUTS_DIR / "rq1_explainability_summary.md"
RQ1_JSON_PATH = RESEARCH_OUTPUTS_DIR / "rq1_explainability_summary.json"
RQ2_MD_PATH = RESEARCH_OUTPUTS_DIR / "rq2_performance_and_hypothesis_summary.md"
RQ2_JSON_PATH = RESEARCH_OUTPUTS_DIR / "rq2_performance_and_hypothesis_summary.json"
RQ_CENTRAL_MD_PATH = RESEARCH_OUTPUTS_DIR / "rq_model_comparison_summary.md"
RQ_CENTRAL_JSON_PATH = RESEARCH_OUTPUTS_DIR / "rq_model_comparison_summary.json"
RESEARCH_SUMMARY_PATH = RESEARCH_OUTPUTS_DIR / "research_summary.json"

SHAP_GLOBAL_PATH = SHAP_OUTPUTS_DIR / "global_mean_abs_shap.csv"
SHAP_TARGET_TOP_PATH = SHAP_OUTPUTS_DIR / f"top_{SHAP_TARGET_CLASS}_features.csv"
SHAP_STRUCTURED_PATH = (
    SHAP_OUTPUTS_DIR / f"{SHAP_TARGET_CLASS}_structured_feature_contributions.csv"
)
SHAP_LOCAL_PATH = SHAP_OUTPUTS_DIR / f"local_{SHAP_TARGET_CLASS}_explanations.csv"
SHAP_SUMMARY_PATH = SHAP_OUTPUTS_DIR / "shap_summary.json"

METRIC_COLUMNS = [
    "ham_precision",
    "ham_recall",
    "ham_f1",
    "spam_precision",
    "spam_recall",
    "spam_f1",
    "smishing_precision",
    "smishing_recall",
    "smishing_f1",
]


def load_model_comparison(path: str | Path) -> pd.DataFrame:
    """Load model comparison table."""
    comparison_path = require_file(path, "Model comparison")
    return pd.read_csv(comparison_path)


def load_shap_summary(path: str | Path) -> dict:
    """Load SHAP run metadata if available."""
    summary_path = Path(path)

    if not summary_path.exists():
        return {}

    return load_json(summary_path)


def format_metric(value: float, digits: int = 4) -> str:
    """Format metric for markdown tables."""
    return f"{value:.{digits}f}"


def build_per_class_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build long-form per-class metrics table."""
    records = []

    for _, row in comparison_df.iterrows():
        model = row["model"]

        for label in LABELS:
            records.append(
                {
                    "model": model,
                    "class": label,
                    "precision": float(row[f"{label}_precision"]),
                    "recall": float(row[f"{label}_recall"]),
                    "f1": float(row[f"{label}_f1"]),
                }
            )

    return pd.DataFrame(records)


def build_overall_ranking(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by macro and weighted F1."""
    ranking = comparison_df[
        ["model", "accuracy", "macro_f1", "weighted_f1"]
    ].copy()

    return ranking.sort_values(
        by=["macro_f1", "weighted_f1", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def find_best_model_per_metric(
    comparison_df: pd.DataFrame,
    metric_columns: list[str],
) -> dict[str, str]:
    """Return best model name for each metric column."""
    best = {}

    for column in metric_columns:
        best_index = comparison_df[column].idxmax()
        best[column] = str(comparison_df.loc[best_index, "model"])

    return best


def build_rq1_json(
    shap_summary: dict,
    structured_df: pd.DataFrame,
    top_tfidf_df: pd.DataFrame,
    local_samples: list[dict],
) -> dict:
    """Build JSON payload for RQ1."""
    return {
        "research_question": (
            "Subquestion 1: What explainability results are produced by the "
            "proposed SHAP-guided MLR model for the smishing class?"
        ),
        "target_class": SHAP_TARGET_CLASS,
        "shap_metadata": shap_summary,
        "structured_feature_contributions": structured_df.to_dict(orient="records"),
        "top_tfidf_features": top_tfidf_df.to_dict(orient="records"),
        "sample_local_explanations": local_samples,
    }


def build_rq1_markdown(
    shap_summary: dict,
    structured_df: pd.DataFrame,
    top_tfidf_df: pd.DataFrame,
    local_df: pd.DataFrame,
    top_n: int,
    local_examples: int,
) -> str:
    """Build markdown narrative for RQ1."""
    lines = [
        "# Research Question 1 — Explainability (MLR + SHAP)",
        "",
        "**Subquestion:** What explainability results are produced by the proposed "
        "SHAP-guided MLR model in predicting the smishing class?",
        "",
        "## Overview",
        "",
        "SHAP (`LinearExplainer`) was applied to the trained Multinomial Logistic "
        "Regression model. The summaries below focus on the **smishing** class and "
        "cover mean absolute SHAP values, structured indicator contributions, influential "
        "TF-IDF terms, and local explanation examples.",
        "",
    ]

    if shap_summary:
        lines.extend(
            [
                "## SHAP Run Metadata",
                "",
                f"- Background rows: {shap_summary.get('background_rows', 'N/A')}",
                f"- Explained rows: {shap_summary.get('explained_rows', 'N/A')}",
                f"- Feature count: {shap_summary.get('feature_count', 'N/A')}",
                f"- Class labels: {', '.join(shap_summary.get('class_labels', []))}",
                "",
            ]
        )

    lines.extend(
        [
            "## URL, EMAIL, and PHONE Indicator Contributions",
            "",
            "Mean absolute SHAP values for structured message-level features "
            f"(target class: **{SHAP_TARGET_CLASS}**):",
            "",
            "| Feature | Mean Abs SHAP |",
            "| --- | ---: |",
        ]
    )

    for _, row in structured_df.iterrows():
        lines.append(
            f"| {row['feature_name']} | {format_metric(row['mean_abs_shap'])} |"
        )

    lines.extend(
        [
            "",
            "**Interpretation:** Higher values indicate stronger average influence on "
            f"{SHAP_TARGET_CLASS} predictions. PHONE, URL, and related count features "
            "typically rank among the most influential structured indicators in smishing "
            "messages.",
            "",
            f"## Top {top_n} Influential TF-IDF Terms (Smishing Class)",
            "",
            "| Rank | Term | Mean Abs SHAP |",
            "| ---: | --- | ---: |",
        ]
    )

    for rank, (_, row) in enumerate(top_tfidf_df.head(top_n).iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature_name']} | {format_metric(row['mean_abs_shap'])} |"
        )

    lines.extend(
        [
            "",
            f"## Local SHAP Examples (Top {local_examples} Messages)",
            "",
            "Each example lists the strongest feature contributions for one message "
            f"with respect to the **{SHAP_TARGET_CLASS}** class.",
            "",
        ]
    )

    if local_df.empty:
        lines.append("_No local explanations were available._")
    else:
        sample_ranks = sorted(local_df["sample_rank"].unique())[:local_examples]

        for sample_rank in sample_ranks:
            sample_rows = local_df[local_df["sample_rank"] == sample_rank].sort_values(
                "feature_rank"
            )
            first_row = sample_rows.iloc[0]

            lines.extend(
                [
                    f"### Example {sample_rank}",
                    "",
                    f"- True label: `{first_row['true_label']}`",
                    f"- Predicted label: `{first_row['predicted_label']}`",
                    f"- {SHAP_TARGET_CLASS} probability: "
                    f"{format_metric(first_row['target_probability'])}",
                    f"- Clean text: _{first_row['clean_text'][:180]}..._"
                    if len(str(first_row["clean_text"])) > 180
                    else f"- Clean text: _{first_row['clean_text']}_",
                    "",
                    "| Feature | SHAP Value | Effect |",
                    "| --- | ---: | --- |",
                ]
            )

            for _, feature_row in sample_rows.iterrows():
                effect = (
                    "Increases smishing score"
                    if feature_row["shap_value"] > 0
                    else "Decreases smishing score"
                )
                lines.append(
                    f"| {feature_row['feature_name']} | "
                    f"{format_metric(feature_row['shap_value'])} | {effect} |"
                )

            lines.append("")

    lines.extend(
        [
            "## Source Files",
            "",
            f"- `{SHAP_GLOBAL_PATH.relative_to(RESULTS_DIR.parent)}`",
            f"- `{SHAP_TARGET_TOP_PATH.relative_to(RESULTS_DIR.parent)}`",
            f"- `{SHAP_STRUCTURED_PATH.relative_to(RESULTS_DIR.parent)}`",
            f"- `{SHAP_LOCAL_PATH.relative_to(RESULTS_DIR.parent)}`",
            "",
        ]
    )

    return "\n".join(lines)


def load_hypothesis_summary(path: str | Path) -> dict | None:
    """Load statistical hypothesis test summary if available."""
    summary_path = Path(path)

    if not summary_path.exists():
        return None

    return load_json(summary_path)


def build_rq2_json(
    comparison_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    best_per_metric: dict[str, str],
    evaluation_summary: dict,
    hypothesis_summary: dict | None,
) -> dict:
    """Build JSON payload for RQ2."""
    if hypothesis_summary:
        hypothesis_block = {
            "null_hypothesis": hypothesis_summary.get("null_hypothesis"),
            "method": hypothesis_summary.get("method"),
            "statistical_testing_status": "completed",
            "alpha": hypothesis_summary.get("alpha"),
            "h01_rejected": hypothesis_summary.get("h01_rejected"),
            "summary_statement": hypothesis_summary.get("summary_statement"),
            "pairwise_comparisons": hypothesis_summary.get("pairwise_comparisons", []),
        }
    else:
        hypothesis_block = {
            "null_hypothesis": (
                "H01: There is no significant difference in class-specific performance "
                "of NB, SVM, and MLR for ham, spam, and smishing."
            ),
            "statistical_testing_status": "pending",
            "note": "Run `python -m src.statistics` to generate hypothesis tests.",
        }

    return {
        "research_question": (
            "Subquestion 2: Is there a significant difference between NB, SVM, and MLR "
            "in class-specific precision, recall, and F1-score?"
        ),
        "hypothesis": hypothesis_block,
        "evaluation_metadata": evaluation_summary,
        "overall_ranking": ranking_df.to_dict(orient="records"),
        "per_class_metrics": per_class_df.to_dict(orient="records"),
        "best_model_per_metric": best_per_metric,
        "model_comparison": comparison_df.to_dict(orient="records"),
    }


def build_rq2_markdown(
    comparison_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    best_per_metric: dict[str, str],
    hypothesis_summary: dict | None,
) -> str:
    """Build markdown narrative for RQ2."""
    lines = [
        "# Research Question 2 — Model Performance",
        "",
        "**Subquestion:** Is there a significant difference between the baseline NB "
        "model, baseline SVM model, and proposed MLR model in terms of class-specific "
        "performance for ham, spam, and smishing?",
        "",
        "## Overall Model Ranking",
        "",
        "Models ranked by macro F1, then weighted F1, then accuracy:",
        "",
        "| Rank | Model | Accuracy | Macro F1 | Weighted F1 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]

    for rank, (_, row) in enumerate(ranking_df.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['model'].upper()} | "
            f"{format_metric(row['accuracy'])} | "
            f"{format_metric(row['macro_f1'])} | "
            f"{format_metric(row['weighted_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Class-Specific Performance",
            "",
        ]
    )

    for label in LABELS:
        class_df = per_class_df[per_class_df["class"] == label]

        lines.extend(
            [
                f"### {label.capitalize()}",
                "",
                "| Model | Precision | Recall | F1-Score |",
                "| --- | ---: | ---: | ---: |",
            ]
        )

        for _, row in class_df.iterrows():
            lines.append(
                f"| {row['model'].upper()} | "
                f"{format_metric(row['precision'])} | "
                f"{format_metric(row['recall'])} | "
                f"{format_metric(row['f1'])} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Best Model per Metric",
            "",
            "| Metric | Best Model |",
            "| --- | --- |",
        ]
    )

    for metric, model in best_per_metric.items():
        readable = metric.replace("_", " ").title()
        lines.append(f"| {readable} | {model.upper()} |")

    lines.extend(
        [
            "",
            "## Hypothesis Testing (H01)",
            "",
        ]
    )

    if hypothesis_summary:
        lines.extend(
            [
                f"**Conclusion:** {hypothesis_summary.get('summary_statement', '')}",
                "",
                "Pairwise **McNemar tests** were run on overall correct/incorrect "
                "predictions for the same test set (alpha = 0.05).",
                "",
                "| Comparison | p-value | Significant | Result |",
                "| --- | ---: | --- | --- |",
            ]
        )

        for row in hypothesis_summary.get("pairwise_comparisons", []):
            significant = "Yes" if row["significant_at_0_05"] else "No"
            lines.append(
                f"| {row['comparison']} | {row['p_value']:.4f} | {significant} | "
                f"{row['result']} |"
            )

        lines.append("")
    else:
        lines.extend(
            [
                "Run statistical tests to evaluate H01:",
                "",
                "```bash",
                "python -m src.statistics",
                "python -m src.research_report",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Source Files",
            "",
            f"- `{MODEL_COMPARISON_PATH.relative_to(RESULTS_DIR.parent)}`",
            f"- `{EVALUATION_SUMMARY_PATH.relative_to(RESULTS_DIR.parent)}`",
        ]
    )

    if hypothesis_summary:
        lines.append(
            f"- `{MODEL_PAIR_COMPARISONS_PATH.relative_to(RESULTS_DIR.parent)}`"
        )
        lines.append(
            f"- `{HYPOTHESIS_SUMMARY_PATH.relative_to(RESULTS_DIR.parent)}`"
        )

    lines.append("")

    return "\n".join(lines)


def build_central_rq_markdown(
    comparison_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> str:
    """Build markdown for the central research question."""
    top_model = ranking_df.iloc[0]
    mlr_row = comparison_df[comparison_df["model"] == "mlr"].iloc[0]

    lines = [
        "# Central Research Question — Model Comparison",
        "",
        "**Question:** How does the proposed SHAP-guided Multinomial Logistic "
        "Regression (MLR) model compare with baseline Naive Bayes (NB) and Support "
        "Vector Machine (SVM) models in multiclass SMS classification?",
        "",
        "## Summary",
        "",
        f"- Highest overall macro F1 on the held-out test set: **{top_model['model'].upper()}** "
        f"({format_metric(top_model['macro_f1'])})",
        f"- Proposed MLR macro F1: **{format_metric(mlr_row['macro_f1'])}**",
        f"- MLR smishing F1: **{format_metric(mlr_row['smishing_f1'])}**",
        "",
        "The proposed MLR model provides direct class probabilities and serves as the "
        "primary explainable model via SHAP. Baseline NB and SVM are included for "
        "performance comparison under the same preprocessing and feature pipeline.",
        "",
        "## Model Comparison (Test Set)",
        "",
        "| Model | Accuracy | Macro F1 | Ham F1 | Spam F1 | Smishing F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['model'].upper()} | "
            f"{format_metric(row['accuracy'])} | "
            f"{format_metric(row['macro_f1'])} | "
            f"{format_metric(row['ham_f1'])} | "
            f"{format_metric(row['spam_f1'])} | "
            f"{format_metric(row['smishing_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Related Reports",
            "",
            "- `rq1_explainability_summary.md` — SHAP explainability for smishing",
            "- `rq2_performance_and_hypothesis_summary.md` — per-class metrics and H01",
            "",
        ]
    )

    return "\n".join(lines)


def build_local_sample_records(local_df: pd.DataFrame, local_examples: int) -> list[dict]:
    """Convert local SHAP dataframe into compact JSON records."""
    if local_df.empty:
        return []

    records = []
    sample_ranks = sorted(local_df["sample_rank"].unique())[:local_examples]

    for sample_rank in sample_ranks:
        sample_rows = local_df[local_df["sample_rank"] == sample_rank].sort_values(
            "feature_rank"
        )
        first_row = sample_rows.iloc[0]

        records.append(
            {
                "sample_rank": int(sample_rank),
                "true_label": str(first_row["true_label"]),
                "predicted_label": str(first_row["predicted_label"]),
                "target_probability": float(first_row["target_probability"]),
                "clean_text": str(first_row["clean_text"]),
                "top_features": [
                    {
                        "feature_name": str(feature_row["feature_name"]),
                        "feature_type": str(feature_row["feature_type"]),
                        "shap_value": float(feature_row["shap_value"]),
                    }
                    for _, feature_row in sample_rows.iterrows()
                ],
            }
        )

    return records


def generate_research_reports(
    results_dir: str | Path = RESULTS_DIR,
    shap_dir: str | Path = SHAP_OUTPUTS_DIR,
    research_dir: str | Path = RESEARCH_OUTPUTS_DIR,
    top_n: int = 15,
    local_examples: int = 3,
) -> dict:
    """Generate research question summaries from existing result files."""
    results_path = Path(results_dir)
    shap_path = Path(shap_dir)
    output_dir = ensure_dir(research_dir)

    comparison_path = results_path / MODEL_COMPARISON_PATH.name
    evaluation_path = results_path / EVALUATION_SUMMARY_PATH.name

    comparison_df = load_model_comparison(comparison_path)
    evaluation_summary = (
        load_json(evaluation_path) if evaluation_path.exists() else {}
    )
    hypothesis_summary = load_hypothesis_summary(HYPOTHESIS_SUMMARY_PATH)

    shap_summary = load_shap_summary(shap_path / SHAP_SUMMARY_PATH.name)

    structured_path = shap_path / SHAP_STRUCTURED_PATH.name
    target_top_path = shap_path / SHAP_TARGET_TOP_PATH.name
    local_path = shap_path / SHAP_LOCAL_PATH.name

    require_file(structured_path, "Structured SHAP contributions")
    require_file(target_top_path, "Top smishing SHAP features")
    require_file(local_path, "Local smishing SHAP explanations")

    structured_df = pd.read_csv(structured_path)
    target_top_df = pd.read_csv(target_top_path)
    local_df = pd.read_csv(local_path)

    top_tfidf_df = target_top_df[target_top_df["feature_type"] == "tfidf"].copy()
    top_tfidf_df = top_tfidf_df.sort_values("mean_abs_shap", ascending=False)

    per_class_df = build_per_class_table(comparison_df)
    ranking_df = build_overall_ranking(comparison_df)
    best_per_metric = find_best_model_per_metric(comparison_df, METRIC_COLUMNS)
    local_samples = build_local_sample_records(local_df, local_examples)

    rq1_json = build_rq1_json(
        shap_summary=shap_summary,
        structured_df=structured_df,
        top_tfidf_df=top_tfidf_df.head(top_n),
        local_samples=local_samples,
    )
    rq2_json = build_rq2_json(
        comparison_df=comparison_df,
        per_class_df=per_class_df,
        ranking_df=ranking_df,
        best_per_metric=best_per_metric,
        evaluation_summary=evaluation_summary,
        hypothesis_summary=hypothesis_summary,
    )
    central_json = {
        "research_question": (
            "How does the proposed SHAP-guided MLR model compare with baseline NB "
            "and SVM in multiclass SMS classification?"
        ),
        "overall_ranking": ranking_df.to_dict(orient="records"),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "related_reports": {
            "rq1": str(RQ1_JSON_PATH),
            "rq2": str(RQ2_JSON_PATH),
        },
    }

    rq1_md = build_rq1_markdown(
        shap_summary=shap_summary,
        structured_df=structured_df,
        top_tfidf_df=top_tfidf_df,
        local_df=local_df,
        top_n=top_n,
        local_examples=local_examples,
    )
    rq2_md = build_rq2_markdown(
        comparison_df=comparison_df,
        per_class_df=per_class_df,
        ranking_df=ranking_df,
        best_per_metric=best_per_metric,
        hypothesis_summary=hypothesis_summary,
    )
    central_md = build_central_rq_markdown(comparison_df, ranking_df)

    save_json(rq1_json, output_dir / RQ1_JSON_PATH.name)
    save_json(rq2_json, output_dir / RQ2_JSON_PATH.name)
    save_json(central_json, output_dir / RQ_CENTRAL_JSON_PATH.name)

    (output_dir / RQ1_MD_PATH.name).write_text(rq1_md, encoding="utf-8")
    (output_dir / RQ2_MD_PATH.name).write_text(rq2_md, encoding="utf-8")
    (output_dir / RQ_CENTRAL_MD_PATH.name).write_text(central_md, encoding="utf-8")

    summary = {
        "research_outputs_dir": str(output_dir),
        "files": {
            "central_summary_md": str(output_dir / RQ_CENTRAL_MD_PATH.name),
            "central_summary_json": str(output_dir / RQ_CENTRAL_JSON_PATH.name),
            "rq1_summary_md": str(output_dir / RQ1_MD_PATH.name),
            "rq1_summary_json": str(output_dir / RQ1_JSON_PATH.name),
            "rq2_summary_md": str(output_dir / RQ2_MD_PATH.name),
            "rq2_summary_json": str(output_dir / RQ2_JSON_PATH.name),
        },
        "inputs": {
            "model_comparison": str(comparison_path),
            "evaluation_summary": str(evaluation_path),
            "shap_outputs_dir": str(shap_path),
        },
    }

    save_json(summary, output_dir / RESEARCH_SUMMARY_PATH.name)

    return summary


def print_report_summary(summary: dict) -> None:
    """Print compact console summary."""
    print("Research reports generated.")
    print(f"Output directory: {summary['research_outputs_dir']}")

    print("\nGenerated files:")
    for label, path in summary["files"].items():
        print(f"- {label}: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate research question summaries from evaluation and SHAP outputs."
    )

    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Directory containing evaluation outputs. Default: {RESULTS_DIR}",
    )

    parser.add_argument(
        "--shap-dir",
        default=str(SHAP_OUTPUTS_DIR),
        help=f"Directory containing SHAP outputs. Default: {SHAP_OUTPUTS_DIR}",
    )

    parser.add_argument(
        "--research-dir",
        default=str(RESEARCH_OUTPUTS_DIR),
        help=f"Directory for research summaries. Default: {RESEARCH_OUTPUTS_DIR}",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top TF-IDF terms to include. Default: 15",
    )

    parser.add_argument(
        "--local-examples",
        type=int,
        default=3,
        help="Number of local SHAP examples in the report. Default: 3",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    summary = generate_research_reports(
        results_dir=args.results_dir,
        shap_dir=args.shap_dir,
        research_dir=args.research_dir,
        top_n=args.top_n,
        local_examples=args.local_examples,
    )

    print_report_summary(summary)


if __name__ == "__main__":
    main()
