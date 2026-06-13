# SmishKaBa — Project Guide

Concise reference for how the project fits together. For setup commands, see [README.md](README.md).

## What it does

SmishKaBa classifies SMS messages as **ham**, **spam**, or **smishing**. It trains three models (NB, SVM, MLR) on the same features, compares them on a held-out test set, and explains MLR predictions with **SHAP**. MLR is the proposed explainable model; NB and SVM are baselines.

**Dataset:** English SMS only (`data/raw/sms_dataset.csv`).

## Pipeline flow

```text
raw CSV → preprocessing → train (NB, SVM, MLR) → evaluate
                                              ↓
                         explain (SHAP on MLR) → statistics (H01) → research_report
```

Run everything:

```bash
python scripts/run_pipeline.py
```

## Research questions → outputs

| Question | Read this |
| -------- | --------- |
| Central — How does MLR compare with NB and SVM? | `results/research/rq_model_comparison_summary.md` |
| RQ1 — SHAP explainability for smishing | `results/research/rq1_explainability_summary.md` |
| RQ2 — Class-specific performance + H01 | `results/research/rq2_performance_and_hypothesis_summary.md` |

JSON versions of the same summaries are in `results/research/` for tables or copy-paste into the paper.

## Features

Each message becomes:

- **Text:** TF-IDF unigrams and bigrams (English stop words removed)
- **Structured indicators:** `URL`, `EMAIL`, `PHONE` flags and counts

All three models use this same pipeline from `src/features.py`.

## Metrics (for the paper)

Report **precision, recall, and F1-score** — not accuracy.

- **Macro** averages across ham / spam / smishing
- **Per-class** tables for each label
- Primary comparison file: `results/model_comparison.csv`
- H01 uses pairwise **McNemar tests** on overall correct/incorrect predictions (`results/statistical_tests/`)

## SHAP (RQ1)

SHAP is computed for **MLR only**.

**Interpret in this order:**

1. Structured features (`PHONE`, `URL`, counts) — strongest smishing signals
2. Scam-related terms (`claim`, `prize`, `verify`, …)
3. Ignore generic English function words — they are excluded from TF-IDF via stop words

Key files: `results/shap_outputs/top_smishing_features.csv`, `smishing_structured_feature_contributions.csv`, `local_smishing_explanations.csv`

## Folder map

| Path | Purpose |
| ---- | ------- |
| `src/` | Pipeline modules |
| `app/streamlit_app.py` | Prototype UI (analyze SMS + view research results) |
| `artifacts/` | Saved NB, SVM, MLR pipelines |
| `results/` | Metrics, SHAP, statistics, research summaries |
| `data/processed/` | Cleaned data and train/test splits |
| `tests/` | Smoke tests |

## Streamlit app

```bash
streamlit run app/streamlit_app.py
```

- **Analyze SMS** — classify a message and show SHAP for the predicted class (session-only; no messages saved)
- **Research Results** — model comparison (P/R/F1), H01, confusion matrix, global SHAP chart
- **About** — short methodology summary

## Further reading

| File | When to use it |
| ---- | -------------- |
| [archive/CONTEXT.md](archive/CONTEXT.md) | Concept paper background and requirements |
| [archive/CURRENT.md](archive/CURRENT.md) | Full implementation notes (detailed) |
| [archive/RESEARCH_OUTPUTS.md](archive/RESEARCH_OUTPUTS.md) | Extended guide to every results file |
