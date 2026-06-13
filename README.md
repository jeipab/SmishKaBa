# SmishKaBa

SmishKaBa is a research prototype for **SMS smishing detection** using explainable machine learning. It classifies messages as **ham**, **spam**, or **smishing**, compares **Naive Bayes**, **Support Vector Machine**, and **Multinomial Logistic Regression**, and explains MLR predictions with **SHAP**.

**Institution context:** Polytechnic University of the Philippines — COSC 402 concept paper project.

## Features

- Multiclass SMS classification (ham / spam / smishing)
- Shared TF-IDF + URL / EMAIL / PHONE feature pipeline
- Model comparison (NB, SVM, MLR) with evaluation reports
- SHAP explainability for the proposed MLR model
- Research summaries mapped to central RQ, RQ1, and RQ2
- Pairwise McNemar tests for hypothesis H01
- Streamlit app with consent gate, research results tab, and smishing SHAP view

## Documentation

| File | Purpose |
| ---- | ------- |
| [README.md](README.md) | Setup and usage (this file) |
| [CONTEXT.md](CONTEXT.md) | Concept paper and research requirements |
| [CURRENT.md](CURRENT.md) | Full technical implementation reference |
| [RESEARCH_OUTPUTS.md](RESEARCH_OUTPUTS.md) | How to read results for your paper |
| [UPGRADE_PLAN.md](UPGRADE_PLAN.md) | Phased enhancement plan |

## Requirements

- **Python 3.11** (recommended)
- Dependencies in `requirements.txt`
- Dev/test tools in `requirements-dev.txt`

Retrain models if your scikit-learn version differs from the one used to create the saved `.joblib` files.

## Project Structure

```text
SmishKaBa/
├── app/streamlit_app.py
├── artifacts/              # trained NB, SVM, MLR pipelines
├── data/raw/               # input CSV
├── data/processed/         # cleaned data, splits, audit logs
├── results/                # evaluation, SHAP, statistics, research summaries
├── scripts/run_pipeline.py # run full pipeline
├── src/                    # Python modules
└── tests/                  # smoke tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for tests
```

## Dataset

Place the raw CSV at:

```text
data/raw/sms_dataset.csv
```

Expected columns: `LABEL`, `TEXT`, `URL`, `EMAIL`, `PHONE`

Labels: `ham`, `spam`, `smishing`

## Usage

### Full pipeline

```bash
python scripts/run_pipeline.py
```

Individual steps:

```bash
python -m src.preprocessing
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.statistics
python -m src.research_report
```

Skip steps if needed:

```bash
python scripts/run_pipeline.py --skip-preprocessing --skip-train
```

### Streamlit prototype

```bash
streamlit run app/streamlit_app.py
```

Tabs: **Analyze SMS**, **Research Results**, **About**

### Single-message prediction (CLI)

```bash
python -m src.predict "Your account has been locked. Verify now at http://example.com"
python -m src.predict "Your message here" --json
```

### Tests

```bash
pytest tests/
```

## Key Outputs

| Folder / file | Contents |
| ------------- | -------- |
| `artifacts/` | Trained model pipelines |
| `results/model_comparison.csv` | NB / SVM / MLR metrics |
| `results/shap_outputs/` | SHAP CSV/PNG for MLR |
| `results/statistical_tests/` | McNemar tests and H01 summary |
| `results/research/` | Markdown + JSON answers for each research question |

See [RESEARCH_OUTPUTS.md](RESEARCH_OUTPUTS.md) for how each file supports the paper.

## Module Overview

| Module | Role |
| ------ | ---- |
| `preprocessing.py` | Clean raw SMS data |
| `features.py` | TF-IDF + indicator features |
| `train.py` | Train NB, SVM, MLR |
| `evaluate.py` | Metrics and confusion matrices |
| `explain.py` | SHAP outputs for MLR |
| `statistics.py` | McNemar tests for H01 |
| `research_report.py` | RQ summary reports |
| `predict.py` | Single-message inference |
| `shap_utils.py` | Shared SHAP helpers |

## Notes

- The Streamlit app analyzes manually entered SMS only; messages are not saved to disk.
- MLR is the proposed explainable model; SHAP is applied to MLR only.
- Generated artifacts and results are kept in the repository for reproducibility.
