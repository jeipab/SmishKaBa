# SmishKaBa

Explainable SMS smishing detection. SmishKaBa classifies messages as **ham**, **spam**, or **smishing**, compares **Naive Bayes**, **linear SVM**, and **multinomial logistic regression (MLR)**, and explains MLR predictions with **SHAP**.

MLR is the proposed model. NB and SVM are baselines. The Streamlit app classifies a message you type and shows why the model made that call.

## Features

- Multiclass SMS classification (ham / spam / smishing)
- Shared TF-IDF (unigrams + bigrams, English stop words) plus URL, email, and phone indicators
- Model comparison with precision, recall, and F1-score
- SHAP explanations for the MLR model
- Pairwise McNemar tests for overall prediction differences
- Streamlit prototype with a consent gate, live analysis, and saved research results
- CLI for single-message prediction

## Requirements

- **Python 3.11** (recommended)
- Runtime packages: `requirements.txt`
- Test packages: `requirements-dev.txt`

Retrain the models if your scikit-learn version differs from the one used to save the `.joblib` artifacts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS / Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # optional, for tests
```

Use one activate command for your OS. Install packages **after** activating `.venv` so `python` points at the project environment.

## Dataset

Place the raw CSV at `data/raw/sms_dataset.csv`.

| Column                  | Role                         |
| ----------------------- | ---------------------------- |
| `LABEL`                 | `ham`, `spam`, or `smishing` |
| `TEXT`                  | Message body                 |
| `URL`, `EMAIL`, `PHONE` | Presence flags               |

The training corpus is English-only. Preprocessing writes a cleaned dataset and train/test splits under `data/processed/`.

## Usage

### Full pipeline

```bash
python scripts/run_pipeline.py
```

This runs preprocessing, training, evaluation, SHAP, McNemar tests, and research summaries.

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

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

| Tab                  | What it does                                                      |
| -------------------- | ----------------------------------------------------------------- |
| **Analyze SMS**      | Classify a message and show SHAP for the predicted class          |
| **Research Results** | Model comparison, hypothesis tests, confusion matrix, global SHAP |
| **About**            | Short methodology summary                                         |

Messages stay in the session. The app does not write raw SMS to disk.

### Single-message prediction

```bash
python -m src.predict "Your account has been locked. Verify now at http://example.com"
python -m src.predict "Your message here" --json
```

### Tests

```bash
pytest tests/
```

## Project structure

```text
SmishKaBa/
├── app/streamlit_app.py     # prototype UI
├── artifacts/               # trained NB, SVM, MLR pipelines
├── data/raw/                # input CSV
├── data/processed/          # cleaned data, splits, audit logs
├── results/                 # metrics, SHAP, statistics, RQ summaries
├── scripts/run_pipeline.py  # end-to-end runner
├── src/                     # pipeline modules
└── tests/                   # smoke tests
```

| Module               | Role                           |
| -------------------- | ------------------------------ |
| `preprocessing.py`   | Clean raw SMS data             |
| `features.py`        | TF-IDF + indicator features    |
| `train.py`           | Train NB, SVM, MLR             |
| `evaluate.py`        | Metrics and confusion matrices |
| `explain.py`         | SHAP outputs for MLR           |
| `statistics.py`      | McNemar tests                  |
| `research_report.py` | Research-question summaries    |
| `predict.py`         | Single-message inference       |
| `shap_utils.py`      | Shared SHAP helpers            |

## Outputs

| Path                           | Contents                                             |
| ------------------------------ | ---------------------------------------------------- |
| `artifacts/`                   | Trained model pipelines                              |
| `results/model_comparison.csv` | NB / SVM / MLR precision, recall, F1                 |
| `results/shap_outputs/`        | SHAP tables and plots for MLR                        |
| `results/statistical_tests/`   | McNemar tests and hypothesis summary                 |
| `results/research/`            | Markdown + JSON summaries for each research question |

Deeper pipeline and research-output notes are in [DOCUMENTATION.md](DOCUMENTATION.md).

## Notes

- Report **precision, recall, and F1-score** when comparing models — not accuracy.
- SHAP is applied to MLR only.
- Trained artifacts and results are kept in the repository so the app and reports can run without retraining.
- This prototype is machine-learning assistance, not final cybersecurity advice.
