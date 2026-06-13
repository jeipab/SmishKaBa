# SmishKaBa

SmishKaBa is a Streamlit-based prototype for detecting SMS messages as **ham**, **spam**, or **smishing**. It uses a scikit-learn machine learning pipeline with TF-IDF text features, URL/email/phone indicators, and SHAP-based explanations for model interpretability.

## Features

- SMS text classification into ham, spam, or smishing
- TF-IDF-based text feature extraction
- URL, email, and phone number indicator features
- Model comparison using Naive Bayes, Support Vector Machine, and Multinomial Logistic Regression
- SHAP-based explanation for prediction results
- Research question summaries linking evaluation and SHAP outputs to RQ1 and RQ2
- Streamlit web interface for manual SMS input

## Recommended Python Version

Use **Python 3.11**.

## Project Structure

```text
smishkaba/
├── app/
│   └── streamlit_app.py
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── research_report.py
│   ├── statistics.py
│   └── shap_utils.py
├── scripts/
│   └── run_pipeline.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset inside:

```text
data/raw/
```

Expected dataset content:

- SMS message text
- Class label: `ham`, `spam`, or `smishing`
- Optional URL, email, and phone indicator columns

## Usage

### 1. Preprocess the dataset

```bash
python -m src.preprocessing
```

### 2. Train the models

```bash
python -m src.train
```

### 3. Evaluate the models

```bash
python -m src.evaluate
```

### 4. Generate SHAP explanations

```bash
python -m src.explain
```

### 5. Generate research question summaries

```bash
python -m src.research_report
```

This step reads existing evaluation and SHAP outputs and writes markdown/JSON summaries under `results/research/` for the central research question, RQ1 (explainability), and RQ2 (performance).

### 6. Run statistical hypothesis tests

```bash
python -m src.statistics
python -m src.research_report
```

`statistics` runs three pairwise McNemar tests on overall classification accuracy. Re-run `research_report` afterward to include H01 in the RQ2 summary.

### 7. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Or run the full pipeline with one command:

```bash
python scripts/run_pipeline.py
```

Use `--skip-*` flags to skip individual steps, for example `--skip-explain`.

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Output Files

Trained model pipelines are saved in:

```text
artifacts/
```

Evaluation reports and explanation outputs are saved in:

```text
results/
```

Research question summaries are saved in:

```text
results/research/
```

Key files:

- `rq_model_comparison_summary.md` — central research question
- `rq1_explainability_summary.md` — SHAP explainability (smishing focus)
- `rq2_performance_and_hypothesis_summary.md` — per-class metrics for NB, SVM, MLR

Statistical test outputs:

```text
results/statistical_tests/
```

Key files:

- `model_pair_comparisons.csv` — pairwise McNemar test results (NB vs SVM vs MLR)
- `hypothesis_test_summary.json` — H01 conclusion

## Notes

- The Streamlit app uses saved trained model pipelines.
- The same preprocessing and feature extraction logic should be reused across training, evaluation, explanation, and prediction.
- Manual SMS input is used for the web prototype.
