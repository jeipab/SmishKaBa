# SmishKaBa

SmishKaBa is a Streamlit-based prototype for detecting SMS messages as **ham**, **spam**, or **smishing**. It uses a scikit-learn machine learning pipeline with TF-IDF text features, URL/email/phone indicators, and SHAP-based explanations for model interpretability.

## Features

- SMS text classification into ham, spam, or smishing
- TF-IDF-based text feature extraction
- URL, email, and phone number indicator features
- Model comparison using Naive Bayes, Support Vector Machine, and Multinomial Logistic Regression
- SHAP-based explanation for prediction results
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
│   └── predict.py
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

### 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
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

## Notes

- The Streamlit app uses saved trained model pipelines.
- The same preprocessing and feature extraction logic should be reused across training, evaluation, explanation, and prediction.
- Manual SMS input is used for the web prototype.
