# src/config.py

from pathlib import Path


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DIR = PROCESSED_DATA_DIR / "splits"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = PROCESSED_DATA_DIR / "audit"

APP_DIR = PROJECT_ROOT / "app"


# Data files
RAW_DATA_PATH = RAW_DATA_DIR / "sms_dataset.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_sms_dataset.csv"

TRAIN_SPLIT_PATH = SPLIT_DIR / "train_split.csv"
TEST_SPLIT_PATH = SPLIT_DIR / "test_split.csv"


# Audit files
PREPROCESSING_REMOVED_ROWS_PATH = AUDIT_DIR / "preprocessing_removed_rows.csv"
PREPROCESSING_DUPLICATE_REPORT_PATH = AUDIT_DIR / "preprocessing_duplicate_clean_text_report.csv"
PREPROCESSING_SUMMARY_PATH = AUDIT_DIR / "preprocessing_summary.json"


# Result files
TRAINING_SUMMARY_PATH = RESULTS_DIR / "training_summary.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
SHAP_OUTPUTS_DIR = RESULTS_DIR / "shap_outputs"
RESEARCH_OUTPUTS_DIR = RESULTS_DIR / "research"
STATISTICAL_TESTS_DIR = RESULTS_DIR / "statistical_tests"

MODEL_PAIR_COMPARISONS_PATH = STATISTICAL_TESTS_DIR / "model_pair_comparisons.csv"
HYPOTHESIS_SUMMARY_PATH = STATISTICAL_TESTS_DIR / "hypothesis_test_summary.json"


# Model files
NB_PIPELINE_PATH = ARTIFACTS_DIR / "nb_pipeline.joblib"
SVM_PIPELINE_PATH = ARTIFACTS_DIR / "svm_pipeline.joblib"
MLR_PIPELINE_PATH = ARTIFACTS_DIR / "mlr_pipeline.joblib"


# Dataset columns
TEXT_COLUMN = "clean_text"

BINARY_FEATURE_COLUMNS = ["URL", "EMAIL", "PHONE"]
COUNT_FEATURE_COLUMNS = ["url_count", "email_count", "phone_count"]
NUMERIC_FEATURE_COLUMNS = BINARY_FEATURE_COLUMNS + COUNT_FEATURE_COLUMNS

FEATURE_COLUMNS = [TEXT_COLUMN] + NUMERIC_FEATURE_COLUMNS

TARGET_COLUMN = "label"
TARGET_ID_COLUMN = "label_id"


# Labels
LABELS = ["ham", "spam", "smishing"]

LABEL_TO_ID = {
    "ham": 0,
    "spam": 1,
    "smishing": 2,
}

ID_TO_LABEL = {
    0: "ham",
    1: "spam",
    2: "smishing",
}


# Training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_SPLIT_STRATEGY = "group"


# TF-IDF settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


# SHAP settings
SHAP_BACKGROUND_SIZE = 200
SHAP_EXPLAIN_SIZE = 500
SHAP_LOCAL_ROWS = 20
SHAP_LOCAL_TOP_N = 10
SHAP_TARGET_CLASS = "smishing"