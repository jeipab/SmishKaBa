# src/predict.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    clean_text_for_model,
    count_emails,
    count_urls,
    count_valid_phones,
    normalize_text,
)
from src.features import REQUIRED_FEATURE_COLUMNS


DEFAULT_MODEL_PATH = "artifacts/mlr_pipeline.joblib"


def load_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a saved sklearn pipeline."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {path}")

    pipeline = joblib.load(path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded artifact is not a scikit-learn Pipeline.")

    if "features" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain 'features' and 'model' steps.")

    return pipeline


def build_input_dataframe(message: str) -> pd.DataFrame:
    """Convert one SMS message into feature-ready input."""
    normalized_message = normalize_text(message)

    if not normalized_message:
        raise ValueError("Message cannot be empty.")

    clean_text = clean_text_for_model(normalized_message)

    url_count = count_urls(normalized_message)
    email_count = count_emails(normalized_message)
    phone_count = count_valid_phones(normalized_message)

    row = {
        "clean_text": clean_text,
        "URL": int(url_count > 0),
        "EMAIL": int(email_count > 0),
        "PHONE": int(phone_count > 0),
        "url_count": int(url_count),
        "email_count": int(email_count),
        "phone_count": int(phone_count),
    }

    return pd.DataFrame([row], columns=REQUIRED_FEATURE_COLUMNS)


def get_class_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, float]:
    """Return class probabilities when available."""
    model = pipeline.named_steps["model"]
    class_labels = [str(label) for label in model.classes_]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        return {
            label: float(probability)
            for label, probability in zip(class_labels, probabilities)
        }

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)

        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        probabilities = softmax(scores, axis=1)[0]
        return {
            label: float(probability)
            for label, probability in zip(class_labels, probabilities)
        }

    return {label: 0.0 for label in class_labels}


def get_risk_message(predicted_label: str) -> str:
    """Return a short user-facing warning."""
    messages = {
        "ham": "This message appears to be normal.",
        "spam": "This message appears to be unsolicited or promotional.",
        "smishing": "This message may be a phishing attempt. Avoid clicking links or sharing sensitive information.",
    }

    return messages.get(predicted_label, "Unable to determine message risk.")


def predict_sms(message: str, pipeline: Pipeline) -> dict:
    """Predict class and confidence for one SMS."""
    X = build_input_dataframe(message)

    predicted_label = str(pipeline.predict(X)[0])
    probabilities = get_class_probabilities(pipeline, X)
    confidence = probabilities.get(predicted_label, 0.0)

    return {
        "message": normalize_text(message),
        "clean_text": str(X.loc[0, "clean_text"]),
        "prediction": predicted_label,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "risk_message": get_risk_message(predicted_label),
        "features": {
            "URL": int(X.loc[0, "URL"]),
            "EMAIL": int(X.loc[0, "EMAIL"]),
            "PHONE": int(X.loc[0, "PHONE"]),
            "url_count": int(X.loc[0, "url_count"]),
            "email_count": int(X.loc[0, "email_count"]),
            "phone_count": int(X.loc[0, "phone_count"]),
        },
    }


def predict_from_text(message: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load model and predict one SMS."""
    pipeline = load_pipeline(model_path)
    return predict_sms(message, pipeline)


def print_prediction(result: dict) -> None:
    """Print compact prediction result."""
    print("Prediction complete.")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Risk message: {result['risk_message']}")

    print("\nProbabilities:")
    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability:.4f}")

    print("\nDetected features:")
    for feature, value in result["features"].items():
        print(f"{feature}: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Predict SMS class using a saved SmishKaBa model."
    )

    parser.add_argument(
        "message",
        help="SMS message to classify.",
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model pipeline. Default: {DEFAULT_MODEL_PATH}",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    result = predict_from_text(args.message, args.model_path)

    if args.json:
        print(json.dumps(result, indent=4))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main()