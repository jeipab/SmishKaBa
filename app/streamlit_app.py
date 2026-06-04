# app/streamlit_app.py

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy import sparse

# Allow Streamlit to import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import get_feature_names, prepare_feature_dataframe
from src.predict import build_input_dataframe, load_pipeline, predict_sms


DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "mlr_pipeline.joblib"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train_split.csv"

BACKGROUND_SIZE = 200
TOP_N_FEATURES = 8


st.set_page_config(
    page_title="SmishKaBa",
    page_icon="📩",
    layout="centered",
)


def to_dense(matrix) -> np.ndarray:
    """Convert sparse matrix to dense."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def clean_feature_name(feature_name: str) -> str:
    """Remove transformer prefix."""
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


@st.cache_resource
def load_model():
    """Load saved MLR pipeline."""
    return load_pipeline(str(DEFAULT_MODEL_PATH))


@st.cache_data
def load_background_data() -> pd.DataFrame:
    """Load background rows for SHAP."""
    if not DEFAULT_TRAIN_PATH.exists():
        return pd.DataFrame()

    train_df = pd.read_csv(DEFAULT_TRAIN_PATH)
    train_df = train_df.sample(
        n=min(BACKGROUND_SIZE, len(train_df)),
        random_state=42,
    )

    return prepare_feature_dataframe(train_df)


@st.cache_resource
def build_shap_explainer():
    """Build cached SHAP explainer."""
    pipeline = load_model()
    background_df = load_background_data()

    if background_df.empty:
        return None

    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    background_matrix = to_dense(transformer.transform(background_df))
    explainer = shap.LinearExplainer(model, background_matrix)

    return explainer


def get_local_shap_explanation(message: str, predicted_label: str) -> pd.DataFrame:
    """Get top SHAP contributors for one message."""
    pipeline = load_model()
    explainer = build_shap_explainer()

    if explainer is None:
        return pd.DataFrame()

    X = build_input_dataframe(message)

    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    transformed = to_dense(transformer.transform(X))
    feature_names = get_feature_names(transformer)

    shap_values = explainer.shap_values(transformed)
    class_labels = [str(label) for label in model.classes_]

    if predicted_label not in class_labels:
        return pd.DataFrame()

    class_index = class_labels.index(predicted_label)

    if isinstance(shap_values, list):
        row_values = shap_values[class_index][0]
    else:
        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3 and shap_array.shape[0] == 1:
            row_values = shap_array[0, :, class_index]
        elif shap_array.ndim == 3:
            row_values = shap_array[class_index, 0, :]
        else:
            row_values = shap_array[0]

    top_indices = np.argsort(np.abs(row_values))[::-1][:TOP_N_FEATURES]

    records = []

    for index in top_indices:
        records.append(
            {
                "Feature": clean_feature_name(feature_names[index]),
                "Feature Value": float(transformed[0, index]),
                "SHAP Value": float(row_values[index]),
                "Effect": "Increases prediction" if row_values[index] > 0 else "Decreases prediction",
            }
        )

    return pd.DataFrame(records)


def display_prediction_badge(prediction: str) -> None:
    """Show prediction with simple status styling."""
    if prediction == "ham":
        st.success("Prediction: HAM")
    elif prediction == "spam":
        st.warning("Prediction: SPAM")
    elif prediction == "smishing":
        st.error("Prediction: SMISHING")
    else:
        st.info(f"Prediction: {prediction.upper()}")


def main() -> None:
    st.title("SmishKaBa")
    st.caption("SHAP-Based Smishing Detection with Multinomial Logistic Regression")

    st.write(
        "Enter an SMS message to classify it as **ham**, **spam**, or **smishing**."
    )

    message = st.text_area(
        "SMS Message",
        height=160,
        placeholder="Paste or type an SMS message here...",
    )

    analyze_button = st.button("Analyze Message", type="primary")

    if analyze_button:
        if not message.strip():
            st.error("Please enter an SMS message first.")
            return

        try:
            pipeline = load_model()
            result = predict_sms(message, pipeline)

            st.divider()
            display_prediction_badge(result["prediction"])

            st.metric(
                label="Confidence",
                value=f"{result['confidence'] * 100:.2f}%",
            )

            st.write(result["risk_message"])

            st.subheader("Class Probabilities")
            probabilities_df = pd.DataFrame(
                {
                    "Class": list(result["probabilities"].keys()),
                    "Probability": [
                        round(value * 100, 2)
                        for value in result["probabilities"].values()
                    ],
                }
            )

            st.dataframe(probabilities_df, use_container_width=True, hide_index=True)
            st.bar_chart(probabilities_df.set_index("Class"))

            st.subheader("Detected Message Features")
            feature_df = pd.DataFrame(
                {
                    "Feature": list(result["features"].keys()),
                    "Value": list(result["features"].values()),
                }
            )

            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            with st.expander("View cleaned text"):
                st.code(result["clean_text"])

            st.subheader("SHAP Local Explanation")

            with st.spinner("Generating explanation..."):
                shap_df = get_local_shap_explanation(
                    message=message,
                    predicted_label=result["prediction"],
                )

            if shap_df.empty:
                st.info(
                    "SHAP explanation is unavailable. Make sure the train split and MLR model exist."
                )
            else:
                st.write(
                    "Top features that influenced the predicted class for this message:"
                )
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error("An error occurred while analyzing the message.")
            st.exception(error)

    st.divider()

    st.caption(
        "Note: This prototype provides machine learning-based assistance only and should not be treated as final cybersecurity advice."
    )


if __name__ == "__main__":
    main()