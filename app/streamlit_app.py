# app/streamlit_app.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import MLR_PIPELINE_PATH, SHAP_BACKGROUND_SIZE, SHAP_LOCAL_TOP_N, TRAIN_SPLIT_PATH
from src.features import get_feature_names, prepare_feature_dataframe
from src.predict import build_input_dataframe, load_pipeline, predict_sms
from src.shap_utils import clean_feature_name, get_row_shap_values, to_dense


TOP_N_FEATURES = SHAP_LOCAL_TOP_N


st.set_page_config(
    page_title="SmishKaBa",
    page_icon="logo.png",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load saved MLR pipeline."""
    return load_pipeline(MLR_PIPELINE_PATH)


@st.cache_data
def load_background_data() -> pd.DataFrame:
    """Load sampled training rows for SHAP."""
    if not TRAIN_SPLIT_PATH.exists():
        return pd.DataFrame()

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    train_df = train_df.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(train_df)),
        random_state=42,
    )

    return prepare_feature_dataframe(train_df)


@st.cache_resource
def build_shap_explainer():
    """Build SHAP explainer once."""
    background_df = load_background_data()

    if background_df.empty:
        return None

    pipeline = load_model()
    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    background_matrix = to_dense(transformer.transform(background_df))

    return shap.LinearExplainer(model, background_matrix)


def get_local_shap_explanation(
    message: str,
    predicted_label: str,
) -> pd.DataFrame:
    """Get top local SHAP contributors."""
    explainer = build_shap_explainer()

    if explainer is None:
        return pd.DataFrame()

    pipeline = load_model()
    X = build_input_dataframe(message)

    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    transformed = to_dense(transformer.transform(X))
    feature_names = get_feature_names(transformer)
    class_labels = [str(label) for label in model.classes_]

    if predicted_label not in class_labels:
        return pd.DataFrame()

    class_index = class_labels.index(predicted_label)
    shap_values = explainer.shap_values(transformed)
    row_values = get_row_shap_values(shap_values, class_index)

    top_indices = np.argsort(np.abs(row_values))[::-1][:TOP_N_FEATURES]

    records = []

    for index in top_indices:
        records.append(
            {
                "Feature": clean_feature_name(feature_names[index]),
                "Feature Value": float(transformed[0, index]),
                "SHAP Value": float(row_values[index]),
                "Effect": (
                    "Increases prediction"
                    if row_values[index] > 0
                    else "Decreases prediction"
                ),
            }
        )

    return pd.DataFrame(records)


def display_prediction_badge(prediction: str) -> None:
    """Show prediction status."""
    if prediction == "ham":
        st.success("Prediction: HAM")
    elif prediction == "spam":
        st.warning("Prediction: SPAM")
    elif prediction == "smishing":
        st.error("Prediction: SMISHING")
    else:
        st.info(f"Prediction: {prediction.upper()}")


def display_probabilities(probabilities: dict[str, float]) -> None:
    """Show class probabilities."""
    probabilities_df = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [
                round(value * 100, 2)
                for value in probabilities.values()
            ],
        }
    )

    st.dataframe(probabilities_df, width="stretch", hide_index=True)
    st.bar_chart(probabilities_df.set_index("Class"))


def display_detected_features(features: dict[str, int]) -> None:
    """Show detected structured features."""
    feature_df = pd.DataFrame(
        {
            "Feature": list(features.keys()),
            "Value": list(features.values()),
        }
    )

    st.dataframe(feature_df, width="stretch", hide_index=True)


def analyze_message(message: str) -> None:
    """Run prediction and explanation."""
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
    display_probabilities(result["probabilities"])

    st.subheader("Detected Message Features")
    display_detected_features(result["features"])

    with st.expander("View cleaned text"):
        st.code(result["clean_text"])

    st.subheader("SHAP Local Explanation")

    with st.spinner("Generating explanation..."):
        shap_df = get_local_shap_explanation(
            message=message,
            predicted_label=result["prediction"],
        )

    if shap_df.empty:
        st.info("SHAP explanation is unavailable. Make sure the train split and MLR model exist.")
    else:
        st.write("Top features that influenced the predicted class:")
        st.dataframe(shap_df, width="stretch", hide_index=True)


def main() -> None:
    """Render Streamlit app."""
    st.title("SmishKaBa")
    st.caption("SHAP-Based Smishing Detection with Multinomial Logistic Regression")

    st.write("Enter an SMS message to classify it as **ham**, **spam**, or **smishing**.")

    message = st.text_area(
        "SMS Message",
        height=160,
        placeholder="Paste or type an SMS message here...",
    )

    if st.button("Analyze Message", type="primary"):
        if not message.strip():
            st.error("Please enter an SMS message first.")
            return

        try:
            analyze_message(message)
        except Exception as error:
            st.error("An error occurred while analyzing the message.")
            st.exception(error)

    st.divider()
    st.caption(
        "Note: This prototype provides machine learning-based assistance only and should not be treated as final cybersecurity advice."
    )


if __name__ == "__main__":
    main()
