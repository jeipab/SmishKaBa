# app/streamlit_app.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "logo.png"
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    CONFUSION_MATRICES_DIR,
    HYPOTHESIS_SUMMARY_PATH,
    MLR_PIPELINE_PATH,
    MODEL_COMPARISON_PATH,
    SHAP_BACKGROUND_SIZE,
    SHAP_LOCAL_TOP_N,
    SHAP_OUTPUTS_DIR,
    TRAIN_SPLIT_PATH,
)
from src.features import get_feature_names, prepare_feature_dataframe
from src.predict import build_input_dataframe, load_pipeline, predict_sms
from src.shap_utils import clean_feature_name, get_feature_type, get_row_shap_values, to_dense
from src.utils import load_json


TOP_N_FEATURES = SHAP_LOCAL_TOP_N
STRUCTURED_FEATURES = {"URL", "EMAIL", "PHONE", "url_count", "email_count", "phone_count"}

EXAMPLE_MESSAGES = {
    "Ham — normal message": "Hey, are we still meeting tomorrow at 3 PM?",
    "Spam — promotional message": "Claim your free reward now! Limited time offer.",
    "Smishing — account alert": (
        "Your account has been locked. Verify now at http://example.com to avoid suspension."
    ),
    "Smishing — prize scam": (
        "You won a cash prize! Call now or visit http://example.com to claim before expiry."
    ),
    "Smishing — tax refund scam": (
        "We have recalculated your vehicle tax. You are owed 48.84. "
        "Visit http://example.com to claim your refund."
    ),
    "Smishing — prize notification": (
        "Congratulations! You have won a 900 prize reward. "
        "Call now or visit http://example.com to claim before expiry."
    ),
}

PRIVACY_NOTICE = (
    "This prototype analyzes only the SMS text you enter on this device. "
    "Messages are processed locally for classification and explanation. "
    "No message text is saved to disk by this app."
)


st.set_page_config(
    page_title="SmishKaBa",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📱",
    layout="wide",
)


def init_session_state() -> None:
    """Initialize session-only UI state."""
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


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


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load saved model comparison table."""
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_top_smishing_features() -> pd.DataFrame:
    """Load global top smishing SHAP features."""
    path = SHAP_OUTPUTS_DIR / "top_smishing_features.csv"
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def render_consent_gate() -> bool:
    """Show privacy notice and consent controls."""
    st.info(PRIVACY_NOTICE)

    consent = st.checkbox(
        "I consent to local analysis of the SMS message I enter.",
        value=st.session_state.consent_given,
    )
    st.session_state.consent_given = consent

    if st.button("Clear session / opt out"):
        st.session_state.consent_given = False
        st.session_state.last_result = None
        st.rerun()

    return consent


def get_local_shap_explanation(message: str, explain_class: str) -> pd.DataFrame:
    """Get top local SHAP contributors for a selected class."""
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

    if explain_class not in class_labels:
        return pd.DataFrame()

    class_index = class_labels.index(explain_class)
    shap_values = explainer.shap_values(transformed)
    row_values = get_row_shap_values(shap_values, class_index)

    top_indices = np.argsort(np.abs(row_values))[::-1][:TOP_N_FEATURES]
    records = []

    for index in top_indices:
        feature_name = clean_feature_name(feature_names[index])
        shap_value = float(row_values[index])

        records.append(
            {
                "Feature": feature_name,
                "Type": get_feature_type(feature_names[index]),
                "Feature Value": float(transformed[0, index]),
                "SHAP Value": shap_value,
                "Effect": (
                    f"Increases {explain_class} score"
                    if shap_value > 0
                    else f"Decreases {explain_class} score"
                ),
            }
        )

    return pd.DataFrame(records)


def plot_shap_bar(shap_df: pd.DataFrame, title: str) -> None:
    """Render a simple Plotly SHAP bar chart."""
    if shap_df.empty:
        return

    plot_df = shap_df.copy()
    plot_df["Abs SHAP"] = plot_df["SHAP Value"].abs()
    plot_df = plot_df.sort_values("Abs SHAP", ascending=True)

    figure = px.bar(
        plot_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="SHAP Value",
        color_continuous_scale=["#d62728", "#cccccc", "#2ca02c"],
        title=title,
    )
    figure.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(figure, width='stretch')


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
            "Probability (%)": [round(value * 100, 2) for value in probabilities.values()],
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


def build_download_payload(result: dict, shap_df: pd.DataFrame) -> str:
    """Build JSON download content."""
    payload = {
        "prediction": result,
        "shap_explanation": shap_df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def render_analyze_tab() -> None:
    """Render SMS analysis workflow."""
    st.subheader("Analyze SMS")
    st.write(
        "Enter an SMS message to classify it as **ham**, **spam**, or **smishing**, "
        "then review the prediction and SHAP explanation."
    )

    example_label = st.selectbox(
        "Example messages",
        ["Custom message"] + list(EXAMPLE_MESSAGES.keys()),
    )

    default_message = ""
    if example_label != "Custom message":
        default_message = EXAMPLE_MESSAGES[example_label]

    message = st.text_area(
        "SMS Message",
        value=default_message,
        height=160,
        placeholder="Paste or type an SMS message here...",
    )

    analyze_disabled = not st.session_state.consent_given

    if analyze_disabled:
        st.warning("Please provide consent above before analyzing a message.")

    if st.button("Analyze Message", type="primary", disabled=analyze_disabled):
        if not message.strip():
            st.error("Please enter an SMS message first.")
            return

        try:
            pipeline = load_model()
            result = predict_sms(message, pipeline)
            predicted_class = result["prediction"]

            with st.spinner("Generating explanation..."):
                shap_df = get_local_shap_explanation(message, predicted_class)

            st.session_state.last_result = {
                "message": message,
                "result": result,
                "shap_df": shap_df,
            }
        except Exception as error:
            st.error("An error occurred while analyzing the message.")
            st.exception(error)

    if not st.session_state.last_result:
        return

    payload = st.session_state.last_result
    result = payload["result"]
    shap_df = payload["shap_df"]
    predicted_class = result["prediction"]

    st.divider()
    display_prediction_badge(result["prediction"])
    st.metric(label="Confidence", value=f"{result['confidence'] * 100:.2f}%")
    st.write(result["risk_message"])

    left, right = st.columns(2)

    with left:
        st.markdown("#### Class Probabilities")
        display_probabilities(result["probabilities"])

    with right:
        st.markdown("#### Detected Message Features")
        display_detected_features(result["features"])

    with st.expander("View cleaned text"):
        st.code(result["clean_text"])

    st.markdown(f"#### SHAP Explanation ({predicted_class})")

    if shap_df.empty:
        st.info("SHAP explanation is unavailable. Make sure the train split and MLR model exist.")
        return

    structured_df = shap_df[shap_df["Feature"].isin(STRUCTURED_FEATURES)].copy()

    st.markdown("**URL / EMAIL / PHONE contributions**")
    if structured_df.empty:
        st.caption("No structured indicator features ranked in the top contributors.")
    else:
        st.dataframe(structured_df, width="stretch", hide_index=True)

    st.markdown("**Top text and indicator features**")
    st.dataframe(shap_df, width="stretch", hide_index=True)
    plot_shap_bar(shap_df, f"Top SHAP Features for {predicted_class}")

    st.download_button(
        label="Download result JSON",
        data=build_download_payload(result, shap_df),
        file_name="smishkaba_prediction.json",
        mime="application/json",
    )


def build_global_shap_figure(top_features_df: pd.DataFrame):
    """Build a spaced-out horizontal bar chart for global smishing SHAP features."""
    top_plot_df = top_features_df.head(12).sort_values("mean_abs_shap", ascending=True)
    bar_count = len(top_plot_df)

    figure = px.bar(
        top_plot_df,
        x="mean_abs_shap",
        y="feature_name",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean |SHAP|",
            "feature_name": "Feature",
        },
        title="Mean Absolute SHAP Values (Smishing Class)",
    )
    figure.update_layout(
        height=max(460, bar_count * 44),
        margin={"l": 8, "r": 16, "t": 48, "b": 28},
        bargap=0.45,
        title={"x": 0, "xanchor": "left", "font": {"size": 14}},
        yaxis={"tickfont": {"size": 13}, "automargin": True},
        xaxis={"tickfont": {"size": 11}, "title_standoff": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_traces(marker_color="#38bdf8")
    return figure


def render_research_tab() -> None:
    """Render saved research outputs."""
    st.subheader("Research Results")
    st.write("Read-only summary of model evaluation and SHAP outputs from the pipeline.")

    comparison_df = load_model_comparison()
    if comparison_df.empty:
        st.warning("Run `python -m src.evaluate` to generate model comparison results.")
    else:
        st.markdown("#### Model Comparison (Test Set)")
        display_df = comparison_df[
            ["model", "accuracy", "macro_f1", "ham_f1", "spam_f1", "smishing_f1"]
        ].copy()
        display_df["model"] = display_df["model"].str.upper()
        st.dataframe(display_df, width="stretch", hide_index=True)

    if HYPOTHESIS_SUMMARY_PATH.exists():
        hypothesis = load_json(HYPOTHESIS_SUMMARY_PATH)
        st.markdown("#### Hypothesis Testing (H01)")
        st.write(hypothesis.get("summary_statement", ""))

        pairs_df = pd.DataFrame(hypothesis.get("pairwise_comparisons", []))
        if not pairs_df.empty:
            st.dataframe(
                pairs_df[["comparison", "p_value", "significant_at_0_05", "result"]],
                width="stretch",
                hide_index=True,
            )

    confusion_col, shap_col = st.columns([1, 1.15], gap="large")

    with confusion_col:
        st.markdown("#### Confusion Matrix")
        model_choice = st.selectbox("Select model", ["mlr", "svm", "nb"], key="confusion_model")

    with shap_col:
        st.markdown("#### Top Global Smishing Features")

    top_features_df = load_top_smishing_features()
    shap_plot = SHAP_OUTPUTS_DIR / "top_smishing_features.png"
    matrix_png = CONFUSION_MATRICES_DIR / f"{model_choice}_confusion_matrix.png"

    with confusion_col:
        if matrix_png.exists():
            st.image(
                str(matrix_png),
                caption=f"{model_choice.upper()} confusion matrix",
                width='stretch',
            )
        else:
            st.caption("Confusion matrix image not found. Run evaluation first.")

    with shap_col:
        if top_features_df.empty:
            if shap_plot.exists():
                st.image(
                    str(shap_plot),
                    caption="Saved SHAP summary plot",
                    width='stretch',
                )
            else:
                st.caption("Global smishing SHAP features not found. Run `python -m src.explain` first.")
        else:
            figure = build_global_shap_figure(top_features_df)
            st.plotly_chart(figure, width='stretch', config={"displayModeBar": False})


def render_about_tab() -> None:
    """Render project and methodology summary."""
    st.subheader("About SmishKaBa")
    st.write(
        "SmishKaBa is an explainable multiclass SMS classification prototype for detecting "
        "**ham**, **spam**, and **smishing** messages using Multinomial Logistic Regression "
        "with SHAP-based explanations."
    )

    st.markdown(
        """
#### Research Questions

1. **Central question:** How does the proposed MLR model compare with baseline NB and SVM?
2. **RQ1:** What SHAP explainability results are produced for the smishing class?
3. **RQ2:** Is there a significant difference in class-specific performance across models?

#### Pipeline

1. Preprocess SMS dataset
2. Train NB, SVM, and MLR models
3. Evaluate on the held-out test split
4. Generate SHAP explanations for MLR
5. Run McNemar tests and build research summaries

#### Privacy

- Manual SMS input only in this prototype
- Local processing during analysis
- Session-only consent; no raw SMS stored by the app
"""
    )


def main() -> None:
    """Render Streamlit app."""
    init_session_state()

    st.title("SmishKaBa")
    st.caption("SHAP-Based Smishing Detection with Multinomial Logistic Regression")

    render_consent_gate()

    analyze_tab, research_tab, about_tab = st.tabs(
        ["Analyze SMS", "Research Results", "About"]
    )

    with analyze_tab:
        render_analyze_tab()

    with research_tab:
        render_research_tab()

    with about_tab:
        render_about_tab()

    st.divider()
    st.caption(
        "Note: This prototype provides machine learning-based assistance only and should "
        "not be treated as final cybersecurity advice."
    )


if __name__ == "__main__":
    main()
