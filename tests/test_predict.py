# tests/test_predict.py

import pytest

from src.config import MLR_PIPELINE_PATH
from src.predict import build_input_dataframe, predict_from_text


def test_build_input_dataframe_detects_url():
    frame = build_input_dataframe("Visit http://example.com now")
    assert frame.loc[0, "URL"] == 1
    assert frame.loc[0, "url_count"] == 1


@pytest.mark.skipif(
    not MLR_PIPELINE_PATH.exists(),
    reason="Trained MLR pipeline not available.",
)
def test_predict_from_text_returns_expected_keys():
    try:
        result = predict_from_text("Your account has been locked. Verify now.")
    except AttributeError:
        pytest.skip("Saved model is incompatible with the installed scikit-learn version.")

    assert result["prediction"] in {"ham", "spam", "smishing"}
    assert set(result["probabilities"]) == {"ham", "spam", "smishing"}
    assert "confidence" in result
    assert "features" in result
