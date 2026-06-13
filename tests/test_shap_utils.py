# tests/test_shap_utils.py

import numpy as np

from src.shap_utils import clean_feature_name, get_row_shap_values, normalize_shap_values


def test_clean_feature_name_removes_prefix():
    assert clean_feature_name("tfidf__claim") == "claim"
    assert clean_feature_name("URL") == "URL"


def test_normalize_shap_values_list_input():
    values = normalize_shap_values(
        shap_values=[np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])],
        class_labels=["ham", "spam"],
        n_samples=1,
        n_features=2,
    )

    assert set(values) == {"ham", "spam"}
    assert values["ham"].shape == (1, 2)


def test_get_row_shap_values_from_list():
    shap_values = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])]
    row = get_row_shap_values(shap_values, class_index=1)
    assert np.allclose(row, np.array([0.3, 0.4]))
