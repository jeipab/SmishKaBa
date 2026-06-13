# tests/test_preprocessing.py

from src.preprocessing import clean_text_for_model, safe_standardize_label


def test_safe_standardize_label_aliases():
    assert safe_standardize_label("phishing") == "smishing"
    assert safe_standardize_label("HAM") == "ham"
    assert safe_standardize_label("promo") == "spam"


def test_clean_text_for_model_lowercases_and_tokenizes_urls():
    cleaned = clean_text_for_model("Verify now at https://example.com")
    assert "http" not in cleaned
    assert "urltoken" in cleaned
