# tests/test_features.py

import pandas as pd

from src.features import build_feature_transformer, prepare_feature_dataframe


def test_feature_transformer_fits_small_sample():
    df = pd.DataFrame(
        {
            "clean_text": ["hello world", "claim your prize now"],
            "URL": [0, 1],
            "EMAIL": [0, 0],
            "PHONE": [0, 1],
            "url_count": [0, 1],
            "email_count": [0, 0],
            "phone_count": [0, 1],
        }
    )

    X = prepare_feature_dataframe(df)
    transformer = build_feature_transformer(max_features=100, min_df=1)
    matrix = transformer.fit_transform(X)

    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0


def test_tfidf_stop_words_are_excluded():
    df = pd.DataFrame(
        {
            "clean_text": [
                "you can claim your prize in the account now",
                "you should verify your account and claim the reward",
            ],
            "URL": [1, 1],
            "EMAIL": [0, 0],
            "PHONE": [1, 1],
            "url_count": [1, 1],
            "email_count": [0, 0],
            "phone_count": [1, 1],
        }
    )

    X = prepare_feature_dataframe(df)
    transformer = build_feature_transformer(max_features=100, min_df=1)
    transformer.fit(X)

    tfidf = transformer.named_transformers_["tfidf"]
    text_features = tfidf.get_feature_names_out().tolist()

    excluded_tokens = {"you", "your", "in", "the", "and", "to"}
    assert excluded_tokens.isdisjoint(set(text_features))
    assert "claim" in text_features or "prize" in text_features
