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
