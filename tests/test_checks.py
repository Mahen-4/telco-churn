import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import pandas as pd
from telco_churn.pipeline import ALL_FEATURES, build_pipeline, parse_features
from telco_churn.splits import ID_COL, TARGET, make_splits

SOURCE = Path("data/raw/telco.csv")
LABELS = {"No": 0, "Yes": 1}


@pytest.fixture(scope="module")
def setup():
    df = pd.read_csv(SOURCE)
    train_df, valid_df, test_df = make_splits(df)
    X_train = parse_features(train_df)
    y_train = train_df[TARGET].map(LABELS)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    return {"train": train_df, "valid": valid_df, "test": test_df, "X_train": X_train, "pipe": pipe}


def _num_imputer(pipe):
    return pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def test_ids_do_not_overlap(setup):
    a, b, c = (set(setup[k][ID_COL]) for k in ("train", "valid", "test"))
    assert not (a & b) and not (a & c) and not (b & c)


def test_imputer_uses_training_data_only(setup):
    expected = setup["X_train"][["tenure", "MonthlyCharges", "TotalCharges"]].median().values
    assert np.allclose(_num_imputer(setup["pipe"]).statistics_, expected)


def test_predict_does_not_change_statistics(setup):
    before = _num_imputer(setup["pipe"]).statistics_.copy()
    setup["pipe"].predict(parse_features(setup["valid"]))
    assert np.array_equal(before, _num_imputer(setup["pipe"]).statistics_)


def test_missing_numeric_and_unknown_category(setup):
    row = setup["valid"][ALL_FEATURES].head(1).copy()
    row["TotalCharges"] = " "                 # blanc -> manquant, accepté
    row["Contract"] = "Contrat inconnu"       # catégorie jamais vue -> ignorée
    pred = setup["pipe"].predict(parse_features(row))
    assert pred.shape == (1,)


def test_missing_required_feature_fails(setup):
    bad = setup["valid"][ALL_FEATURES].head(3).drop(columns=["Contract"])
    with pytest.raises(KeyError):
        parse_features(bad)


def test_malformed_text_fails(setup):
    bad = setup["valid"][ALL_FEATURES].head(3).copy()
    bad["TotalCharges"] = "abc"
    with pytest.raises(ValueError):
        parse_features(bad)


def test_save_reload_preserves_predictions(setup, tmp_path):
    X = parse_features(setup["valid"].head(50))
    path = tmp_path / "p.joblib"
    joblib.dump(setup["pipe"], path)
    loaded = joblib.load(path)
    assert np.allclose(setup["pipe"].predict_proba(X), loaded.predict_proba(X), atol=1e-9)