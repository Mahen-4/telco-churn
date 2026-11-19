import numpy as np
import pytest

from telco_churn.pipeline import parse_features


def test_probability_contract(fitted_pipeline, sample_features):
    X = parse_features(sample_features)
    proba = fitted_pipeline.predict_proba(X)

    assert list(fitted_pipeline.classes_) == [0, 1]
    assert proba.shape == (len(sample_features), 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.sum(axis=1) == pytest.approx(1.0)

    # Le score positif est la colonne de la classe 1, trouvée via classes_
    idx = list(fitted_pipeline.classes_).index(1)
    assert np.array_equal(proba[:, idx], proba[:, 1])
