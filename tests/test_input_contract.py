import numpy as np
import pytest

from telco_churn.pipeline import parse_features


def test_missing_required_feature_is_rejected(sample_features):
    broken = sample_features.drop(columns=["Contract"])
    with pytest.raises(KeyError, match="Missing required features.*Contract"):
        parse_features(broken)


def test_supported_variation_is_predicted(fitted_pipeline, sample_features):
    batch = sample_features.copy()          # copie : on ne modifie pas la fixture
    batch.loc[0, "tenure"] = np.nan         # numérique manquant -> imputé
    batch.loc[1, "TotalCharges"] = " "      # blanc -> manquant
    batch.loc[2, "PaymentMethod"] = "Crypto"  # catégorie jamais vue à l'entraînement

    assert "Crypto" not in sample_features["PaymentMethod"].unique()  # vraiment absente

    proba = fitted_pipeline.predict_proba(parse_features(batch))
    assert proba.shape == (len(batch), 2)
    assert np.isfinite(proba).all()