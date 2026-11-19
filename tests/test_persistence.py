import joblib
import numpy as np

from telco_churn.pipeline import parse_features


def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_features):
    X = parse_features(sample_features)
    artifact = tmp_path / "pipeline.joblib"

    before_proba = fitted_pipeline.predict_proba(X)
    before_class = fitted_pipeline.predict(X)

    joblib.dump(fitted_pipeline, artifact)
    loaded = joblib.load(artifact)          # artefact de confiance : le nôtre

    assert np.array_equal(before_class, loaded.predict(X))
    assert np.allclose(before_proba, loaded.predict_proba(X), atol=1e-12)