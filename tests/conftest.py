import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.pipeline import build_pipeline, parse_features


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """18 clients inventés avec les 6 colonnes d'entrée (pas de Churn, pas de customerID)."""
    n = 18
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)"]
    tenure = np.arange(1, n + 1) * 3.0
    monthly = 20.0 + np.arange(n) * 4.0
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly)
            .round(2)
            .astype(str),  # texte, comme dans le CSV
            "Contract": [contracts[i % 3] for i in range(n)],
            "InternetService": [internet[i % 3] for i in range(n)],
            "PaymentMethod": [payments[i % 3] for i in range(n)],
        }
    )


@pytest.fixture
def sample_target(sample_features) -> pd.Series:
    """Cible binaire avec les deux classes (churn plus probable quand tenure est faible)."""
    return pd.Series((sample_features["tenure"] < 30).astype(int), name="Churn")


@pytest.fixture
def fitted_pipeline(sample_features, sample_target):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(parse_features(sample_features), sample_target)
    return pipe
