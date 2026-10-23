import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parsing sans état, identique à l'entraînement et à l'inférence.

    - Une colonne manquante lève une KeyError (échec visible).
    - TotalCharges : vide ou espaces -> valeur manquante (NaN).
    - Un texte non vide et non numérique lève une ValueError
      (errors="raise"), il n'est pas converti en NaN en silence.
    """
    out = df[ALL_FEATURES].copy()

    total = out["TotalCharges"].astype("string").str.strip()
    total = total.mask(total.eq(""), pd.NA)
    out["TotalCharges"] = pd.to_numeric(total, errors="raise").astype("float64")

    for col in ["tenure", "MonthlyCharges"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out


def build_pipeline(classifier) -> Pipeline:
    """Construit un Pipeline NON entraîné (ne lit aucune donnée, n'appelle pas fit)."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])