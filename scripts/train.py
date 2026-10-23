import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from telco_churn.dataset import  sha256_of
from telco_churn.pipeline import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline, parse_features,
)
from telco_churn.splits import TARGET, make_splits, save_allocation
from telco_churn.training import evaluate_model, make_classifier

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
args = parser.parse_args()
cfg = json.loads(args.config.read_text(encoding="utf-8"))

source = Path(cfg["source"])
SOURCE = Path("data/raw/telco.csv")
df = pd.read_csv(SOURCE)
train_df, valid_df, test_df = make_splits(df, cfg["seed"])
save_allocation(train_df, valid_df, test_df, Path("reports/split_allocation.json"), cfg["seed"])

label_map = {"No": 0, "Yes": 1}
X_train, y_train = parse_features(train_df), train_df[TARGET].map(label_map)
X_valid, y_valid = parse_features(valid_df), valid_df[TARGET].map(label_map)

rows, fitted = [], {}
for name in ["baseline", *cfg["models"]]:
    params = cfg["models"].get(name, {})
    pipe = build_pipeline(make_classifier(name, params))   # preprocessing neuf à chaque fois
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = round(time.perf_counter() - t0, 3)
    metrics = evaluate_model(pipe, X_valid, y_valid, cfg["threshold"])
    rows.append({"model": name, "params": params, "fit_seconds": fit_seconds, **metrics})
    fitted[name] = pipe
    print(f"{name:15s} AP={metrics['average_precision']:.4f} "
          f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

# Règle fixée à l'avance : meilleure average precision sur la validation (baseline exclue)
learned = [r for r in rows if r["model"] != "baseline"]
selected = max(learned, key=lambda r: r["average_precision"])["model"]
print("Modèle sélectionné :", selected)

Path("reports").mkdir(exist_ok=True)
Path("reports/model_comparison.json").write_text(json.dumps({
    "split": "validation",
    "positive_class": "Churn = Yes (1)",
    "primary_metric": cfg["primary_metric"],
    "threshold": cfg["threshold"],
    "selected": selected,
    "rows": rows,
}, indent=2), encoding="utf-8")

# Artefact : prétraitement + classifieur ensemble
Path("models").mkdir(exist_ok=True)
joblib.dump(fitted[selected], "models/pipeline.joblib")

try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
except Exception:
    commit = None

metadata = {
    "selected_model": selected,
    "source": {"path": str(source), "sha256": sha256_of(source)},
    "seed": cfg["seed"],
    "split_allocation_file": "reports/split_allocation.json",
    "positive_class": "Churn = Yes (1)",
    "threshold": cfg["threshold"],
    "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
    "preprocessing": {
        "numeric": "median imputation + StandardScaler",
        "categorical": "most_frequent imputation + OneHotEncoder(handle_unknown='ignore')",
        "TotalCharges": "blank -> missing; other non-numeric text raises",
    },
    "model_params": next(r["params"] for r in rows if r["model"] == selected),
    "validation_metrics": next(r for r in rows if r["model"] == selected),
    "environment": {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
        "pandas": version("pandas"),
        "joblib": version("joblib"),
    },
    "git_commit": commit,
}
Path("models/metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Prédictions de référence pour tester le rechargement
sample = valid_df[ALL_FEATURES].head(20)
reference = {
    "inputs": json.loads(sample.to_json(orient="records")),
    "scores": pipe_scores if (pipe_scores := fitted[selected].predict_proba(parse_features(sample))[:, 1].tolist()) else [],
}
Path("models/reference_predictions.json").write_text(json.dumps(reference), encoding="utf-8")
print("Artefact sauvegardé dans models/")