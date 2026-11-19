import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from telco_churn.pipeline import parse_features

mlflow.set_tracking_uri("http://127.0.0.1:5000")
info = json.loads(Path("reports/tp03_registered.json").read_text(encoding="utf-8"))

pipe = mlflow.sklearn.load_model("models:/telco-churn@candidate")
new_customers = pd.DataFrame(info["reference"]["inputs"])   # six features, sans Churn ni customerID
X = parse_features(new_customers)                           # même parsing qu'à l'entraînement

idx = list(pipe.classes_).index(1)
scores = pipe.predict_proba(X)[:, idx]
expected = np.array(info["reference"]["scores"])

print("version :", info["version"], "| run :", info["run_id"])
print("écart max des scores :", float(np.max(np.abs(scores - expected))))
assert np.allclose(scores, expected, atol=1e-9), "Les prédictions diffèrent !"
assert np.array_equal(scores >= 0.5, expected >= 0.5), "Les classes prédites diffèrent !"
print("OK : prédictions identiques (tolérance 1e-9)")