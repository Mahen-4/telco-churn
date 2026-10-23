import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from telco_churn.pipeline import parse_features

pipe = joblib.load("models/pipeline.joblib")   # artefact de confiance uniquement
ref = json.loads(Path("models/reference_predictions.json").read_text(encoding="utf-8"))

new_customers = pd.DataFrame(ref["inputs"])     # six features, pas de Churn
scores = pipe.predict_proba(parse_features(new_customers))[:, list(pipe.classes_).index(1)]

print("max écart vs scores d'origine :", float(np.max(np.abs(scores - np.array(ref["scores"])))))
assert np.allclose(scores, ref["scores"], atol=1e-9), "Les prédictions ont changé après rechargement !"
print("OK : rechargement identique (tolérance 1e-9)")