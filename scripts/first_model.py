import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
from telco_churn.splits import TARGET, make_splits, save_allocation

SOURCE = Path("data/raw/telco.csv")
FEATURES = ["tenure", "MonthlyCharges"]

df = pd.read_csv(SOURCE)
train_df, valid_df, test_df = make_splits(df)
save_allocation(train_df, valid_df, test_df, Path("reports/split_allocation.json"))

label_map = {"No": 0, "Yes": 1}  # positif = Churn "Yes" = 1
X_train, y_train = train_df[FEATURES], train_df[TARGET].map(label_map)
X_valid, y_valid = valid_df[FEATURES], valid_df[TARGET].map(label_map)

results = {}
models = {
    "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_2_features": LogisticRegression(max_iter=1000),
}
for name, model in models.items():
    model.fit(X_train, y_train)          # baseline aussi entraînée sur le train
    pred = model.predict(X_valid)
    results[name] = {
        "accuracy": float(accuracy_score(y_valid, pred)),
        # lignes = vrai, colonnes = prédit, ordre [0=No, 1=Yes]
        # [[TN, FP], [FN, TP]]
        "confusion_matrix": confusion_matrix(y_valid, pred, labels=[0, 1]).tolist(),
    }

out = {
    "positive_class": "Churn = Yes (1)",
    "matrix_layout": "rows=true [No, Yes], cols=predicted [No, Yes]",
    "features": FEATURES,
    "results": results,
}
Path("reports").mkdir(exist_ok=True)
Path("reports/first_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))