from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
from telco_churn.pipeline import build_pipeline, parse_features
from telco_churn.splits import TARGET, make_splits

SOURCE = Path("data/raw/telco.csv")
df = pd.read_csv(SOURCE)
train_df, valid_df, test_df = make_splits(df)

label_map = {"No": 0, "Yes": 1}
X_train, y_train = parse_features(train_df), train_df[TARGET].map(label_map)
X_valid, y_valid = parse_features(valid_df), valid_df[TARGET].map(label_map)

pipe = build_pipeline(LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)          # un seul fit pour tout
pred = pipe.predict(X_valid)        # colonnes d'origine en entrée

print("accuracy:", accuracy_score(y_valid, pred))
print(confusion_matrix(y_valid, pred, labels=[0, 1]))

# Contrôle : la médiane apprise vient bien du train, pas de tout le dataset
imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
print("médianes apprises :", imputer.statistics_)
print("médianes du train :", X_train[["tenure", "MonthlyCharges", "TotalCharges"]].median().values)