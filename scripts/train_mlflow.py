import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd

from telco_churn.dataset import read_telco, sha256_of
from telco_churn.pipeline import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline, parse_features,
)
from telco_churn.splits import TARGET, make_splits
from telco_churn.training import evaluate_model, make_classifier


def git_output(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/tp03.json"))
args = parser.parse_args()
cfg = json.loads(args.config.read_text(encoding="utf-8"))

mlflow.set_tracking_uri(cfg["tracking_uri"])
mlflow.set_experiment(cfg["experiment"])

source = Path(cfg["source"])
data_sha = sha256_of(source)
commit = git_output("rev-parse", "HEAD")
dirty = bool(git_output("status", "--porcelain"))
if dirty:
    print("ATTENTION : modifications non commitées, le commit enregistré ne décrit pas tout le code.")

# Même split que TP02 (même seed -> mêmes clients)
df = read_telco(source)
train_df, valid_df, _test_df = make_splits(df, cfg["seed"])   # le test n'est pas utilisé
alloc_path = Path("reports/split_allocation.json")
alloc_sha = sha256_of(alloc_path) if alloc_path.exists() else None

label_map = {"No": 0, "Yes": 1}
X_train, y_train = parse_features(train_df), train_df[TARGET].map(label_map)
X_valid, y_valid = parse_features(valid_df), valid_df[TARGET].map(label_map)

train_dataset = mlflow.data.from_pandas(train_df, name="telco_train", targets=TARGET)
valid_dataset = mlflow.data.from_pandas(valid_df, name="telco_validation", targets=TARGET)

for cand in cfg["candidates"]:
    with mlflow.start_run(run_name=cand["run_name"]) as run:
        mlflow.log_params({
            "model_type": cand["model"],
            "seed": cfg["seed"],
            "threshold": cfg["threshold"],
            "features": ",".join(ALL_FEATURES),
            **{f"hp_{k}": v for k, v in cand["params"].items()},
        })
        mlflow.set_tags({
            "git_commit": commit or "unknown",
            "git_dirty": str(dirty),
            "data_sha256": data_sha,
        })
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(valid_dataset, context="validation")

        pipe = build_pipeline(make_classifier(cand["model"], cand["params"]))  # neuf à chaque run
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        mlflow.log_metric("fit_seconds", round(time.perf_counter() - t0, 3))

        m = evaluate_model(pipe, X_valid, y_valid, cfg["threshold"])
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if k != "confusion_matrix"})

        # Matrice de confusion : lignes = vrai, colonnes = prédit, ordre [No, Yes]
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrix_validation.csv"
            pd.DataFrame(
                m["confusion_matrix"], index=["true_No", "true_Yes"], columns=["pred_No", "pred_Yes"]
            ).to_csv(cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        mlflow.log_dict({
            "data": {"path": str(source), "sha256": data_sha},
            "split": {"seed": cfg["seed"], "allocation_file": str(alloc_path), "allocation_sha256": alloc_sha},
            "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
            "class_mapping": label_map,
            "positive_class": "Churn = Yes (1)",
            "evaluation": {
                "split": "validation",
                "primary_metric": "average_precision (scores continus de la classe 1)",
                "threshold": cfg["threshold"],
                "confusion_matrix_layout": "rows=true [No, Yes], cols=predicted [No, Yes]",
            },
            "code": {"git_commit": commit, "uncommitted_changes": dirty},
        }, "provenance.json")

        # Prédictions de référence pour le test de rechargement
        sample = valid_df[ALL_FEATURES].head(20)
        idx = list(pipe.classes_).index(1)
        ref_scores = pipe.predict_proba(parse_features(sample))[:, idx].tolist()
        mlflow.log_dict(
            {"inputs": json.loads(sample.to_json(orient="records")), "scores": ref_scores},
            "reference_predictions.json",
        )

        # Pipeline complet (prétraitement + classifieur)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="churn_pipeline",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        print(f"{cand['run_name']:15s} run_id={run.info.run_id} "
              f"AP={m['average_precision']:.4f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"model_uri={model_info.model_uri}")