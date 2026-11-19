import json
from pathlib import Path

import mlflow

URI = "http://127.0.0.1:5000"
EXPERIMENT = "telco-churn-tp03"
NAME = "telco-churn"

mlflow.set_tracking_uri(URI)
exp = mlflow.get_experiment_by_name(EXPERIMENT)
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="params.model_type != 'baseline'",
    order_by=["metrics.val_average_precision DESC"],
)
best = runs.iloc[0]
run_id = best["run_id"]
model_uri = best["tags.logged_model_uri"]
print("Meilleur run :", best["tags.mlflow.runName"], run_id, "AP =", best["metrics.val_average_precision"])

version = mlflow.register_model(model_uri, NAME)
client = mlflow.MlflowClient()
client.set_registered_model_alias(NAME, "candidate", version.version)
print(f"Enregistré : {NAME} version {version.version}, alias 'candidate'")

reference = mlflow.artifacts.load_dict(f"runs:/{run_id}/reference_predictions.json")
Path("reports").mkdir(exist_ok=True)
Path("reports/tp03_registered.json").write_text(json.dumps({
    "run_id": run_id,
    "model_uri": model_uri,
    "registered_name": NAME,
    "version": str(version.version),
    "alias": "candidate",
    "git_commit": best.get("tags.git_commit"),
    "data_sha256": best.get("tags.data_sha256"),
    "reference": reference,
}, indent=2), encoding="utf-8")