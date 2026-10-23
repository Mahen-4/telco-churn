import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(source: Path) -> dict:
    df = pd.read_csv(source)

    # Valeurs vides ou composées d'espaces (différent de isna())
    as_str = df.astype("string")
    blank = as_str.apply(lambda s: s.str.strip().eq("")).sum()

    target_counts = df["Churn"].value_counts(dropna=False)
    target_props = df["Churn"].value_counts(dropna=False, normalize=True)

    return {
        "source": {"path": str(source), "sha256": sha256_of(source)},
        "environment": {
            "python": sys.version,
            "pandas": importlib.metadata.version("pandas"),
        },
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "proportions": {str(k): float(v) for k, v in target_props.items()},
        },
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "missing_isna": {c: int(v) for c, v in df.isna().sum().items()},
        "blank_or_whitespace": {c: int(v) for c, v in blank.items()},
    }


def write_report(source: Path, output: Path) -> None:
    report = inspect(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")