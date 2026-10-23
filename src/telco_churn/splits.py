import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "Churn"
ID_COL = "customerID"


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Split stratifié 60/20/20. Même seed = mêmes clients dans chaque jeu."""
    train_df, rest_df = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=seed
    )
    valid_df, test_df = train_test_split(
        rest_df, test_size=0.5, stratify=rest_df[TARGET], random_state=seed
    )
    return train_df, valid_df, test_df


def save_allocation(train_df, valid_df, test_df, path: Path, seed: int = SEED) -> None:
    """Sauvegarde les customerID de chaque jeu (pour prouver qu'ils sont disjoints)."""
    allocation = {
        "seed": seed,
        "train": sorted(train_df[ID_COL]),
        "validation": sorted(valid_df[ID_COL]),
        "test": sorted(test_df[ID_COL]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allocation), encoding="utf-8")