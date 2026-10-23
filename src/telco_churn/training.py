import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    "baseline": lambda p: DummyClassifier(strategy="most_frequent", **p),
    "logistic": lambda p: LogisticRegression(**p),
    "decision_tree": lambda p: DecisionTreeClassifier(**p),
    "random_forest": lambda p: RandomForestClassifier(**p),
}


def make_classifier(name: str, params: dict):
    """Retourne un classifieur NON entraîné, neuf à chaque appel."""
    return CLASSIFIERS[name](params)


def positive_scores(pipe, X) -> np.ndarray:
    """Score de la classe positive (1), trouvé via classes_ et non supposé."""
    idx = list(pipe.classes_).index(1)
    return pipe.predict_proba(X)[:, idx]


def evaluate_model(pipe, X, y, threshold: float = 0.5) -> dict:
    scores = positive_scores(pipe, X)
    pred = (scores >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        # lignes = vrai [No, Yes], colonnes = prédit [No, Yes]
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }