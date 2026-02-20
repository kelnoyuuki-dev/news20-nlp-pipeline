# src/part2_embeddings.py
from __future__ import annotations

from typing import Dict

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_part2_models(seed: int = 42) -> Dict[str, object]:
    """
    Same 4 classifiers as Part 1.
    Note: MNB is not a good match for dense embeddings (expects nonnegative count-ish features).
    We keep it for completeness per assignment.
    """
    return {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=4.0,
        ),
        "LinearSVC": LinearSVC(C=2.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def build_embedding_pipeline(model_name: str, estimator: object) -> object:
    """
    For dense embeddings:
    - Scaling helps LR/SVM (and often improves stability).
    - Scaling is NOT necessary for RandomForest.
    - MNB generally expects nonnegative features; embeddings contain negatives -> poor fit.
      We still run it for completeness.
    """
    if model_name in {"LogisticRegression", "LinearSVC"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                ("model", estimator),
            ]
        )
    else:
        # MNB / RF run directly
        return estimator
