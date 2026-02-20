# src/part1_classic.py
from __future__ import annotations

from typing import Dict, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def get_part1_models(seed: int = 42) -> Dict[str, object]:
    """
    Returns the 4 required classifiers with reasonable defaults.
    """
    return {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",   # supports multiclass
            C=4.0,
        ),
        "LinearSVC": LinearSVC(C=2.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def build_classic_pipeline(
    estimator: object,
    vectorizer_type: str = "tfidf",
    max_features: int = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    stop_words: Optional[str] = "english",
) -> Pipeline:
    """
    Builds a sklearn Pipeline: vectorizer -> estimator.
    Using Pipeline prevents leakage (vectorizer fit only on train during .fit()).
    """
    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=stop_words,
        )
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=stop_words,
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown vectorizer_type: {vectorizer_type}")

    return Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("model", estimator),
        ]
    )
