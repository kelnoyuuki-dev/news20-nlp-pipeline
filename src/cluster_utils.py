# src/cluster_utils.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Sequence

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_elbow_inertia(X: np.ndarray, ks: List[int], seed: int = 42) -> List[float]:
    inertias: List[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X)
        inertias.append(float(km.inertia_))
    return inertias


def choose_k_by_max_curvature(ks: List[int], inertias: List[float]) -> int:
    """
    Simple elbow selection via max curvature (discrete 2nd difference).
    """
    if len(ks) != len(inertias) or len(ks) < 3:
        return int(ks[0])

    y = np.array(inertias, dtype=float)
    second = []
    for i in range(1, len(y) - 1):
        second.append(y[i - 1] - 2 * y[i] + y[i + 1])
    second = np.array(second)

    best_i = int(np.argmax(second)) + 1
    return int(ks[best_i])


def save_elbow_plot(ks: List[int], inertias: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia (KMeans)")
    plt.title("Elbow Method (Inertia vs K)")
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return km


def representatives_closest_to_centroid(
    X: np.ndarray,
    indices: np.ndarray,
    centroid: np.ndarray,
    texts: Sequence[str],
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """
    Pick representative documents in a cluster as those closest to centroid (Euclidean).
    """
    if indices.size == 0:
        return []

    Xc = X[indices]
    d = np.linalg.norm(Xc - centroid.reshape(1, -1), axis=1)
    order = np.argsort(d)[: min(top_n, indices.size)]

    reps: List[Dict[str, Any]] = []
    for rank, local_i in enumerate(order):
        global_i = int(indices[local_i])
        text = str(texts[global_i])
        snippet = " ".join(text.split())[:600]
        reps.append(
            {
                "global_index": global_i,
                "rank": int(rank),
                "distance": float(d[local_i]),
                "text": snippet,
            }
        )
    return reps


def tfidf_fallback_label(texts: Sequence[str], top_k: int = 8) -> Dict[str, Any]:
    """
    Fallback label using top TF-IDF terms inside the cluster.

    Important: clusters can be small, so we auto-adjust min_df to avoid errors.
    """
    texts = [str(t) for t in texts if str(t).strip()]
    n = len(texts)

    if n == 0:
        return {"label": "empty cluster", "keywords": []}

    # Auto-adjust min_df so small clusters don't crash
    # - if n < 5, min_df=1
    # - else min_df=5 (your original choice)
    min_df = 1 if n < 5 else 5

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.6,
        min_df=min_df,
        token_pattern=r"\b[a-zA-Z]{4,}\b",
        ngram_range=(1, 2),
        max_features=20000,
    )

    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # If still fails (rare), relax constraints
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            token_pattern=r"\b[a-zA-Z]{3,}\b",
            ngram_range=(1, 1),
            max_features=20000,
        )
        X = vectorizer.fit_transform(texts)

    mean_scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top_k = min(int(top_k), len(terms))
    if top_k == 0:
        return {"label": "no terms", "keywords": []}

    top_idx = mean_scores.argsort()[::-1][:top_k]
    keywords = [terms[i] for i in top_idx]

    label = " / ".join(keywords[:3]) if len(keywords) >= 3 else " / ".join(keywords)
    return {"label": label, "keywords": keywords}

