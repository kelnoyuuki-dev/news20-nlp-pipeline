# src/cluster_viz.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

def plot_umap_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP Projection of Clusters",
) -> None:
    """
    UMAP 2D projection + scatter plot colored by cluster label.
    X: (n_samples, dim) embeddings (float32 ok)
    labels: (n_samples,) cluster ids
    """
    try:
        import umap  # umap-learn
    except ImportError as e:
        raise ImportError("Please `pip install umap-learn` to use UMAP visualization.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=6, alpha=0.75)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
