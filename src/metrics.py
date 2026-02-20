# src/metrics.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt


def evaluate_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Fit model, time it, predict test, compute accuracy + macro-F1 + confusion matrix.
    Returns metrics plus y_pred and confusion for saving artifacts.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    y_pred = model.predict(X_test)
    t2 = time.time()

    acc = float(accuracy_score(y_test, y_pred))
    macro = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro,
        "train_seconds": float(t1 - t0),
        "predict_seconds": float(t2 - t1),
        "y_pred": y_pred,
        "confusion": cm,
    }


def top_confusion_pairs(
    confusion: np.ndarray,
    target_names: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Finds top off-diagonal confusion counts (true -> predicted).
    Returns list of dicts sorted descending by count.
    """
    cm = confusion.copy()
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = int(cm[i, j])
            if c > 0:
                pairs.append(
                    {
                        "true_idx": i,
                        "pred_idx": j,
                        "true_label": target_names[i],
                        "pred_label": target_names[j],
                        "count": c,
                    }
                )

    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_k]


def save_confusion_matrix_png(
    confusion: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "Confusion Matrix",
    max_classes_to_show: int = 20,
) -> None:
    """
    Saves confusion matrix as a PNG.
    For 20 Newsgroups it's exactly 20 classes; keep it readable via size/rotation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Safety: if someone subsets categories, class_names length may differ
    n = confusion.shape[0]
    names = class_names[:n]
    if n > max_classes_to_show:
        # Not expected for 20NG, but keep safe
        names = names[:max_classes_to_show]
        confusion = confusion[:max_classes_to_show, :max_classes_to_show]

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, names, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
