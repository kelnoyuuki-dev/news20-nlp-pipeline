# src/data_loader.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Sequence
import re

from sklearn.datasets import fetch_20newsgroups


_WHITESPACE_RE = re.compile(r"\s+")


def minimal_clean(text: str) -> str:
    """
    Minimal, consistent cleaning:
    - normalize whitespace
    - strip leading/trailing spaces
    (Do NOT aggressively remove punctuation / casing; vectorizers handle that.)
    """
    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_20newsgroups_train_test(
    remove: Tuple[str, ...] = ("headers", "footers", "quotes"),
    categories: Optional[Sequence[str]] = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Loads 20 Newsgroups using the official train/test split.
    Returns cleaned texts + labels + target names.
    """
    train = fetch_20newsgroups(
        subset="train",
        remove=remove,
        categories=categories,
        shuffle=shuffle,
        random_state=random_state,
    )
    test = fetch_20newsgroups(
        subset="test",
        remove=remove,
        categories=categories,
        shuffle=shuffle,
        random_state=random_state,
    )

    X_train = [minimal_clean(t) for t in train.data]
    X_test = [minimal_clean(t) for t in test.data]

    return {
        "X_train": X_train,
        "y_train": train.target,
        "X_test": X_test,
        "y_test": test.target,
        "target_names": train.target_names,  # same mapping as test
    }
