# src/embedding_cache.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple, Sequence, Optional

import numpy as np


def _stable_hash(obj: dict) -> str:
    """
    Stable hash for cache keys (order-independent via sort_keys).
    """
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(s).hexdigest()  # ok for cache keys


def _cache_paths(
    cache_dir: Path,
    st_model_name: str,
    split_name: str,
    remove: Tuple[str, ...],
    batch_size: int,
    seed: int,
    normalize: bool,
) -> tuple[Path, Path]:
    """
    Returns (npz_path, meta_path)
    """
    cfg = {
        "st_model": st_model_name,
        "split": split_name,
        "remove": list(remove),
        "batch_size": batch_size,
        "seed": seed,
        "normalize": normalize,
    }
    key = _stable_hash(cfg)
    stem = f"emb_{split_name}_{st_model_name.replace('/', '__')}_{key}"
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"
    return npz_path, meta_path


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def embed_texts_cached(
    texts: Sequence[str],
    st_model_name: str,
    cache_dir: Path,
    split_name: str,
    remove: Tuple[str, ...],
    batch_size: int,
    seed: int,
    normalize: bool,
    dtype=np.float32,
) -> np.ndarray:
    """
    Loads cached embeddings if present; otherwise encodes with SentenceTransformer and saves to disk.

    Saves:
      - NPZ with array 'emb'
      - JSON meta with config + shape
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _cache_paths(
        cache_dir=cache_dir,
        st_model_name=st_model_name,
        split_name=split_name,
        remove=remove,
        batch_size=batch_size,
        seed=seed,
        normalize=normalize,
    )

    if npz_path.exists():
        data = np.load(npz_path)
        emb = data["emb"]
        return emb.astype(dtype, copy=False)

    # Lazy import so Part 1 doesn't require sentence-transformers installed
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(st_model_name)

    # Encode in batches; convert to float32 for memory
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll control normalization ourselves
    ).astype(dtype, copy=False)

    if normalize:
        emb = _l2_normalize(emb).astype(dtype, copy=False)

    np.savez_compressed(npz_path, emb=emb)
    meta = {
        "st_model": st_model_name,
        "split": split_name,
        "remove": list(remove),
        "batch_size": batch_size,
        "seed": seed,
        "normalize": normalize,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return emb
