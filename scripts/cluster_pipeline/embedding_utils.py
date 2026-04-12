from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def _encode_with_fallback(
    docs: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, str]:
    """Encode with preferred device and fallback to cpu when needed."""
    try:
        encoder = SentenceTransformer(model_name, device=device)
        used_device = device
    except Exception:
        used_device = "cpu"
        encoder = SentenceTransformer(model_name, device=used_device)

    emb = encoder.encode(
        list(docs),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = normalize(emb, norm="l2")
    return emb, used_device


def load_or_create_embeddings(
    docs: Sequence[str],
    cache_path: str | Path,
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, bool, str]:
    """Load cached embeddings or compute and persist them.

    Returns: (embeddings, cache_hit, used_device)
    """
    cache = Path(cache_path)
    if cache.exists():
        emb = np.load(cache)
        return emb, True, device

    emb, used_device = _encode_with_fallback(docs, model_name, device, batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb, False, used_device
