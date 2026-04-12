from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from bertopic.backend import BaseEmbedder
from sklearn.preprocessing import normalize


@dataclass
class Specter2AdapterBackend(BaseEmbedder):
    """BERTopic embedding backend that uses the SPECTER2 adapter stack."""

    base_repo_id: str
    adapter_repo_id: str
    cache_root: Path
    adapter_name: str = "proximity"
    batch_size: int = 16
    max_length: int = 512
    device: str = "cpu"

    def __post_init__(self) -> None:
        super().__init__()

        try:
            import torch
            from adapters import AutoAdapterModel
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BERTopic representation backend requires adapters, huggingface-hub, transformers, and torch."
            ) from exc

        self._torch = torch

        base_dir = self.cache_root / "allenai__specter2_base"
        adapter_dir = self.cache_root / "allenai__specter2"

        base_path = snapshot_download(repo_id=self.base_repo_id, local_dir=str(base_dir))
        adapter_path = snapshot_download(repo_id=self.adapter_repo_id, local_dir=str(adapter_dir))

        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.model = AutoAdapterModel.from_pretrained(base_path)
        self.model.load_adapter(adapter_path, load_as=self.adapter_name, set_active=True)
        self.model.set_active_adapters(self.adapter_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._hf_model = self.base_repo_id

    def _encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        bs = int(batch_size or self.batch_size)
        encoded_batches: List[np.ndarray] = []

        with self._torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = texts[start : start + bs]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                encoded_batches.append(batch_emb)

        embeddings_out = np.vstack(encoded_batches)
        return normalize(embeddings_out, norm="l2")

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        return self._encode(documents)

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        return self._encode(words)

    def embed_documents(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        return self._encode(documents)


_ENCODER_CACHE: Specter2AdapterBackend | None = None


def get_specter2_representation_encoder(
    *,
    base_model: str,
    adapter_model: str,
    cache_root: Path,
    batch_size: int,
    device: str,
    force_reload: bool = False,
) -> Specter2AdapterBackend:
    """Return a cached SPECTER2 representation backend instance."""
    global _ENCODER_CACHE
    if force_reload or _ENCODER_CACHE is None:
        _ENCODER_CACHE = Specter2AdapterBackend(
            base_repo_id=base_model,
            adapter_repo_id=adapter_model,
            cache_root=cache_root,
            batch_size=batch_size,
            device=device,
        )
    return _ENCODER_CACHE
