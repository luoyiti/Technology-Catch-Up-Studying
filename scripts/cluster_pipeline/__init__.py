"""Utilities to keep notebooks lean and structured."""

from .config_utils import load_json_config, set_reproducibility, unpack_runtime_config
from .data_prep import load_and_clean_papers
from .device_utils import select_device
from .embedding_utils import load_or_create_embeddings
from .agglom_pipeline import (
    build_agglomerative_model,
    compute_topic_confidence,
    plot_topic_centroid_dendrogram,
    run_agglomerative_search,
)
from .specter2_backend import get_specter2_representation_encoder

__all__ = [
    "load_json_config",
    "set_reproducibility",
    "unpack_runtime_config",
    "load_and_clean_papers",
    "select_device",
    "load_or_create_embeddings",
    "build_agglomerative_model",
    "compute_topic_confidence",
    "plot_topic_centroid_dendrogram",
    "run_agglomerative_search",
    "get_specter2_representation_encoder",
]
