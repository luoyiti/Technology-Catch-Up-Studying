from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def load_json_config(config_path: str | Path) -> Dict[str, Any]:
    """Load JSON config from a path and return a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_reproducibility(seed: int) -> None:
    """Set reproducibility knobs used across notebook runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_output_dirs(output_dir: str | Path, extra_subdirs: Iterable[str] = ()) -> Dict[str, Path]:
    """Create output root and requested subdirectories."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    created: Dict[str, Path] = {"output_dir": root}
    for sub in extra_subdirs:
        p = root / sub
        p.mkdir(parents=True, exist_ok=True)
        created[sub] = p
    return created


def build_custom_stopwords(vectorizer_cfg: Dict[str, Any], english_stopwords: Iterable[str]) -> list[str]:
    """Merge sklearn stopwords with custom groups from config."""
    groups = vectorizer_cfg.get("extra_stopword_groups", {})
    merged = set(english_stopwords)
    for key in ("academic", "nuclear_background", "noise"):
        merged |= set(groups.get(key, []))
    return sorted(merged)


def unpack_runtime_config(cfg: Dict[str, Any], english_stopwords: Iterable[str]) -> Dict[str, Any]:
    """Flatten frequently used config values for notebook globals."""
    paths_cfg = cfg["paths"]
    minimax_cfg = cfg["minimax"]
    sk_cfg = cfg["specter2_keybert"]
    umap_cfg = cfg["umap_defaults"]
    agg_cfg = cfg["agglomerative_defaults"]
    search_cfg = cfg["agglo_search"]
    vec_cfg = cfg["vectorizer"]

    return {
        "SEED": int(cfg["reproducibility"]["seed"]),
        "DATA_PATH": paths_cfg["data_csv"],
        "OUTPUT_DIR": paths_cfg["output_dir"],
        "HIERARCHICAL_SUBDIR": paths_cfg["hierarchical_subdir"],
        "SPECTER_MODEL": cfg["specter"]["clustering_model"],
        "SPECTER2_BASE_MODEL": sk_cfg["base_model"],
        "SPECTER2_ADAPTER_MODEL": sk_cfg["adapter_model"],
        "SPECTER2_CACHE_ROOT": Path(sk_cfg["cache_root"]),
        "SPECTER2_KEYBERT_BATCH_SIZE": int(sk_cfg["batch_size"]),
        "TOP_N_KEYWORDS": int(sk_cfg["top_n_keywords"]),
        "MINIMAX_API_BASE": minimax_cfg["api_base"],
        "MINIMAX_CHAT_ENDPOINT": minimax_cfg["chat_endpoint"],
        "MINIMAX_MODEL": minimax_cfg["model"],
        "MINIMAX_API_KEY_ENV": minimax_cfg.get("api_key_env", "MINIMAX_API_KEY"),
        "MINIMAX_TIMEOUT": int(minimax_cfg["timeout_seconds"]),
        "MINIMAX_MAX_RETRIES": int(minimax_cfg["max_retries"]),
        "MINIMAX_RETRY_SLEEP": float(minimax_cfg["retry_sleep_seconds"]),
        "TOPIC_SUMMARY_CACHE_FILENAME": minimax_cfg["topic_summary_cache_filename"],
        "UMAP_N_NEIGHBORS": int(umap_cfg["n_neighbors"]),
        "UMAP_N_COMPONENTS": int(umap_cfg["n_components"]),
        "UMAP_MIN_DIST": float(umap_cfg["min_dist"]),
        "UMAP_METRIC": str(umap_cfg["metric"]),
        "AGGLO_N_CLUSTERS": int(agg_cfg["n_clusters"]),
        "AGGLO_LINKAGE": str(agg_cfg["linkage"]),
        "AGGLO_METRIC": str(agg_cfg["metric"]),
        "AGGLO_SEARCH_SUBSET_N": int(search_cfg["subset_n"]),
        "AGGLO_SEARCH_TOPK_FULL": int(search_cfg["topk_full"]),
        "AGGLO_SEARCH_MAX_EVAL_POINTS": int(search_cfg["max_eval_points"]),
        "AGGLO_SEARCH_LINKAGE_METRIC_COARSE": [tuple(pair) for pair in search_cfg["linkage_metric_coarse"]],
        "AGGLO_SEARCH_LINKAGE_METRIC_FULL": [tuple(pair) for pair in search_cfg["linkage_metric_full"]],
        "CUSTOM_STOPWORDS": build_custom_stopwords(vec_cfg, english_stopwords),
        "NGRAM_RANGE": tuple(int(x) for x in vec_cfg["ngram_range"]),
        "MAX_FEATURES": int(vec_cfg["max_features"]),
        "BATCH_SIZE": int(cfg["embedding"]["batch_size"]),
        "TOP_K_TOPICS": int(cfg["visualization"]["top_k_topics"]),
    }
