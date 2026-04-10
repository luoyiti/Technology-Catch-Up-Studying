# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research-oriented analysis project studying **technological catch-up** using Web of Science paper data. It examines China's position relative to the United States in nuclear-science-related research through topic modeling, semantic similarity networks, citation networks, and temporal analysis.

Core research questions:
- Whether China and the US occupy similar or different technology topic spaces
- Whether China catches up through semantic proximity or asymmetric citation dependency
- How capability gaps manifest over time

## Environment Setup

```bash
uv sync
```

Dependencies are declared in `pyproject.toml`. Core packages include: `sentence-transformers`, `bertopic`, `umap-learn`, `hdbscan`, `faiss-cpu`, `pandas`, `scikit-learn`, `statsmodels`, `pyarrow`.

## Running Notebooks

The project is notebook-driven. Execute a notebook with:
```bash
jupyter nbconvert --to notebook --execute <notebook>.ipynb --output <notebook>.ipynb
```

## Core Notebooks

| Notebook | Purpose |
|----------|---------|
| `cluster.ipynb` | CN-US topic modeling, capability gaps, temporal evolution, lead-lag analysis |
| `cluster-China-Japan.ipynb` | CN-JP comparison (secondary reference) |
| `cluster_kmeans.ipynb`, `cluster_gmm.ipynb`, `cluster_fuzzy.ipynb` | Alternative clustering approaches |
| `paper_knn_graph.ipynb` / `data/paper_knn_graph_2.ipynb` | Semantic KNN graph construction and network analysis |
| `paper_pure_citation_net.ipynb`, `paper_citation_graph.ipynb` | Citation network from WoS CR field (three-layer matching) |
| `time_varying_cn_us_citation_trends.ipynb` | Year-by-year logistic regression for citation mechanism analysis |

## Architecture

- **Notebooks are the primary artifacts** — they contain the full analysis pipeline from data loading to visualization
- Python generation scripts (`_gen2.py`, `_gen_citation_notebook.py`) programmatically construct notebooks using `nbformat`
- Notebooks depend on cached intermediate results (embeddings, topic models) — run in published order for reproducibility
- Notebooks are often run independently and write outputs to `output/` subdirectories

## Key Input Data

- `data/dataCleanSCIE.csv` — Main dataset (~25,794 papers from WoS)
- `data/dataCleanSCIE_Cut.csv` — Smaller subset
- Country labels are normalized to `CN / US / JP / Other`

## Key Output Directories

- `output/cluster_results/` — Topic modeling, capability gap metrics, temporal analysis
- `output/graph/` — Semantic KNN graph results
- `output/citation_graph/` — Citation network (DOI/strict/loose matching)
- `output/compare_networks/` — Semantic vs citation network overlap analysis
- `output/timeaware_similarity/` — Time-constrained semantic similarity evaluation

## Pipeline Summary

1. **Topic Space**: SPECTER2 embeddings → BERTopic → UMAP+HDBSCAN → 63 topics
2. **Semantic Graph**: KNN on embeddings → Leiden communities → modularity ~0.83
3. **Citation Graph**: WoS CR field → three-layer matching → directed graph ~49k edges
4. **Time Analysis**: Rolling window logit fitting → lead-lag detection → time-aware precision improvement (0.26 → 0.41)

## Notebooks Execution Order (for full reproducibility)

1. `cluster.ipynb`
2. `paper_knn_graph.ipynb`
3. `paper_pure_citation_net.ipynb`
4. `time_varying_cn_us_citation_trends.ipynb`
5. `cluster-China-Japan.ipynb`
