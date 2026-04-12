from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from umap import UMAP


def build_agglomerative_model(
    n_clusters: int,
    linkage_name: str,
    metric_name: str,
    connectivity=None,
) -> AgglomerativeClustering:
    """Build AgglomerativeClustering with sklearn-version-compatible args."""
    linkage_name = str(linkage_name)
    metric_name = str(metric_name)
    if linkage_name == "ward":
        metric_name = "euclidean"

    kwargs: dict[str, Any] = {
        "n_clusters": int(n_clusters),
        "linkage": linkage_name,
    }
    if connectivity is not None:
        kwargs["connectivity"] = connectivity

    try:
        return AgglomerativeClustering(metric=metric_name, **kwargs)
    except TypeError:
        return AgglomerativeClustering(affinity=metric_name, **kwargs)


def compute_topic_confidence(embeddings_in, topics_in):
    """Surrogate topic confidence based on cosine(sample, topic-centroid)."""
    emb = np.asarray(embeddings_in, dtype=float)
    labels = np.asarray(topics_in)
    n = len(labels)
    conf = np.full(n, np.nan, dtype=float)
    if n == 0:
        return conf

    centroids = {}
    for topic_id in np.unique(labels):
        idx = np.where(labels == topic_id)[0]
        if idx.size == 1:
            centroids[topic_id] = emb[idx[0]]
        elif idx.size > 1:
            centroids[topic_id] = emb[idx].mean(axis=0)

    for i, topic_id in enumerate(labels):
        centroid = centroids.get(topic_id)
        if centroid is None:
            continue
        vec = emb[i]
        denom = np.linalg.norm(vec) * np.linalg.norm(centroid)
        conf[i] = 1.0 if denom <= 1e-12 else float(np.dot(vec, centroid) / denom)

    conf = np.clip((conf + 1.0) / 2.0, 0.0, 1.0)

    vc = pd.Series(labels).value_counts()
    singleton_topics = set(vc[vc == 1].index.tolist())
    if singleton_topics:
        singleton_mask = pd.Series(labels).isin(singleton_topics).values
        conf[singleton_mask] = 1.0

    return conf


def compute_topic_size_stats(topics_in):
    """Return topic-size statistics for hierarchical clustering labels."""
    labels = pd.Series(topics_in)
    sizes = labels.value_counts().sort_values(ascending=False)
    if sizes.empty:
        return {
            "n_topics": 0,
            "topic_size_min": np.nan,
            "topic_size_median": np.nan,
            "topic_size_max": np.nan,
            "max_topic_share": np.nan,
            "topic_size_cv": np.nan,
            "outlier_ratio": 0.0,
        }
    return {
        "n_topics": int(sizes.shape[0]),
        "topic_size_min": float(sizes.min()),
        "topic_size_median": float(sizes.median()),
        "topic_size_max": float(sizes.max()),
        "max_topic_share": float(sizes.max() / sizes.sum()),
        "topic_size_cv": float(sizes.std(ddof=0) / (sizes.mean() + 1e-12)),
        "outlier_ratio": 0.0,
    }


def _safe_cluster_scores(reduced, labels, seed=42, max_eval_points=5000):
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return np.nan, np.nan

    n = labels.shape[0]
    if n > max_eval_points:
        rs = np.random.RandomState(seed)
        idx = rs.choice(np.arange(n), size=max_eval_points, replace=False)
        x_eval = reduced[idx]
        y_eval = labels[idx]
        if np.unique(y_eval).size < 2:
            x_eval = reduced
            y_eval = labels
    else:
        x_eval = reduced
        y_eval = labels

    try:
        sil = float(silhouette_score(x_eval, y_eval))
    except Exception:
        sil = np.nan
    try:
        dbi = float(davies_bouldin_score(x_eval, y_eval))
    except Exception:
        dbi = np.nan
    return sil, dbi


def _score_penalty_for_topic_count(n_topics, low=20, high=120):
    if pd.isna(n_topics):
        return 2.0
    if low <= n_topics <= high:
        return 0.0
    if n_topics < low:
        return float((low - n_topics) / max(1, low))
    return float((n_topics - high) / max(1, high))


def _add_rank_aggregate(df_in):
    df = df_in.copy()
    df["rank_silhouette"] = df["silhouette"].rank(ascending=False, method="average", na_option="bottom")
    df["rank_dbi"] = df["dbi"].rank(ascending=True, method="average", na_option="bottom")
    df["rank_size_cv"] = df["topic_size_cv"].rank(ascending=True, method="average", na_option="bottom")
    df["rank_max_share"] = df["max_topic_share"].rank(ascending=True, method="average", na_option="bottom")
    df["topic_count_penalty"] = df["n_topics"].map(_score_penalty_for_topic_count)

    if "cn_coverage" in df.columns:
        df["rank_cn_coverage"] = df["cn_coverage"].rank(ascending=False, method="average", na_option="bottom")
    else:
        df["rank_cn_coverage"] = 0.0

    if "us_coverage" in df.columns:
        df["rank_us_coverage"] = df["us_coverage"].rank(ascending=False, method="average", na_option="bottom")
    else:
        df["rank_us_coverage"] = 0.0

    if "lead_lag_n_topics" in df.columns:
        df["rank_lead_lag_n_topics"] = df["lead_lag_n_topics"].rank(ascending=False, method="average", na_option="bottom")
    else:
        df["rank_lead_lag_n_topics"] = 0.0

    df["rank_score"] = (
        df["rank_silhouette"]
        + df["rank_dbi"]
        + df["rank_size_cv"]
        + df["rank_max_share"]
        + df["topic_count_penalty"]
        + 0.25 * df["rank_cn_coverage"]
        + 0.25 * df["rank_us_coverage"]
        + 0.15 * df["rank_lead_lag_n_topics"]
    )
    return df


def _evaluate_config_with_reduced(
    reduced_embeddings,
    cfg,
    seed=42,
    max_eval_points=5000,
    evaluate_gap_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
):
    rec = {
        "status": "ok",
        "error_msg": "",
        "umap_n_neighbors": int(cfg["umap_n_neighbors"]),
        "umap_n_components": int(cfg["umap_n_components"]),
        "umap_min_dist": float(cfg["umap_min_dist"]),
        "umap_metric": str(cfg["umap_metric"]),
        "agglom_n_clusters": int(cfg["agglom_n_clusters"]),
        "agglom_linkage": str(cfg["agglom_linkage"]),
        "agglom_metric": str(cfg["agglom_metric"]),
    }

    try:
        n_samples = reduced_embeddings.shape[0]
        n_neighbors_conn = max(2, min(30, n_samples - 1))
        conn = kneighbors_graph(reduced_embeddings, n_neighbors=n_neighbors_conn, include_self=False)

        clusterer = build_agglomerative_model(
            n_clusters=cfg["agglom_n_clusters"],
            linkage_name=cfg["agglom_linkage"],
            metric_name=cfg["agglom_metric"],
            connectivity=conn,
        )

        labels = clusterer.fit_predict(reduced_embeddings)
        sil, dbi = _safe_cluster_scores(
            reduced_embeddings,
            labels,
            seed=seed,
            max_eval_points=max_eval_points,
        )

        rec.update(compute_topic_size_stats(labels))
        rec["silhouette"] = sil
        rec["dbi"] = dbi
        rec["labels"] = labels

        if callable(evaluate_gap_fn):
            try:
                gap = evaluate_gap_fn(labels)
                if isinstance(gap, dict):
                    rec.update(gap)
            except Exception:
                pass

    except Exception as e:
        rec.update(
            {
                "status": "error",
                "error_msg": f"{type(e).__name__}: {e}",
                "silhouette": np.nan,
                "dbi": np.nan,
                "n_topics": np.nan,
                "topic_size_min": np.nan,
                "topic_size_median": np.nan,
                "topic_size_max": np.nan,
                "max_topic_share": np.nan,
                "topic_size_cv": np.nan,
                "outlier_ratio": 0.0,
                "labels": None,
            }
        )

    return rec


def plot_agglomerative_search(search_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = search_df.copy()
    d = d[d["status"] == "ok"].copy()
    if d.empty:
        print("No valid agglomerative search rows to plot.")
        return

    markers = {"cosine": "o", "euclidean": "s"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for (lnk, met), grp in d.groupby(["agglom_linkage", "agglom_metric"], dropna=False):
        ax.scatter(
            grp["agglom_n_clusters"],
            grp["silhouette"],
            label=f"{lnk}/{met}",
            alpha=0.85,
            marker=markers.get(str(met), "o"),
            s=52,
        )
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("silhouette")
    ax.set_title("Agglomerative Search: silhouette vs n_clusters")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "agglom_search_silhouette.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (lnk, met), grp in d.groupby(["agglom_linkage", "agglom_metric"], dropna=False):
        ax.scatter(
            grp["agglom_n_clusters"],
            grp["dbi"],
            label=f"{lnk}/{met}",
            alpha=0.85,
            marker=markers.get(str(met), "o"),
            s=52,
        )
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("Davies-Bouldin Index (lower better)")
    ax.set_title("Agglomerative Search: DBI vs n_clusters")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "agglom_search_dbi.png", dpi=180)
    plt.close(fig)

    topk = d.sort_values("rank_score", ascending=True).head(12).copy()
    topk["cfg"] = topk.apply(
        lambda r: (
            f"U({int(r['umap_n_neighbors'])},{int(r['umap_n_components'])},{r['umap_min_dist']}) | "
            f"A({int(r['agglom_n_clusters'])},{r['agglom_linkage']},{r['agglom_metric']})"
        ),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(topk["cfg"][::-1], topk["rank_score"][::-1], color="#3C6FE0", alpha=0.85)
    ax.set_xlabel("rank_score (lower is better)")
    ax.set_title("Agglomerative Search: Top-K by aggregate rank score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "agglom_search_topk.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        d["silhouette"],
        d["dbi"],
        s=(d["max_topic_share"].fillna(0.0) * 1000 + 40),
        c=d["n_topics"],
        cmap="viridis",
        alpha=0.75,
    )
    ax.set_xlabel("silhouette (higher better)")
    ax.set_ylabel("DBI (lower better)")
    ax.set_title("Agglomerative Search Bubble: size=max_topic_share, color=n_topics")
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("n_topics")
    plt.tight_layout()
    fig.savefig(out_dir / "agglom_search_heatmap.png", dpi=180)
    plt.close(fig)

    try:
        import plotly.express as px

        fig_html = px.scatter(
            d,
            x="agglom_n_clusters",
            y="silhouette",
            color="agglom_linkage",
            symbol="agglom_metric",
            hover_data=[
                "umap_n_neighbors",
                "umap_n_components",
                "umap_min_dist",
                "dbi",
                "n_topics",
                "rank_score",
            ],
            title="Agglomerative Search (interactive): silhouette vs n_clusters",
        )
        fig_html.write_html(out_dir / "agglom_search_silhouette.html")
    except Exception:
        pass

    print(f"Search plots saved to: {out_dir}")


def plot_topic_centroid_dendrogram(embeddings_in, topics_in, out_png):
    out_png = Path(out_png)
    labels = np.asarray(topics_in)
    emb = np.asarray(embeddings_in, dtype=float)
    unique_topics = sorted(pd.unique(labels))
    if len(unique_topics) < 2:
        print("Skip centroid dendrogram: fewer than 2 topics.")
        return None

    centroids = []
    topic_labels = []
    topic_sizes = pd.Series(labels).value_counts()
    for topic_id in unique_topics:
        idx = np.where(labels == topic_id)[0]
        if idx.size == 0:
            continue
        centroids.append(emb[idx].mean(axis=0))
        topic_labels.append(f"T{int(topic_id)} ({int(topic_sizes.get(topic_id, 0))})")

    if len(centroids) < 2:
        print("Skip centroid dendrogram: valid centroid count < 2.")
        return None

    centroids = np.vstack(centroids)
    z = linkage(centroids, method="average", metric="cosine")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        z,
        labels=topic_labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )
    ax.set_title("Topic Centroid Dendrogram (average linkage, cosine distance)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved centroid dendrogram: {out_png}")
    return out_png


def run_agglomerative_search(
    embeddings_all,
    seed: int,
    output_dir,
    umap_metric: str,
    search_subset_n: int,
    search_topk_full: int,
    search_max_eval_points: int,
    linkage_metric_coarse,
    linkage_metric_full,
    umap_neighbors_grid,
    umap_components_grid,
    umap_min_dist_grid,
    cluster_counts,
    evaluate_gap_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
):
    """Two-stage auto search for UMAP + Agglomerative clustering."""
    rs = np.random.RandomState(seed)
    emb_all = np.asarray(embeddings_all)
    n_docs = emb_all.shape[0]

    if n_docs > 10000:
        subset_n = min(int(search_subset_n), n_docs)
        subset_idx = rs.choice(np.arange(n_docs), size=subset_n, replace=False)
        emb_subset = emb_all[np.sort(subset_idx)]
        print(f"Stage A (coarse): sampled {subset_n:,}/{n_docs:,} docs.")
    else:
        emb_subset = emb_all
        print(f"Stage A (coarse): use full data ({n_docs:,} docs).")

    umap_grid = list(itertools.product(umap_neighbors_grid, umap_components_grid, umap_min_dist_grid))
    linkage_metric_grid = linkage_metric_coarse if n_docs > 10000 else linkage_metric_full
    agg_grid = list(itertools.product(cluster_counts, linkage_metric_grid))
    print(f"Stage A grids: UMAP={len(umap_grid)}, Agglo={len(agg_grid)}")

    coarse_records = []
    for nn, nc, mdist in umap_grid:
        try:
            umap_model_tmp = UMAP(
                n_neighbors=int(nn),
                n_components=int(nc),
                min_dist=float(mdist),
                metric=umap_metric,
                random_state=seed,
                low_memory=True,
            )
            reduced = umap_model_tmp.fit_transform(emb_subset)
        except Exception as e:
            print(f"UMAP coarse failed for ({nn}, {nc}, {mdist}, {umap_metric}): {e}")
            continue

        for nclust, (lnk, met) in agg_grid:
            cfg = {
                "umap_n_neighbors": int(nn),
                "umap_n_components": int(nc),
                "umap_min_dist": float(mdist),
                "umap_metric": str(umap_metric),
                "agglom_n_clusters": int(nclust),
                "agglom_linkage": str(lnk),
                "agglom_metric": str(met),
            }
            rec = _evaluate_config_with_reduced(
                reduced,
                cfg,
                seed=seed,
                max_eval_points=search_max_eval_points,
            )
            rec["stage"] = "coarse"
            coarse_records.append(rec)

    coarse_df = pd.DataFrame(coarse_records)
    if coarse_df.empty:
        raise RuntimeError("Agglomerative coarse search failed: no valid configurations.")

    coarse_df = _add_rank_aggregate(coarse_df)
    coarse_df = coarse_df.sort_values("rank_score", ascending=True).reset_index(drop=True)

    top_k = max(1, int(search_topk_full))
    candidates = coarse_df[coarse_df["status"] == "ok"].head(top_k).copy()
    print(f"Stage B (full): evaluating top {len(candidates)} configurations on full data.")

    full_records = []
    for _, row in candidates.iterrows():
        cfg = {
            "umap_n_neighbors": int(row["umap_n_neighbors"]),
            "umap_n_components": int(row["umap_n_components"]),
            "umap_min_dist": float(row["umap_min_dist"]),
            "umap_metric": str(row["umap_metric"]),
            "agglom_n_clusters": int(row["agglom_n_clusters"]),
            "agglom_linkage": str(row["agglom_linkage"]),
            "agglom_metric": str(row["agglom_metric"]),
        }
        rec = {"status": "ok", "error_msg": "", **cfg}

        try:
            umap_model_full = UMAP(
                n_neighbors=cfg["umap_n_neighbors"],
                n_components=cfg["umap_n_components"],
                min_dist=cfg["umap_min_dist"],
                metric=cfg["umap_metric"],
                random_state=seed,
                low_memory=True,
            )
            reduced_full = umap_model_full.fit_transform(emb_all)
            _tmp = _evaluate_config_with_reduced(
                reduced_full,
                cfg,
                seed=seed,
                max_eval_points=search_max_eval_points,
                evaluate_gap_fn=evaluate_gap_fn,
            )
            rec.update(_tmp)
        except Exception as e:
            rec.update(
                {
                    "status": "error",
                    "error_msg": f"{type(e).__name__}: {e}",
                    "silhouette": np.nan,
                    "dbi": np.nan,
                    "n_topics": np.nan,
                    "topic_size_min": np.nan,
                    "topic_size_median": np.nan,
                    "topic_size_max": np.nan,
                    "max_topic_share": np.nan,
                    "topic_size_cv": np.nan,
                    "outlier_ratio": 0.0,
                }
            )

        rec["stage"] = "full"
        full_records.append(rec)

    full_df = pd.DataFrame(full_records)
    if not full_df.empty:
        full_df = _add_rank_aggregate(full_df)
        full_df = full_df.sort_values("rank_score", ascending=True).reset_index(drop=True)

    if not full_df.empty and (full_df["status"] == "ok").any():
        final_table = full_df.copy()
        best_row = final_table[final_table["status"] == "ok"].iloc[0]
        selection_basis = "full-data refinement"
    else:
        final_table = coarse_df.copy()
        best_row = final_table[final_table["status"] == "ok"].iloc[0]
        selection_basis = "coarse search fallback"

    best_params = {
        "umap_n_neighbors": int(best_row["umap_n_neighbors"]),
        "umap_n_components": int(best_row["umap_n_components"]),
        "umap_min_dist": float(best_row["umap_min_dist"]),
        "umap_metric": str(best_row["umap_metric"]),
        "agglom_n_clusters": int(best_row["agglom_n_clusters"]),
        "agglom_linkage": str(best_row["agglom_linkage"]),
        "agglom_metric": str(best_row["agglom_metric"]),
        "selection_basis": selection_basis,
        "silhouette": float(best_row["silhouette"]) if pd.notna(best_row["silhouette"]) else None,
        "dbi": float(best_row["dbi"]) if pd.notna(best_row["dbi"]) else None,
        "rank_score": float(best_row["rank_score"]) if pd.notna(best_row["rank_score"]) else None,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = pd.concat([coarse_df, full_df], axis=0, ignore_index=True, sort=False)
    all_results.to_csv(output_dir / "agglom_search_results.csv", index=False)
    all_results.to_csv(output_dir / "results_agglomerative_grid.csv", index=False)

    top10 = all_results[all_results["status"] == "ok"].sort_values("rank_score", ascending=True).head(10)
    top10.to_csv(output_dir / "agglom_search_top10.csv", index=False)

    pd.DataFrame([best_params]).to_csv(output_dir / "best_agglomerative_params.csv", index=False)
    pd.DataFrame([best_params]).to_csv(output_dir / "best_agglom_params.csv", index=False)

    with (output_dir / "best_agglom_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    with (output_dir / "best_agglomerative_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    plot_agglomerative_search(all_results, output_dir)

    print("\nTop 10 parameter combinations (rank score ascending):")
    display_cols = [
        "stage",
        "umap_n_neighbors",
        "umap_n_components",
        "umap_min_dist",
        "agglom_n_clusters",
        "agglom_linkage",
        "agglom_metric",
        "silhouette",
        "dbi",
        "n_topics",
        "max_topic_share",
        "topic_size_cv",
        "rank_score",
    ]
    print(top10[display_cols].to_string(index=False))

    print("\nBest agglomerative params:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(
        "Reason: selected by aggregate rank score with preference for "
        "higher silhouette, lower DBI, lower size imbalance, and reasonable topic count."
    )

    return all_results, best_params
