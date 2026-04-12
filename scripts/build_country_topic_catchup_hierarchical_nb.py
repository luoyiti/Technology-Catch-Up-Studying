#!/usr/bin/env python3
"""Build country_topic_catchup_hierarchical.ipynb from the DTW template (one-shot)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "country_topic_catchup_dtw.ipynb"
DST = ROOT / "country_topic_catchup_hierarchical.ipynb"

MD0 = """## 1. 研究目标（层次聚类轨迹）

对每个 **(国家, 主题)**，在多年份上观察相对研究前沿的强度（`ratio_to_frontier`），经 **时间滚动平滑** 后得到一条一维时间序列。本 notebook 使用 **Hierarchical Clustering（层次聚类）** 对这些轨迹聚类，距离模式由 `HIER_DISTANCE_MODE` 控制：

- **euclidean**：列向标准化后的欧氏距离（特征空间 linkage；可与 `ward` / `average` / `complete` / `single` 搭配）。
- **dtw**：两两 **DTW** 预计算距离矩阵 + linkage（推荐 **average**；需要 `tslearn`）。
- **shape**：行内 **z-normalization** 后，用 **correlation distance** \\(1 - \\mathrm{corr}(x,y)\\) 作为形状距离（推荐 **average**）。

默认经验组合：**dtw + average**、**shape + average**、**euclidean + ward 或 average**。

**输出目录**：`output/catchup_round2/`（写入 `trajectory_*_hier_<mode>.csv/png` 等）。**不修改** `output/catchup_mvp/` 下已有 MVP 文件；亦**不覆盖** `country_topic_catchup_dtw.ipynb` 对应的 `*_dtw.*` 文件。"""

MD1 = """## 2. 数据依赖

- **配置**：[`config/cluster_keybert_from_cluster.json`](config/cluster_keybert_from_cluster.json)（数据 CSV 路径、`paper_topics.csv` 所在目录、rolling 窗口与随机种子）。
- **面板**：优先读取 `output/catchup_mvp/topic_country_year_panel.csv`（与 MVP 一致）；若不存在且 frontier 为 `top3_mean`，则从 `paper_topics.csv` 或配置中的主 CSV 重建；若仍缺文件会报错并提示先运行上游 notebook。

**内核**：建议使用仓库 `.venv`（含 `umap-learn`；**dtw** 模式另需 `tslearn`）。在 Jupyter 中选择与项目一致的内核，或执行：  
`uv run python -m ipykernel install --user --name=catchup-venv --display-name='catch-up (uv venv)'` 后刷新内核列表。

§5 树状图使用 **scipy**（`linkage` / `dendrogram`）。"""

MD5 = """## 4. 轨迹矩阵构建

从面板得到每个 **(country, topic, year)** 的 `ratio_to_frontier`（相对前沿比值），再按年透视、对缺失年填 0 后做 **rolling 平滑**，得到 `wide_roll_raw`：形状为 **(样本数, 年份数)**，每一行是一条待聚类的时间序列。随后用 `MIN_ACTIVE_YEARS` 与峰值份额阈值筛掉不可靠单元。

**数值稳定**：`ratio = share / max(frontier_value, FRONTIER_RATIO_EPS)`，不改动 `share` 单纯形；读入已有 MVP 面板 CSV 时若含 `share` 与 `frontier_value` 会用同一公式重算 `ratio_to_frontier`。透视与 rolling 后对轨迹做有限幅裁剪，防止历史数据中的 `inf` 进入后续距离计算。"""

MD7 = """## 5. 层次聚类（Hierarchical Clustering）与导出

- **模型**：`sklearn.cluster.AgglomerativeClustering`；`euclidean` 在标准化特征上直接拟合；`dtw` / `shape` 使用 **metric=\"precomputed\"** 与两两距离矩阵。
- **全 k 诊断**：在 `TRAJ_K_RANGE` 内对每个 k 拟合一次，在二维特征矩阵 `X_score` 上计算 **Davies–Bouldin** 与 **轮廓系数**（`silhouette` 失败则记为 NaN），保存 `trajectory_k_search_metrics_hier_<mode>.csv`、绘制 `trajectory_k_diagnostics_hier_<mode>.png`（仅 k–DB 与 k–Silhouette，并标出 `k_final`）。
- **最终 k**：`FINAL_K_STRATEGY` 为 `metrics_lex`（先最小化 DB，再最大化 silhouette）或 `fixed`（`FINAL_K_OVERRIDE`）；**不再**使用 inertia 肘部。
- **树状图**：`scipy.cluster.hierarchy` + 大样本时可截断显示；导出 `trajectory_dendrogram_hier_<mode>.png`。
- **产出**：`trajectory_clusters_hier_<mode>.csv`、`trajectory_prototypes_hier_<mode>.png`、`trajectory_umap_hier_<mode>.png` 等。**dtw** 模式下两两 DTW 为 **O(n²)** 时间与内存，样本量大时慎用上界较大的 `TRAJ_K_RANGE`。"""

MD10 = """## 6. 备注

若本 notebook 报错缺少 `paper_topics.csv` 或主数据 CSV，请先在同一环境中运行上游 **KeyBERT 聚类 / 导出** 相关 notebook，使 `config/cluster_keybert_from_cluster.json` 中的路径可用，并生成 `output/catchup_mvp/topic_country_year_panel.csv`（可选，但可加速读取）。

**dtw** 模式若提示缺少 `tslearn`，请执行 `uv sync` 或 `pip install tslearn`。"""

CELL4 = r'''# 与配置中的随机种子一致（上一单元已从 JSON 写入 RANDOM_STATE）。
RANDOM_STATE = RANDOM_STATE

# 轨迹样本至少需要在多少年份上有发文，才进入聚类（避免极稀疏噪声单元）。
MIN_ACTIVE_YEARS = 6

# 对 ratio_to_frontier 做滚动平均的窗口长度与最小有效窗口（与 MVP / KeyBERT 配置对齐）。
MAIN_ROLLING_WINDOW = CONFIG_ROLLING_WINDOW
ROLLING_MIN_PERIODS = CONFIG_ROLLING_MIN

# 前沿强度定义：在每个 (topic, year) 上取发文量 Top-K 国家的平均 share 作为 frontier（top3_mean）。
FRONTIER_TOP_K = 3
MAIN_FRONTIER_MODE: Literal["top3_mean", "max"] = "top3_mean"

# 层次聚类：距离模式与 linkage；在 k 的网格上搜索（dtw/shape 的预计算距离为 O(n²)，上界过大将显著变慢）。
HIER_DISTANCE_MODE: Literal["euclidean", "dtw", "shape"] = "dtw"
HIER_LINKAGE_METHOD: Literal["ward", "average", "complete", "single"] = "average"
TRAJ_K_RANGE = range(2, 11)

# 最终聚类数：metrics_lex = 在 X_score 上 (DB 升序, Silhouette 降序) 字典序最优；fixed = 使用 FINAL_K_OVERRIDE。
FINAL_K_STRATEGY: Literal["metrics_lex", "fixed"] = "metrics_lex"
FINAL_K_OVERRIDE: int | None = None  # only when FINAL_K_STRATEGY == "fixed"

# 设为非空列表（如 [3, 5, 7]）时，在 §5 末尾用各窗口重建轨迹并以当前 k_final 各拟合一次，输出对比小表（默认关闭）。
SWEEP_ROLLING_WINDOWS: list[int] = []

# 过滤弱信号单元：全时段内「单年最大主题份额」低于该阈值的 (country, topic) 剔除（减少长期接近 0 的轨迹）。
MIN_PEAK_TOPIC_YEAR_SHARE = 0.05
# 若 >0，则还要求「有发文年份上的平均份额」不低于该阈值；0 表示关闭此约束。
MIN_MEAN_SHARE_ACTIVE = 0.0

# UMAP 仅用于二维可视化展平后的 rolling 特征（或与 shape 评分一致的 z-score 矩阵），不参与聚类。
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# ratio_to_frontier = share / frontier 的分母下界：避免 frontier 为极小正数时比值爆炸为 inf（不修改 share）。
FRONTIER_RATIO_EPS = 1e-10
# 透视与 rolling 之后将非有限值压到该上界，防止历史 CSV 中残留 inf 进入距离计算。
RATIO_ROLLING_MAX_CLIP = 50.0


def validate_hierarchical_params(distance_mode: str, linkage_method: str) -> None:
    """Validate HIER_DISTANCE_MODE x HIER_LINKAGE_METHOD combinations."""
    if distance_mode not in ("euclidean", "dtw", "shape"):
        raise ValueError(f"Invalid HIER_DISTANCE_MODE: {distance_mode!r}")
    if linkage_method not in ("ward", "average", "complete", "single"):
        raise ValueError(f"Invalid HIER_LINKAGE_METHOD: {linkage_method!r}")
    if distance_mode in ("dtw", "shape") and linkage_method == "ward":
        raise ValueError("ward linkage is not valid with precomputed DTW or shape distances; use average/complete/single.")


validate_hierarchical_params(HIER_DISTANCE_MODE, HIER_LINKAGE_METHOD)

plt.rcParams["figure.dpi"] = 120
print(
    "主要参数:",
    "MAIN_ROLLING_WINDOW=",
    MAIN_ROLLING_WINDOW,
    "MAIN_FRONTIER_MODE=",
    MAIN_FRONTIER_MODE,
    "HIER_DISTANCE_MODE=",
    HIER_DISTANCE_MODE,
    "HIER_LINKAGE_METHOD=",
    HIER_LINKAGE_METHOD,
    "TRAJ_K_RANGE=",
    list(TRAJ_K_RANGE),
    "FINAL_K_STRATEGY=",
    FINAL_K_STRATEGY,
    "FINAL_K_OVERRIDE=",
    FINAL_K_OVERRIDE,
    "SWEEP_ROLLING_WINDOWS=",
    SWEEP_ROLLING_WINDOWS,
    "MIN_PEAK_TOPIC_YEAR_SHARE=",
    MIN_PEAK_TOPIC_YEAR_SHARE,
    "FRONTIER_RATIO_EPS=",
    FRONTIER_RATIO_EPS,
    "RATIO_ROLLING_MAX_CLIP=",
    RATIO_ROLLING_MAX_CLIP,
)
'''

# Section 5 full code cell (part 1: defs through fit_hierarchical; part 2: driver)
CELL8 = r'''def plot_trajectory_prototypes(
    wide_roll: pd.DataFrame,
    labels: np.ndarray,
    years: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Mean rolling ratio_to_frontier curves per cluster (prototype interpretation)."""
    clusters = np.sort(np.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in clusters:
        idx = labels == c
        mean_curve = wide_roll.values[idx].mean(axis=0)
        ax.plot(years, mean_curve, label=f"C{int(c)} (n={int(idx.sum())})")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean rolling ratio_to_frontier")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trajectory_umap(
    wide_roll: pd.DataFrame,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    random_state: int,
    out_png: Path,
    X_umap_feat: np.ndarray | None = None,
) -> None:
    """UMAP on scaled rolling features, or on precomputed X_umap_feat (e.g. z-score rows for shape mode)."""
    from umap import UMAP

    _ = meta_df
    if X_umap_feat is None:
        X = StandardScaler().fit_transform(wide_roll.values.astype(float))
    else:
        X = np.asarray(X_umap_feat, dtype=float)
    n = X.shape[0]
    if n < 5:
        warnings.warn("样本过少，跳过 UMAP")
        return
    nn = max(2, min(UMAP_N_NEIGHBORS, n - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.72, label=f"C{int(c)} (n={int(m.sum())})")
    ax.set_title("Trajectory UMAP (visualization features)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def score_labels_flat(X_flat: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Davies–Bouldin and silhouette on flat Euclidean space (X_score)."""
    uq = np.unique(labels)
    if len(uq) < 2 or len(X_flat) < uq.max() + 2:
        return np.inf, np.nan
    try:
        sil = float(silhouette_score(X_flat, labels, metric="euclidean"))
    except Exception:
        sil = np.nan
    db = float(davies_bouldin_score(X_flat, labels))
    return db, sil


def cluster_size_entropy(labels: np.ndarray) -> float:
    vc = pd.Series(labels).value_counts(normalize=True)
    return float(-(vc * np.log(vc + 1e-15)).sum())


def z_normalize_rows(wide_roll: pd.DataFrame) -> np.ndarray:
    """Per-row z-score; constant or invalid rows become zeros (finite, safe for correlation)."""
    X = wide_roll.to_numpy(dtype=float, copy=True)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sig = np.nanstd(X, axis=1, ddof=0, keepdims=True)
    bad = ~np.isfinite(sig) | (sig < 1e-12)
    Z = (X - mu) / np.where(bad, 1.0, sig)
    Z = np.where(bad, 0.0, Z)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


def build_pairwise_distance_matrix(
    wide_roll: pd.DataFrame, distance_mode: str
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Build distances for hierarchical clustering and the 2D matrix used for scoring / visualization.

    Returns
    -------
    D : ndarray or None
        Square precomputed distance matrix (symmetric, zero diagonal) for dtw/shape; None for euclidean.
    X_used : ndarray
        Feature matrix aligned with distance_mode (column-standardized rolling for euclidean/dtw;
        row z-normalized trajectories for shape).
    """
    X_raw = wide_roll.to_numpy(dtype=float, copy=False)
    if distance_mode == "euclidean":
        X_used = StandardScaler().fit_transform(X_raw)
        return None, X_used
    if distance_mode == "dtw":
        try:
            from tslearn.metrics import cdist_dtw
        except ImportError as e:
            raise ImportError("tslearn is required for HIER_DISTANCE_MODE='dtw'. Install with: uv add tslearn") from e
        X_used = StandardScaler().fit_transform(X_raw)
        X3 = X_raw[:, :, np.newaxis]
        D = np.asarray(cdist_dtw(X3, X3), dtype=float)
        D = (D + D.T) * 0.5
        np.fill_diagonal(D, 0.0)
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return D, X_used
    if distance_mode == "shape":
        Z = z_normalize_rows(wide_roll)
        X_used = Z
        # Pearson correlation matrix; distance = 1 - corr
        with np.errstate(invalid="ignore"):
            C = np.corrcoef(Z)
        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
        D = (D + D.T) * 0.5
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, 2.0)
        return D, X_used
    raise ValueError(distance_mode)


def fit_hierarchical_single_k(
    wide_roll: pd.DataFrame,
    k: int,
    distance_mode: str,
    linkage_method: str,
    D: np.ndarray | None,
    X_used: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one AgglomerativeClustering with k clusters; labels are contiguous integers from 0."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(wide_roll)
    if k < 2 or k >= n:
        raise ValueError(f"invalid k={k} for n={n}")
    if distance_mode == "euclidean":
        model = AgglomerativeClustering(
            n_clusters=k, metric="euclidean", linkage=linkage_method
        )
        labels = model.fit_predict(X_used)
    else:
        if D is None:
            raise ValueError("D required for dtw/shape")
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=linkage_method
        )
        labels = model.fit_predict(D)
    labels = np.asarray(labels, dtype=int)
    # Remap to 0..C-1 if sklearn returns unused ids (defensive)
    uniq = np.unique(labels)
    if len(uniq) != len(np.arange(uniq.min(), uniq.max() + 1)):
        labels = pd.factorize(labels, sort=True)[0].astype(int)
    extras: dict[str, Any] = {
        "distance_mode": distance_mode,
        "linkage_method": linkage_method,
        "distance_matrix": D,
    }
    return labels, extras


def hierarchical_search_all_k(
    wide_roll: pd.DataFrame,
    k_list: list[int],
    distance_mode: str,
    linkage_method: str,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """One precomputed D per run; metrics rows include k, db, silhouette, cluster_sizes, size_entropy."""
    D, X_used = build_pairwise_distance_matrix(wide_roll, distance_mode)
    X_score = X_used
    n = len(wide_roll)
    rows: list[dict[str, Any]] = []
    for k in k_list:
        if k < 2 or k >= n:
            rows.append({"k": k, "error": "k_out_of_range"})
            continue
        try:
            labels, _ = fit_hierarchical_single_k(
                wide_roll, k, distance_mode, linkage_method, D, X_used
            )
            uq = np.unique(labels)
            if len(uq) < 2:
                raise ValueError("single_cluster")
            db, sil = score_labels_flat(X_score, labels)
            sizes = pd.Series(labels).value_counts().sort_index()
            sz_str = ",".join(f"{int(i)}:{int(sizes.loc[i])}" for i in sizes.index)
            rows.append(
                {
                    "k": k,
                    "db": db,
                    "silhouette": sil,
                    "cluster_sizes": sz_str,
                    "size_entropy": cluster_size_entropy(labels),
                    "distance_mode": distance_mode,
                    "linkage_method": linkage_method,
                }
            )
        except Exception as e:
            rows.append({"k": k, "error": str(e)})
    return pd.DataFrame(rows), D, X_used


def pick_best_k_from_metrics(
    metrics_df: pd.DataFrame, strategy: str, override: int | None, n_samples: int
) -> tuple[int, str]:
    """metrics_lex: min (db, -silhouette); fixed: use override if present in successful search rows."""
    if strategy == "fixed":
        if override is None:
            raise ValueError("FINAL_K_STRATEGY=fixed requires FINAL_K_OVERRIDE")
        kf = int(override)
        if kf < 2 or kf >= n_samples:
            raise ValueError(f"FINAL_K_OVERRIDE out of range: {kf}")
        ok_k = set(metrics_df.loc[metrics_df["db"].notna(), "k"].astype(int))
        if kf not in ok_k:
            raise ValueError(f"k={kf} had no successful fit in metrics_df; adjust TRAJ_K_RANGE or k")
        return kf, "fixed"
    if strategy == "metrics_lex":
        sub = metrics_df.loc[metrics_df["db"].notna()].copy()
        if sub.empty:
            raise RuntimeError("no valid k for metrics_lex")
        sub["_neg_sil"] = -np.nan_to_num(sub["silhouette"].astype(float), nan=-1e9)
        sub = sub.sort_values(["db", "_neg_sil", "k"])
        return int(sub.iloc[0]["k"]), "metrics_lex"
    raise ValueError(strategy)


def plot_hier_k_diagnostics(
    metrics_df: pd.DataFrame, k_final: int | None, out_path: Path, title: str
) -> None:
    sub = metrics_df.loc[metrics_df["db"].notna()].sort_values("k")
    if sub.empty:
        warnings.warn("No valid k rows for diagnostics plot; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax0, ax1 = axes
    ax0.plot(sub["k"], sub["db"], "o-", color="C1", lw=1.2)
    ax0.set_xlabel("k")
    ax0.set_ylabel("Davies–Bouldin (X_score)")
    ax0.set_title("DB index")
    sil = sub["silhouette"].astype(float)
    ax1.plot(sub["k"], sil, "o-", color="C2", lw=1.2)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette (X_score)")
    ax1.set_title("Silhouette")
    if k_final is not None:
        for ax in axes:
            ax.axvline(
                k_final,
                color="crimson",
                ls="--",
                lw=1.0,
                alpha=0.85,
                label=f"k_final={k_final}",
            )
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_dendrogram(
    wide_roll: pd.DataFrame,
    distance_mode: str,
    linkage_method: str,
    D: np.ndarray | None,
    X_used: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Linkage + dendrogram; truncated when many leaves for readability."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform

    if distance_mode == "euclidean":
        if linkage_method == "ward":
            Z = linkage(X_used, method="ward")
        else:
            condensed = pdist(X_used, metric="euclidean")
            Z = linkage(condensed, method=linkage_method)
    else:
        if D is None:
            warnings.warn("No D for dendrogram; skip")
            return
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method=linkage_method)

    n = len(wide_roll)
    fig, ax = plt.subplots(figsize=(12, 5))
    if n > 80:
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=50, leaf_rotation=90.0, no_labels=True)
    else:
        dendrogram(Z, ax=ax, leaf_rotation=90.0, no_labels=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# === Cell 1: grid search and k selection ===
_file_suffix = HIER_DISTANCE_MODE
_k_list = [int(k) for k in TRAJ_K_RANGE]
search_df, D_main, X_used_main = hierarchical_search_all_k(
    wide_roll_raw, _k_list, HIER_DISTANCE_MODE, HIER_LINKAGE_METHOD
)
_metrics_path = CATCHUP_ROUND2_DIR / f"trajectory_k_search_metrics_hier_{_file_suffix}.csv"
search_df.to_csv(_metrics_path, index=False)
print(f"Hierarchical k search (saved {_metrics_path.name}):\n", search_df.to_string())

k_final, k_strategy_used = pick_best_k_from_metrics(
    search_df, FINAL_K_STRATEGY, FINAL_K_OVERRIDE, len(wide_roll_raw)
)
print(f"\nk_final={k_final} (strategy={k_strategy_used}, configured={FINAL_K_STRATEGY})")

for _, row in search_df.iterrows():
    if pd.notna(row.get("cluster_sizes")):
        print(f"  k={int(row['k'])} cluster_sizes: {row['cluster_sizes']}")

plot_hier_k_diagnostics(
    search_df,
    k_final,
    CATCHUP_ROUND2_DIR / f"trajectory_k_diagnostics_hier_{_file_suffix}.png",
    f"Hierarchical k diagnostics ({HIER_DISTANCE_MODE}, {HIER_LINKAGE_METHOD})",
)
print("Grid search & k selection done. Run the next cell for final clustering outputs.")
'''

CELL9 = r'''# === Cell 2: final hierarchical fit, export, and plots ===
labels_hier, hier_extras = fit_hierarchical_single_k(
    wide_roll_raw,
    k_final,
    HIER_DISTANCE_MODE,
    HIER_LINKAGE_METHOD,
    D_main,
    X_used_main,
)

_out_suffix = HIER_DISTANCE_MODE
pd.DataFrame(
    {"country": meta["country"], "topic": meta["topic"], "trajectory_cluster": labels_hier}
).to_csv(CATCHUP_ROUND2_DIR / f"trajectory_clusters_hier_{_out_suffix}.csv", index=False)

plot_trajectory_prototypes(
    wide_roll_raw,
    labels_hier,
    years_all,
    CATCHUP_ROUND2_DIR / f"trajectory_prototypes_hier_{_out_suffix}.png",
    f"Hierarchical prototypes (k={k_final}, mode={HIER_DISTANCE_MODE})",
)
_umap_X = X_used_main if HIER_DISTANCE_MODE == "shape" else None
plot_trajectory_umap(
    wide_roll_raw,
    labels_hier,
    meta,
    RANDOM_STATE,
    CATCHUP_ROUND2_DIR / f"trajectory_umap_hier_{_out_suffix}.png",
    X_umap_feat=_umap_X,
)

plot_trajectory_dendrogram(
    wide_roll_raw,
    HIER_DISTANCE_MODE,
    HIER_LINKAGE_METHOD,
    D_main,
    X_used_main,
    CATCHUP_ROUND2_DIR / f"trajectory_dendrogram_hier_{_out_suffix}.png",
    f"Dendrogram ({HIER_DISTANCE_MODE}, {HIER_LINKAGE_METHOD}, n={len(wide_roll_raw)})",
)

print("Final hierarchical cluster sizes:\n", pd.Series(labels_hier).value_counts().sort_index())

if SWEEP_ROLLING_WINDOWS:
    sweep_rows = []
    for rw in SWEEP_ROLLING_WINDOWS:
        wr, _, _, _ = build_trajectory_matrix(
            panel_main, years_all, rw, ROLLING_MIN_PERIODS, MIN_ACTIVE_YEARS
        )
        if k_final >= len(wr):
            sweep_rows.append({"rolling_window": rw, "error": "n_samples_too_small_for_k"})
            continue
        D_sw, Xu_sw = build_pairwise_distance_matrix(wr, HIER_DISTANCE_MODE)
        lab, _ = fit_hierarchical_single_k(
            wr, k_final, HIER_DISTANCE_MODE, HIER_LINKAGE_METHOD, D_sw, Xu_sw
        )
        X_score_sw = Xu_sw
        db, sil = score_labels_flat(X_score_sw, lab)
        sizes = pd.Series(lab).value_counts().sort_index()
        sz_str = ",".join(f"{int(i)}:{int(sizes.loc[i])}" for i in sizes.index)
        sweep_rows.append(
            {
                "rolling_window": rw,
                "k": k_final,
                "n_samples": len(wr),
                "db": db,
                "silhouette": sil,
                "size_entropy": cluster_size_entropy(lab),
                "cluster_sizes": sz_str,
                "distance_mode": HIER_DISTANCE_MODE,
                "linkage_method": HIER_LINKAGE_METHOD,
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    print("\nRolling-window sweep (fixed k_final):\n", sweep_df.to_string())
    sweep_df.to_csv(
        CATCHUP_ROUND2_DIR / f"hier_{HIER_DISTANCE_MODE}_rolling_window_sweep.csv", index=False
    )
'''


def to_src_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [text]


def clear_outputs(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = to_src_lines(MD0)
    nb["cells"][1]["source"] = to_src_lines(MD1)
    nb["cells"][5]["source"] = to_src_lines(MD5)
    nb["cells"][7]["source"] = to_src_lines(MD7)
    nb["cells"][10]["source"] = to_src_lines(MD10)

    nb["cells"][4]["source"] = to_src_lines(CELL4)

    # Insert z_normalize_rows + comment tweak in cell 6 (trajectory build); define once for §5 shape distance.
    c6 = "".join(nb["cells"][6]["source"])
    needle = "\n\nyears_all = np.sort(panel_main[\"year\"].unique())\n# wide_roll_raw：DTW 输入使用的原始滚动水平（不做行内 z-score，以保留与前沿的绝对距离信息）。\n"
    insert = '''

def z_normalize_rows(wide_roll: pd.DataFrame) -> np.ndarray:
    """Per-row z-score; constant or invalid rows become zeros (finite, safe for correlation)."""
    X = wide_roll.to_numpy(dtype=float, copy=True)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sig = np.nanstd(X, axis=1, ddof=0, keepdims=True)
    bad = ~np.isfinite(sig) | (sig < 1e-12)
    Z = (X - mu) / np.where(bad, 1.0, sig)
    Z = np.where(bad, 0.0, Z)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


years_all = np.sort(panel_main["year"].unique())
# wide_roll_raw：层次聚类输入使用的原始滚动水平（不做行内 z-score，以保留与前沿的绝对距离信息；shape 模式在 §5 另行 z-normalize）。
'''
    if needle not in c6:
        raise SystemExit("cell 6 needle not found; DTW notebook structure changed")
    c6 = c6.replace(needle, insert)
    nb["cells"][6]["source"] = to_src_lines(c6)

    zblock = '''

def z_normalize_rows(wide_roll: pd.DataFrame) -> np.ndarray:
    """Per-row z-score; constant or invalid rows become zeros (finite, safe for correlation)."""
    X = wide_roll.to_numpy(dtype=float, copy=True)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sig = np.nanstd(X, axis=1, ddof=0, keepdims=True)
    bad = ~np.isfinite(sig) | (sig < 1e-12)
    Z = (X - mu) / np.where(bad, 1.0, sig)
    Z = np.where(bad, 0.0, Z)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


'''
    s8 = CELL8.replace(zblock, "\n\n")
    nb["cells"][8]["source"] = to_src_lines(s8)
    nb["cells"][9]["source"] = to_src_lines(CELL9)

    for cell in nb["cells"]:
        clear_outputs(cell)

    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "catch-up (.venv)",
        "language": "python",
        "name": "catchup-catch-up",
    }

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", DST)


if __name__ == "__main__":
    main()
