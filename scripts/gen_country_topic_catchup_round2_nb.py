# Generates country_topic_catchup_round2.ipynb (DTW-only trajectory clustering).
# Run from repo root: uv run python scripts/gen_country_topic_catchup_round2_nb.py
from __future__ import annotations

from pathlib import Path

import nbformat
import nbformat.v4 as nbf


def md(text: str):
    return nbf.new_markdown_cell(text)


def code(text: str):
    return nbf.new_code_cell(text)


ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "country_topic_catchup_round2.ipynb"

cells: list = []

cells.append(
    md(
        """## 1. 研究目标（DTW 轨迹聚类）

对每个 **(国家, 主题)**，在多年份上观察相对研究前沿的强度（`ratio_to_frontier`），经 **时间滚动平滑** 后得到一条一维时间序列。本 notebook **仅** 使用 `tslearn` 的 `TimeSeriesKMeans(metric="dtw")`（DTW 距离）对这些轨迹做 **k 均值聚类**，以允许时间轴上的局部错位，更贴近「不同国家进入高平台期的年份不同」这类追赶节奏。

**输出目录**：`output/catchup_round2/`（写入 `trajectory_clusters_dtw.csv`、原型曲线图与 UMAP 图）。**不修改** `output/catchup_mvp/` 下已有 MVP 文件。"""
    )
)

cells.append(
    md(
        """## 2. 数据依赖

- **配置**：[`config/cluster_keybert_from_cluster.json`](config/cluster_keybert_from_cluster.json)（数据 CSV 路径、`paper_topics.csv` 所在目录、rolling 窗口与随机种子）。
- **面板**：优先读取 `output/catchup_mvp/topic_country_year_panel.csv`（与 MVP 一致）；若不存在且 frontier 为 `top3_mean`，则从 `paper_topics.csv` 或配置中的主 CSV 重建；若仍缺文件会报错并提示先运行上游 notebook。

**内核**：建议使用仓库 `.venv`（含 `tslearn`、`umap-learn`）。在 Jupyter 中选择与项目一致的内核，或执行：  
`uv run python -m ipykernel install --user --name=catchup-venv --display-name='catch-up (uv venv)'` 后刷新内核列表。"""
    )
)

cells.append(
    code(
        r'''from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# 屏蔽部分库的未来弃用警告，避免输出干扰阅读（不影响数值结果）。
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 路径与输出目录（相对仓库根目录）---
_CONFIG_PATH = Path("config/cluster_keybert_from_cluster.json")
CATCHUP_MVP_DIR = Path("output/catchup_mvp")
CATCHUP_ROUND2_DIR = Path("output/catchup_round2")
CATCHUP_ROUND2_DIR.mkdir(parents=True, exist_ok=True)


def load_json_config(path: Path) -> dict[str, Any]:
    """读取聚类流水线 JSON 配置。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_reproducibility(seed: int) -> None:
    """固定 Python / NumPy 随机种子，便于结果复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_cfg = load_json_config(_CONFIG_PATH)
_paths = _cfg["paths"]
DATA_PATH = Path(_paths["data_csv"])
CLUSTER_OUTPUT_DIR = Path(_paths["output_dir"])
PAPER_TOPICS_PATH = CLUSTER_OUTPUT_DIR / "paper_topics.csv"
_time = _cfg.get("time_evolution", {})
CONFIG_ROLLING_WINDOW = int(_time.get("rolling_window_size", 5))
CONFIG_ROLLING_MIN = int(_time.get("rolling_min_periods", 3))
RANDOM_STATE = int(_cfg["reproducibility"]["seed"])
set_reproducibility(RANDOM_STATE)

print("CATCHUP_MVP_DIR exists:", CATCHUP_MVP_DIR.is_dir())
print("CATCHUP_ROUND2_DIR:", CATCHUP_ROUND2_DIR.resolve())
'''
    )
)

cells.append(md("## 3. 参数区（集中管理）"))

cells.append(
    code(
        r'''# 与配置中的随机种子一致（上一单元已从 JSON 写入 RANDOM_STATE）。
RANDOM_STATE = RANDOM_STATE

# 轨迹样本至少需要在多少年份上有发文，才进入聚类（避免极稀疏噪声单元）。
MIN_ACTIVE_YEARS = 6

# 对 ratio_to_frontier 做滚动平均的窗口长度与最小有效窗口（与 MVP / KeyBERT 配置对齐）。
MAIN_ROLLING_WINDOW = CONFIG_ROLLING_WINDOW
ROLLING_MIN_PERIODS = CONFIG_ROLLING_MIN

# 前沿强度定义：在每个 (topic, year) 上取发文量 Top-K 国家的平均 share 作为 frontier（top3_mean）。
FRONTIER_TOP_K = 3
MAIN_FRONTIER_MODE: Literal["top3_mean", "max"] = "top3_mean"

# DTW 聚类：在 k 的网格上搜索；每个 k 用 DB 指数为主、轮廓系数为辅挑选最优模型（与原先三法 notebook 中 DTW 分支一致）。
TRAJ_K_RANGE = range(3, 7)
DTW_N_INIT = 5
TS_MAX_ITER = 20

# 过滤弱信号单元：全时段内「单年最大主题份额」低于该阈值的 (country, topic) 剔除（减少长期接近 0 的轨迹）。
MIN_PEAK_TOPIC_YEAR_SHARE = 0.05
# 若 >0，则还要求「有发文年份上的平均份额」不低于该阈值；0 表示关闭此约束。
MIN_MEAN_SHARE_ACTIVE = 0.0

# UMAP 仅用于二维可视化展平后的 rolling 特征，不参与聚类。
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# ratio_to_frontier = share / frontier 的分母下界：避免 frontier 为极小正数时比值爆炸为 inf（不修改 share）。
FRONTIER_RATIO_EPS = 1e-10
# 透视与 rolling 之后将非有限值压到该上界，防止历史 CSV 中残留 inf 进入 DTW。
RATIO_ROLLING_MAX_CLIP = 50.0

plt.rcParams["figure.dpi"] = 120
print(
    "主要参数:",
    "MAIN_ROLLING_WINDOW=", MAIN_ROLLING_WINDOW,
    "MAIN_FRONTIER_MODE=", MAIN_FRONTIER_MODE,
    "TRAJ_K_RANGE=", list(TRAJ_K_RANGE),
    "MIN_PEAK_TOPIC_YEAR_SHARE=", MIN_PEAK_TOPIC_YEAR_SHARE,
    "FRONTIER_RATIO_EPS=", FRONTIER_RATIO_EPS,
    "RATIO_ROLLING_MAX_CLIP=", RATIO_ROLLING_MAX_CLIP,
)
'''
    )
)

cells.append(
    md(
        """## 4. 轨迹矩阵构建

从面板得到每个 **(country, topic, year)** 的 `ratio_to_frontier`（相对前沿比值），再按年透视、对缺失年填 0 后做 **rolling 平滑**，得到 `wide_roll_raw`：形状为 **(样本数, 年份数)**，每一行是一条待聚类的时间序列。随后用 `MIN_ACTIVE_YEARS` 与峰值份额阈值筛掉不可靠单元。

**数值稳定**：`ratio = share / max(frontier_value, FRONTIER_RATIO_EPS)`，不改动 `share` 单纯形；读入已有 MVP 面板 CSV 时若含 `share` 与 `frontier_value` 会用同一公式重算 `ratio_to_frontier`。透视与 rolling 后对轨迹做有限幅裁剪，防止历史数据中的 `inf` 进入 DTW。"""
    )
)

cells.append(
    code(
        r'''# 列名候选：兼容 WoS 导出或中间表的不同命名。
TOPIC_CANDIDATES = ("topic", "Topic", "topic_id")
YEAR_CANDIDATES = ("year", "Year", "pub_year", "publication_year")
COUNTRY_CANDIDATES = ("country_code", "country", "Country", "nation")


def _first_present(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def resolve_topic_country_year_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    out["topic"] = _first_present(df, TOPIC_CANDIDATES) or ""
    out["year"] = _first_present(df, YEAR_CANDIDATES) or ""
    out["country_src"] = _first_present(df, COUNTRY_CANDIDATES) or ""
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise KeyError(f"Missing columns for: {missing}. Available: {list(df.columns)[:40]}...")
    return out  # type: ignore[return-value]


def load_paper_level_table() -> tuple[pd.DataFrame, str]:
    """优先读聚类输出的 paper_topics；否则读配置中的主 CSV。"""
    if PAPER_TOPICS_PATH.exists():
        df = pd.read_csv(PAPER_TOPICS_PATH, low_memory=False)
        return df, str(PAPER_TOPICS_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Neither {PAPER_TOPICS_PATH} nor {DATA_PATH} exists. "
            "Run cluster_keybert_from_cluster.ipynb to export paper_topics.csv."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if _first_present(df, TOPIC_CANDIDATES) is None:
        raise ValueError(f"No topic column in {DATA_PATH}.")
    return df, str(DATA_PATH)


def frontier_topk_mean_share(shares: pd.Series, k: int) -> float:
    """在每个 (topic, year) 内，对各国 share 取 top-k 正的国家的均值，作为前沿参考。"""
    s = shares.astype(float).sort_values(ascending=False)
    s = s[s > 0] if (s > 0).any() else s
    if s.empty:
        return 0.0
    top = s.head(min(k, len(s)))
    return float(top.mean())


def frontier_max_share(shares: pd.Series) -> float:
    """备选前沿：该国在该主题该年的最大 share（更激进）。"""
    s = shares.astype(float)
    if s.empty:
        return 0.0
    return float(s.max())


def build_topic_country_year_panel_with_frontier(
    papers: pd.DataFrame,
    frontier_mode: Literal["top3_mean", "max"],
    top_k: int,
) -> pd.DataFrame:
    """构造完整 (topic, country, year) 网格，计算 count、share、rank、frontier_value、ratio_to_frontier。"""
    counts = (
        papers.groupby(["topic", "country", "year"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    topics = counts["topic"].unique()
    countries = counts["country"].unique()
    years = np.arange(int(counts["year"].min()), int(counts["year"].max()) + 1, dtype=int)
    full = pd.MultiIndex.from_product(
        [topics, countries, years], names=["topic", "country", "year"]
    ).to_frame(index=False)
    panel = full.merge(counts, on=["topic", "country", "year"], how="left")
    panel["count"] = panel["count"].fillna(0).astype(int)
    tot = panel.groupby(["topic", "year"])["count"].transform("sum")
    panel["share"] = np.where(tot > 0, panel["count"] / tot, 0.0)
    panel["rank"] = panel.groupby(["topic", "year"])["count"].rank(ascending=False, method="min")
    if frontier_mode == "top3_mean":
        panel["frontier_value"] = panel.groupby(["topic", "year"])["share"].transform(
            lambda s: frontier_topk_mean_share(s, top_k)
        )
    elif frontier_mode == "max":
        panel["frontier_value"] = panel.groupby(["topic", "year"])["share"].transform(frontier_max_share)
    else:
        raise ValueError(frontier_mode)
    # 安全除法：分母不低于 FRONTIER_RATIO_EPS，避免极小 frontier_value 导致 inf（share 不加扰动）。
    fv = panel["frontier_value"].astype(float)
    sh = panel["share"].astype(float)
    denom = np.maximum(fv, FRONTIER_RATIO_EPS)
    panel["ratio_to_frontier"] = (sh / denom).astype(float)
    panel["gap_to_frontier"] = panel["frontier_value"] - panel["share"]
    return panel


def load_or_build_panel(frontier_mode: str, top_k: int) -> tuple[pd.DataFrame, str]:
    """若 MVP 已生成面板且模式为 top3_mean，则直接读取；否则从论文表重建。"""
    mvp_panel = CATCHUP_MVP_DIR / "topic_country_year_panel.csv"
    if mvp_panel.exists() and frontier_mode == "top3_mean":
        panel = pd.read_csv(mvp_panel, low_memory=False)
        # 与重建路径一致：用分母下界重算 ratio，修复旧 CSV 中因极小 frontier 产生的 inf。
        if {"share", "frontier_value"}.issubset(panel.columns):
            fv = panel["frontier_value"].astype(float)
            sh = panel["share"].astype(float)
            panel["ratio_to_frontier"] = (sh / np.maximum(fv, FRONTIER_RATIO_EPS)).astype(float)
        src = f"read:{mvp_panel}"
    else:
        papers, _src = load_paper_level_table()
        cols = resolve_topic_country_year_columns(papers)
        tc, yc, csc = cols["topic"], cols["year"], cols["country_src"]
        if csc != "country" and "country" in papers.columns:
            papers = papers.drop(columns=["country"], errors="ignore")
        papers = papers.rename(columns={tc: "topic", yc: "year", csc: "country"})
        papers["topic"] = pd.to_numeric(papers["topic"], errors="coerce")
        papers["year"] = pd.to_numeric(papers["year"], errors="coerce").astype("Int64")
        papers = papers.dropna(subset=["topic", "year", "country"]).copy()
        papers["topic"] = papers["topic"].astype(int)
        papers["year"] = papers["year"].astype(int)
        papers = papers[papers["topic"] != -1]
        papers["country"] = papers["country"].astype(str).str.strip()
        papers = papers[papers["country"].str.len() > 0]
        panel = build_topic_country_year_panel_with_frontier(papers, frontier_mode, top_k)  # type: ignore[arg-type]
        src = f"rebuild:{frontier_mode}"
        if not mvp_panel.exists():
            out_rebuilt = CATCHUP_ROUND2_DIR / "topic_country_year_panel_rebuilt.csv"
            panel.to_csv(out_rebuilt, index=False)
            print("Saved rebuilt panel to", out_rebuilt)
    return panel, src


panel_main, panel_src = load_or_build_panel(MAIN_FRONTIER_MODE, FRONTIER_TOP_K)
print("Panel source:", panel_src)
print("shape", panel_main.shape, "cols", list(panel_main.columns))
print(panel_main.isna().mean().sort_values(ascending=False).head(8))
print(panel_main.head(3))


def build_trajectory_matrix(
    panel: pd.DataFrame,
    years_all: np.ndarray,
    rolling_window: int,
    min_periods: int,
    min_active_years: int,
    min_peak_share: float | None = None,
    min_mean_share_active: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """由面板得到 rolling 后的 ratio 轨迹及辅助宽表；返回的 wide_roll 行索引为 MultiIndex (country, topic)。"""
    year_cols = [str(y) for y in years_all]
    thr = MIN_PEAK_TOPIC_YEAR_SHARE if min_peak_share is None else float(min_peak_share)
    mthr = MIN_MEAN_SHARE_ACTIVE if min_mean_share_active is None else float(min_mean_share_active)
    ratio_pivot = panel.pivot_table(
        index=["country", "topic"], columns="year", values="ratio_to_frontier", aggfunc="first"
    )
    ratio_pivot = ratio_pivot.reindex(columns=years_all)
    ratio_filled = ratio_pivot.fillna(0.0)
    # 将 nan/inf 压到有限区间，避免读入历史面板时残留 inf 进入 rolling。
    ratio_filled = pd.DataFrame(
        np.nan_to_num(
            ratio_filled.to_numpy(dtype=float, copy=True),
            nan=0.0,
            posinf=RATIO_ROLLING_MAX_CLIP,
            neginf=0.0,
        ),
        index=ratio_filled.index,
        columns=ratio_filled.columns,
    )
    # 按时间轴 rolling：平滑单年噪声，使 DTW 更关注趋势与阶段而非单点毛刺。
    roll = ratio_filled.T.rolling(rolling_window, min_periods=min_periods).mean().T.fillna(0.0)
    roll = pd.DataFrame(
        np.nan_to_num(
            roll.to_numpy(dtype=float, copy=True),
            nan=0.0,
            posinf=RATIO_ROLLING_MAX_CLIP,
            neginf=0.0,
        ),
        index=roll.index,
        columns=roll.columns,
    )
    count_pivot = (
        panel.pivot_table(index=["country", "topic"], columns="year", values="count", aggfunc="first")
        .reindex(columns=years_all)
        .fillna(0)
        .astype(int)
    )
    share_pivot = panel.pivot_table(
        index=["country", "topic"], columns="year", values="share", aggfunc="first"
    ).reindex(columns=years_all)
    peak_share = share_pivot.max(axis=1).fillna(0.0)
    mask_peak = peak_share >= thr
    active_years = (count_pivot > 0).sum(axis=1)
    mask_active = active_years >= min_active_years
    ps_active = peak_share.reindex(active_years.index).fillna(0.0)
    print("peak_share among active_years-eligible units:\n", ps_active.loc[mask_active].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    count_pos = count_pivot > 0
    mean_share_active = share_pivot.where(count_pos).mean(axis=1).fillna(0.0)
    if mthr > 0:
        mean_ok = mean_share_active.reindex(active_years.index).fillna(0.0) >= mthr
        print("mean_share_active (count>0 years) among active-eligible:\n", mean_share_active.reindex(active_years.index).fillna(0.0).loc[mask_active].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    else:
        mean_ok = pd.Series(True, index=active_years.index)
    peak_ok = mask_peak.reindex(active_years.index).fillna(False)
    mask = mask_active & peak_ok & mean_ok
    n_active = int(mask_active.sum())
    n_final = int(mask.sum())
    n_drop_peak = int((mask_active & ~peak_ok).sum())
    n_drop_mean = int((mask_active & peak_ok & ~mean_ok).sum()) if mthr > 0 else 0
    print(
        f"Trajectory filter: active_years>={min_active_years}: {n_active}, "
        f"after peak_share>={thr} and mean_share_active>={mthr}: {n_final}, "
        f"dropped_by_peak_only: {n_drop_peak}, dropped_by_mean_only: {n_drop_mean}"
    )
    wide_roll = roll.loc[mask].copy()
    wide_roll.columns = year_cols
    wide_ratio = ratio_filled.loc[mask].copy()
    wide_ratio.columns = year_cols
    active_f = active_years.loc[mask]
    count_sub = count_pivot.loc[mask].copy()
    count_sub.columns = year_cols
    return wide_roll, wide_ratio, active_f, count_sub


years_all = np.sort(panel_main["year"].unique())
# wide_roll_raw：DTW 输入使用的原始滚动水平（不做行内 z-score，以保留与前沿的绝对距离信息）。
wide_roll_raw, wide_ratio_raw, active_years_s, count_wide = build_trajectory_matrix(
    panel_main, years_all, MAIN_ROLLING_WINDOW, ROLLING_MIN_PERIODS, MIN_ACTIVE_YEARS
)
meta = wide_roll_raw.reset_index()[["country", "topic"]]
print("Trajectory units:", len(wide_roll_raw), "T:", wide_roll_raw.shape[1], "years", years_all[0], "..", years_all[-1])
'''
    )
)

cells.append(
    md(
        """## 5. DTW 轨迹聚类与导出

- **模型**：`TimeSeriesKMeans(..., metric="dtw")`，输入张量形状为 `(n_samples, n_timesteps, 1)`，由 `wide_roll_raw` 扩展最后一维得到。
- **选 k**：在 `TRAJ_K_RANGE` 内对每个 k 拟合；将展平并标准化后的特征用于计算 **Davies–Bouldin**（越小越好）与 **轮廓系数**（越大越好），以 `(DB, -Silhouette)` 字典序择优（与原 notebook DTW 分支一致）。
- **产出**：`trajectory_clusters_dtw.csv`、`trajectory_prototypes_dtw.png`、`trajectory_umap_dtw.png`。"""
    )
)

cells.append(
    code(
        r'''def plot_trajectory_prototypes(
    wide_roll: pd.DataFrame,
    labels: np.ndarray,
    years: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """按簇绘制平均 ratio_to_frontier（rolling）曲线，用于解释各簇典型追赶形态。"""
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
) -> None:
    """将展平后的 rolling 向量标准化后做 UMAP 二维嵌入，仅作探索性可视化（meta_df 预留与下游标注联用）。"""
    from umap import UMAP

    _ = meta_df  # 保留签名便于与旧代码或后续扩展一致
    X = StandardScaler().fit_transform(wide_roll.values.astype(float))
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
    ax.set_title("Trajectory UMAP (scaled rolling features)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def score_labels_flat(X_flat: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """在展平特征上计算 DB 与轮廓系数，用于比较不同 k 的聚类质量（与 DTW 分配标签对齐）。"""
    uq = np.unique(labels)
    if len(uq) < 2 or len(X_flat) < uq.max() + 2:
        return np.inf, np.nan
    try:
        sil = float(silhouette_score(X_flat, labels, metric="euclidean"))
    except Exception:
        sil = np.nan
    db = float(davies_bouldin_score(X_flat, labels))
    return db, sil


def fit_dtw_trajectory_clusters(
    wide_roll: pd.DataFrame, random_state: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """对每条轨迹使用 DTW 度量的时间序列 k-means；在 k 网格上搜索并返回最优标签与诊断表。"""
    from tslearn.clustering import TimeSeriesKMeans

    # tslearn 期望 (n, T, d)；此处 d=1。
    X = wide_roll.values.astype(float)[:, :, np.newaxis]
    # 与聚类标签对齐的「扁平」特征，仅用于经典几何指标（非 DTW 空间下的 silhouette）。
    X_flat = StandardScaler().fit_transform(wide_roll.values.astype(float))
    best = None
    best_key = None
    best_labels: np.ndarray | None = None
    rows = []
    for k in TRAJ_K_RANGE:
        if k >= len(X):
            continue
        try:
            km = TimeSeriesKMeans(
                n_clusters=k,
                metric="dtw",
                n_init=DTW_N_INIT,
                max_iter=TS_MAX_ITER,
                random_state=random_state,
                verbose=False,
            )
            labels = km.fit_predict(X)
            uq = np.unique(labels)
            if len(uq) < 2:
                raise ValueError("single_cluster")
            db, sil = score_labels_flat(X_flat, labels)
            rows.append({"k": k, "db": db, "silhouette": sil, "inertia": float(km.inertia_)})
            key = (db, -np.nan_to_num(sil, nan=-1e9))
            if best is None or key < best_key:
                best = km
                best_key = key
                best_labels = labels
        except Exception as e:
            rows.append({"k": k, "error": str(e)})
    search_df = pd.DataFrame(rows)
    print("DTW search:\n", search_df)
    if best is None or best_labels is None:
        raise RuntimeError("DTW clustering failed for all k")
    return best_labels, {"model": best, "search_df": search_df}


labels_dtw, dtw_info = fit_dtw_trajectory_clusters(wide_roll_raw, RANDOM_STATE)

pd.DataFrame({"country": meta["country"], "topic": meta["topic"], "trajectory_cluster": labels_dtw}).to_csv(
    CATCHUP_ROUND2_DIR / "trajectory_clusters_dtw.csv", index=False
)

plot_trajectory_prototypes(
    wide_roll_raw, labels_dtw, years_all, CATCHUP_ROUND2_DIR / "trajectory_prototypes_dtw.png", "DTW prototypes"
)
plot_trajectory_umap(wide_roll_raw, labels_dtw, meta, RANDOM_STATE, CATCHUP_ROUND2_DIR / "trajectory_umap_dtw.png")

print("DTW cluster sizes:\n", pd.Series(labels_dtw).value_counts().sort_index())
'''
    )
)

cells.append(
    md(
        """## 6. 复现提示

若本 notebook 报错缺少 `paper_topics.csv` 或主数据 CSV，请先在同一环境中运行上游 **KeyBERT 聚类 / 导出** 相关 notebook，使 `config/cluster_keybert_from_cluster.json` 中的路径可用，并生成 `output/catchup_mvp/topic_country_year_panel.csv`（可选，但可加速读取）。"""
    )
)

nb = nbf.new_notebook(
    metadata={
        "kernelspec": {
            "display_name": "catch-up (uv venv)",
            "language": "python",
            "name": "catchup-venv",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    cells=cells,
)
nbformat.write(nb, OUT_NB)
print("Wrote", OUT_NB)
