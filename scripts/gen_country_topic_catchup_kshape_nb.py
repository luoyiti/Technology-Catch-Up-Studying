# Generates country_topic_catchup_kshape.ipynb (run from repo root: uv run python scripts/gen_country_topic_catchup_kshape_nb.py)
from __future__ import annotations

from pathlib import Path

import nbformat
import nbformat.v4 as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "country_topic_catchup_kshape.ipynb"

    md_intro = r"""## 1. 研究目标（k-Shape 轨迹聚类，替代 DTW / GMM 轨迹主线）

对每个 **(国家, 主题)**，在多年份上观察相对研究前沿的强度（`ratio_to_frontier`），经与 MVP/DTW 一致的 **rolling 平滑** 后得到一条一维时间序列。

在本项目中：

- **GMM 轨迹聚类**（`country_topic_catchup_gmm.ipynb`）把展平后的 rolling 轨迹当作高维欧氏向量，更敏感于**绝对水平**与多维联合结构；
- **DTW 轨迹聚类**（`country_topic_catchup_dtw.ipynb`）允许时间轴局部错位，但仍主要在**原始 rolling 水平**上比较轨迹，容易让「长期高位 vs 长期低位但形状相似」的单元分到不同类。

**k-Shape**（`tslearn.clustering.KShape`）在**行内 z-score 标准化**后的序列上聚类，目标更接近 **shape subtyping**（平台型、持续爬升、先升后平、波动型等），弱化「离前沿绝对距离」的主导作用。

**本 notebook 的定位**：形成一套可顺序运行、可解释、可导出的 **k-Shape 版轨迹聚类**流程；**不预设 k-Shape 优于 GMM/DTW**——模块 G 会结合原型与簇规模如实讨论主方法 vs 补充方法的适用边界。

**输出目录**：`output/catchup_kshape/`（**不写入** `output/catchup_mvp/` 与 `output/catchup_round2/`，不修改其他 notebook）。"""

    md_deps = r"""## 2. 数据依赖

- **配置**：`config/cluster_keybert_from_cluster.json`（随机种子、rolling 窗口）。
- **面板**：优先 `output/catchup_mvp/topic_country_year_panel.csv`；缺失时按与 DTW 相同逻辑从 `paper_topics.csv` 或主 CSV 重建。
- **机制标签（可选）**：`output/catchup_mvp/mechanism_cluster_labels.csv`（由 GMM notebook 生成），用于轨迹–机制交叉。

**环境**：仓库 `uv sync` 后使用 `.venv`；需已安装 `tslearn`、`umap-learn`。

**内核**：本 notebook 元数据指定内核名 `catchup-project`（指向 `.venv` 中的 Python）。若 Jupyter 列表中看不到，请在仓库根目录执行一次：  
`./.venv/bin/python -m ipykernel install --prefix=.venv --name=catchup-project --display-name="catch-up (.venv)"`  
批处理执行（`nbconvert`）时设置 `JUPYTER_PATH` 为 `.venv/share/jupyter` 以便解析该内核。"""

    code_setup = r'''from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_CONFIG_PATH = Path("config/cluster_keybert_from_cluster.json")
CATCHUP_MVP_DIR = Path("output/catchup_mvp")
EXPORT_DIR = Path("output/catchup_kshape")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MECH_LABELS_PATH = CATCHUP_MVP_DIR / "mechanism_cluster_labels.csv"
MECH_FEATURES_PATH = CATCHUP_MVP_DIR / "mechanism_features.csv"
GMM_TRAJ_PATH = CATCHUP_MVP_DIR / "trajectory_cluster_labels.csv"
DTW_TRAJ_PATH = Path("output/catchup_round2/trajectory_clusters_dtw.csv")


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_reproducibility(seed: int) -> None:
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

print("EXPORT_DIR:", EXPORT_DIR.resolve())
print("CATCHUP_MVP_DIR exists:", CATCHUP_MVP_DIR.is_dir())
'''

    code_params = r'''# Central parameters (edit here)
RANDOM_STATE = RANDOM_STATE
MIN_ACTIVE_YEARS = 6
ROLLING_WINDOW = CONFIG_ROLLING_WINDOW
ROLLING_MIN_PERIODS = CONFIG_ROLLING_MIN
MIN_PEAK_TOPIC_YEAR_SHARE = 0.05
MIN_MEAN_SHARE_ACTIVE = 0.0
FRONTIER_TOP_K = 3
MAIN_FRONTIER_MODE: Literal["top3_mean", "max"] = "top3_mean"
FRONTIER_RATIO_EPS = 1e-10
RATIO_ROLLING_MAX_CLIP = 50.0

KSHAPE_K_RANGE = range(2, 9)  # k = 2..8
KSHAPE_N_INIT = 5
KSHAPE_MAX_ITER = 30

# Minimum cluster size (absolute and fraction of n) for "viable" k in automatic pick
MIN_CLUSTER_ABS = 15
MIN_CLUSTER_FRAC = 0.03

# Override automatic k (None = use picker)
K_FINAL_OVERRIDE: int | None = None

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

plt.rcParams["figure.dpi"] = 120
print(
    "Params:",
    "ROLLING_WINDOW=", ROLLING_WINDOW,
    "KSHAPE_K_RANGE=", list(KSHAPE_K_RANGE),
    "MIN_ACTIVE_YEARS=", MIN_ACTIVE_YEARS,
)
'''

    code_panel_traj = r'''# --- Column resolution & panel load (aligned with country_topic_catchup_dtw.ipynb) ---
TOPIC_CANDIDATES = ("topic", "Topic", "topic_id")
YEAR_CANDIDATES = ("year", "Year", "pub_year", "publication_year")
COUNTRY_CANDIDATES = ("country_code", "country", "Country", "nation")
RATIO_CANDIDATES = ("ratio_to_frontier", "rolling_ratio_to_frontier", "ratio_roll")


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
    if PAPER_TOPICS_PATH.exists():
        df = pd.read_csv(PAPER_TOPICS_PATH, low_memory=False)
        return df, str(PAPER_TOPICS_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Neither {PAPER_TOPICS_PATH} nor {DATA_PATH} exists. "
            "Run upstream clustering notebook to export paper_topics.csv."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if _first_present(df, TOPIC_CANDIDATES) is None:
        raise ValueError(f"No topic column in {DATA_PATH}.")
    return df, str(DATA_PATH)


def frontier_topk_mean_share(shares: pd.Series, k: int) -> float:
    s = shares.astype(float).sort_values(ascending=False)
    s = s[s > 0] if (s > 0).any() else s
    if s.empty:
        return 0.0
    top = s.head(min(k, len(s)))
    return float(top.mean())


def frontier_max_share(shares: pd.Series) -> float:
    s = shares.astype(float)
    if s.empty:
        return 0.0
    return float(s.max())


def build_topic_country_year_panel_with_frontier(
    papers: pd.DataFrame,
    frontier_mode: Literal["top3_mean", "max"],
    top_k: int,
) -> pd.DataFrame:
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
    fv = panel["frontier_value"].astype(float)
    sh = panel["share"].astype(float)
    denom = np.maximum(fv, FRONTIER_RATIO_EPS)
    panel["ratio_to_frontier"] = (sh / denom).astype(float)
    panel["gap_to_frontier"] = panel["frontier_value"] - panel["share"]
    return panel


def load_or_build_panel(frontier_mode: str, top_k: int) -> tuple[pd.DataFrame, str]:
    mvp_panel = CATCHUP_MVP_DIR / "topic_country_year_panel.csv"
    if mvp_panel.exists() and frontier_mode == "top3_mean":
        panel = pd.read_csv(mvp_panel, low_memory=False)
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
            out_rebuilt = EXPORT_DIR / "topic_country_year_panel_rebuilt.csv"
            panel.to_csv(out_rebuilt, index=False)
            print("Saved rebuilt panel to", out_rebuilt)
    return panel, src


def build_trajectory_matrix(
    panel: pd.DataFrame,
    years_all: np.ndarray,
    rolling_window: int,
    min_periods: int,
    min_active_years: int,
    min_peak_share: float | None = None,
    min_mean_share_active: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    year_cols = [str(y) for y in years_all]
    thr = MIN_PEAK_TOPIC_YEAR_SHARE if min_peak_share is None else float(min_peak_share)
    mthr = MIN_MEAN_SHARE_ACTIVE if min_mean_share_active is None else float(min_mean_share_active)
    ratio_col = _first_present(panel, RATIO_CANDIDATES)
    if not ratio_col:
        raise KeyError(f"No ratio column in panel; tried {RATIO_CANDIDATES}")
    if ratio_col != "ratio_to_frontier":
        panel = panel.rename(columns={ratio_col: "ratio_to_frontier"})
    ratio_pivot = panel.pivot_table(
        index=["country", "topic"], columns="year", values="ratio_to_frontier", aggfunc="first"
    )
    ratio_pivot = ratio_pivot.reindex(columns=years_all)
    ratio_filled = ratio_pivot.fillna(0.0)
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
    print("peak_share among active_years-eligible:\n", ps_active.loc[mask_active].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    count_pos = count_pivot > 0
    mean_share_active = share_pivot.where(count_pos).mean(axis=1).fillna(0.0)
    if mthr > 0:
        mean_ok = mean_share_active.reindex(active_years.index).fillna(0.0) >= mthr
    else:
        mean_ok = pd.Series(True, index=active_years.index)
    peak_ok = mask_peak.reindex(active_years.index).fillna(False)
    mask = mask_active & peak_ok & mean_ok
    n_active = int(mask_active.sum())
    n_final = int(mask.sum())
    print(
        f"Trajectory filter: active_years>={min_active_years}: {n_active}, "
        f"after peak_share>={thr}: {n_final}"
    )
    wide_roll = roll.loc[mask].copy()
    wide_roll.columns = year_cols
    wide_ratio = ratio_filled.loc[mask].copy()
    wide_ratio.columns = year_cols
    active_f = active_years.loc[mask]
    count_sub = count_pivot.loc[mask].copy()
    count_sub.columns = year_cols
    return wide_roll, wide_ratio, active_f, count_sub


def row_zscore_matrix(wide_roll: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score over time (shape-focused input for k-Shape)."""
    arr = wide_roll.values.astype(float)
    mu = arr.mean(axis=1, keepdims=True)
    sig = arr.std(axis=1, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    z = (arr - mu) / sig
    return pd.DataFrame(z, index=wide_roll.index, columns=wide_roll.columns)


def score_labels_flat(X_flat: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    uq = np.unique(labels)
    if len(uq) < 2 or len(X_flat) < uq.max() + 2:
        return np.inf, np.nan
    try:
        sil = float(silhouette_score(X_flat, labels, metric="euclidean"))
    except Exception:
        sil = np.nan
    try:
        db = float(davies_bouldin_score(X_flat, labels))
    except Exception:
        db = np.inf
    return db, sil


def cluster_size_entropy(labels: np.ndarray) -> float:
    vc = pd.Series(labels).value_counts(normalize=True)
    return float(-(vc * np.log(vc + 1e-15)).sum())


panel_main, panel_src = load_or_build_panel(MAIN_FRONTIER_MODE, FRONTIER_TOP_K)
print("Panel source:", panel_src)
print("shape", panel_main.shape, "cols", list(panel_main.columns))
print("NA rate (top):\n", panel_main.isna().mean().sort_values(ascending=False).head(8))
print(panel_main.head(3))

years_all = np.sort(panel_main["year"].unique())
wide_roll_raw, wide_ratio_raw, active_years_s, count_wide = build_trajectory_matrix(
    panel_main, years_all, ROLLING_WINDOW, ROLLING_MIN_PERIODS, MIN_ACTIVE_YEARS
)
wide_roll_z = row_zscore_matrix(wide_roll_raw)
meta = wide_roll_raw.reset_index()[["country", "topic"]]
print("Trajectory units:", len(wide_roll_raw), "T:", wide_roll_raw.shape[1], "years", int(years_all[0]), "..", int(years_all[-1]))
'''

    code_module_a_diag = r'''# --- Module A: optional existing artifacts ---
def _peek_csv(p: Path, name: str) -> None:
    print(f"\n=== {name} ({p}) exists={p.exists()} ===")
    if not p.exists():
        return
    df = pd.read_csv(p, nrows=5, low_memory=False)
    full = pd.read_csv(p, low_memory=False)
    print("shape", full.shape, "columns", list(full.columns))
    print(full.head(3))
    print("na_frac (top)", full.isna().mean().sort_values(ascending=False).head(6))


_peek_csv(MECH_LABELS_PATH, "mechanism_cluster_labels")
_peek_csv(MECH_FEATURES_PATH, "mechanism_features")
_peek_csv(GMM_TRAJ_PATH, "GMM trajectory_cluster_labels")
_peek_csv(DTW_TRAJ_PATH, "DTW trajectory_clusters")
'''

    code_ksearch = r'''from tslearn.clustering import KShape


def run_kshape_search(
    wide_z: pd.DataFrame,
    random_state: int,
    k_list: list[int],
    n_init: int,
    max_iter: int,
) -> pd.DataFrame:
    X = wide_z.values.astype(float)[:, :, np.newaxis]
    n = X.shape[0]
    X_flat = wide_z.values.astype(float).reshape(n, -1)
    X_flat_s = StandardScaler().fit_transform(X_flat)
    rows: list[dict[str, Any]] = []
    for k in k_list:
        if k < 2 or k >= n:
            rows.append({"k": k, "status": "skip", "error": "k_out_of_range"})
            continue
        try:
            ks = KShape(
                n_clusters=k,
                max_iter=max_iter,
                random_state=random_state,
                n_init=n_init,
                init="random",
                verbose=False,
            )
            labels = ks.fit_predict(X)
            uq, counts = np.unique(labels, return_counts=True)
            if len(uq) < 2:
                raise ValueError("single_cluster")
            min_sz = int(counts.min())
            db, sil = score_labels_flat(X_flat_s, labels)
            sizes = pd.Series(labels).value_counts().sort_index()
            sz_str = ",".join(f"{int(i)}:{int(sizes.loc[i])}" for i in sizes.index)
            n_iter = int(getattr(ks, "n_iter_", -1) or -1)
            ent = cluster_size_entropy(labels)
            balance = min_sz / max(n / k, 1e-9)
            note = f"k={k}: min_size={min_sz}, entropy={ent:.3f}"
            rows.append(
                {
                    "k": k,
                    "status": "ok",
                    "db": db,
                    "silhouette": sil,
                    "inertia": np.nan,
                    "n_iter": n_iter,
                    "cluster_sizes": sz_str,
                    "min_cluster_size": min_sz,
                    "size_entropy": ent,
                    "balance_min_over_uniform": float(balance),
                    "interpretability_note": note,
                    "error": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "k": k,
                    "status": "error",
                    "db": np.nan,
                    "silhouette": np.nan,
                    "inertia": np.nan,
                    "n_iter": -1,
                    "cluster_sizes": "",
                    "min_cluster_size": 0,
                    "size_entropy": np.nan,
                    "balance_min_over_uniform": np.nan,
                    "interpretability_note": "",
                    "error": str(e),
                }
            )
    return pd.DataFrame(rows)


def pick_k_final_kshape(search_df: pd.DataFrame, n_samples: int) -> tuple[int, str]:
    ok = search_df["status"] == "ok"
    sub = search_df.loc[ok].copy()
    if sub.empty:
        raise RuntimeError("No successful k in k-Shape search; relax parameters or check data.")
    thr_sz = max(MIN_CLUSTER_ABS, int(MIN_CLUSTER_FRAC * n_samples))
    sub["viable"] = sub["min_cluster_size"] >= thr_sz
    viable = sub[sub["viable"]]
    pick_from = viable if len(viable) else sub
    pick_from = pick_from.sort_values(["db", "silhouette", "k"], ascending=[True, False, True])
    k_star = int(pick_from.iloc[0]["k"])
    if len(viable):
        rationale = (
            f"Among k with min_cluster_size>={thr_sz}, choose lexicographic min DB then max silhouette; "
            f"tie-breaker prefers smaller k."
        )
    else:
        rationale = (
            f"No k met min_cluster_size>={thr_sz}; fell back to global min DB / max silhouette. "
            f"Consider lowering MIN_CLUSTER_ABS or inspecting rare-shape clusters."
        )
    return k_star, rationale


_k_list = [int(k) for k in KSHAPE_K_RANGE]
search_df = run_kshape_search(wide_roll_z, RANDOM_STATE, _k_list, KSHAPE_N_INIT, KSHAPE_MAX_ITER)
search_path = EXPORT_DIR / "trajectory_kshape_k_search.csv"
search_df.to_csv(search_path, index=False)
print("Saved", search_path)
print(search_df.to_string())

if K_FINAL_OVERRIDE is not None:
    k_final = int(K_FINAL_OVERRIDE)
    k_pick_note = f"Manual override K_FINAL_OVERRIDE={k_final}"
else:
    k_final, k_pick_note = pick_k_final_kshape(search_df, len(wide_roll_z))
    k_pick_note = "Automatic: " + k_pick_note

print(f"\nk_final={k_final}\n{k_pick_note}")
'''

    code_fit_plot = r'''def plot_kshape_prototypes(
    model,
    years: np.ndarray,
    wide_roll_raw: pd.DataFrame,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    centers = np.asarray(model.cluster_centers_)
    if centers.ndim == 3:
        centers = centers[:, :, 0]
    clusters = np.sort(np.unique(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axz, axr = axes
    for c in clusters:
        cc = int(c)
        axz.plot(years, centers[cc], lw=1.8, label=f"C{cc} (center)")
        idx = labels == c
        axr.plot(
            years,
            wide_roll_raw.values[idx].mean(axis=0),
            lw=1.5,
            alpha=0.85,
            label=f"C{cc} mean raw roll (n={int(idx.sum())})",
        )
    axz.axhline(0.0, color="gray", ls="--", lw=0.8)
    axz.set_title("k-Shape centers (row-z-score space)")
    axz.set_xlabel("Year")
    axz.set_ylabel("Center (z)")
    axz.legend(fontsize=7, ncol=2)
    axr.axhline(1.0, color="gray", ls="--", lw=0.8)
    axr.set_title("Mean rolling ratio (original scale)")
    axr.set_xlabel("Year")
    axr.set_ylabel("Mean rolling ratio_to_frontier")
    axr.legend(fontsize=7, ncol=2)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fit_kshape_final(wide_z: pd.DataFrame, k: int, random_state: int, n_init: int, max_iter: int):
    X = wide_z.values.astype(float)[:, :, np.newaxis]
    ks = KShape(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        n_init=n_init,
        init="random",
        verbose=False,
    )
    labels = ks.fit_predict(X)
    return labels, ks


labels_k, kshape_model = fit_kshape_final(
    wide_roll_z, k_final, RANDOM_STATE, KSHAPE_N_INIT, KSHAPE_MAX_ITER
)

out_labels = pd.DataFrame(
    {
        "country": meta["country"].values,
        "topic": meta["topic"].values,
        "trajectory_cluster_kshape": labels_k.astype(int),
    }
)
out_labels.to_csv(EXPORT_DIR / "trajectory_clusters_kshape.csv", index=False)

size_summary = (
    out_labels.groupby("trajectory_cluster_kshape")
    .size()
    .rename("n")
    .reset_index()
    .sort_values("trajectory_cluster_kshape")
)
size_summary["fraction"] = size_summary["n"] / size_summary["n"].sum()
size_summary.to_csv(EXPORT_DIR / "trajectory_cluster_size_summary_kshape.csv", index=False)
print("Cluster sizes:\n", size_summary)

plot_kshape_prototypes(
    kshape_model,
    years_all,
    wide_roll_raw,
    labels_k,
    EXPORT_DIR / "trajectory_prototypes_kshape.png",
    f"k-Shape trajectory prototypes (k={k_final})",
)
print("Saved trajectory_prototypes_kshape.png")
'''

    code_umap_pca = r'''def plot_kshape_embedding(
    wide_for_embed: pd.DataFrame,
    labels: np.ndarray,
    random_state: int,
    out_umap: Path,
    out_pca: Path,
) -> None:
    from umap import UMAP

    X = StandardScaler().fit_transform(wide_for_embed.values.astype(float))
    n = X.shape[0]
    if n < 5:
        warnings.warn("Too few samples; skip UMAP/PCA plots")
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
    ax.set_title("Trajectory UMAP (flattened row-z-score; visualization only)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_umap, dpi=160)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=random_state)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=16, alpha=0.72, label=f"C{int(c)} (n={int(m.sum())})")
    ax.set_title("Trajectory PCA (same features; conservative view)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_pca, dpi=160)
    plt.close(fig)


plot_kshape_embedding(
    wide_roll_z,
    labels_k,
    RANDOM_STATE,
    EXPORT_DIR / "trajectory_umap_kshape.png",
    EXPORT_DIR / "trajectory_pca_kshape.png",
)
print("Saved trajectory_umap_kshape.png and trajectory_pca_kshape.png")
'''

    code_crosswalk = r'''def resolve_mechanism_column(df: pd.DataFrame) -> str | None:
    for c in ("mechanism_cluster", "mech_cluster", "mechanism_label"):
        if c in df.columns:
            return c
    return None


def build_crosswalk_with_mechanism(
    traj: pd.DataFrame,
    mech_path: Path,
    out_csv: Path,
    out_png: Path,
) -> pd.DataFrame | None:
    if not mech_path.exists():
        print("Mechanism labels not found; skip crosswalk:", mech_path)
        return None
    mech = pd.read_csv(mech_path, low_memory=False)
    mcc = resolve_mechanism_column(mech)
    if not mcc:
        print("No mechanism cluster column in", list(mech.columns))
        return None
    mech = mech.rename(columns={mcc: "mechanism_cluster"})
    cross = traj.merge(mech[["country", "topic", "mechanism_cluster"]], on=["country", "topic"], how="inner")
    cross.to_csv(out_csv, index=False)
    print("Crosswalk rows:", len(cross), "saved", out_csv)

    ct = pd.crosstab(cross["trajectory_cluster_kshape"], cross["mechanism_cluster"])
    fig, ax = plt.subplots(figsize=(max(6, ct.shape[1] * 0.5), max(4, ct.shape[0] * 0.45)))
    im = ax.imshow(ct.values.astype(float), aspect="auto", cmap="Blues")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([str(c) for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels([str(r) for r in ct.index])
    ax.set_xlabel("Mechanism cluster")
    ax.set_ylabel("k-Shape trajectory cluster")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Trajectory (k-Shape) vs mechanism (counts)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Saved", out_png)
    return cross


cross = build_crosswalk_with_mechanism(
    out_labels,
    MECH_LABELS_PATH,
    EXPORT_DIR / "trajectory_mechanism_crosswalk_kshape.csv",
    EXPORT_DIR / "trajectory_mechanism_crosswalk_kshape.png",
)
'''

    code_summary = r'''# Optional agreement with GMM / DTW (read-only reference files)
def _safe_ari_nmi(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    try:
        return float(adjusted_rand_score(a, b)), float(normalized_mutual_info_score(a, b))
    except Exception:
        return None, None


if GMM_TRAJ_PATH.exists():
    gmm = pd.read_csv(GMM_TRAJ_PATH, low_memory=False)
    tcol = "trajectory_cluster" if "trajectory_cluster" in gmm.columns else None
    if tcol:
        m = out_labels.merge(gmm[["country", "topic", tcol]], on=["country", "topic"], how="inner")
        if len(m):
            ari, nmi = _safe_ari_nmi(m["trajectory_cluster_kshape"].values, m[tcol].values)
            print("vs GMM trajectory — ARI:", ari, "NMI:", nmi)
if DTW_TRAJ_PATH.exists():
    dtw = pd.read_csv(DTW_TRAJ_PATH, low_memory=False)
    tcol = "trajectory_cluster" if "trajectory_cluster" in dtw.columns else None
    if tcol:
        m = out_labels.merge(dtw[["country", "topic", tcol]], on=["country", "topic"], how="inner")
        if len(m):
            ari, nmi = _safe_ari_nmi(m["trajectory_cluster_kshape"].values, m[tcol].values)
            print("vs DTW trajectory — ARI:", ari, "NMI:", nmi)

summary = {
    "notebook": "country_topic_catchup_kshape.ipynb",
    "k_final": int(k_final),
    "k_pick_note": k_pick_note,
    "export_dir": str(EXPORT_DIR.resolve()),
    "outputs": sorted(p.name for p in EXPORT_DIR.glob("*") if p.is_file()),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
'''

    md_b = r"""## 3. 模块 B：k-Shape 输入矩阵

- **原始矩阵** `wide_roll_raw`：rolling 后的 `ratio_to_frontier`，用于**解释**（与前沿的绝对距离）及均值曲线对照图。
- **行内 z-score 矩阵** `wide_roll_z`：**k-Shape 的唯一聚类输入**。每条 `(country, topic)` 在时间维上标准化到零均值单位方差，使算法关注**形状**而非绝对幅值，避免「长期低水平但形态相似」被水平差异完全支配。

缺失年仍按项目惯例在透视后填 **0** 再 rolling（与 `country_topic_catchup_dtw.ipynb` 一致）。"""

    md_c = r"""## 4. 模块 C：k 搜索与指标

对每个 `k`，在 **row-z-score** 张量上拟合 `KShape`。**轮廓系数 / Davies–Bouldin** 在**展平且整体标准化**后的欧氏特征上计算，仅作辅助比较（**不是** k-Shape 目标函数）。

`k_final` 默认规则：优先只考虑 **最小簇规模** 达到阈值（绝对个数与占样本比例）的候选；在其中按 **DB 升序、轮廓降序** 选取。若无一满足阈值，则退回全局并打印说明。可通过 `K_FINAL_OVERRIDE` 强制指定 k。"""

    md_d = r"""## 5. 模块 D：最终聚类与原型

导出 `trajectory_clusters_kshape.csv` 与 `trajectory_cluster_size_summary_kshape.csv`。原型图左列为 **k-Shape center（z 空间）**，右列为 **原始 rolling 的簇内均值**（便于与业务解释衔接）。"""

    md_e = r"""## 6. 模块 E：低维可视化（非聚类依据）

在展平的 **row-z-score** 上做 `StandardScaler` + **UMAP** 与 **PCA**。UMAP 更利于展示局部结构；PCA 更保守。二者均**不参与** k-Shape 分配。"""

    md_f = r"""## 7. 模块 F：与机制聚类对齐

合并 `mechanism_cluster_labels.csv`（若存在），输出交叉表与热图。"""

    md_g = r"""## 8. 模块 G：结果解释与定位

请结合上文的 **簇规模表**、**原型图**、**UMAP/PCA** 与（若存在）**轨迹–机制交叉表**自行归纳各 trajectory type。

**与 GMM / DTW 的差异**：

- GMM：展平欧氏空间，水平与多时间点联合模式共同驱动。
- DTW：允许时间错位，仍在**原始 rolling 水平**上比较。
- k-Shape：在**去水平后的形状空间**聚类，更易得到「形状子类型」，但**弱化绝对前沿差距**；对噪声与初始化更敏感，小簇解释成本可能更高。

**是否作为主方法**：若 k-Shape 原型在 z 空间清晰、且与机制交叉结构稳定，可作为**主叙事中的形状维度**；若簇高度不平衡或原型难以命名，更适合作为 **DTW/GMM 的补充（shape supplement）**——以本 notebook 实际输出为准。"""

    cells = [
        nbf.new_markdown_cell(md_intro),
        nbf.new_markdown_cell(md_deps),
        nbf.new_code_cell(code_setup),
        nbf.new_markdown_cell("## 参数区（集中管理）"),
        nbf.new_code_cell(code_params),
        nbf.new_markdown_cell("## 模块 A：读取与检查"),
        nbf.new_code_cell(code_panel_traj),
        nbf.new_code_cell(code_module_a_diag),
        nbf.new_markdown_cell(md_b),
        nbf.new_markdown_cell(md_c),
        nbf.new_code_cell(code_ksearch),
        nbf.new_markdown_cell(md_d),
        nbf.new_code_cell(code_fit_plot),
        nbf.new_markdown_cell(md_e),
        nbf.new_code_cell(code_umap_pca),
        nbf.new_markdown_cell(md_f),
        nbf.new_code_cell(code_crosswalk),
        nbf.new_markdown_cell(md_g),
        nbf.new_code_cell(code_summary),
    ]

    nb = nbf.new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    # Use venv-local kernelspec so nbconvert resolves tslearn (not system/anaconda python3).
    nb["metadata"]["kernelspec"] = {
        "display_name": "catch-up (.venv)",
        "language": "python",
        "name": "catchup-project",
    }
    nbformat.write(nb, out.open("w", encoding="utf-8"))
    print("Wrote", out)


if __name__ == "__main__":
    main()
