#!/usr/bin/env python3
"""Build country_topic_catchup_autoencoder.ipynb from hierarchical template cells + AE pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
HIER = ROOT / "country_topic_catchup_hierarchical.ipynb"
DST = ROOT / "country_topic_catchup_autoencoder.ipynb"

MD0 = """## 1. 研究目标（Autoencoder 表示学习、聚类与异常轨迹检测）

对每个 **(国家, 主题)**，在多年份上观察相对研究前沿的强度（`ratio_to_frontier`），经 **时间滚动平滑** 后得到一条一维时间序列。本 notebook 在 **与 `country_topic_catchup_hierarchical.ipynb` 相同的数据与轨迹定义** 前提下，增加一条对照分析路径：

1. 使用 **Conv1D Autoencoder**（`tensorflow.keras`）将每条轨迹编码为固定维度 **latent vector**；
2. 在 **latent 空间** 上对轨迹做 **层次聚类**（与原先在原始轨迹距离空间上的层次聚类形成对照）；
3. 利用 **重构误差** 与 **latent 空间离群度**（默认 kNN 距离；可选 LOF / IsolationForest）标记 **异常时间序列**；
4. 输出与 round2 目录习惯一致的 **CSV / PNG**，文件名统一带 `ae` / `latent` 前缀，**不覆盖** `*_hier_*` 产物。

**关系说明**：`country_topic_catchup_hierarchical.ipynb` 仍以 **原始 rolling 轨迹 + 预定义距离（euclidean/dtw/shape）+ 层次聚类** 为主线；本 notebook 是 **表示学习 + latent 聚类 + 双路异常检测** 的增强/对照方案，**不替代**原方法。

**输出目录**：`output/catchup_round2/`。**不修改** `output/catchup_mvp/`。"""

MD1 = """## 2. 数据依赖

- **配置**：[`config/cluster_keybert_from_cluster.json`](config/cluster_keybert_from_cluster.json)（数据 CSV 路径、`paper_topics.csv` 所在目录、rolling 窗口与随机种子）。
- **面板**：优先读取 `output/catchup_mvp/topic_country_year_panel.csv`（与 MVP 一致）；若不存在且 frontier 为 `top3_mean`，则从 `paper_topics.csv` 或配置中的主 CSV 重建；若仍缺文件会报错并提示先运行上游 notebook。

**内核**：建议使用仓库 `.venv`；首次请 `uv sync`（已包含 **TensorFlow** 与 `umap-learn`）。本 notebook 的 `kernelspec` 为 **`catchup-venv`**（需先注册到该 venv，见下）。

在仓库根目录执行一次（将内核安装到 `.venv/share/jupyter`，便于 `nbconvert` 与 Jupyter 发现）：  
`./.venv/bin/python -m ipykernel install --prefix=.venv --name=catchup-venv --display-name='catch-up (venv)'`

**nbconvert**（使用仓库解释器与 TensorFlow）：  
`JUPYTER_PATH=\"$PWD/.venv/share/jupyter\" MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m nbconvert --execute country_topic_catchup_autoencoder.ipynb --ExecutePreprocessor.kernel_name=catchup-venv --inplace`"""

MD3 = """## 3. 参数区（集中管理）"""

MD5 = """## 4. 轨迹矩阵构建

从面板得到每个 **(country, topic, year)** 的 `ratio_to_frontier`（相对前沿比值），再按年透视、对缺失年填 0 后做 **rolling 平滑**，得到 `wide_roll_raw`：形状为 **(样本数, 年份数)**，每一行是一条时间序列。随后用 `MIN_ACTIVE_YEARS` 与峰值份额阈值筛掉不可靠单元。

**数值稳定**：`ratio = share / max(frontier_value, FRONTIER_RATIO_EPS)`，不改动 `share` 单纯形；读入已有 MVP 面板 CSV 时若含 `share` 与 `frontier_value` 会用同一公式重算 `ratio_to_frontier`。透视与 rolling 后对轨迹做有限幅裁剪，防止历史数据中的 `inf` 进入后续建模。

本段逻辑与 **hierarchical** notebook **一致**，以保证可比性。"""

MD7 = """## 5. Autoencoder：输入预处理、训练与 latent 提取

- **输入**：`wide_roll_raw` 经 `prepare_autoencoder_input` 得到 `X_ae_scaled`（形状 `(n, T, 1)`）。
- **模型**：轻量 **Conv1D** 编码器–解码器（同长度卷积 + `Dense` 瓶颈），`AE_LATENT_DIM` 维 latent（聚类与异常检测用完整 `Z`）。
- **训练**：`EarlyStopping` 监控验证损失；导出 `ae_training_loss.png`。
- **输出**：`latent_vectors.csv`、`ae_reconstruction_errors.csv`（重构 MSE 逐样本）。
- **二维散点图**：使用 `Z[:, :2]`（encoder 前两维）；若需整条表示链严格 2 维，将 `AE_LATENT_DIM=2`。"""

MD9 = """## 6. Latent 层次聚类与 k 诊断

在 latent 矩阵 `Z` 上对每条样本做 **AgglomerativeClustering**（欧氏距离；`ward` / `average` / `complete` / `single`）。对 `TRAJ_K_RANGE` 内每个 `k` 记录 DB、Silhouette、`cluster_sizes`、`size_entropy`。

- **`FINAL_N_CLUSTERS`** 为 `None` 时：按 `FINAL_K_STRATEGY` 选 `k_final`（`metrics_lex` = DB 升序、Silhouette 降序；`fixed` = `FINAL_K_OVERRIDE`）。
- **`FINAL_N_CLUSTERS`** 为整数时：**强制**使用该簇数作为 `k_final`（仍会输出 k 搜索表与诊断图供对照）。"""

MD11 = """## 7. 异常检测与总表

- **重构异常**：逐样本 `mean((x - \\hat{x})^2)`，阈值支持 `robust_z` 或 `quantile`。
- **Latent 异常**：默认 **kNN** 平均距离；可选 **LOF** / **IsolationForest**；对分数再做 robust z 并打标。
- **综合**：`is_anomalous = recon_anomaly_flag OR latent_anomaly_flag`，写入 `trajectory_anomalies_ae.csv`。"""

MD13 = """## 8. 可视化

- **Latent 二维散点**：使用 **encoder 输出的前两维** `Z[:, :2]`（按簇 / 按综合异常各一图；文件名仍为 `latent_umap_*.png` 以保持路径习惯，但已非 UMAP）。
- **簇原型曲线**（仍在 **原始 rolling ratio** 空间画各簇均值曲线，便于与 hierarchical 解读对齐）；
- **重构示例**（正常 vs 异常若干条）。
- 原 **UMAP** 实现保留在绘图代码单元的 `if False:` 块中，便于对照与恢复。"""

MD15 = """## 9. 小结

- **Hierarchical（原 notebook）**：直接在轨迹空间定义距离并聚类。
- **本 notebook（AE）**：先学习低维表示，再在 latent 上聚类，并用重构与 latent 离群度发现异常轨迹。两者可并列用于论文中的方法对照。"""

CELL4 = r'''# 与配置中的随机种子一致（上一单元已从 JSON 写入 RANDOM_STATE）。
RANDOM_STATE = RANDOM_STATE

# 轨迹样本至少需要在多少年份上有发文，才进入后续建模。
MIN_ACTIVE_YEARS = 6

# 对 ratio_to_frontier 做滚动平均的窗口长度与最小有效窗口（与 MVP / KeyBERT 配置对齐）。
MAIN_ROLLING_WINDOW = CONFIG_ROLLING_WINDOW
ROLLING_MIN_PERIODS = CONFIG_ROLLING_MIN

# 前沿强度定义：在每个 (topic, year) 上取发文量 Top-K 国家的平均 share 作为 frontier（top3_mean）。
FRONTIER_TOP_K = 3
MAIN_FRONTIER_MODE: Literal["top3_mean", "max"] = "top3_mean"

# --- Autoencoder 参数 ---
AE_MODEL_TYPE: Literal["conv1d"] = "conv1d"
AE_LATENT_DIM = 8
AE_EPOCHS = 200
AE_BATCH_SIZE = 16
AE_LEARNING_RATE = 1e-3
AE_VALIDATION_SPLIT = 0.2
AE_EARLY_STOPPING_PATIENCE = 20
AE_STANDARDIZE_MODE: Literal["per_series", "global"] = "per_series"
AE_RECON_LOSS: Literal["mse", "mae"] = "mse"

# --- latent 聚类参数 ---
LATENT_CLUSTER_METHOD: Literal["hierarchical"] = "hierarchical"
LATENT_HIER_LINKAGE: Literal["ward", "average", "complete", "single"] = "ward"
TRAJ_K_RANGE = range(2, 11)
FINAL_K_STRATEGY: Literal["metrics_lex", "fixed"] = "metrics_lex"
FINAL_K_OVERRIDE: int | None = None
# 最终聚类数（簇个数 k）开关：None = 按 FINAL_K_STRATEGY / FINAL_K_OVERRIDE 从搜索结果中选；整数 = 强制使用该 k（仍会跑 k 诊断表）。
FINAL_N_CLUSTERS: int | None = None

# --- latent 异常检测参数 ---
LATENT_ANOMALY_METHOD: Literal["knn", "lof", "isolation_forest"] = "knn"
LATENT_KNN_K = 5

# --- reconstruction 异常阈值 ---
RECON_ANOMALY_RULE: Literal["robust_z", "quantile"] = "robust_z"
RECON_ANOMALY_Z_THRESHOLD = 3.0
RECON_ANOMALY_QUANTILE = 0.95

# 设为非空列表时，可在轨迹重建上做 rolling 对比（默认关闭；本 AE 主线不依赖）。
SWEEP_ROLLING_WINDOWS: list[int] = []

# 过滤弱信号单元。
MIN_PEAK_TOPIC_YEAR_SHARE = 0.05
MIN_MEAN_SHARE_ACTIVE = 0.0

# UMAP 仅用于 latent 或特征可视化。
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

FRONTIER_RATIO_EPS = 1e-10
RATIO_ROLLING_MAX_CLIP = 50.0


def validate_ae_params() -> None:
    """Basic sanity checks for AE + clustering + anomaly settings."""
    if AE_MODEL_TYPE != "conv1d":
        raise ValueError(f"Unsupported AE_MODEL_TYPE: {AE_MODEL_TYPE!r}")
    if AE_STANDARDIZE_MODE not in ("per_series", "global"):
        raise ValueError(AE_STANDARDIZE_MODE)
    if AE_RECON_LOSS not in ("mse", "mae"):
        raise ValueError(AE_RECON_LOSS)
    if LATENT_CLUSTER_METHOD != "hierarchical":
        raise ValueError(LATENT_CLUSTER_METHOD)
    if LATENT_HIER_LINKAGE not in ("ward", "average", "complete", "single"):
        raise ValueError(LATENT_HIER_LINKAGE)
    if LATENT_ANOMALY_METHOD not in ("knn", "lof", "isolation_forest"):
        raise ValueError(LATENT_ANOMALY_METHOD)
    if RECON_ANOMALY_RULE not in ("robust_z", "quantile"):
        raise ValueError(RECON_ANOMALY_RULE)
    if not (0.0 < AE_VALIDATION_SPLIT < 0.5):
        raise ValueError("AE_VALIDATION_SPLIT should be in (0, 0.5)")
    if AE_LATENT_DIM < 2:
        raise ValueError("AE_LATENT_DIM should be >= 2")
    if FINAL_N_CLUSTERS is not None and int(FINAL_N_CLUSTERS) < 2:
        raise ValueError("FINAL_N_CLUSTERS must be >= 2 when set")


validate_ae_params()

plt.rcParams["figure.dpi"] = 120
print(
    "主要参数:",
    "MAIN_ROLLING_WINDOW=",
    MAIN_ROLLING_WINDOW,
    "AE_LATENT_DIM=",
    AE_LATENT_DIM,
    "AE_STANDARDIZE_MODE=",
    AE_STANDARDIZE_MODE,
    "LATENT_HIER_LINKAGE=",
    LATENT_HIER_LINKAGE,
    "TRAJ_K_RANGE=",
    list(TRAJ_K_RANGE),
    "FINAL_K_STRATEGY=",
    FINAL_K_STRATEGY,
    "FINAL_N_CLUSTERS=",
    FINAL_N_CLUSTERS,
    "LATENT_ANOMALY_METHOD=",
    LATENT_ANOMALY_METHOD,
    "RECON_ANOMALY_RULE=",
    RECON_ANOMALY_RULE,
)
'''

CELL8 = r'''# --- AE: preprocessing, model, training, latent + recon export ---

EPS = 1e-9


def prepare_autoencoder_input(
    wide_roll: pd.DataFrame, standardize_mode: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert rolling trajectory matrix to Conv1D input tensor.

    Returns
    -------
    X_raw : ndarray, shape (n, T)
        Row-aligned copy of wide_roll values.
    X_3d : ndarray, shape (n, T, 1)
        Scaled tensor for the autoencoder.
    prep_info : dict
        Scales / means for inverse transform or auditing.
    """
    X_raw = wide_roll.to_numpy(dtype=np.float64, copy=True)
    n, T = X_raw.shape
    prep_info: dict[str, Any] = {"standardize_mode": standardize_mode, "T": int(T)}
    if standardize_mode == "global":
        mu = float(np.nanmean(X_raw))
        sig = float(np.nanstd(X_raw))
        if not np.isfinite(sig) or sig < EPS:
            sig = 1.0
        Xs = (X_raw - mu) / sig
        prep_info["global_mu"] = mu
        prep_info["global_sig"] = sig
    elif standardize_mode == "per_series":
        mu = np.nanmean(X_raw, axis=1, keepdims=True)
        sig = np.nanstd(X_raw, axis=1, keepdims=True)
        bad = (~np.isfinite(sig)) | (sig < EPS)
        sig_safe = np.where(bad, 1.0, sig)
        Xs = (X_raw - mu) / sig_safe
        Xs = np.where(bad, 0.0, Xs)
        prep_info["per_series_mu"] = mu.ravel().tolist()
        prep_info["per_series_sig"] = sig_safe.ravel().tolist()
    else:
        raise ValueError(standardize_mode)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    X_3d = Xs.reshape(n, T, 1).astype(np.float32)
    return X_raw.astype(np.float32), X_3d, prep_info


def build_conv1d_autoencoder(
    seq_len: int, latent_dim: int, learning_rate: float, recon_loss: str
) -> tuple[Any, Any]:
    """Lightweight same-length Conv1D autoencoder + encoder head.

    Returns
    -------
    autoencoder, encoder Keras models.
    """
    from tensorflow.keras import Model, layers, optimizers

    inp = layers.Input(shape=(seq_len, 1), name="input")
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    bottleneck = int(seq_len * 64)
    latent = layers.Dense(latent_dim, name="latent")(x)

    enc = Model(inp, latent, name="encoder")

    dec_in = layers.Input(shape=(latent_dim,), name="latent_in")
    d = layers.Dense(bottleneck, activation="relu")(dec_in)
    d = layers.Reshape((seq_len, 64))(d)
    d = layers.Conv1D(32, 5, padding="same", activation="relu")(d)
    out = layers.Conv1D(1, 5, padding="same", name="recon")(d)
    dec = Model(dec_in, out, name="decoder")

    full_in = layers.Input(shape=(seq_len, 1), name="ae_in")
    z = enc(full_in)
    full_out = dec(z)
    autoencoder = Model(full_in, full_out, name="conv1d_ae")

    loss = "mse" if recon_loss == "mse" else "mae"
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )
    return autoencoder, enc


def fit_autoencoder_model(
    autoencoder: Any,
    X: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
    early_stopping_patience: int,
    random_state: int,
) -> Any:
    """Train autoencoder; adapts validation split for small n."""
    from tensorflow.keras import callbacks

    n = X.shape[0]
    tf.random.set_seed(int(random_state))
    if n < 10:
        vs = 0.0
        es_monitor = "loss"
        patience = max(5, early_stopping_patience // 2)
    else:
        vs = float(validation_split)
        if int(max(2, round(vs * n))) >= n:
            vs = 0.0
            es_monitor = "loss"
        else:
            es_monitor = "val_loss"
        patience = int(early_stopping_patience)
    cb = [
        callbacks.EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-6,
        )
    ]
    hist = autoencoder.fit(
        X,
        X,
        epochs=int(epochs),
        batch_size=min(int(batch_size), n),
        validation_split=vs,
        callbacks=cb,
        verbose=2,
    )
    return hist


def compute_reconstruction_errors(X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """Per-sample mean squared error over time and channel."""
    d = (X_true.astype(np.float64) - X_pred.astype(np.float64)) ** 2
    return np.mean(d, axis=(1, 2))


# Prepare tensors
X_ae_raw, X_ae_scaled, prep_ae = prepare_autoencoder_input(wide_roll_raw, AE_STANDARDIZE_MODE)
seq_len = int(X_ae_scaled.shape[1])
n_samples = int(X_ae_scaled.shape[0])

autoencoder, encoder = build_conv1d_autoencoder(
    seq_len, AE_LATENT_DIM, AE_LEARNING_RATE, AE_RECON_LOSS
)
autoencoder.summary()
encoder.summary()

history = fit_autoencoder_model(
    autoencoder,
    X_ae_scaled,
    AE_EPOCHS,
    AE_BATCH_SIZE,
    AE_VALIDATION_SPLIT,
    AE_EARLY_STOPPING_PATIENCE,
    RANDOM_STATE,
)

# Training curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history["loss"], label="train_loss")
if "val_loss" in history.history:
    ax.plot(history.history["val_loss"], label="val_loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Autoencoder training")
ax.legend()
fig.tight_layout()
fig.savefig(CATCHUP_ROUND2_DIR / "ae_training_loss.png", dpi=160)
plt.close(fig)

X_pred = autoencoder.predict(X_ae_scaled, batch_size=min(AE_BATCH_SIZE, n_samples), verbose=0)
Z = encoder.predict(X_ae_scaled, batch_size=min(AE_BATCH_SIZE, n_samples), verbose=0)

recon_err = compute_reconstruction_errors(X_ae_scaled, X_pred)
recon_df = pd.DataFrame(
    {
        "country": meta["country"].values,
        "topic": meta["topic"].values,
        "recon_error": recon_err,
    }
)
recon_df.to_csv(CATCHUP_ROUND2_DIR / "ae_reconstruction_errors.csv", index=False)

latent_cols = [f"z{i}" for i in range(Z.shape[1])]
latent_df = pd.concat(
    [meta.reset_index(drop=True), pd.DataFrame(Z, columns=latent_cols)], axis=1
)
latent_df.to_csv(CATCHUP_ROUND2_DIR / "latent_vectors.csv", index=False)

print("AE training done. latent shape:", Z.shape)
'''

CELL10 = r'''# --- Latent hierarchical clustering + k search ---

from sklearn.cluster import AgglomerativeClustering


def cluster_size_entropy(labels: np.ndarray) -> float:
    vc = pd.Series(labels).value_counts(normalize=True)
    return float(-(vc * np.log(vc + 1e-15)).sum())


def score_labels_flat(X_flat: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Davies–Bouldin and silhouette on Euclidean space; silhouette failures -> NaN."""
    uq = np.unique(labels)
    if len(uq) < 2 or len(X_flat) < uq.max() + 2:
        return np.inf, np.nan
    try:
        sil = float(silhouette_score(X_flat, labels, metric="euclidean"))
    except Exception:
        sil = np.nan
    db = float(davies_bouldin_score(X_flat, labels))
    return db, sil


def cluster_latent_vectors(
    Z: np.ndarray, method: str, k: int, linkage: str
) -> np.ndarray:
    """Cluster rows of Z; labels in 0..k-1. method must be 'hierarchical'."""
    if method != "hierarchical":
        raise ValueError(method)
    n = Z.shape[0]
    if k < 2 or k >= n:
        raise ValueError(f"invalid k={k} for n={n}")
    kw: dict[str, Any] = {"n_clusters": int(k), "linkage": linkage}
    if linkage != "ward":
        kw["metric"] = "euclidean"
    clf = AgglomerativeClustering(**kw)
    lab = clf.fit_predict(Z.astype(np.float64))
    lab = np.asarray(lab, dtype=int)
    uq = np.unique(lab)
    remap = {old: i for i, old in enumerate(np.sort(uq))}
    return np.vectorize(remap.get)(lab)


def latent_search_all_k(
    Z: np.ndarray, k_range: range, method: str, linkage: str
) -> pd.DataFrame:
    """Metrics per k on latent vectors (same columns style as hierarchical k-search)."""
    n = Z.shape[0]
    rows: list[dict[str, Any]] = []
    X_score = Z.astype(np.float64)
    for k in k_range:
        k = int(k)
        if k < 2 or k >= n:
            rows.append({"k": k, "error": "k_out_of_range"})
            continue
        try:
            labels = cluster_latent_vectors(Z, method, k, linkage)
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
                    "latent_linkage": linkage,
                }
            )
        except Exception as e:
            rows.append({"k": k, "error": str(e)})
    return pd.DataFrame(rows)


def pick_best_k_from_metrics(
    metrics_df: pd.DataFrame, strategy: str, override: int | None, n_samples: int
) -> tuple[int, str]:
    """metrics_lex: sort by DB asc, -silhouette asc; fixed: override."""
    if strategy == "fixed":
        if override is None:
            raise ValueError("FINAL_K_STRATEGY=fixed requires FINAL_K_OVERRIDE")
        kf = int(override)
        if kf < 2 or kf >= n_samples:
            raise ValueError(f"FINAL_K_OVERRIDE out of range: {kf}")
        ok_k = set(metrics_df.loc[metrics_df["db"].notna(), "k"].astype(int))
        if kf not in ok_k:
            raise ValueError(f"k={kf} had no successful fit in metrics_df")
        return kf, "fixed"
    if strategy == "metrics_lex":
        sub = metrics_df.loc[metrics_df["db"].notna()].copy()
        if sub.empty:
            raise RuntimeError("no valid k for metrics_lex")
        sub["_neg_sil"] = -np.nan_to_num(sub["silhouette"].astype(float), nan=-1e9)
        sub = sub.sort_values(["db", "_neg_sil", "k"])
        return int(sub.iloc[0]["k"]), "metrics_lex"
    raise ValueError(strategy)


def plot_latent_k_diagnostics(
    metrics_df: pd.DataFrame, k_final: int | None, out_path: Path, title: str
) -> None:
    sub = metrics_df.loc[metrics_df["db"].notna()].sort_values("k")
    if sub.empty:
        warnings.warn("No valid k rows for latent diagnostics plot; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax0, ax1 = axes
    ax0.plot(sub["k"], sub["db"], "o-", color="C1", lw=1.2)
    ax0.set_xlabel("k")
    ax0.set_ylabel("Davies–Bouldin (latent)")
    ax0.set_title("DB index")
    sil = sub["silhouette"].astype(float)
    ax1.plot(sub["k"], sil, "o-", color="C2", lw=1.2)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette (latent)")
    ax1.set_title("Silhouette")
    if k_final is not None:
        for ax in axes:
            ax.axvline(
                int(k_final),
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


search_latent_df = latent_search_all_k(
    Z, TRAJ_K_RANGE, LATENT_CLUSTER_METHOD, LATENT_HIER_LINKAGE
)
search_latent_df.to_csv(CATCHUP_ROUND2_DIR / "latent_k_search_metrics.csv", index=False)
print("Latent k search:\n", search_latent_df.to_string())

if FINAL_N_CLUSTERS is not None:
    kf = int(FINAL_N_CLUSTERS)
    if kf < 2 or kf >= n_samples:
        raise ValueError(f"FINAL_N_CLUSTERS out of range: {kf} for n_samples={n_samples}")
    k_final, k_strategy_used = kf, "final_n_clusters_fixed"
else:
    k_final, k_strategy_used = pick_best_k_from_metrics(
        search_latent_df, FINAL_K_STRATEGY, FINAL_K_OVERRIDE, n_samples
    )
print(f"k_final={k_final} (strategy={k_strategy_used})")

plot_latent_k_diagnostics(
    search_latent_df,
    k_final,
    CATCHUP_ROUND2_DIR / "latent_k_diagnostics.png",
    f"Latent hierarchical k diagnostics ({LATENT_HIER_LINKAGE})",
)

labels_final = cluster_latent_vectors(Z, LATENT_CLUSTER_METHOD, k_final, LATENT_HIER_LINKAGE)

cluster_out = meta.reset_index(drop=True).copy()
cluster_out["cluster_id"] = labels_final
cluster_out["active_years"] = active_years_s.loc[wide_roll_raw.index].values
cluster_out.to_csv(CATCHUP_ROUND2_DIR / "trajectory_clusters_ae_latent.csv", index=False)
print("cluster sizes:\n", cluster_out["cluster_id"].value_counts().sort_index())
'''

CELL12 = r'''# --- Anomaly scores + combined table ---

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float, float]:
    """Median/MAD robust z; MAD scaled to match std of normal."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    denom = 1.4826 * mad + eps
    if denom < eps:
        denom = float(np.std(x) + eps)
    z = (x.astype(np.float64) - med) / denom
    return z, med, denom


def score_reconstruction_anomaly(
    recon_error: np.ndarray,
    rule: str,
    z_thr: float,
    q: float,
) -> pd.DataFrame:
    """recon_error: per-sample MSE; adds z and flag."""
    e = recon_error.astype(np.float64)
    rz, _, _ = robust_zscore(e)
    df = pd.DataFrame({"recon_error": e, "recon_error_z": rz})
    if rule == "robust_z":
        df["recon_anomaly_flag"] = df["recon_error_z"] > float(z_thr)
    elif rule == "quantile":
        thr = float(np.quantile(e, float(q)))
        df["recon_anomaly_flag"] = e >= thr
    else:
        raise ValueError(rule)
    return df


def score_latent_anomaly(Z: np.ndarray, method: str, knn_k: int, random_state: int) -> pd.DataFrame:
    """Latent-space anomaly scores + robust z + flag (threshold via robust_z on scores)."""
    n = Z.shape[0]
    X = Z.astype(np.float64)
    if method == "knn":
        # kneighbors includes the point itself at distance 0 as first neighbor
        kk = int(min(max(2, knn_k + 1), n))
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        score = dists[:, 1:].mean(axis=1) if dists.shape[1] >= 2 else dists[:, 0]
    elif method == "lof":
        clf = LocalOutlierFactor(n_neighbors=min(20, max(2, n // 5)), novelty=False)
        pred = clf.fit_predict(X)
        score = -clf.negative_outlier_factor_
        # higher = more abnormal for LOF neg factor inverted
    elif method == "isolation_forest":
        iso = IsolationForest(
            n_estimators=200,
            max_samples=min(256, n),
            random_state=random_state,
            contamination="auto",
        )
        s = iso.fit_predict(X)
        score = -iso.score_samples(X)
        _ = s
    else:
        raise ValueError(method)
    score = np.asarray(score, dtype=np.float64)
    z, _, _ = robust_zscore(score)
    # flag: high score in robust_z sense
    df = pd.DataFrame({"latent_anomaly_score": score, "latent_anomaly_z": z})
    df["latent_anomaly_flag"] = df["latent_anomaly_z"] > 3.0
    return df


recon_part = score_reconstruction_anomaly(
    recon_err, RECON_ANOMALY_RULE, RECON_ANOMALY_Z_THRESHOLD, RECON_ANOMALY_QUANTILE
)
latent_part = score_latent_anomaly(Z, LATENT_ANOMALY_METHOD, LATENT_KNN_K, RANDOM_STATE)

summary = meta.reset_index(drop=True).copy()
summary["cluster_id"] = labels_final
summary["active_years"] = active_years_s.loc[wide_roll_raw.index].values
summary = pd.concat([summary, recon_part, latent_part], axis=1)
summary["is_anomalous"] = summary["recon_anomaly_flag"] | summary["latent_anomaly_flag"]
summary.to_csv(CATCHUP_ROUND2_DIR / "trajectory_anomalies_ae.csv", index=False)
print("Saved trajectory_anomalies_ae.csv; n_anomalous=", int(summary["is_anomalous"].sum()))
'''

CELL14 = r'''# --- Plots: latent 2D (encoder dims 0–1), prototypes, reconstruction examples ---


def plot_trajectory_prototypes(
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


def latent_encoder_coords_2d(Z: np.ndarray) -> np.ndarray:
    """First two columns of encoder latent (same weights as full Z)."""
    if Z.shape[1] < 2:
        raise ValueError("AE_LATENT_DIM must be >= 2 for 2D latent scatter")
    return np.asarray(Z[:, :2], dtype=float)


def plot_latent_encoder2d_scatter(
    Z: np.ndarray,
    hue: np.ndarray,
    out_png: Path,
    title: str,
    label_fn,
) -> None:
    """Scatter plot on Z[:, :2] (no UMAP)."""
    emb = latent_encoder_coords_2d(Z)
    n = emb.shape[0]
    if n < 2:
        warnings.warn("样本过少，跳过 latent 2D 图")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    hue_arr = np.asarray(hue)
    for v in np.sort(np.unique(hue_arr)):
        m = hue_arr == v
        lab = label_fn(v)
        ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.72, label=f"{lab} (n={int(m.sum())})")
    ax.set_xlabel("latent dim 0")
    ax.set_ylabel("latent dim 1")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


if False:  # legacy UMAP visualization (disabled; kept for reference)

    def plot_latent_umap_colored(
        Z: np.ndarray,
        hue: np.ndarray,
        random_state: int,
        out_png: Path,
        title: str,
        label_fn,
    ) -> None:
        from umap import UMAP

        X = np.asarray(Z, dtype=float)
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
        hue_arr = np.asarray(hue)
        for v in np.sort(np.unique(hue_arr)):
            m = hue_arr == v
            lab = label_fn(v)
            ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.72, label=f"{lab} (n={int(m.sum())})")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)


plot_trajectory_prototypes(
    wide_roll_raw,
    labels_final,
    years_all,
    CATCHUP_ROUND2_DIR / "trajectory_prototypes_ae_latent.png",
    f"AE latent clusters — prototypes in ratio space (k={k_final})",
)

plot_latent_encoder2d_scatter(
    Z,
    labels_final,
    CATCHUP_ROUND2_DIR / "latent_umap_clusters.png",
    "Encoder latent dims 0–1 (by cluster_id; not UMAP)",
    lambda v: f"C{int(v)}",
)

anom_int = summary["is_anomalous"].astype(bool).values
plot_latent_encoder2d_scatter(
    Z,
    anom_int.astype(int),
    CATCHUP_ROUND2_DIR / "latent_umap_anomalies.png",
    "Encoder latent dims 0–1 (by is_anomalous; not UMAP)",
    lambda v: "normal" if int(v) == 0 else "anomalous",
)

# Reconstruction examples: normal vs anomalous
rng = np.random.default_rng(RANDOM_STATE)
n_show = 4
idx_normal = np.where(~summary["is_anomalous"].values)[0]
idx_anom = np.where(summary["is_anomalous"].values)[0]
pick_n = min(n_show, len(idx_normal))
pick_a = min(n_show, len(idx_anom))
sel = []
if pick_n:
    sel.extend(rng.choice(idx_normal, size=pick_n, replace=False).tolist())
if pick_a:
    sel.extend(rng.choice(idx_anom, size=pick_a, replace=False).tolist())
if not sel:
    sel = list(range(min(4, n_samples)))

fig, axes = plt.subplots(len(sel), 1, figsize=(9, 2.2 * len(sel)), sharex=True)
if len(sel) == 1:
    axes = [axes]
years_plot = years_all.astype(int)
for ax, i in zip(axes, sel):
    ax.plot(years_plot, X_ae_scaled[i, :, 0], label="scaled true", color="C0", lw=1.2)
    ax.plot(years_plot, X_pred[i, :, 0], label="recon", color="C3", ls="--", lw=1.0)
    tag = "ANOM" if summary["is_anomalous"].iloc[i] else "ok"
    ax.set_ylabel(f"{meta.iloc[i]['country']}|{meta.iloc[i]['topic']}|{tag}")
    ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel("Year")
fig.suptitle("AE reconstruction examples (scaled input space)", y=1.01)
fig.tight_layout()
fig.savefig(CATCHUP_ROUND2_DIR / "ae_reconstruction_examples.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("All AE exports written to", CATCHUP_ROUND2_DIR.resolve())
'''


def main() -> None:
    hier = json.loads(HIER.read_text(encoding="utf-8"))
    imp = "".join(hier["cells"][2]["source"])
    if "tensorflow" not in imp:
        imp = imp.rstrip() + "\n\n# TensorFlow / Keras (after RANDOM_STATE is set)\nos.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')\nimport tensorflow as tf\ntf.random.set_seed(RANDOM_STATE)\n"
    traj = "".join(hier["cells"][6]["source"])

    cells = [
        new_markdown_cell(MD0),
        new_markdown_cell(MD1),
        new_code_cell(imp),
        new_markdown_cell(MD3),
        new_code_cell(CELL4),
        new_markdown_cell(MD5),
        new_code_cell(traj),
        new_markdown_cell(MD7),
        new_code_cell(CELL8),
        new_markdown_cell(MD9),
        new_code_cell(CELL10),
        new_markdown_cell(MD11),
        new_code_cell(CELL12),
        new_markdown_cell(MD13),
        new_code_cell(CELL14),
        new_markdown_cell(MD15),
    ]

    md = dict(hier.get("metadata", {}))
    md["kernelspec"] = {
        "display_name": "catch-up (venv)",
        "language": "python",
        "name": "catchup-venv",
    }
    if "language_info" not in md and "language_info" in hier.get("metadata", {}):
        md["language_info"] = hier["metadata"]["language_info"]
    nb = new_notebook(cells=cells, metadata=md)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    DST.write_text(nbformat.writes(nb), encoding="utf-8")
    print("Wrote", DST)


if __name__ == "__main__":
    main()
