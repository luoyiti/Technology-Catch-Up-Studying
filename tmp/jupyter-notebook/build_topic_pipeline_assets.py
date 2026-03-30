from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path("/Users/luoyiti/Project/catch-up")
NOTEBOOK_DIR = ROOT / "output/jupyter-notebook/topic_pipeline"
EXTRACT_DIR = ROOT / "tmp/jupyter-notebook/extracted"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def read_extract(filename: str) -> str:
    return (EXTRACT_DIR / filename).read_text(encoding="utf-8")


def indent_block(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text.rstrip() + "\n", prefix)


COMMON_IMPORT = """
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PIPELINE_NOTEBOOK_DIR = Path("output/jupyter-notebook/topic_pipeline").resolve()
if str(PIPELINE_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_NOTEBOOK_DIR))
"""


NB1 = [
    md(
        """
        # 01 数据向量化

        这一册只负责数据读取、字段规整、文本拼接与文档向量缓存。

        ## 本册定位
        - 目标：把原始 SCIE 数据整理成稳定的数据表与文档向量。
        - 输入：`data/dataCleanSCIE.csv`
        - 输出：清洗后的数据表、`doc_embeddings_*.npy`、`vectorize_data` manifest
        - 风险点：若本地没有现成 embeddings，且 `RUN_SPECTER2_ENCODING=False`，则只会校验并提示如何开启编码。
        - 运行后检查：确认 manifest 中的 `clean_rows`、`embedding_dim` 与缓存路径都已生成。
        """
    ),
    code(
        COMMON_IMPORT
        + """
from topic_pipeline_common import (
    build_paths,
    doc_embeddings_path,
    encode_papers_specter2,
    load_clean_dataset,
    load_numpy,
    save_numpy,
    save_stage_manifest,
)
"""
    ),
    md(
        """
        ## 参数区

        这里统一管理本册的输入源、缓存行为和 SPECTER2 编码参数。
        """
    ),
    code(
        """
PIPELINE_NAME = "specter2_proximity"
DATA_PATH = "data/dataCleanSCIE.csv"
MIN_TEXT_LENGTH = 20
RUN_SPECTER2_ENCODING = False
FORCE_REBUILD_EMBEDDINGS = False
SPECTER2_BATCH_SIZE = 64
SPECTER2_MAX_LENGTH = 512

paths = build_paths(pipeline_name=PIPELINE_NAME, root=".")
stage_dir = paths.stage_dir("01_vectorize_data", method="shared")
"""
    ),
    md(
        """
        ## 主执行区

        先加载并清洗数据，再优先复用已有 embeddings 缓存；只有在显式打开 `RUN_SPECTER2_ENCODING` 时才重新编码。
        """
    ),
    code(
        """
df, data_summary = load_clean_dataset(
    data_path=DATA_PATH,
    min_text_length=MIN_TEXT_LENGTH,
    paths=paths,
    write_outputs=True,
)

embedding_path = doc_embeddings_path(paths)
if embedding_path.exists() and not FORCE_REBUILD_EMBEDDINGS:
    embeddings = load_numpy(embedding_path)
    embedding_source = "cache"
else:
    if not RUN_SPECTER2_ENCODING:
        raise RuntimeError(
            "未发现现成文档向量缓存。若要重建，请把 RUN_SPECTER2_ENCODING 改为 True 后重新执行本册。"
        )
    embeddings = encode_papers_specter2(
        title_list=df["title"].fillna("").tolist(),
        abstract_list=df["abstract"].fillna("").tolist(),
        batch_size=SPECTER2_BATCH_SIZE,
        max_length=SPECTER2_MAX_LENGTH,
    )
    save_numpy(embedding_path, embeddings)
    embedding_source = "fresh_encode"

vectorize_manifest_path = save_stage_manifest(
    paths=paths,
    stage="01_vectorize_data",
    method="shared",
    payload={
        "data_path": str(Path(DATA_PATH).resolve()),
        "clean_dataset_path": str((paths.shared_dir / "clean_dataset.csv").resolve()),
        "clean_dataset_summary_path": str((paths.shared_dir / "clean_dataset_summary.json").resolve()),
        "embeddings_path": str(embedding_path.resolve()),
        "embedding_source": embedding_source,
        "clean_rows": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "min_text_length": int(MIN_TEXT_LENGTH),
    },
)

print("clean rows:", len(df))
print("embeddings shape:", embeddings.shape)
print("manifest:", vectorize_manifest_path)
display(df.head(3))
"""
    ),
    md(
        """
        ## 结果校验

        建议核对：
        - `clean_rows` 是否与预期样本规模一致；
        - `embedding_dim` 是否稳定；
        - `output/embeddings/` 下是否已生成对应 `.npy` 文件。
        """
    ),
    code(
        """
manifest = json.loads(Path(vectorize_manifest_path).read_text(encoding="utf-8"))
display(pd.DataFrame([manifest]))
"""
    ),
    md(
        """
        ## 下一册读取什么

        下一册 `02_umap_reduce.ipynb` 只读取本册生成的 `01_vectorize_data__shared.json` manifest，不依赖当前内存状态。
        """
    ),
]


NB2 = [
    md(
        """
        # 02 UMAP 降维

        这一册只负责把文档向量压缩成可聚类与可视化的低维表示。

        ## 本册定位
        - 目标：统一生成 `reduced_embeddings` 和 `umap_2d_embeddings`
        - 输入：`01_vectorize_data` manifest
        - 输出：降维数组与 `umap_reduce` manifest
        - 风险点：如果环境没有安装 `umap-learn` 且本地又没有缓存，会在运行期报错
        - 运行后检查：确认 `reduced_embeddings` 与 `umap_2d_embeddings` 的 shape 正确
        """
    ),
    code(
        COMMON_IMPORT
        + """
from topic_pipeline_common import (
    build_paths,
    load_numpy,
    load_stage_manifest,
    reduced_embeddings_path,
    reduce_with_umap,
    save_numpy,
    save_stage_manifest,
    umap_2d_path,
)
"""
    ),
    md("## 参数区"),
    code(
        """
PIPELINE_NAME = "specter2_proximity"
FORCE_REBUILD_REDUCED = False
FORCE_REBUILD_UMAP_2D = False

UMAP_N_NEIGHBORS = 30
UMAP_N_COMPONENTS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

UMAP_2D_N_NEIGHBORS = 30
UMAP_2D_MIN_DIST = 0.05
UMAP_2D_METRIC = "cosine"

paths = build_paths(pipeline_name=PIPELINE_NAME, root=".")
"""
    ),
    md(
        """
        ## 主执行区

        这一册完全通过上游 manifest 定位输入文件，不依赖上一册运行时保留的变量。
        """
    ),
    code(
        """
vectorize_manifest = load_stage_manifest(paths, stage="01_vectorize_data", method="shared")
embeddings = load_numpy(vectorize_manifest["embeddings_path"])

reduced_path = reduced_embeddings_path(paths)
if reduced_path.exists() and not FORCE_REBUILD_REDUCED:
    reduced_embeddings = load_numpy(reduced_path)
    reduced_source = "cache"
else:
    reduced_embeddings = reduce_with_umap(
        embeddings,
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    save_numpy(reduced_path, reduced_embeddings)
    reduced_source = "fresh_umap"

xy_path = umap_2d_path(paths)
if xy_path.exists() and not FORCE_REBUILD_UMAP_2D:
    xy = load_numpy(xy_path)
    xy_source = "cache"
else:
    try:
        xy = reduce_with_umap(
            embeddings,
            n_neighbors=UMAP_2D_N_NEIGHBORS,
            n_components=2,
            min_dist=UMAP_2D_MIN_DIST,
            metric=UMAP_2D_METRIC,
            random_state=UMAP_RANDOM_STATE,
        )
    except Exception:
        xy = reduced_embeddings[:, :2]
    save_numpy(xy_path, xy)
    xy_source = "fresh_umap"

umap_manifest_path = save_stage_manifest(
    paths=paths,
    stage="02_umap_reduce",
    method="shared",
    payload={
        "vectorize_manifest_path": str(paths.stage_manifest_path("01_vectorize_data", "shared").resolve()),
        "embeddings_path": vectorize_manifest["embeddings_path"],
        "reduced_embeddings_path": str(reduced_path.resolve()),
        "umap_2d_path": str(xy_path.resolve()),
        "reduced_source": reduced_source,
        "umap_2d_source": xy_source,
        "reduced_shape": list(reduced_embeddings.shape),
        "umap_2d_shape": list(xy.shape),
        "umap_params": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "n_components": UMAP_N_COMPONENTS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
        },
    },
)

print("reduced shape:", reduced_embeddings.shape)
print("2d shape:", xy.shape)
print("manifest:", umap_manifest_path)
"""
    ),
    md("## 结果校验"),
    code(
        """
manifest = json.loads(Path(umap_manifest_path).read_text(encoding="utf-8"))
display(pd.DataFrame([manifest]))
"""
    ),
    md(
        """
        ## 下一册读取什么

        `03_cluster_topics.ipynb` 会读取 `02_umap_reduce__shared.json` manifest，并默认使用 `reduced_embeddings` 作为聚类输入。
        """
    ),
]


NB3 = [
    md(
        """
        # 03 聚类

        这一册统一承载 KMeans 与 Fuzzy C-Means 两条聚类分支。

        ## 本册定位
        - 目标：把聚类算法差异收敛到单一参数 `CLUSTER_METHOD`
        - 输入：`01_vectorize_data` 与 `02_umap_reduce` manifests
        - 输出：聚类结果数组、质量摘要、带聚类列的数据表、`cluster_topics` manifest
        - 主分析口径：
          - `kmeans` 使用硬标签作为主结果
          - `fuzzy` 使用 soft membership 作为主结果，同时保留 hard anchor 兼容列
        """
    ),
    code(
        COMMON_IMPORT
        + """
from topic_pipeline_common import (
    build_doc_topic_profiles,
    build_paths,
    evaluate_hard_partition,
    load_dataframe,
    load_numpy,
    load_stage_manifest,
    normalized_entropy,
    plot_kmeans_search,
    prepare_cluster_input,
    run_fuzzy_cmeans,
    run_kmeans_clustering,
    run_kmeans_search,
    safe_row_normalize,
    save_dataframe,
    save_numpy,
    save_stage_manifest,
)
"""
    ),
    md("## 参数区"),
    code(
        """
PIPELINE_NAME = "specter2_proximity"
CLUSTER_METHOD = "kmeans"  # 可选: "kmeans" | "fuzzy"

KMEANS_N_CLUSTERS = 50
RUN_K_SEARCH = True
K_SEARCH_VALUES = [20, 30, 40, 50, 60, 80]

FUZZY_N_CLUSTERS = 50
FUZZY_M = 2.0
FUZZY_MAXITER = 300
FUZZY_ERROR = 1e-5
FUZZY_RANDOM_STATE = 42
FUZZY_DOC_MIN_WEIGHT = 0.03
FUZZY_DOC_MIN_TOPICS = 4
FUZZY_DOC_MAX_TOPICS = 8
FUZZY_DOC_COVERAGE = 0.85

paths = build_paths(pipeline_name=PIPELINE_NAME, root=".")
stage_dir = paths.stage_dir("03_cluster_topics", method=CLUSTER_METHOD)
"""
    ),
    md("## 主执行区"),
    code(
        """
vectorize_manifest = load_stage_manifest(paths, stage="01_vectorize_data", method="shared")
umap_manifest = load_stage_manifest(paths, stage="02_umap_reduce", method="shared")

df = load_dataframe(vectorize_manifest["clean_dataset_path"])
cluster_input = load_numpy(umap_manifest["reduced_embeddings_path"])
cluster_input = prepare_cluster_input(cluster_input, l2_normalize=True)

artifact_payload = {
    "vectorize_manifest_path": str(paths.stage_manifest_path("01_vectorize_data", "shared").resolve()),
    "umap_manifest_path": str(paths.stage_manifest_path("02_umap_reduce", "shared").resolve()),
    "cluster_input_path": umap_manifest["reduced_embeddings_path"],
}

if CLUSTER_METHOD == "kmeans":
    if RUN_K_SEARCH and len(K_SEARCH_VALUES) > 0:
        kmeans_search_df = run_kmeans_search(cluster_input, k_values=K_SEARCH_VALUES, random_state=42)
        kmeans_search_path = save_dataframe(stage_dir / "kmeans_search_metrics.csv", kmeans_search_df)
        kmeans_search_fig_path = plot_kmeans_search(kmeans_search_df, stage_dir / "kmeans_search_metrics.png")
        artifact_payload["kmeans_search_path"] = str(kmeans_search_path.resolve())
        artifact_payload["kmeans_search_fig_path"] = str(kmeans_search_fig_path.resolve())

    result = run_kmeans_clustering(cluster_input, n_clusters=KMEANS_N_CLUSTERS, random_state=42)
    df["topic"] = result["labels"]
    df["topic_anchor"] = df["topic"].astype(int)
    quality_df = pd.DataFrame([result["quality"]])
    quality_path = save_dataframe(stage_dir / "kmeans_quality_summary.csv", quality_df)
    labels_path = save_numpy(stage_dir / "kmeans_labels.npy", result["labels"].astype(np.int32))
    centers_path = save_numpy(stage_dir / "kmeans_centers.npy", result["centers"])
    clustered_dataset_path = save_dataframe(stage_dir / "clustered_dataset.csv", df)

    cluster_manifest_path = save_stage_manifest(
        paths=paths,
        stage="03_cluster_topics",
        method="kmeans",
        payload={
            **artifact_payload,
            "n_topics": int(KMEANS_N_CLUSTERS),
            "label_column": "topic",
            "primary_label_col": "topic",
            "quality_summary_path": str(quality_path.resolve()),
            "labels_path": str(labels_path.resolve()),
            "centers_path": str(centers_path.resolve()),
            "clustered_dataset_path": str(clustered_dataset_path.resolve()),
        },
    )
    display(quality_df)
else:
    fuzzy_result = run_fuzzy_cmeans(
        x=cluster_input,
        n_clusters=FUZZY_N_CLUSTERS,
        m=FUZZY_M,
        error=FUZZY_ERROR,
        maxiter=FUZZY_MAXITER,
        random_state=FUZZY_RANDOM_STATE,
    )

    membership_matrix = safe_row_normalize(fuzzy_result["membership"]).astype(np.float32)
    fuzzy_centers = np.asarray(fuzzy_result["centers"], dtype=np.float32)
    topic_anchor = np.argmax(membership_matrix, axis=1).astype(int)

    sorted_membership = np.sort(membership_matrix, axis=1)[:, ::-1]
    top1_membership = sorted_membership[:, 0]
    top2_membership = sorted_membership[:, 1] if membership_matrix.shape[1] > 1 else np.zeros(len(df), dtype=np.float32)
    topic_top2_gap = (top1_membership - top2_membership).astype(np.float32)
    topic_entropy = normalized_entropy(membership_matrix).astype(np.float32)
    effective_topic_count = (1.0 / np.clip(np.square(membership_matrix).sum(axis=1), 1e-12, None)).astype(np.float32)

    df["topic_anchor"] = topic_anchor
    df["topic"] = df["topic_anchor"].astype(int)
    df["topic_top2_gap"] = topic_top2_gap
    df["topic_entropy"] = topic_entropy
    df["effective_topic_count"] = effective_topic_count
    df["topic_anchor_strength"] = top1_membership.astype(np.float32)

    doc_topic_weight_df, doc_topic_summary_df = build_doc_topic_profiles(
        membership=membership_matrix,
        min_weight=FUZZY_DOC_MIN_WEIGHT,
        min_topics=FUZZY_DOC_MIN_TOPICS,
        max_topics=FUZZY_DOC_MAX_TOPICS,
        coverage=FUZZY_DOC_COVERAGE,
        paper_ids=df["paper_id"].tolist() if "paper_id" in df.columns else None,
        years=df["year"].tolist() if "year" in df.columns else None,
    )

    df["n_topics_active"] = doc_topic_summary_df["n_topics_kept"].astype(int).values
    df["topic_retained_weight"] = doc_topic_summary_df["retained_weight"].astype(np.float32).values
    df["topic_multi_ids_json"] = doc_topic_summary_df["topic_ids_json"].values
    df["topic_multi_weights_json"] = doc_topic_summary_df["topic_weights_json"].values
    df["topic_multi_selected_shares_json"] = doc_topic_summary_df["topic_selected_shares_json"].values

    fuzzy_quality_summary = {
        "library": "scikit-fuzzy",
        "n_iter": int(fuzzy_result.get("n_iter", len(fuzzy_result.get("objective_trace", [])))),
        "fpc": float(fuzzy_result.get("fpc", np.mean(np.square(membership_matrix).sum(axis=1)))),
        "mean_entropy": float(np.mean(topic_entropy)),
        "mean_top2_gap": float(np.mean(topic_top2_gap)),
        "mean_effective_topic_count": float(np.mean(effective_topic_count)),
        "mean_n_topics_active": float(doc_topic_summary_df["n_topics_kept"].mean()),
        "mean_retained_weight": float(doc_topic_summary_df["retained_weight"].mean()),
        "row_sum_mean": float(membership_matrix.sum(axis=1).mean()),
        "row_sum_std": float(membership_matrix.sum(axis=1).std()),
    }
    fuzzy_anchor_metrics = evaluate_hard_partition(cluster_input, topic_anchor)
    quality_df = pd.DataFrame([
        {**{"metric_group": "fuzzy"}, **fuzzy_quality_summary},
        {**{"metric_group": "anchor"}, **fuzzy_anchor_metrics},
    ])

    quality_path = save_dataframe(stage_dir / "fuzzy_quality_summary.csv", quality_df)
    membership_path = save_numpy(stage_dir / "membership_matrix.npy", membership_matrix)
    centers_path = save_numpy(stage_dir / "fuzzy_centers.npy", fuzzy_centers)
    anchor_path = save_numpy(stage_dir / "topic_anchor.npy", topic_anchor.astype(np.int32))
    doc_topic_weight_path = save_dataframe(stage_dir / "doc_topic_weight_df_fuzzy.csv", doc_topic_weight_df)
    doc_topic_summary_path = save_dataframe(stage_dir / "doc_topic_summary_df_fuzzy.csv", doc_topic_summary_df)
    clustered_dataset_path = save_dataframe(stage_dir / "clustered_dataset.csv", df)

    cluster_manifest_path = save_stage_manifest(
        paths=paths,
        stage="03_cluster_topics",
        method="fuzzy",
        payload={
            **artifact_payload,
            "n_topics": int(FUZZY_N_CLUSTERS),
            "label_column": "topic_anchor",
            "primary_label_col": "topic_anchor",
            "quality_summary_path": str(quality_path.resolve()),
            "membership_matrix_path": str(membership_path.resolve()),
            "centers_path": str(centers_path.resolve()),
            "topic_anchor_path": str(anchor_path.resolve()),
            "doc_topic_weight_path": str(doc_topic_weight_path.resolve()),
            "doc_topic_summary_path": str(doc_topic_summary_path.resolve()),
            "clustered_dataset_path": str(clustered_dataset_path.resolve()),
        },
    )
    display(quality_df)

print("cluster manifest:", cluster_manifest_path)
display(df.head(3))
"""
    ),
    md("## 结果校验"),
    code(
        """
manifest = json.loads(Path(cluster_manifest_path).read_text(encoding="utf-8"))
display(pd.DataFrame([manifest]))
"""
    ),
    md(
        """
        ## 下一册读取什么

        `04_ctfidf_and_hierarchy.ipynb` 只通过 `03_cluster_topics__{method}.json` manifest 读取聚类结果，不依赖这一册的任何运行态变量。
        """
    ),
]


NB4_IMPORT = COMMON_IMPORT + """
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import jensenshannon, pdist
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import TheilSenRegressor

from topic_pipeline_common import (
    CountVectorizer,
    aggregate_topic_term_matrix,
    build_paths,
    build_topic_docs_df_anchor,
    build_topic_docs_df_fuzzy,
    compute_c_tf_idf,
    enrich_doc_topic_summary_with_names,
    extract_top_words_per_topic,
    load_dataframe,
    load_numpy,
    load_stage_manifest,
    prepare_cluster_input,
    safe_row_normalize,
    save_dataframe,
    save_json,
    save_sparse,
    save_stage_manifest,
)
"""

NB4 = [
    md(
        """
        # 04 c-TF-IDF 与层次结构

        这一册负责主题词提取与层次主题树构建，不再混入向量化、降维或时间演化逻辑。

        ## 本册定位
        - 目标：构造 `topic_info`、层次映射、层级命名和供下游读取的 hierarchy manifest
        - 输入：`03_cluster_topics` manifest
        - 输出：topic info CSV、层次树节点表、层级规格表、带 `topic_L*` 列的数据表、hierarchy manifest
        - 主分析口径：
          - `kmeans` 以 hard topic 为主线
          - `fuzzy` 同时保留 `anchor` 对照与 `weighted` 主版本，并保留 `method_a/method_b` 两套层次树
        """
    ),
    code(NB4_IMPORT),
    md("## 参数区"),
    code(
        """
PIPELINE_NAME = "specter2_proximity"
CLUSTER_METHOD = "kmeans"  # 可选: "kmeans" | "fuzzy"
PRIMARY_TOPIC_MODE = "weighted"

TOP_WORDS = 10
VECTORIZER_STOP_WORDS = "english"
FUZZY_ASSIGN_THRESHOLD = 0.08
FUZZY_TOPIC_TOP_DOCS = 300

HIER_DISTANCE = "cosine"
HIER_LINKAGE_METHOD = "average"
HIER_LEVEL_TARGETS = [50, 35, 20, 10, 5]
HIER_TOP_WORDS = 8
HIER_LABEL_WORDS = 4
HIER_TOPK_VIS = 30
HIER_COLOR_THRESHOLD = None
HIER_PRIMARY_METHOD = "method_a"

paths = build_paths(pipeline_name=PIPELINE_NAME, root=".")
stage_dir = paths.stage_dir("04_ctfidf_and_hierarchy", method=CLUSTER_METHOD)
OUTPUT_DIR = str(stage_dir)
SEED = 42
"""
    ),
    md(
        """
        ## 输入解析

        本册只读取上游 manifest 和对应产物。这里把 `clustered_dataset.csv` 重新加载成显式输入，避免依赖第三册的内存状态。
        """
    ),
    code(
        """
cluster_manifest = load_stage_manifest(paths, stage="03_cluster_topics", method=CLUSTER_METHOD)
df = load_dataframe(cluster_manifest["clustered_dataset_path"])

if CLUSTER_METHOD == "kmeans":
    df["topic"] = df[cluster_manifest["label_column"]].astype(int)
else:
    membership_matrix = load_numpy(cluster_manifest["membership_matrix_path"])
    fuzzy_centers = load_numpy(cluster_manifest["centers_path"])
    doc_topic_summary_df = load_dataframe(cluster_manifest["doc_topic_summary_path"])
    df["topic_anchor"] = df["topic_anchor"].astype(int)
    df["topic"] = df["topic_anchor"].astype(int)

print("cluster rows:", len(df))
display(df.head(3))
"""
    ),
    md(
        """
        ## 主题文本与 c-TF-IDF 构建

        这里统一把聚类结果转换成 topic-level 文本，再计算 c-TF-IDF。
        """
    ),
    code(
        """
vectorizer = CountVectorizer(stop_words=VECTORIZER_STOP_WORDS)

if CLUSTER_METHOD == "kmeans":
    topic_docs_df = build_topic_docs_df_anchor(
        df_in=df,
        n_topics=cluster_manifest["n_topics"],
        topic_col="topic",
        text_col="text",
    )
    X_counts = vectorizer.fit_transform(topic_docs_df["topic_text"].fillna("").astype(str))
    words = vectorizer.get_feature_names_out()
    c_tf_idf_matrix = compute_c_tf_idf(X_counts)
    topic_info_df = extract_top_words_per_topic(topic_docs_df, c_tf_idf_matrix, words, top_n=TOP_WORDS)
    topic_name_map = dict(zip(topic_info_df["topic"], topic_info_df["name"]))
    df["topic_name"] = df["topic"].map(topic_name_map)

    save_dataframe(stage_dir / "topic_docs_df_kmeans.csv", topic_docs_df)
    save_sparse(stage_dir / "topic_counts_kmeans.npz", X_counts)
    save_dataframe(stage_dir / "topic_info_df_kmeans.csv", topic_info_df)
else:
    topic_docs_df_anchor = build_topic_docs_df_anchor(
        df_in=df,
        n_topics=cluster_manifest["n_topics"],
        topic_col="topic_anchor",
        text_col="text",
    )
    topic_docs_df_fuzzy = build_topic_docs_df_fuzzy(
        df_in=df,
        membership=membership_matrix,
        threshold=FUZZY_ASSIGN_THRESHOLD,
        top_docs=FUZZY_TOPIC_TOP_DOCS,
        text_col="text",
    )

    X_counts_anchor = vectorizer.fit_transform(topic_docs_df_anchor["topic_text"].fillna("").astype(str))
    X_doc_term = vectorizer.transform(df["text"].fillna("").astype(str))
    X_counts_fuzzy = aggregate_topic_term_matrix(X_doc_term, membership_matrix)
    words = vectorizer.get_feature_names_out()

    c_tf_idf_matrix_anchor = compute_c_tf_idf(X_counts_anchor)
    weighted_c_tf_idf_matrix = compute_c_tf_idf(X_counts_fuzzy)

    topic_info_df_anchor = extract_top_words_per_topic(topic_docs_df_anchor, c_tf_idf_matrix_anchor, words, top_n=TOP_WORDS)
    topic_info_df_fuzzy = extract_top_words_per_topic(topic_docs_df_fuzzy, weighted_c_tf_idf_matrix, words, top_n=TOP_WORDS)

    topic_name_map_anchor = dict(zip(topic_info_df_anchor["topic"], topic_info_df_anchor["name"]))
    topic_name_map_fuzzy = dict(zip(topic_info_df_fuzzy["topic"], topic_info_df_fuzzy["name"]))
    doc_topic_summary_named_df = enrich_doc_topic_summary_with_names(doc_topic_summary_df, topic_name_map_fuzzy)

    df["topic_name"] = df["topic_anchor"].map(topic_name_map_fuzzy)
    df["topic_name_anchor"] = df["topic_anchor"].map(topic_name_map_anchor)
    df["topic_name_fuzzy"] = df["topic_anchor"].map(topic_name_map_fuzzy)
    df["topic_multi_names_json"] = doc_topic_summary_named_df["topic_names_json"].values
    df["topic_multi_name_weight_pairs_json"] = doc_topic_summary_named_df["topic_name_weight_pairs_json"].values

    topic_docs_df = topic_docs_df_fuzzy.copy()
    X_counts = X_counts_fuzzy
    c_tf_idf_matrix = weighted_c_tf_idf_matrix
    topic_info_df = topic_info_df_fuzzy.copy()
    topic_name_map = dict(topic_name_map_fuzzy)

    save_dataframe(stage_dir / "topic_docs_df_anchor.csv", topic_docs_df_anchor)
    save_dataframe(stage_dir / "topic_docs_df_fuzzy.csv", topic_docs_df_fuzzy)
    save_sparse(stage_dir / "topic_counts_anchor.npz", X_counts_anchor)
    save_sparse(stage_dir / "topic_counts_fuzzy.npz", X_counts_fuzzy)
    save_dataframe(stage_dir / "topic_info_df_anchor.csv", topic_info_df_anchor)
    save_dataframe(stage_dir / "topic_info_df_fuzzy.csv", topic_info_df_fuzzy)
    save_dataframe(stage_dir / "doc_topic_summary_named_df_fuzzy.csv", doc_topic_summary_named_df)

display(topic_info_df.head(10))
"""
    ),
    md(
        """
        ## 层次结构参数与 helper

        下两段代码保留原 notebook 的核心层次树实现，但放在单独册子中，并且所有输入都来自本册显式准备的对象。
        """
    ),
    code(
        """
TIME_DIR = Path(globals().get("TIME_DIR", Path(OUTPUT_DIR) / "time_evolution"))
FIGS_DIR = Path(globals().get("FIGS_DIR", TIME_DIR / "figs"))
HIER_DIR = FIGS_DIR / "hierarchical_topics"
TIME_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
HIER_DIR.mkdir(parents=True, exist_ok=True)

HIER_METHOD_SPECS = {
    "method_a": {
        "label": "topic_representation_distribution",
        "description": "直接把 fuzzy weighted topic representation 视为分布向量建树",
        "metric_label": "jensen_shannon_on_topic_representation",
        "dir_name": "method_a_topic_representation",
    },
    "method_b": {
        "label": "fuzzy_centers",
        "description": "直接把 fuzzy centers 当作 topic-topic 距离输入建树",
        "metric_label": "cosine_on_fuzzy_centers",
        "dir_name": "method_b_fuzzy_centers",
    },
}

print("hierarchy root:", HIER_DIR)
"""
    ),
    code(read_extract("cluster_fuzzy.ipynb.39.code.txt")),
    code(
        f"""
if CLUSTER_METHOD == "kmeans":
{indent_block(read_extract("cluster_kmeans.ipynb.38.code.txt"))}
else:
{indent_block(read_extract("cluster_fuzzy.ipynb.40.code.txt"))}
"""
    ),
    md("## Manifest 导出"),
    code(
        """
hierarchy_dataset_path = save_dataframe(stage_dir / "hierarchy_dataset.csv", df)

topic_name_map_paths = {}
for level_name, name_map in hier_topic_name_maps.items():
    out_path = stage_dir / f"topic_name_map_{level_name}.json"
    save_json(out_path, name_map)
    topic_name_map_paths[level_name] = str(out_path.resolve())

topic_info_paths = {}
for level_name, info_df in hier_level_topic_info.items():
    out_path = stage_dir / f"hier_topic_info_{level_name}.csv"
    save_dataframe(out_path, info_df)
    topic_info_paths[level_name] = str(out_path.resolve())

membership_by_level_paths = {}
if CLUSTER_METHOD == "fuzzy":
    for level_name, membership_level in membership_by_level.items():
        out_path = stage_dir / f"membership_{level_name}.npy"
        np.save(out_path, membership_level)
        membership_by_level_paths[level_name] = str(out_path.resolve())

hierarchy_manifest_path = save_stage_manifest(
    paths=paths,
    stage="04_ctfidf_and_hierarchy",
    method=CLUSTER_METHOD,
    payload={
        "cluster_manifest_path": str(paths.stage_manifest_path("03_cluster_topics", CLUSTER_METHOD).resolve()),
        "hierarchy_dataset_path": str(hierarchy_dataset_path.resolve()),
        "hier_node_info_path": str((HIER_DIR / "hier_node_info.csv").resolve()) if CLUSTER_METHOD == "kmeans" else str((HIER_DIR / HIER_METHOD_SPECS[HIER_PRIMARY_METHOD]["dir_name"] / "hier_node_info.csv").resolve()),
        "hier_level_specs_path": str((HIER_DIR / "hier_level_specs.csv").resolve()) if CLUSTER_METHOD == "kmeans" else str((HIER_DIR / HIER_METHOD_SPECS[HIER_PRIMARY_METHOD]["dir_name"] / "hier_level_specs.csv").resolve()),
        "topic_info_paths": topic_info_paths,
        "topic_name_map_paths": topic_name_map_paths,
        "membership_by_level_paths": membership_by_level_paths,
        "primary_method": HIER_PRIMARY_METHOD if CLUSTER_METHOD == "fuzzy" else "kmeans_single_tree",
        "hierarchy_method_comparison_path": str((HIER_DIR / "hierarchy_method_comparison.csv").resolve()) if CLUSTER_METHOD == "fuzzy" else None,
        "primary_topic_mode": PRIMARY_TOPIC_MODE,
    },
)

print("hierarchy manifest:", hierarchy_manifest_path)
display(pd.DataFrame([json.loads(Path(hierarchy_manifest_path).read_text(encoding="utf-8"))]))
"""
    ),
    md(
        """
        ## 下一册读取什么

        `05_cn_us_tech_landscape.ipynb` 会只读取本册的 hierarchy manifest，并从其中解析层级名称映射、层级数据表与 fuzzy membership。
        """
    ),
]


NB5_IMPORT = COMMON_IMPORT + """
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.distance import jensenshannon
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import TheilSenRegressor

from topic_pipeline_common import (
    build_paths,
    load_dataframe,
    load_numpy,
    load_stage_manifest,
    save_stage_manifest,
)
"""

NB5 = [
    md(
        """
        # 05 中美技术版图可视化

        这一册只负责层级主题在中美之间的份额差、滚动差与累计差分析，以及 fuzzy 专属的 soft 诊断。

        ## 本册定位
        - 目标：在统一层级结果上输出中美技术版图相关表格与图像
        - 输入：`04_ctfidf_and_hierarchy` manifest
        - 输出：各层级 annual / rolling / cumulative gap 结果、比较图、专项诊断结果、最终 manifest
        - 主分析口径：
          - `kmeans` 只运行 hard-count 主流程
          - `fuzzy` 默认运行 weighted 主流程，并保留 hard baseline 对照
        """
    ),
    code(NB5_IMPORT),
    md("## 参数区"),
    code(
        """
PIPELINE_NAME = "specter2_proximity"
CLUSTER_METHOD = "kmeans"  # 可选: "kmeans" | "fuzzy"
PRIMARY_TOPIC_MODE = "weighted"

GAP_START_YEAR = 1990
ROLLING_WINDOW = 3
ROLLING_MIN_PERIODS = None
TREND_MIN_YEARS = 6
TREND_EPS = 1e-4
LEVEL_COMPARISON_TOPK = 10
HIER_TOPK_VIS = 30
SEED = 42

paths = build_paths(pipeline_name=PIPELINE_NAME, root=".")
stage_dir = paths.stage_dir("05_cn_us_tech_landscape", method=CLUSTER_METHOD)
OUTPUT_DIR = str(stage_dir)
TIME_DIR = Path(OUTPUT_DIR) / "time_evolution"
FIGS_DIR = TIME_DIR / "figs"
TIME_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
"""
    ),
    md("## 输入解析"),
    code(
        """
cluster_manifest = load_stage_manifest(paths, stage="03_cluster_topics", method=CLUSTER_METHOD)
hierarchy_manifest = load_stage_manifest(paths, stage="04_ctfidf_and_hierarchy", method=CLUSTER_METHOD)

df = load_dataframe(hierarchy_manifest["hierarchy_dataset_path"])
hier_topic_name_maps = {
    level_name: json.loads(Path(path).read_text(encoding="utf-8"))
    for level_name, path in hierarchy_manifest["topic_name_map_paths"].items()
}

membership_by_level = {}
for level_name, path in hierarchy_manifest.get("membership_by_level_paths", {}).items():
    membership_by_level[level_name] = load_numpy(path)

display(df.head(3))
print("available levels:", list(hier_topic_name_maps.keys()))
"""
    ),
    md("## 通用 helper"),
    code(read_extract("cluster_kmeans.ipynb.40.code.txt")),
    code(read_extract("cluster_fuzzy.ipynb.43.code.txt")),
    md("## 主执行区"),
    code(
        f"""
if CLUSTER_METHOD == "kmeans":
{indent_block(read_extract("cluster_kmeans.ipynb.41.code.txt"))}
{indent_block(read_extract("cluster_kmeans.ipynb.42.code.txt"))}
{indent_block(read_extract("cluster_kmeans.ipynb.43.code.txt"))}
else:
{indent_block(read_extract("cluster_fuzzy.ipynb.45.code.txt"))}
{indent_block(read_extract("cluster_fuzzy.ipynb.46.code.txt"))}
{indent_block(read_extract("cluster_fuzzy.ipynb.48.code.txt"))}
"""
    ),
    md("## Manifest 导出"),
    code(
        """
landscape_manifest_path = save_stage_manifest(
    paths=paths,
    stage="05_cn_us_tech_landscape",
    method=CLUSTER_METHOD,
    payload={
        "hierarchy_manifest_path": str(paths.stage_manifest_path("04_ctfidf_and_hierarchy", CLUSTER_METHOD).resolve()),
        "time_output_dir": str(TIME_DIR.resolve()),
        "fig_output_dir": str(FIGS_DIR.resolve()),
        "primary_topic_mode": PRIMARY_TOPIC_MODE,
    },
)

print("landscape manifest:", landscape_manifest_path)
display(pd.DataFrame([json.loads(Path(landscape_manifest_path).read_text(encoding="utf-8"))]))
"""
    ),
    md(
        """
        ## 运行后检查

        建议检查：
        - `time_evolution/` 目录下是否生成了 annual / rolling / cumulative 结果；
        - `fuzzy` 模式下是否同时产出了 hard baseline 与 fuzzy weighted 主版本；
        - 层级比较图是否与 `hier_topic_name_maps` 中的层级数一致。
        """
    ),
]


NOTEBOOK_SPECS = {
    "01_vectorize_data.ipynb": NB1,
    "02_umap_reduce.ipynb": NB2,
    "03_cluster_topics.ipynb": NB3,
    "04_ctfidf_and_hierarchy.ipynb": NB4,
    "05_cn_us_tech_landscape.ipynb": NB5,
}


def rewrite_notebook(path: Path, cells: list[dict]) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "cells": [],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    payload["cells"] = cells
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    for filename, cells in NOTEBOOK_SPECS.items():
        rewrite_notebook(NOTEBOOK_DIR / filename, cells)
        print(f"rewrote {NOTEBOOK_DIR / filename}")


if __name__ == "__main__":
    main()
