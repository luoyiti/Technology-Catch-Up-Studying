import json
import textwrap
from pathlib import Path


def lines(text: str):
    text = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in text.split("\n")]


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


NOTEBOOK_PATH = Path("/backup/cluster_fuzzy.ipynb")
nb = json.loads(NOTEBOOK_PATH.read_text())
cells = nb["cells"]

cells[0] = markdown_cell(
    """
    # 层次主题建模（Fuzzy C-Means 主分析 + Hard Anchor 对照）

    在保留原有 embeddings、UMAP、hierarchical topics 与时间演化主线的前提下，
    这个版本把 `fuzzy membership` 作为默认分析口径，把 `hard anchor` 收缩为次要对照：

    - `membership_matrix`：主分析对象。一篇论文默认同时连接多个主题，并以权重形式参与所有下游统计。
    - `hard anchor`：仅作为兼容旧流程、对照 sanity check、以及展示“如果强行单标签会怎样”的 baseline。
    - `hierarchical topics`：不再只围绕 hard topic，而是直接对 fuzzy topic representation 建树。
    - `time evolution`：不再把论文简单记为“属于某主题”，而是按 membership 权重计算国家-年份-主题份额。

    notebook 的核心研究问题也因此从“谁占了更多主题篇数”升级为：

    1. 哪个国家在某个主题上的**权重份额**更高；
    2. 哪个国家的论文更**多主题交叉**；
    3. 哪些主题本身更像**交叉枢纽**；
    4. 父主题内部的技术结构差异，是否在 soft composition 下依然显著。
    """
)

cells[13] = markdown_cell(
    """
    ## 三、Fuzzy C-Means 聚类

    ### 本节目标

    这一部分的主目标不再是给每篇论文分配唯一 topic，而是为每篇论文生成完整的 `membership vector`：

    1. 用 `scikit-fuzzy` 生成文档到主题的 `membership matrix`。
    2. 保留 `topic_anchor = argmax(membership)`，但只把它当作兼容旧代码和对照分析的辅助变量。

    ### 主次关系

    请始终按下面的优先级理解后续结果：

    - `membership_matrix`：主版本，一篇论文可以同时属于多个主题，并且每个主题都有不同权重。
    - `doc_topic_weight_df` / `doc_topic_summary_df`：主版本的展开表，供解释和统计使用。
    - `topic_anchor`：次要的 baseline/control，仅表示“如果把该论文硬塞进唯一主题，它会落在哪一类”。
    """
)

cells[16] = markdown_cell(
    """
    ### Fuzzy 聚类参数区

    ### 参数解读

    这一段参数建议分成三组理解：

    - 聚类本体参数：控制 Fuzzy C-Means 本身怎么收敛。
    - 多主题展开参数：控制“一篇论文最终保留多少个主题摘要给人看”。
    - 主题摘要参数：控制“一个主题要吸收哪些论文来做关键词抽取”。

    其中最关键的是 `FUZZY_M`：

    - `m` 越接近 1，分配会越“硬”，更像单标签聚类；
    - `m` 越大，分配会越“软”，一篇论文会更容易同时连向多个主题。

    本 notebook 默认接受“一篇论文天然是多主题的”，因此后续分析默认使用完整 membership，而不是 argmax 标签。
    """
)

cells[20] = markdown_cell(
    """
    ### 聚类实现与结果评估

    ### 本节输出对象

    这一步会生成三类核心对象：

    1. `membership_matrix`
       每一行对应一篇论文，每一列对应一个主题，行和为 1，是后续所有 soft 统计的基础。

    2. `doc_topic_weight_df`
       long format，表示“某篇论文以多大权重连向某个主题”，适合后续做统计、筛选、可视化。

    3. `doc_topic_summary_df`
       one-row-per-doc 的多主题摘要表，便于直接查看“这篇论文主要涉及哪些主题，以及各自权重”。

    `topic_anchor` 仍会保留，但只作为对照输出，不再作为主分析口径。
    """
)

cells[24] = markdown_cell(
    """
    ## 四、c-tf-idf 主题提取

    ### 本节目标

    这一部分仍然同时保留两种主题表达，但主次关系已经明确切换：

    - `fuzzy weighted`：主版本。每篇论文会按 membership 权重同时贡献给多个主题。
    - `hard anchor`：对照版本。每篇论文只投给一个主题，用于兼容旧解释方式。

    如果你关心 fuzzy 聚类的真实主题语义，请优先看：

    - `weighted_c_tf_idf_matrix`
    - `topic_info_df_fuzzy`
    - `topic_name_map_fuzzy`

    hard anchor 版本只保留为 baseline，不作为默认结论来源。
    """
)

cells[35] = markdown_cell(
    """
    ### Fuzzy 诊断与多主题解释

    ### 本节目标

    下面这个 section 主要回答四个问题：

    - `membership_matrix` 的每一行是否真的像概率分布一样稳定；
    - 这些 soft membership 是否让文档表现出明显的“多主题性”；
    - 某些主题是否天然更像“交叉主题”；
    - `hard anchor` 与 `soft membership` 分别应该扮演什么角色。

    理解方式可以简单记成：

    - `soft membership` 负责保留真实的多主题结构；
    - `hard anchor` 负责提供对照和兼容接口；
    - 真正的研究结论默认以 fuzzy 版本为准。
    """
)

cells[37] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell H1 — Hierarchical Topic Parameters
    # ══════════════════════════════════════════════════════════════════════════════
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from matplotlib.colors import TwoSlopeNorm
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import jensenshannon, pdist
    from scipy.sparse import csr_matrix, issparse
    from sklearn.linear_model import TheilSenRegressor

    # ── Hierarchical topics 集中参数区 ──────────────────────────────────────────────
    # 这里不再只跑单一层次树，而是同时尝试两种 fuzzy topic representation：
    # 1) method_a：把 topic 的 fuzzy representation 视为“分布向量”直接建树
    # 2) method_b：直接对 fuzzy centers 做 topic-topic 距离
    HIER_DISTANCE = "cosine"
    HIER_LINKAGE_METHOD = "average"
    HIER_LEVEL_TARGETS = [50, 35, 20, 10, 5]
    HIER_TOP_WORDS = 8
    HIER_LABEL_WORDS = 4
    HIER_TOPK_VIS = 30
    HIER_COLOR_THRESHOLD = None
    HIER_PRIMARY_METHOD = "method_a"

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

    # ── 中美技术差距分析集中参数区 ────────────────────────────────────────────────
    GAP_START_YEAR = 1990
    ROLLING_WINDOW = 3
    ROLLING_MIN_PERIODS = None
    TREND_MIN_YEARS = 6
    TREND_EPS = 1e-4
    LEVEL_COMPARISON_TOPK = 10

    TIME_DIR = Path(globals().get("TIME_DIR", Path(OUTPUT_DIR) / "time_evolution"))
    FIGS_DIR = Path(globals().get("FIGS_DIR", TIME_DIR / "figs"))
    HIER_DIR = FIGS_DIR / "hierarchical_topics"
    TIME_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    HIER_DIR.mkdir(parents=True, exist_ok=True)

    if ROLLING_MIN_PERIODS is None:
        ROLLING_MIN_PERIODS = ROLLING_WINDOW

    hier_params = {
        "distance": HIER_DISTANCE,
        "linkage_method": HIER_LINKAGE_METHOD,
        "level_targets": HIER_LEVEL_TARGETS,
        "top_words": HIER_TOP_WORDS,
        "label_words": HIER_LABEL_WORDS,
        "topk_vis": HIER_TOPK_VIS,
        "color_threshold": HIER_COLOR_THRESHOLD,
        "primary_method": HIER_PRIMARY_METHOD,
        "gap_start_year": GAP_START_YEAR,
        "rolling_window": ROLLING_WINDOW,
        "rolling_min_periods": ROLLING_MIN_PERIODS,
    }

    print("✅ Hierarchical topic 参数已初始化")
    print(pd.Series(hier_params))
    print("✅ Hierarchical 方法配置：")
    display(pd.DataFrame(HIER_METHOD_SPECS).T)
    """
)

cells[39] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell H3 — Build Hierarchical Topics / Dendrogram / Level Maps
    # ══════════════════════════════════════════════════════════════════════════════
    topic_docs_df_ref = _resolve_required_object("topic_docs_df_fuzzy", fallback_names=["topic_docs_df"])
    X_counts_ref = _resolve_required_object("X_counts_fuzzy", fallback_names=["X_counts"])
    c_tf_idf_matrix_ref = _resolve_required_object("weighted_c_tf_idf_matrix", fallback_names=["c_tf_idf_matrix"])
    words_ref = _resolve_required_object("words")

    topic_name_map_base = dict(_resolve_required_object("topic_name_map_fuzzy", fallback_names=["topic_name_map", "topic_name_map_L0"]))
    if len(topic_name_map_base) == 0 and "topic_info_df_fuzzy" in globals():
        topic_name_map_base = dict(zip(topic_info_df_fuzzy["topic"], topic_info_df_fuzzy["name"]))


    def make_distributional_topic_representation(topic_score_matrix):
        \"\"\"把 topic representation 转成逐行和为 1 的分布矩阵。

        设计原因：
        1. fuzzy weighted c-TF-IDF 本身是 topic 的语义表示；
        2. 但它不是天然概率分布，因此这里先截断负值，再逐行归一化；
        3. 这样后面就可以用 Jensen-Shannon distance 来衡量 topic-topic 的语义差异。
        \"\"\"
        topic_score_matrix = np.asarray(_ensure_dense(topic_score_matrix), dtype=np.float64)
        topic_score_matrix = np.nan_to_num(topic_score_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        topic_score_matrix[topic_score_matrix < 0] = 0.0
        row_sums = topic_score_matrix.sum(axis=1, keepdims=True)
        zero_mask = (row_sums <= 0).ravel()
        if zero_mask.any():
            topic_score_matrix[zero_mask] = 1.0
            row_sums = topic_score_matrix.sum(axis=1, keepdims=True)
        return topic_score_matrix / np.clip(row_sums, 1e-12, None)


    def js_distance(u, v):
        \"\"\"对两个 topic 分布向量计算 Jensen-Shannon distance。\"\"\"
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return float(jensenshannon(u + 1e-12, v + 1e-12, base=2.0))


    def build_membership_by_level_from_map(membership_leaf, topic_to_cluster, n_level_topics):
        \"\"\"把 L0 的 soft membership 按层次映射聚合成更粗层级的 membership。

        数学含义：
        - 已知每篇论文对叶子主题的 soft membership：u_{ik}
        - 若叶子主题 k 属于粗层级 cluster g，则把该论文对 g 的权重定义为：
          u_{ig}^{(level)} = sum_{k in g} u_{ik}
        - 聚合后再做一次行归一化，保证每篇论文在该层级上的 membership 和仍为 1。
        \"\"\"
        membership_leaf = safe_row_normalize(membership_leaf)
        membership_level = np.zeros((membership_leaf.shape[0], int(n_level_topics)), dtype=np.float32)
        for leaf_topic, cluster_id in topic_to_cluster.items():
            membership_level[:, int(cluster_id)] += membership_leaf[:, int(leaf_topic)]
        return safe_row_normalize(membership_level).astype(np.float32)


    def summarize_hierarchy_method(method_name, method_result):
        \"\"\"把某个 hierarchy method 的关键结果整理成易比较的摘要表。\"\"\"
        rows = []
        for level_name, name_map in method_result["topic_name_maps"].items():
            rows.append({
                "method": method_name,
                "representation_label": method_result["representation_label"],
                "metric_label": method_result["metric_label"],
                "level": level_name,
                "n_topics": int(len(name_map)),
                "example_topics": " | ".join(list(name_map.values())[:3]),
            })
        return pd.DataFrame(rows)


    # method_a：直接对 topic representation 分布建树
    topic_representation_distribution = make_distributional_topic_representation(c_tf_idf_matrix_ref)

    # method_b：直接对 fuzzy centers 建树
    fuzzy_center_representation = np.asarray(_ensure_dense(fuzzy_centers), dtype=np.float64)
    fuzzy_center_representation = np.nan_to_num(fuzzy_center_representation, nan=0.0, posinf=0.0, neginf=0.0)

    hierarchy_results_by_method = {}
    hierarchy_method_summary_dfs = []

    for method_name, method_spec in HIER_METHOD_SPECS.items():
        method_dir = HIER_DIR / method_spec["dir_name"]
        method_dir.mkdir(parents=True, exist_ok=True)

        if method_name == "method_a":
            representation_matrix = topic_representation_distribution
            distance_metric = js_distance
        else:
            representation_matrix = fuzzy_center_representation
            distance_metric = HIER_DISTANCE

        hierarchy_bundle_method = build_hierarchical_topic_tree(
            topic_docs_df=topic_docs_df_ref,
            X_counts=X_counts_ref,
            c_tf_idf_matrix=c_tf_idf_matrix_ref,
            words=words_ref,
            base_topic_name_map=topic_name_map_base,
            distance=distance_metric,
            linkage_method=HIER_LINKAGE_METHOD,
            top_words=HIER_TOP_WORDS,
            label_words=HIER_LABEL_WORDS,
            representation_matrix=representation_matrix,
            representation_name=method_spec["label"],
        )

        hier_linkage_matrix_method = hierarchy_bundle_method["linkage_matrix"]
        hier_leaf_topic_ids_method = hierarchy_bundle_method["leaf_topic_ids"]
        hier_leafset_to_node_id_method = hierarchy_bundle_method["leafset_to_node_id"]
        hier_node_info_df_method = hierarchy_bundle_method["node_info_df"].copy()
        hier_node_info_df_method["method"] = method_name

        hier_level_counts_method = _resolve_level_cluster_counts(
            n_topics=len(hier_leaf_topic_ids_method),
            target_counts=HIER_LEVEL_TARGETS,
            max_levels=5,
        )

        hier_level_maps_method = {"L0": {int(t): int(t) for t in hier_leaf_topic_ids_method}}
        hier_topic_name_maps_method = {"L0": dict(topic_name_map_base)}
        hier_level_topic_info_method = {
            "L0": topic_info_df_fuzzy.copy() if "topic_info_df_fuzzy" in globals() else pd.DataFrame({
                "topic": hier_leaf_topic_ids_method,
                "name": [topic_name_map_base.get(int(t), f"topic_{int(t)}") for t in hier_leaf_topic_ids_method],
            })
        }
        membership_by_level_method = {"L0": safe_row_normalize(membership_matrix).astype(np.float32)}

        level_records_method = [{
            "method": method_name,
            "representation_label": method_spec["label"],
            "metric_label": method_spec["metric_label"],
            "level": "L0",
            "topic_col": "topic_L0",
            "n_clusters": len(hier_topic_name_maps_method["L0"]),
        }]

        for level_idx, n_clusters in enumerate(hier_level_counts_method[1:], start=1):
            level_name = f"L{level_idx}"
            topic_to_cluster, level_topic_info_df, level_topic_name_map = build_hierarchy_level(
                topic_docs_df=topic_docs_df_ref,
                X_counts=X_counts_ref,
                words=words_ref,
                linkage_matrix=hier_linkage_matrix_method,
                leafset_to_node_id=hier_leafset_to_node_id_method,
                n_clusters=n_clusters,
                top_words=HIER_TOP_WORDS,
                label_words=HIER_LABEL_WORDS,
            )

            hier_level_maps_method[level_name] = topic_to_cluster
            hier_topic_name_maps_method[level_name] = level_topic_name_map
            hier_level_topic_info_method[level_name] = level_topic_info_df
            membership_by_level_method[level_name] = build_membership_by_level_from_map(
                membership_leaf=membership_matrix,
                topic_to_cluster=topic_to_cluster,
                n_level_topics=level_topic_info_df["topic"].nunique(),
            )

            level_records_method.append({
                "method": method_name,
                "representation_label": method_spec["label"],
                "metric_label": method_spec["metric_label"],
                "level": level_name,
                "topic_col": f"topic_{level_name}",
                "n_clusters": int(level_topic_info_df["topic"].nunique()),
            })

        hier_level_specs_method = pd.DataFrame(level_records_method)
        dendrogram_path = method_dir / f"topic_hierarchy_dendrogram_{method_name}.png"
        plot_hierarchical_dendrogram(
            linkage_matrix=hier_linkage_matrix_method,
            leaf_topic_ids=hier_leaf_topic_ids_method,
            leaf_name_map=hier_topic_name_maps_method["L0"],
            node_info_df=hier_node_info_df_method,
            out_path=dendrogram_path,
            linkage_method=HIER_LINKAGE_METHOD,
            linkage_metric=method_spec["metric_label"],
            color_threshold=HIER_COLOR_THRESHOLD,
        )

        hier_node_info_df_method.to_csv(method_dir / "hier_node_info.csv", index=False)
        hier_level_specs_method.to_csv(method_dir / "hier_level_specs.csv", index=False)
        for level_name, info_df in hier_level_topic_info_method.items():
            info_df.to_csv(method_dir / f"hier_topic_info_{level_name}.csv", index=False)
        for level_name, membership_level in membership_by_level_method.items():
            np.save(method_dir / f"membership_{level_name}.npy", membership_level)

        hierarchy_results_by_method[method_name] = {
            "method": method_name,
            "representation_label": method_spec["label"],
            "metric_label": method_spec["metric_label"],
            "description": method_spec["description"],
            "dir": method_dir,
            "bundle": hierarchy_bundle_method,
            "level_maps": hier_level_maps_method,
            "topic_name_maps": hier_topic_name_maps_method,
            "level_topic_info": hier_level_topic_info_method,
            "membership_by_level": membership_by_level_method,
            "level_specs": hier_level_specs_method,
            "dendrogram_path": dendrogram_path,
        }
        hierarchy_method_summary_dfs.append(summarize_hierarchy_method(method_name, hierarchy_results_by_method[method_name]))

    hierarchy_method_comparison_df = pd.concat(hierarchy_method_summary_dfs, ignore_index=True)
    hierarchy_method_comparison_df.to_csv(HIER_DIR / "hierarchy_method_comparison.csv", index=False)

    primary_hierarchy = hierarchy_results_by_method[HIER_PRIMARY_METHOD]
    hierarchy_bundle = primary_hierarchy["bundle"]
    hier_level_maps = primary_hierarchy["level_maps"]
    hier_topic_name_maps = primary_hierarchy["topic_name_maps"]
    hier_level_topic_info = primary_hierarchy["level_topic_info"]
    membership_by_level = primary_hierarchy["membership_by_level"]
    hier_level_specs = primary_hierarchy["level_specs"]

    # notebook 默认使用主方法 method_a 的层级映射结果。
    for level_name, topic_map in hier_level_maps.items():
        if level_name == "L0":
            df["topic_L0"] = df["topic_anchor"].astype(int)
        else:
            df[f"topic_{level_name}"] = df["topic_anchor"].map(topic_map).astype(int)
        globals()[f"topic_name_map_{level_name}"] = hier_topic_name_maps[level_name]
        globals()[f"membership_{level_name}"] = membership_by_level[level_name]

    topic_name_map = dict(hier_topic_name_maps["L0"])
    topic_info_df = hier_level_topic_info["L0"].copy()

    globals()["hierarchy_results_by_method"] = hierarchy_results_by_method
    globals()["hierarchy_method_comparison_df"] = hierarchy_method_comparison_df
    globals()["hierarchy_bundle"] = hierarchy_bundle
    globals()["hier_level_maps"] = hier_level_maps
    globals()["hier_topic_name_maps"] = hier_topic_name_maps
    globals()["hier_level_topic_info"] = hier_level_topic_info
    globals()["membership_by_level"] = membership_by_level
    globals()["hier_level_specs"] = hier_level_specs

    print("✅ 已完成双层次树构建：method_a + method_b")
    print(f"✅ 主展示方法：{HIER_PRIMARY_METHOD}")
    display(hierarchy_method_comparison_df)
    display(hier_level_specs)
    """
)

cells[40] = markdown_cell(
    """
    ## 五、中美技术版图可视化与 fuzzy 时间演化

    ### 主分析口径

    本部分把 `fuzzy weighted` 年度份额、rolling gap、cumulative gap 作为主分析；
    `hard anchor` 版本保留为次要对照。

    对国家 `c`、年份 `t`、主题 `k`，统一使用：

    $$
    w_{c,t,k} = \\sum_{i \\in (c,t)} u_{ik}
    $$

    $$
    share_{c,t,k} =
    \\frac{\\sum_{i \\in (c,t)} u_{ik}}
    {\\sum_{i \\in (c,t)} \\sum_j u_{ij}}
    $$

    因为每篇论文的 membership 行和为 1，所以分母本质上就是该国家该年份的论文数；
    但分子不再是“属于该主题的论文篇数”，而是“投向该主题的总权重”。
    """
)

cells[42] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell T1b — Fuzzy Gap Helpers / Soft Diagnostics
    # ══════════════════════════════════════════════════════════════════════════════
    def prepare_gap_base_df_with_index(df_in, topic_cols=None, year_col="year"):
        '''整理中美时间演化分析的基础表，并保留文档索引。

        设计目标：
        1. 后续 fuzzy membership 需要按原始文档索引回查，所以这里必须保留 `doc_index`；
        2. 国家列可能叫 `country` 或 `country_code`，这里统一映射为 `country2`；
        3. `topic_cols` 在 hard baseline 中需要，但在纯 fuzzy 指标分析中可以为空。
        '''
        topic_cols = list(topic_cols or [])
        base_df = df_in.copy().reset_index(drop=True)
        if "doc_index" not in base_df.columns:
            base_df["doc_index"] = np.arange(len(base_df))

        if "country2" not in base_df.columns:
            if "country_code" in base_df.columns:
                base_df["country2"] = base_df["country_code"].map(
                    lambda x: "CN" if str(x).upper() in {"CN", "CHN", "CHINA"} else
                    ("US" if str(x).upper() in {"US", "USA", "UNITED STATES"} else "OTHER")
                )
            elif "country" in base_df.columns:
                def _map_country(x):
                    s = str(x).upper()
                    if ("CHINA" in s) or (s in {"CN", "CHN"}):
                        return "CN"
                    if ("UNITED STATES" in s) or (s in {"US", "USA"}):
                        return "US"
                    return "OTHER"
                base_df["country2"] = base_df["country"].map(_map_country)
            else:
                raise RuntimeError("没有找到国家列（country / country_code）。")

        base_df["year_int"] = pd.to_numeric(base_df[year_col], errors="coerce")
        required_cols = ["doc_index", "country2", "year_int"] + topic_cols
        base_df = base_df.dropna(subset=required_cols).copy()
        base_df["doc_index"] = base_df["doc_index"].astype(int)
        base_df["year_int"] = base_df["year_int"].astype(int)

        for col in topic_cols:
            base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
            base_df = base_df.dropna(subset=[col]).copy()
            base_df[col] = base_df[col].astype(int)

        base_df = base_df[base_df["country2"].isin(["CN", "US"])].copy()
        base_df = base_df[(base_df["year_int"] >= 1900) & (base_df["year_int"] <= 2100)].copy()
        return base_df.reset_index(drop=True)


    def run_gap_suite_with_dirs(run_fn, time_dir, fig_dir, **kwargs):
        '''临时重定向 TIME_DIR / FIGS_DIR，便于 hard / fuzzy 结果分目录保存。'''
        global TIME_DIR, FIGS_DIR
        old_time_dir = TIME_DIR
        old_fig_dir = FIGS_DIR
        TIME_DIR = Path(time_dir)
        FIGS_DIR = Path(fig_dir)
        TIME_DIR.mkdir(parents=True, exist_ok=True)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            return run_fn(**kwargs)
        finally:
            TIME_DIR = old_time_dir
            FIGS_DIR = old_fig_dir


    def build_fuzzy_topic_time_series(
        df_in,
        membership_level,
        year_col="year_int",
        country_col="country2",
        doc_index_col="doc_index",
    ):
        '''按 fuzzy membership 计算国家-年份-主题时间面板。

        数学定义：
        - 主题权重：w_{c,t,k} = sum_{i in (c,t)} u_{ik}
        - 主题份额：share_{c,t,k} = w_{c,t,k} / sum_j w_{c,t,j}

        解释重点：
        - 分子不再是“落到 topic k 的论文篇数”，而是“投向 topic k 的总权重”；
        - 因为每篇论文的 membership 行和为 1，所以分母等价于该国家该年份的论文数；
        - 这正是 soft/fuzzy 版本与 hard count 版本的根本区别。
        '''
        tmp = df_in[[doc_index_col, year_col, country_col]].copy()
        tmp = tmp[tmp[country_col].isin(["CN", "US"])].dropna().copy()
        tmp[doc_index_col] = tmp[doc_index_col].astype(int)
        tmp[year_col] = tmp[year_col].astype(int)

        membership_level = safe_row_normalize(membership_level)
        valid_mask = (tmp[doc_index_col] >= 0) & (tmp[doc_index_col] < membership_level.shape[0])
        tmp = tmp[valid_mask].copy()

        years = sorted(tmp[year_col].unique().tolist())
        n_topics = membership_level.shape[1]
        rows = []

        for year in years:
            tmp_year = tmp[tmp[year_col] == year].copy()
            mem_year = membership_level[tmp_year[doc_index_col].values]

            cn_mask = (tmp_year[country_col].values == "CN")
            us_mask = (tmp_year[country_col].values == "US")

            cn_weights = mem_year[cn_mask].sum(axis=0) if cn_mask.any() else np.zeros(n_topics, dtype=np.float64)
            us_weights = mem_year[us_mask].sum(axis=0) if us_mask.any() else np.zeros(n_topics, dtype=np.float64)
            n_cn_papers = int(cn_mask.sum())
            n_us_papers = int(us_mask.sum())

            total_cn = float(cn_weights.sum())
            total_us = float(us_weights.sum())

            for topic_id in range(n_topics):
                share_cn = float(cn_weights[topic_id] / total_cn) if total_cn > 0 else 0.0
                share_us = float(us_weights[topic_id] / total_us) if total_us > 0 else 0.0
                delta = share_cn - share_us
                rows.append({
                    "topic": int(topic_id),
                    "year": int(year),
                    "weight_CN": float(cn_weights[topic_id]),
                    "weight_US": float(us_weights[topic_id]),
                    "paper_count_CN": n_cn_papers,
                    "paper_count_US": n_us_papers,
                    "share_CN": share_cn,
                    "share_US": share_us,
                    "delta": delta,
                    "abs_delta": abs(delta),
                })

        return pd.DataFrame(rows).sort_values(["topic", "year"]).reset_index(drop=True)


    def build_fuzzy_windowed_share_gap(annual_df, window_size=3, min_periods=None):
        '''在 fuzzy annual panel 上计算 rolling gap。

        计算逻辑：
        1. 先按 topic-year 对 `weight_CN` / `weight_US` 做 rolling sum；
        2. 再在每个年份横截面内，把 rolling 权重归一化成份额；
        3. 最后比较 `share_CN - share_US`。
        '''
        if min_periods is None:
            min_periods = window_size

        tmp = annual_df.copy().sort_values(["topic", "year"]).reset_index(drop=True)
        if tmp.empty:
            return tmp

        all_topics = sorted(tmp["topic"].unique().tolist())
        all_years = sorted(tmp["year"].unique().tolist())
        full_idx = pd.MultiIndex.from_product([all_topics, all_years], names=["topic", "year"])
        tmp = (
            tmp.set_index(["topic", "year"])
            .reindex(full_idx, fill_value=0.0)
            .reset_index()
            .sort_values(["topic", "year"])
        )

        for country in ["CN", "US"]:
            tmp[f"window_weight_{country}"] = (
                tmp.groupby("topic")[f"weight_{country}"]
                .transform(lambda s: s.rolling(window=window_size, min_periods=min_periods).sum())
            )

        tmp = tmp.dropna(subset=["window_weight_CN", "window_weight_US"]).copy()
        total_cn = tmp.groupby("year")["window_weight_CN"].transform("sum")
        total_us = tmp.groupby("year")["window_weight_US"].transform("sum")

        tmp["share_CN"] = tmp["window_weight_CN"] / total_cn.replace(0, np.nan)
        tmp["share_US"] = tmp["window_weight_US"] / total_us.replace(0, np.nan)
        tmp["share_CN"] = tmp["share_CN"].fillna(0.0)
        tmp["share_US"] = tmp["share_US"].fillna(0.0)
        tmp["delta"] = tmp["share_CN"] - tmp["share_US"]
        tmp["abs_delta"] = tmp["delta"].abs()

        return tmp[[
            "topic", "year", "window_weight_CN", "window_weight_US",
            "share_CN", "share_US", "delta", "abs_delta"
        ]].reset_index(drop=True)


    def build_fuzzy_cumulative_gap(panel_df, topic_name_map, start_year=1990):
        '''在 fuzzy rolling panel 上复用累计差距逻辑。'''
        return build_cumulative_gap(panel_df, topic_name_map=topic_name_map, start_year=start_year)


    def run_fuzzy_gap_suite(
        df_in,
        level_name,
        membership_level,
        topic_name_map,
        start_year=1990,
        rolling_window=3,
        rolling_min_periods=None,
        topk_vis=30,
        trend_min_years=6,
        trend_eps=1e-4,
    ):
        '''对指定层级运行 fuzzy weighted annual / rolling / cumulative gap。

        这个函数是时间演化部分的主版本：
        - annual：看每年主题份额差
        - rolling：看平滑后的中短期差距
        - cumulative：看长期累计领先/滞后
        '''
        rolling_min_periods = rolling_window if rolling_min_periods is None else rolling_min_periods

        level_time_dir = TIME_DIR / f"level_{level_name.lower()}"
        level_fig_dir = FIGS_DIR / f"level_{level_name.lower()}"
        level_time_dir.mkdir(parents=True, exist_ok=True)
        level_fig_dir.mkdir(parents=True, exist_ok=True)

        annual_df = build_fuzzy_topic_time_series(df_in, membership_level=membership_level)
        annual_df = annual_df[annual_df["year"] >= start_year].copy()
        annual_df["topic_name"] = annual_df["topic"].map(topic_name_map)
        annual_trend_summary = classify_gap_trends(
            annual_df,
            topic_name_map=topic_name_map,
            value_col="delta",
            min_years=trend_min_years,
            eps=trend_eps,
            slope_col="slope_delta",
            start_col="delta_start",
            end_col="delta_end",
        )

        rolling_df = build_fuzzy_windowed_share_gap(
            annual_df,
            window_size=rolling_window,
            min_periods=rolling_min_periods,
        )
        rolling_df = rolling_df[rolling_df["year"] >= start_year].copy()
        rolling_df["topic_name"] = rolling_df["topic"].map(topic_name_map)
        rolling_trend_summary = classify_gap_trends(
            rolling_df,
            topic_name_map=topic_name_map,
            value_col="delta",
            min_years=trend_min_years,
            eps=trend_eps,
            slope_col="slope_delta",
            start_col="delta_start",
            end_col="delta_end",
        )

        cumulative_df = build_fuzzy_cumulative_gap(
            rolling_df,
            topic_name_map=topic_name_map,
            start_year=start_year,
        )
        cumulative_trend_summary = classify_gap_trends(
            cumulative_df,
            topic_name_map=topic_name_map,
            value_col="cum_delta",
            min_years=trend_min_years,
            eps=trend_eps,
            slope_col="slope_cum_delta",
            start_col="cum_start",
            end_col="cum_end",
        )

        annual_df.to_csv(level_time_dir / "topic_share_yearly.csv", index=False)
        annual_trend_summary.to_csv(level_time_dir / "topic_trend_summary_yearly.csv", index=False)
        rolling_df.to_csv(level_time_dir / f"topic_share_gap_roll{rolling_window}.csv", index=False)
        rolling_trend_summary.to_csv(level_time_dir / f"topic_trend_summary_roll{rolling_window}.csv", index=False)
        cumulative_df.to_csv(level_time_dir / f"topic_cumulative_gap_from_{start_year}.csv", index=False)
        cumulative_trend_summary.to_csv(level_time_dir / f"topic_cumulative_trend_summary_from_{start_year}.csv", index=False)

        annual_latest_summary = export_latest_gap_summary(
            annual_df,
            topic_name_map=topic_name_map,
            rank_col="abs_delta",
            value_cols=["weight_CN", "weight_US", "share_CN", "share_US", "delta", "abs_delta"],
            topn=20,
        )
        rolling_latest_summary = export_latest_gap_summary(
            rolling_df,
            topic_name_map=topic_name_map,
            rank_col="abs_delta",
            value_cols=["share_CN", "share_US", "delta", "abs_delta", "window_weight_CN", "window_weight_US"],
            topn=20,
        )
        cumulative_latest_summary = export_latest_gap_summary(
            cumulative_df,
            topic_name_map=topic_name_map,
            rank_col="cum_abs_delta",
            value_cols=["share_CN", "share_US", "delta", "cum_delta", "cum_abs_delta"],
            topn=20,
        )

        annual_latest_summary.to_csv(level_time_dir / "summary_latest_gap_topics_yearly.csv", index=False)
        rolling_latest_summary.to_csv(level_time_dir / f"summary_latest_gap_topics_roll{rolling_window}.csv", index=False)
        cumulative_latest_summary.to_csv(level_time_dir / f"summary_latest_cumulative_gap_topics_from_{start_year}.csv", index=False)

        title_prefix = f"{level_name} Fuzzy Weighted"
        if level_name == "L0":
            figure_paths = {
                "annual_heatmap": plot_gap_heatmap(
                    annual_df,
                    topic_name_map=topic_name_map,
                    value_col="delta",
                    rank_col="abs_delta",
                    topk=topk_vis,
                    title=f"{title_prefix} Topic-Year Share Gap Heatmap (Annual, CN − US)",
                    colorbar_label="Δ share = share_CN − share_US",
                    out_path=level_fig_dir / "annual_gap_heatmap.png",
                ),
                "annual_bar_latest": plot_gap_bar(
                    annual_df,
                    topic_name_map=topic_name_map,
                    year=int(annual_df["year"].max()) if not annual_df.empty else None,
                    topk=min(20, topk_vis),
                    title=f"{title_prefix} Latest Annual Topic Gap",
                    out_path=level_fig_dir / "annual_gap_bar_latest.png",
                    figsize=(12, max(6, min(20, topk_vis) * 0.38)),
                ),
                "rolling_heatmap": plot_gap_heatmap(
                    rolling_df,
                    topic_name_map=topic_name_map,
                    value_col="delta",
                    rank_col="abs_delta",
                    topk=topk_vis,
                    title=f"{title_prefix} Topic-Year Share Gap Heatmap (Rolling {rolling_window}Y, CN − US)",
                    colorbar_label="Δ share = share_CN − share_US",
                    out_path=level_fig_dir / f"rolling_gap_heatmap_roll{rolling_window}.png",
                ),
                "rolling_overview": plot_gap_overview(
                    rolling_df,
                    topic_name_map=topic_name_map,
                    value_col="delta",
                    rank_col="abs_delta",
                    topk=topk_vis,
                    title=f"{title_prefix} Rolling {rolling_window}Y Gap Overview",
                    ylabel="Δ share (CN − US)",
                    out_path=level_fig_dir / f"rolling_gap_overview_top_{topk_vis}.png",
                ),
                "cumulative_heatmap": plot_gap_heatmap(
                    cumulative_df,
                    topic_name_map=topic_name_map,
                    value_col="cum_delta",
                    rank_col="cum_abs_delta",
                    topk=topk_vis,
                    title=f"{title_prefix} Topic-Year Cumulative Gap Heatmap (since {start_year})",
                    colorbar_label="Cumulative Δ share",
                    out_path=level_fig_dir / f"cumulative_gap_heatmap_from_{start_year}.png",
                ),
                "cumulative_overview": plot_gap_overview(
                    cumulative_df,
                    topic_name_map=topic_name_map,
                    value_col="cum_delta",
                    rank_col="cum_abs_delta",
                    topk=topk_vis,
                    title=f"{title_prefix} Cumulative Gap Overview (since {start_year})",
                    ylabel=f"Cumulative Δ share since {start_year}",
                    out_path=level_fig_dir / f"cumulative_gap_overview_top_{topk_vis}.png",
                ),
            }
        else:
            figure_paths = {
                "annual_topic_series": export_topic_gap_timeseries_batch(
                    panel_df=annual_df,
                    trend_summary_df=annual_trend_summary,
                    topic_name_map=topic_name_map,
                    out_dir=level_fig_dir / "annual_gap_by_topic",
                    value_col="delta",
                    title_prefix=f"{title_prefix} Annual Delta",
                    ylabel="Δ share (CN − US)",
                    filename_prefix="annual_gap",
                    show=False,
                ),
                "rolling_topic_series": export_topic_gap_timeseries_batch(
                    panel_df=rolling_df,
                    trend_summary_df=rolling_trend_summary,
                    topic_name_map=topic_name_map,
                    out_dir=level_fig_dir / f"rolling_gap_by_topic_roll{rolling_window}",
                    value_col="delta",
                    title_prefix=f"{title_prefix} Rolling {rolling_window}Y Delta",
                    ylabel="Δ share (CN − US)",
                    filename_prefix=f"rolling_gap_roll{rolling_window}",
                    show=False,
                ),
                "cumulative_topic_series": export_topic_gap_timeseries_batch(
                    panel_df=cumulative_df,
                    trend_summary_df=cumulative_trend_summary,
                    topic_name_map=topic_name_map,
                    out_dir=level_fig_dir / f"cumulative_gap_by_topic_from_{start_year}",
                    value_col="cum_delta",
                    title_prefix=f"{title_prefix} Cumulative Delta",
                    ylabel=f"Cumulative Δ share (CN − US)",
                    filename_prefix=f"cumulative_gap_from_{start_year}",
                    show=False,
                ),
            }

        return {
            "mode": "fuzzy_weighted",
            "level": level_name,
            "topic_name_map": topic_name_map,
            "annual": annual_df,
            "annual_trend_summary": annual_trend_summary,
            "rolling": rolling_df,
            "rolling_trend_summary": rolling_trend_summary,
            "cumulative": cumulative_df,
            "cumulative_trend_summary": cumulative_trend_summary,
            "annual_latest_summary": annual_latest_summary,
            "rolling_latest_summary": rolling_latest_summary,
            "cumulative_latest_summary": cumulative_latest_summary,
            "figure_paths": figure_paths,
            "time_dir": level_time_dir,
            "fig_dir": level_fig_dir,
            "n_topics": int(len(topic_name_map)),
        }


    def build_country_multitopic_yearly_metrics(df_in, start_year=1990):
        '''按国家、按年份汇总论文层面的多主题性指标。

        指标解释：
        - entropy 均值：论文主题分布越均匀，说明越跨主题；
        - effective_topic_count 均值：论文“有效连接了多少主题”；
        - top1_top2_gap 均值：越小表示主题边界越模糊、越多主题交叉。
        '''
        tmp = df_in[(df_in["country2"].isin(["CN", "US"])) & (df_in["year_int"] >= start_year)].copy()
        out = (
            tmp.groupby(["country2", "year_int"])
            .agg(
                n_papers=("doc_index", "size"),
                mean_topic_entropy=("topic_entropy", "mean"),
                mean_effective_topic_count=("effective_topic_count", "mean"),
                mean_top1_top2_gap=("topic_top2_gap", "mean"),
            )
            .reset_index()
            .rename(columns={"year_int": "year"})
            .sort_values(["country2", "year"])
            .reset_index(drop=True)
        )
        return out


    def plot_country_multitopic_yearly_metrics(yearly_df, out_dir):
        '''把国家层面的多主题性指标画成年度折线图。'''
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = [
            ("mean_topic_entropy", "Mean Entropy"),
            ("mean_effective_topic_count", "Mean Effective Topic Count"),
            ("mean_top1_top2_gap", "Mean Top1-Top2 Gap"),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.8))
        colors = {"CN": "#E03C31", "US": "#3C6FE0"}

        for ax, (metric, title) in zip(axes, metrics):
            for country in ["CN", "US"]:
                sub = yearly_df[yearly_df["country2"] == country]
                if sub.empty:
                    continue
                ax.plot(sub["year"], sub[metric], marker="o", linewidth=1.8, label=country, color=colors[country])
            ax.set_title(title)
            ax.set_xlabel("Year")
            ax.grid(True, alpha=0.25)
        axes[0].legend()
        plt.tight_layout()
        out_path = out_dir / "country_multitopic_yearly_metrics.png"
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return out_path


    def build_topic_boundary_metrics(df_in, membership_level, topic_name_map):
        '''衡量每个 topic 是否更像“交叉枢纽主题”。

        核心思想：
        - 若某个 topic 吸收进来的论文本身 entropy 很高，
          那么这个 topic 往往不是一个高度封闭的单一主题，而更像交叉汇聚点。
        - 因此这里用 membership 作为权重，对论文层面的 entropy / effective_topic_count / gap 做加权平均。
        '''
        tmp = df_in[["doc_index", "topic_entropy", "effective_topic_count", "topic_top2_gap"]].copy()
        tmp["doc_index"] = tmp["doc_index"].astype(int)
        idx = tmp["doc_index"].values
        mem = safe_row_normalize(membership_level)[idx]

        rows = []
        for topic_id in range(mem.shape[1]):
            topic_weight = mem[:, topic_id]
            denom = float(topic_weight.sum())
            if denom <= 0:
                continue
            rows.append({
                "topic": int(topic_id),
                "topic_name": topic_name_map.get(int(topic_id), f"topic_{int(topic_id)}"),
                "topic_weight_total": denom,
                "boundary_entropy_mean": float(np.average(tmp["topic_entropy"], weights=topic_weight)),
                "boundary_effective_topic_count_mean": float(np.average(tmp["effective_topic_count"], weights=topic_weight)),
                "boundary_top1_top2_gap_mean": float(np.average(tmp["topic_top2_gap"], weights=topic_weight)),
            })
        return pd.DataFrame(rows).sort_values(
            ["boundary_entropy_mean", "boundary_effective_topic_count_mean"],
            ascending=[False, False],
        ).reset_index(drop=True)


    def build_topic_country_cross_metrics(df_in, membership_level, topic_name_map):
        '''比较同一 topic 内部，CN 与 US 的论文谁更专门化、谁更交叉融合。

        对每个 topic、每个国家，使用该国家论文投向该 topic 的 membership 作为权重，
        再去加权论文层面的 entropy / effective_topic_count / top1-top2 gap。
        额外加入 `other_topic_share_mean`，表示“进入该 topic 的论文仍有多少权重投向了其他主题”。
        '''
        tmp = df_in[["doc_index", "country2", "topic_entropy", "effective_topic_count", "topic_top2_gap"]].copy()
        tmp["doc_index"] = tmp["doc_index"].astype(int)
        idx = tmp["doc_index"].values
        mem = safe_row_normalize(membership_level)[idx]

        rows = []
        for topic_id in range(mem.shape[1]):
            topic_weight_all = mem[:, topic_id]
            for country in ["CN", "US"]:
                mask = (tmp["country2"].values == country)
                if not mask.any():
                    continue
                w = topic_weight_all[mask]
                denom = float(w.sum())
                if denom <= 0:
                    continue
                rows.append({
                    "topic": int(topic_id),
                    "topic_name": topic_name_map.get(int(topic_id), f"topic_{int(topic_id)}"),
                    "country2": country,
                    "topic_weight_total": denom,
                    "entropy_mean": float(np.average(tmp.loc[mask, "topic_entropy"], weights=w)),
                    "effective_topic_count_mean": float(np.average(tmp.loc[mask, "effective_topic_count"], weights=w)),
                    "top1_top2_gap_mean": float(np.average(tmp.loc[mask, "topic_top2_gap"], weights=w)),
                    "other_topic_share_mean": float(np.average(1.0 - mem[mask, topic_id], weights=w)),
                })

        panel_df = pd.DataFrame(rows)
        diff_df = (
            panel_df.pivot_table(index=["topic", "topic_name"], columns="country2",
                                 values=["entropy_mean", "effective_topic_count_mean", "top1_top2_gap_mean", "other_topic_share_mean"])
            .reset_index()
        )
        diff_df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            if isinstance(col, tuple) else str(col)
            for col in diff_df.columns
        ]
        for metric in ["entropy_mean", "effective_topic_count_mean", "top1_top2_gap_mean", "other_topic_share_mean"]:
            cn_col = f"{metric}_CN"
            us_col = f"{metric}_US"
            if cn_col in diff_df.columns and us_col in diff_df.columns:
                diff_df[f"{metric}_CN_minus_US"] = diff_df[cn_col] - diff_df[us_col]

        diff_df = diff_df.sort_values("entropy_mean_CN_minus_US", ascending=False).reset_index(drop=True)
        return panel_df, diff_df


    def plot_topic_boundary_metrics(boundary_df, out_dir, topk=15):
        '''绘制 topic 边界性排名图。'''
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_df = boundary_df.head(min(topk, len(boundary_df))).iloc[::-1].copy()
        fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.38)))
        ax.barh(plot_df["topic_name"], plot_df["boundary_entropy_mean"], color="#E08A2E", alpha=0.85)
        ax.set_title("Top Boundary Topics by Membership-Weighted Entropy")
        ax.set_xlabel("Membership-weighted mean entropy")
        ax.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        out_path = out_dir / "topic_boundary_entropy_topk.png"
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return out_path


    def plot_topic_country_cross_metrics(diff_df, out_dir, topk=15):
        '''绘制 CN-US 在 topic 内部交叉性的差异排名。'''
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_df = diff_df.nlargest(min(topk, len(diff_df)), "entropy_mean_CN_minus_US").iloc[::-1].copy()
        fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.38)))
        ax.barh(plot_df["topic_name"], plot_df["entropy_mean_CN_minus_US"], color="#8A5BD8", alpha=0.85)
        ax.set_title("CN - US Cross-Topicality Gap within Topic")
        ax.set_xlabel("Entropy mean difference (CN - US)")
        ax.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        out_path = out_dir / "topic_country_entropy_gap_topk.png"
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return out_path
    """
)

# insert markdown cell before time evolution runner
cells.insert(43, markdown_cell(
    """
    ### 五点一、fuzzy 年度份额与多层级时间演化

    这一部分会同时输出：

    - `fuzzy weighted` 的 annual / rolling / cumulative gap：主版本；
    - `hard anchor` 的 annual / rolling / cumulative gap：对照版本。

    默认 notebook 中的主结果对象将指向 fuzzy 版本。
    """
))

cells[44] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell T2 — Run Annual / Rolling / Cumulative Gap Analysis for L0~L4
    # ══════════════════════════════════════════════════════════════════════════════
    level_topic_cols = [
        f"topic_L{i}"
        for i in range(5)
        if f"topic_L{i}" in df.columns and f"L{i}" in hier_topic_name_maps
    ]

    te_df = prepare_gap_base_df_with_index(df, topic_cols=level_topic_cols, year_col="year")
    print(f"✅ 时间演化分析样本数: {len(te_df):,}")
    print(f"   年份范围: {te_df['year_int'].min()} - {te_df['year_int'].max()}")
    print(f"   CN: {(te_df['country2'] == 'CN').sum():,} | US: {(te_df['country2'] == 'US').sum():,}")
    print(f"   可用层级: {level_topic_cols}")

    TIME_ROOT = Path(OUTPUT_DIR) / "time_evolution"
    TIME_DIR_HARD = TIME_ROOT / "hard_anchor_baseline"
    FIGS_DIR_HARD = TIME_DIR_HARD / "figs"
    TIME_DIR_FUZZY = TIME_ROOT / "fuzzy_weighted_main"
    FIGS_DIR_FUZZY = TIME_DIR_FUZZY / "figs"

    gap_results_by_level_hard = {}
    gap_results_by_level_fuzzy = {}

    for idx in range(5):
        level_name = f"L{idx}"
        topic_col = f"topic_{level_name}"
        if topic_col not in te_df.columns or level_name not in hier_topic_name_maps:
            continue

        print()
        print("=" * 88)
        print(f"开始运行 {level_name} | {topic_col} | 主题数={len(hier_topic_name_maps[level_name])}")
        print("=" * 88)

        gap_results_by_level_hard[level_name] = run_gap_suite_with_dirs(
            run_fn=run_gap_suite,
            time_dir=TIME_DIR_HARD,
            fig_dir=FIGS_DIR_HARD,
            df_in=te_df,
            level_name=level_name,
            topic_col=topic_col,
            topic_name_map=hier_topic_name_maps[level_name],
            start_year=GAP_START_YEAR,
            rolling_window=ROLLING_WINDOW,
            rolling_min_periods=ROLLING_MIN_PERIODS,
            topk_vis=HIER_TOPK_VIS,
            trend_min_years=TREND_MIN_YEARS,
            trend_eps=TREND_EPS,
        )

        gap_results_by_level_fuzzy[level_name] = run_gap_suite_with_dirs(
            run_fn=run_fuzzy_gap_suite,
            time_dir=TIME_DIR_FUZZY,
            fig_dir=FIGS_DIR_FUZZY,
            df_in=te_df,
            level_name=level_name,
            membership_level=membership_by_level[level_name],
            topic_name_map=hier_topic_name_maps[level_name],
            start_year=GAP_START_YEAR,
            rolling_window=ROLLING_WINDOW,
            rolling_min_periods=ROLLING_MIN_PERIODS,
            topk_vis=HIER_TOPK_VIS,
            trend_min_years=TREND_MIN_YEARS,
            trend_eps=TREND_EPS,
        )

    # notebook 默认主版本切到 fuzzy，而不是 hard。
    gap_results_by_level = gap_results_by_level_fuzzy

    available_gap_levels_hard = list(gap_results_by_level_hard.keys())
    available_gap_levels_fuzzy = list(gap_results_by_level_fuzzy.keys())

    gap_level_inventory_hard = pd.DataFrame({
        "mode": "hard_anchor_baseline",
        "level": available_gap_levels_hard,
        "topic_col": [gap_results_by_level_hard[level]["topic_col"] for level in available_gap_levels_hard],
        "n_topics": [gap_results_by_level_hard[level]["n_topics"] for level in available_gap_levels_hard],
        "time_dir": [str(gap_results_by_level_hard[level]["time_dir"]) for level in available_gap_levels_hard],
        "fig_dir": [str(gap_results_by_level_hard[level]["fig_dir"]) for level in available_gap_levels_hard],
    })
    gap_level_inventory_fuzzy = pd.DataFrame({
        "mode": "fuzzy_weighted_main",
        "level": available_gap_levels_fuzzy,
        "topic_col": [f"membership_{level}" for level in available_gap_levels_fuzzy],
        "n_topics": [gap_results_by_level_fuzzy[level]["n_topics"] for level in available_gap_levels_fuzzy],
        "time_dir": [str(gap_results_by_level_fuzzy[level]["time_dir"]) for level in available_gap_levels_fuzzy],
        "fig_dir": [str(gap_results_by_level_fuzzy[level]["fig_dir"]) for level in available_gap_levels_fuzzy],
    })

    TIME_ROOT.mkdir(parents=True, exist_ok=True)
    gap_level_inventory_hard.to_csv(TIME_ROOT / "gap_level_inventory_hard.csv", index=False)
    gap_level_inventory_fuzzy.to_csv(TIME_ROOT / "gap_level_inventory_fuzzy.csv", index=False)

    for mode_name, result_dict in [("hard", gap_results_by_level_hard), ("fuzzy", gap_results_by_level_fuzzy)]:
        for level_name, result in result_dict.items():
            result["annual"].to_csv(TIME_ROOT / f"annual_gap_{mode_name}_{level_name}.csv", index=False)
            result["rolling"].to_csv(TIME_ROOT / f"rolling_gap_{mode_name}_{level_name}.csv", index=False)
            result["cumulative"].to_csv(TIME_ROOT / f"cumulative_gap_{mode_name}_{level_name}.csv", index=False)

    # ── 新增 fuzzy 时间指标 1：国家层面的平均多主题性 ─────────────────────────────
    fuzzy_diag_root = TIME_ROOT / "fuzzy_multitopic_diagnostics"
    country_multitopic_dir = fuzzy_diag_root / "country_multitopic"
    topic_boundary_dir = fuzzy_diag_root / "topic_boundary"
    topic_country_dir = fuzzy_diag_root / "topic_country_cross"
    for p in [country_multitopic_dir, topic_boundary_dir, topic_country_dir]:
        p.mkdir(parents=True, exist_ok=True)

    country_multitopic_yearly_df = build_country_multitopic_yearly_metrics(te_df, start_year=GAP_START_YEAR)
    country_multitopic_yearly_df.to_csv(country_multitopic_dir / "country_multitopic_yearly_metrics.csv", index=False)

    latest_country_multitopic_df = (
        country_multitopic_yearly_df
        .sort_values("year")
        .groupby("country2")
        .tail(1)
        .sort_values("country2")
        .reset_index(drop=True)
    )
    latest_country_multitopic_df.to_csv(country_multitopic_dir / "country_multitopic_latest_year.csv", index=False)
    plot_country_multitopic_yearly_metrics(country_multitopic_yearly_df, country_multitopic_dir)

    # ── 新增 fuzzy 时间指标 2：主题层面的边界性 ─────────────────────────────────
    topic_boundary_df = build_topic_boundary_metrics(
        te_df,
        membership_level=membership_by_level["L0"],
        topic_name_map=hier_topic_name_maps["L0"],
    )
    topic_boundary_df.to_csv(topic_boundary_dir / "topic_boundary_metrics.csv", index=False)
    plot_topic_boundary_metrics(topic_boundary_df, topic_boundary_dir, topk=15)

    # ── 新增 fuzzy 时间指标 3：国家-主题交叉性差异 ───────────────────────────────
    topic_country_cross_panel_df, topic_country_cross_diff_df = build_topic_country_cross_metrics(
        te_df,
        membership_level=membership_by_level["L0"],
        topic_name_map=hier_topic_name_maps["L0"],
    )
    topic_country_cross_panel_df.to_csv(topic_country_dir / "topic_country_cross_panel.csv", index=False)
    topic_country_cross_diff_df.to_csv(topic_country_dir / "topic_country_cross_diff.csv", index=False)
    plot_topic_country_cross_metrics(topic_country_cross_diff_df, topic_country_dir, topk=15)

    globals()["TIME_ROOT"] = TIME_ROOT
    globals()["te_df"] = te_df
    globals()["country_multitopic_yearly_df"] = country_multitopic_yearly_df
    globals()["latest_country_multitopic_df"] = latest_country_multitopic_df
    globals()["topic_boundary_df"] = topic_boundary_df
    globals()["topic_country_cross_panel_df"] = topic_country_cross_panel_df
    globals()["topic_country_cross_diff_df"] = topic_country_cross_diff_df

    print()
    print("✅ 多层级 annual / rolling / cumulative gap 分析已完成")
    print("✅ notebook 默认主结果已切换到 fuzzy_weighted_main")
    display(gap_level_inventory_hard)
    display(gap_level_inventory_fuzzy)
    print("✅ 国家层面的平均多主题性（最新年份）")
    display(latest_country_multitopic_df)
    print("✅ 主题层面的边界性 Top 10")
    display(topic_boundary_df.head(10))
    print("✅ 国家-主题交叉性差异 Top 10")
    display(topic_country_cross_diff_df.head(10))
    """
)

cells[45] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell T3 — Cross-Level Gap Comparison Summary
    # ══════════════════════════════════════════════════════════════════════════════
    def summarize_gap_levels(gap_results_by_level, topk=10):
        '''汇总比较不同层级下的中美技术差距强度。'''
        rows = []
        for level_name, result in gap_results_by_level.items():
            rolling_df = result["rolling"].copy()
            cumulative_df = result["cumulative"].copy()
            latest_year = int(rolling_df["year"].max()) if not rolling_df.empty else np.nan
            latest_df = rolling_df[rolling_df["year"] == latest_year].copy() if pd.notna(latest_year) else rolling_df.head(0)

            topic_gap_mean = (
                rolling_df.groupby("topic")["abs_delta"].mean().sort_values(ascending=False)
                if not rolling_df.empty else pd.Series(dtype=float)
            )
            topk_mean_abs_gap = float(topic_gap_mean.head(min(topk, len(topic_gap_mean))).mean()) if len(topic_gap_mean) else np.nan

            rows.append({
                "level": level_name,
                "n_topics": int(result["n_topics"]),
                "rolling_mean_abs_gap": float(rolling_df["abs_delta"].mean()) if not rolling_df.empty else np.nan,
                "rolling_max_abs_gap": float(rolling_df["abs_delta"].max()) if not rolling_df.empty else np.nan,
                "rolling_topk_mean_abs_gap": topk_mean_abs_gap,
                "latest_year": int(latest_year) if pd.notna(latest_year) else np.nan,
                "latest_mean_abs_gap": float(latest_df["abs_delta"].mean()) if not latest_df.empty else np.nan,
                "latest_max_abs_gap": float(latest_df["abs_delta"].max()) if not latest_df.empty else np.nan,
                "cumulative_max_abs_gap": float(cumulative_df["cum_abs_delta"].max()) if not cumulative_df.empty else np.nan,
            })

        out = pd.DataFrame(rows)
        level_order = {f"L{i}": i for i in range(10)}
        out["level_order"] = out["level"].map(level_order)
        out = out.sort_values("level_order").drop(columns="level_order").reset_index(drop=True)
        return out


    def plot_level_comparison(level_summary_df, out_path, title_prefix=""):
        '''绘制不同层级技术差距比较图。'''
        metrics = [
            ("n_topics", "# Topics"),
            ("rolling_mean_abs_gap", "Mean |Gap|"),
            ("rolling_max_abs_gap", "Max |Gap|"),
            ("rolling_topk_mean_abs_gap", f"Top-{LEVEL_COMPARISON_TOPK} Mean |Gap|"),
        ]

        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.8))
        for ax, (metric, title) in zip(axes, metrics):
            ax.bar(level_summary_df["level"], level_summary_df[metric], color="#3C6FE0", alpha=0.85)
            ax.set_title(f"{title_prefix}{title}")
            ax.set_xlabel("Level")
            ax.grid(True, axis="y", alpha=0.25)
            for row in level_summary_df.itertuples(index=False):
                value = getattr(row, metric)
                if pd.notna(value):
                    label = f"{int(value)}" if metric == "n_topics" else f"{value:.3g}"
                    ax.text(row.level, value, label, ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ 已保存跨层级比较图: {out_path}")


    level_gap_comparison_hard_df = summarize_gap_levels(
        gap_results_by_level=gap_results_by_level_hard,
        topk=LEVEL_COMPARISON_TOPK,
    )
    level_gap_comparison_fuzzy_df = summarize_gap_levels(
        gap_results_by_level=gap_results_by_level_fuzzy,
        topk=LEVEL_COMPARISON_TOPK,
    )

    level_gap_comparison_hard_df.to_csv(TIME_ROOT / "hier_level_gap_comparison_hard.csv", index=False)
    level_gap_comparison_fuzzy_df.to_csv(TIME_ROOT / "hier_level_gap_comparison_fuzzy.csv", index=False)

    plot_level_comparison(
        level_gap_comparison_hard_df,
        out_path=TIME_ROOT / "hier_level_gap_comparison_hard.png",
        title_prefix="Hard Anchor Baseline ",
    )
    plot_level_comparison(
        level_gap_comparison_fuzzy_df,
        out_path=TIME_ROOT / "hier_level_gap_comparison_fuzzy.png",
        title_prefix="Fuzzy Weighted Main ",
    )

    comparison_merge_df = level_gap_comparison_hard_df.merge(
        level_gap_comparison_fuzzy_df,
        on="level",
        how="outer",
        suffixes=("_hard", "_fuzzy"),
    )
    comparison_merge_df.to_csv(TIME_ROOT / "hier_level_gap_comparison_hard_vs_fuzzy.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    compare_metrics = [
        ("rolling_mean_abs_gap", "Mean |Gap|"),
        ("rolling_max_abs_gap", "Max |Gap|"),
        ("cumulative_max_abs_gap", "Max Cumulative |Gap|"),
    ]
    x = np.arange(len(comparison_merge_df))
    width = 0.36
    for ax, (metric, title) in zip(axes, compare_metrics):
        ax.bar(x - width / 2, comparison_merge_df[f"{metric}_hard"], width=width, label="Hard Anchor", color="#3C6FE0", alpha=0.82)
        ax.bar(x + width / 2, comparison_merge_df[f"{metric}_fuzzy"], width=width, label="Fuzzy Weighted", color="#E03C31", alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_merge_df["level"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(TIME_ROOT / "hier_level_gap_comparison_hard_vs_fuzzy.png", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print()
    print("✅ Hard / Fuzzy 跨层级比较汇总表")
    display(level_gap_comparison_hard_df)
    display(level_gap_comparison_fuzzy_df)
    """
)

# insert markdown cell before parent structure
cells.insert(46, markdown_cell(
    """
    ### 五点二、父主题内部结构：soft composition 主分析 + hard child count 对照

    这里的核心问题不再是“某个父主题下面硬分了多少篇 child topic”，
    而是：

    - 某篇论文与父主题 `P` 的关系越强，它对 `P` 内部结构的贡献越大；
    - 同时该论文又可以按自身对各 child topic 的 membership 继续拆分。

    因此对父主题 `P`、国家 `country`、子主题 `k`，采用：

    $$
    p_k^{(country|P)} =
    \\frac{\\sum_i u_{iP} \\cdot u_{ik}}{\\sum_i u_{iP}}
    $$

    hard child count 版本会保留，但仅作为 baseline。
    """
))

cells[47] = code_cell(
    """
    # ══════════════════════════════════════════════════════════════════════════════
    # Cell T4 — Parent Internal Structure: Soft Composition Main + Hard Baseline
    # ══════════════════════════════════════════════════════════════════════════════
    def cosine_distance_safe(p, q):
        '''计算两个分布向量的 cosine distance。'''
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        denom = np.linalg.norm(p) * np.linalg.norm(q)
        if denom == 0:
            return 0.0
        return float(1.0 - np.dot(p, q) / denom)


    def analyze_parent_soft_composition(
        df_in,
        parent_membership,
        child_membership,
        parent_name_map,
        start_year=1990,
    ):
        '''用 soft composition 比较父主题内部的子主题结构差异。

        数学定义：
        对某个父主题 P、国家 country、子主题 k，
        p_k^{(country|P)} = sum_i u_{iP} * u_{ik} / sum_i u_{iP}

        解释：
        - 一篇论文如果与父主题 P 的关系更强，它对 P 内部结构的贡献更大；
        - 同一篇论文再按其对子主题 k 的 membership 继续分配；
        - 这比 hard child count 更贴近 fuzzy 主题结构。
        '''
        tmp = df_in[(df_in["year_int"] >= start_year) & (df_in["country2"].isin(["CN", "US"]))].copy()
        tmp["doc_index"] = tmp["doc_index"].astype(int)
        doc_idx = tmp["doc_index"].values

        parent_membership = safe_row_normalize(parent_membership)
        child_membership = safe_row_normalize(child_membership)
        parent_mem = parent_membership[doc_idx]
        child_mem = child_membership[doc_idx]

        rows = []
        composition_rows = []
        total_parent_weight_cn = parent_mem[tmp["country2"].values == "CN"].sum()
        total_parent_weight_us = parent_mem[tmp["country2"].values == "US"].sum()

        for parent_topic in range(parent_mem.shape[1]):
            parent_weight_all = parent_mem[:, parent_topic]
            parent_weight_cn = parent_weight_all[tmp["country2"].values == "CN"].sum()
            parent_weight_us = parent_weight_all[tmp["country2"].values == "US"].sum()
            if float(parent_weight_cn + parent_weight_us) <= 0:
                continue

            country_vectors = {}
            for country in ["CN", "US"]:
                country_mask = (tmp["country2"].values == country)
                w_parent = parent_weight_all[country_mask]
                denom = float(w_parent.sum())
                if denom <= 0:
                    comp = np.zeros(child_mem.shape[1], dtype=np.float64)
                else:
                    comp = ((w_parent[:, None] * child_mem[country_mask]).sum(axis=0) / denom).astype(np.float64)
                country_vectors[country] = comp

                for child_topic, value in enumerate(comp):
                    composition_rows.append({
                        "parent_topic": int(parent_topic),
                        "parent_name": parent_name_map.get(int(parent_topic), f"topic_{int(parent_topic)}"),
                        "country2": country,
                        "child_topic": int(child_topic),
                        "soft_share": float(value),
                    })

            p = country_vectors["CN"]
            q = country_vectors["US"]
            js_div = float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0) ** 2)
            l1_dist = float(np.abs(p - q).sum())
            cos_dist = cosine_distance_safe(p, q)
            child_gap = np.abs(p - q)
            top3_child_gap_mean = float(np.sort(child_gap)[::-1][: min(3, len(child_gap))].mean()) if len(child_gap) else np.nan

            rows.append({
                "parent_topic": int(parent_topic),
                "parent_name": parent_name_map.get(int(parent_topic), f"topic_{int(parent_topic)}"),
                "parent_weight_CN": float(parent_weight_cn),
                "parent_weight_US": float(parent_weight_us),
                "parent_share_CN": float(parent_weight_cn / total_parent_weight_cn) if total_parent_weight_cn > 0 else 0.0,
                "parent_share_US": float(parent_weight_us / total_parent_weight_us) if total_parent_weight_us > 0 else 0.0,
                "parent_delta": float((parent_weight_cn / total_parent_weight_cn) - (parent_weight_us / total_parent_weight_us)) if total_parent_weight_cn > 0 and total_parent_weight_us > 0 else 0.0,
                "parent_abs_delta": float(abs((parent_weight_cn / total_parent_weight_cn) - (parent_weight_us / total_parent_weight_us))) if total_parent_weight_cn > 0 and total_parent_weight_us > 0 else 0.0,
                "js_divergence": js_div,
                "l1_distance": l1_dist,
                "cosine_distance": cos_dist,
                "top3_child_gap_mean": top3_child_gap_mean,
            })

        return (
            pd.DataFrame(rows).sort_values(["js_divergence", "parent_abs_delta"], ascending=[False, False]).reset_index(drop=True),
            pd.DataFrame(composition_rows),
        )


    def analyze_parent_internal_structure_hard(
        df_in,
        parent_col,
        child_col,
        parent_name_map,
        start_year=1990,
    ):
        '''保留 hard child count 版本作为 baseline 对照。'''
        tmp = df_in[(df_in["year_int"] >= start_year) & (df_in["country2"].isin(["CN", "US"]))].copy()
        parent_totals = (
            tmp.groupby([parent_col, "country2"])
            .size()
            .unstack(fill_value=0)
            .rename_axis("parent_topic")
            .reset_index()
        )
        for c in ["CN", "US"]:
            if c not in parent_totals.columns:
                parent_totals[c] = 0
        parent_totals["parent_share_CN"] = parent_totals["CN"] / max(parent_totals["CN"].sum(), 1)
        parent_totals["parent_share_US"] = parent_totals["US"] / max(parent_totals["US"].sum(), 1)
        parent_totals["parent_delta"] = parent_totals["parent_share_CN"] - parent_totals["parent_share_US"]
        parent_totals["parent_abs_delta"] = parent_totals["parent_delta"].abs()

        rows = []
        for parent_topic, grp in tmp.groupby(parent_col):
            child_table = grp.groupby(["country2", child_col]).size().unstack(fill_value=0)
            for country in ["CN", "US"]:
                if country not in child_table.index:
                    child_table.loc[country] = 0
            child_table = child_table.sort_index(axis=1)

            cn_counts = child_table.loc["CN"].astype(float).values
            us_counts = child_table.loc["US"].astype(float).values
            if cn_counts.sum() == 0 and us_counts.sum() == 0:
                continue

            p = cn_counts / cn_counts.sum() if cn_counts.sum() > 0 else np.zeros_like(cn_counts)
            q = us_counts / us_counts.sum() if us_counts.sum() > 0 else np.zeros_like(us_counts)
            js_div = float(jensenshannon(p + 1e-12, q + 1e-12, base=2.0) ** 2)
            l1_dist = float(np.abs(p - q).sum())
            cos_dist = cosine_distance_safe(p, q)
            child_gap = np.abs(p - q)
            top3_child_gap_mean = float(np.sort(child_gap)[::-1][: min(3, len(child_gap))].mean()) if len(child_gap) else np.nan

            rows.append({
                "parent_topic": int(parent_topic),
                "parent_name": parent_name_map.get(int(parent_topic), f"topic_{int(parent_topic)}"),
                "n_child_topics": int((child_table.sum(axis=0) > 0).sum()),
                "js_divergence": js_div,
                "l1_distance": l1_dist,
                "cosine_distance": cos_dist,
                "top3_child_gap_mean": top3_child_gap_mean,
            })

        out = pd.DataFrame(rows)
        out = out.merge(parent_totals, how="left", left_on="parent_topic", right_on="parent_topic")
        out = out.sort_values(["js_divergence", "parent_abs_delta"], ascending=[False, False]).reset_index(drop=True)
        return out


    def plot_parent_structure_scatter(parent_structure_df, parent_level, out_path, title_suffix="Soft Composition"):
        '''绘制父主题总份额差距 vs 内部结构差异散点图。'''
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            parent_structure_df["parent_abs_delta"],
            parent_structure_df["js_divergence"],
            s=80,
            alpha=0.82,
            color="#E03C31",
            edgecolor="white",
            linewidth=0.8,
        )

        to_annotate = parent_structure_df.head(min(12, len(parent_structure_df)))
        for row in to_annotate.itertuples(index=False):
            ax.text(
                row.parent_abs_delta,
                row.js_divergence,
                f"T{row.parent_topic}:{str(row.parent_name)[:18]}",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        ax.set_xlabel("Parent |Δ share| (CN − US)")
        ax.set_ylabel("Internal JS divergence")
        ax.set_title(f"{parent_level} Parent Gap vs Internal Route Difference ({title_suffix})")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ 已保存父主题内部结构差异图: {out_path}")


    parent_level = next((lvl for lvl in ["L4", "L3", "L2"] if lvl in gap_results_by_level_fuzzy), None)
    if parent_level is None:
        print("⚠️ 可用粗层级不足，跳过父主题内部结构差异分析")
    else:
        parent_soft_df, parent_soft_comp_long_df = analyze_parent_soft_composition(
            df_in=te_df,
            parent_membership=membership_by_level[parent_level],
            child_membership=membership_by_level["L0"],
            parent_name_map=hier_topic_name_maps[parent_level],
            start_year=GAP_START_YEAR,
        )
        parent_soft_df.to_csv(TIME_ROOT / f"parent_internal_structure_soft_composition_{parent_level}.csv", index=False)
        parent_soft_comp_long_df.to_csv(TIME_ROOT / f"parent_internal_structure_soft_composition_long_{parent_level}.csv", index=False)

        parent_soft_fig_path = TIME_ROOT / f"parent_internal_structure_soft_composition_{parent_level}.png"
        plot_parent_structure_scatter(
            parent_structure_df=parent_soft_df,
            parent_level=parent_level,
            out_path=parent_soft_fig_path,
            title_suffix="Soft Composition",
        )

        parent_hard_df = analyze_parent_internal_structure_hard(
            df_in=te_df,
            parent_col=f"topic_{parent_level}",
            child_col="topic_L0",
            parent_name_map=hier_topic_name_maps[parent_level],
            start_year=GAP_START_YEAR,
        )
        parent_hard_df.to_csv(TIME_ROOT / f"parent_internal_structure_hard_anchor_{parent_level}.csv", index=False)

        parent_hard_fig_path = TIME_ROOT / f"parent_internal_structure_hard_anchor_{parent_level}.png"
        plot_parent_structure_scatter(
            parent_structure_df=parent_hard_df,
            parent_level=parent_level,
            out_path=parent_hard_fig_path,
            title_suffix="Hard Anchor Baseline",
        )

        globals()["parent_soft_df"] = parent_soft_df
        globals()["parent_soft_comp_long_df"] = parent_soft_comp_long_df
        globals()["parent_hard_df"] = parent_hard_df

        print()
        print("✅ 父主题内部结构差异 summary（Soft Composition 主版本）")
        display(parent_soft_df.head(20))
        print("✅ 父主题内部结构差异 summary（Hard Anchor 对照）")
        display(parent_hard_df.head(20))
    """
)

cells[48] = code_cell(
    """
    # notebook 收尾预览：默认展示 fuzzy 主版本下的关键表
    print("当前默认层次树主方法：", HIER_PRIMARY_METHOD)
    print("当前默认时间演化主版本：fuzzy_weighted_main")

    display(hierarchy_method_comparison_df.head(10))
    display(country_multitopic_yearly_df.head(10))
    display(topic_boundary_df.head(10))
    display(topic_country_cross_diff_df.head(10))
    """
)

# 清理由于插入 cell 后保留的旧尾部重复 cell。
while len(cells) > 49:
    cells.pop()

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Updated notebook: {NOTEBOOK_PATH}")
