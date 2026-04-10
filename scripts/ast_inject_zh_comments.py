"""
Insert # 中文：... lines after each function docstring (or before first stmt if no docstring).
Uses AST lineno/end_lineno (Python 3.8+).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Expr):
        v = stmt.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return True
        if isinstance(v, ast.Str):  # pragma: no cover
            return True
    return False


def _insert_positions_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[int]:
    """Return 0-based line index where a new line should be inserted (insert before this index)."""
    if not node.body:
        return None
    first = node.body[0]
    if _is_docstring_stmt(first):
        # Insert after docstring's last line
        end = getattr(first, "end_lineno", None)
        if end is None:
            return None
        return end  # 0-based index = insert before line (end+1) in 1-based terms
    # No docstring: insert before first statement
    ln = getattr(first, "lineno", None)
    if ln is None:
        return None
    return ln - 1  # 0-based: first line of first stmt


def _strip_zh_annotation_lines(source: str) -> str:
    """Remove # 中文： full lines and end-of-line markers (idempotent re-run)."""
    out_lines = []
    for ln in source.splitlines(keepends=True):
        body = ln.rstrip("\r\n")
        ending = ln[len(body) :]
        if body.lstrip().startswith("# 中文："):
            continue
        if " # 中文：" in body:
            body = body.split(" # 中文：", 1)[0].rstrip()
            ln = body + ending
        out_lines.append(ln)
    return "".join(out_lines)


def inject_after_docstrings(source: str, comments: Dict[str, str]) -> str:
    """comments: function_name -> single-line text without leading # (we add '    # 中文：')."""
    if not comments or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    inserts: List[Tuple[int, str]] = []
    seen_names: set[str] = set()

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in comments:
            if n.name in seen_names:
                continue
            seen_names.add(n.name)
            pos = _insert_positions_for_function(n)
            if pos is None:
                continue
            text = comments[n.name].strip()
            if not text.startswith("#"):
                line = f"    # 中文：{text}\n"
            else:
                line = f"    {text.lstrip()}\n" if not text.startswith("    ") else text + "\n"
            inserts.append((pos, line))

    inserts.sort(key=lambda x: -x[0])
    lines = source.splitlines(keepends=True)
    for pos, line in inserts:
        if pos < 0 or pos > len(lines):
            continue
        # Idempotent: do not insert twice if Chinese line already follows docstring
        if pos < len(lines) and lines[pos].lstrip().startswith("# 中文："):
            continue
        lines.insert(pos, line)

    return "".join(lines)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def set_cell_source(nb: dict, cell_idx: int, new_source: str) -> None:
    nb["cells"][cell_idx]["source"] = new_source.splitlines(keepends=True)


# --- Per-cell comment maps (value = short Chinese after 中文：) ---

CELL8_COMMENTS: Dict[str, str] = {
    "build_agglomerative_model": "封装 AgglomerativeClustering；Ward 仅适用欧氏距离；connectivity 为近邻图以约束聚类。",
    "compute_topic_confidence": "用样本向量与所属主题质心余弦相似度作「置信度」代理，并映射到 [0,1]；单篇主题另作处理。",
    "compute_topic_size_stats": "统计各主题文档数：极值、集中度、变异系数；层次聚类默认无 -1 噪声类。",
    "_safe_cluster_scores": "在降维空间上算轮廓系数与 Davies-Bouldin；大样本时子采样以控耗时。",
    "_score_penalty_for_topic_count": "主题数偏离预设区间时施加惩罚，避免过多或过粗的主题划分。",
    "_add_rank_aggregate": "对各指标做秩变换再线性组合为 rank_score，缓解量纲不可比问题。",
    "_evaluate_config_with_reduced": "在给定 UMAP 约简坐标上跑层次聚类并评分；可注入 gap 相关辅助指标。",
    "plot_agglomerative_search": "将网格搜索结果画成散点/条形/气泡图并导出 PNG/HTML。",
    "plot_topic_centroid_dendrogram": "先按主题求嵌入质心，再对质心做层次聚类树状图，观察主题远近。",
    "run_agglomerative_search": "两阶段：子样本粗搜 UMAP×Agglo；全量精评 Top-K；导出最优参数与 CSV。",
}

CELL50_COMMENTS: Dict[str, str] = {
    "build_topic_time_series": "按年、按国统计各主题发文数；用该国当年总发文作分母得份额；delta=CN 份额−US 份额。",
}

CELL51_COMMENTS: Dict[str, str] = {
    "build_rolling_share": "对年度计数做滚动求和再按国年归一化，得到平滑后的主题份额序列。",
}

CELL52_COMMENTS: Dict[str, str] = {
    "classify_trends": "对每个主题的 delta(t) 用 Theil–Sen 稳健回归估斜率，并标记追赶/拉开/平稳。",
}

CELL53_COMMENTS: Dict[str, str] = {
    "_topic_label": "组合主题编号与 BERTopic 名称，缩短用于图例。",
    "plot_share_over_time": "单主题 roll5 下中美份额随时间折线图。",
    "plot_delta_over_time": "单主题 delta 随时间；可标出由负变正的年份（交叉年）。",
}

CELL55_COMMENTS: Dict[str, str] = {
    "detect_enter_year": "用早期年份份额作基线，高于 μ+kσ 且后续非跌视为「进入」该主题领域。",
}

CELL56_COMMENTS: Dict[str, str] = {
    "compute_lead_lag": "lag=enter_CN−enter_US；并扫描互相关得最优对齐滞后。",
    "cross_corr_lag": "在 ±max_lag 内平移中美份额序列，取 Pearson 相关最大的滞后（正=CN 滞后）。",
}

CELL65_COMMENTS: Dict[str, str] = {
    "build_topic_set_from_row": "从 (国,年) 聚合行提取「出现」的主题集合；可按最小份额或 top-N 截断。",
    "safe_get_vector": "安全读取嵌套字典 vecs[country][year]，缺省返回空。",
    "has_enough_docs": "判断该 (国,年) 单元格总发文是否达到最小阈值，过滤不可靠份额。",
}

CELL67_COMMENTS: Dict[str, str] = {
    "binary_jaccard": "二元 Jaccard：两集合交并比，强调「是否出现」而非权重。",
    "weighted_jaccard": "加权 Jaccard：用最小份额之和归一，兼顾主题频率。",
    "cosine_topic_similarity": "将主题份额视为向量，在共同主题子空间上算余弦相似度。",
}

CELL69_COMMENTS: Dict[str, str] = {
    "_build_country_year_long": "由文献级表聚合为 (国,年,主题) 长表，并附 n_docs、topic_share。",
    "_roll_country_year_long": "对长表按国、年做滚动窗口平滑，再归一化得 rolling 份额。",
    "_build_vectors_from_long": "将长表折叠为 topic_sets / share_vectors / binary_vectors 嵌套字典。",
    "_get_common_topics": "取两向量在足够份额下的主题并集或交集支撑，用于相似度计算。",
    "compute_overlap_panel_curves": "对 lag 网格与多种 metric 扫描，输出面板曲线长表。",
}

CELL71_COMMENTS: Dict[str, str] = {
    "plot_overlap_four_panel": "四宫格：相似度-滞后曲线、最佳滞后/最佳相似度随 CN 年变化。",
}

CELL73_COMMENTS: Dict[str, str] = {
    "_extract_best_lag": "对每个 (metric, rolling_window, cn_year) 取 similarity 最大的 lag 作为 best_lag。",
}

CELL80_COMMENTS: Dict[str, str] = {
    "build_topic_stats": "按主题统计中美篇数、占比、dominance，并拉取关键词用于悬停。",
    "make_topic_map": "用 topic_embeddings_ 做 UMAP 二维散点，颜色=CN 占比，大小=全球发文量。",
}

CELL82_COMMENTS: Dict[str, str] = {
    "export_us_only_topic_pack": "为「仅美国出现」的主题生成 Markdown 报告、图表与邻域/重聚类诊断。",
}

CELL86_COMMENTS: Dict[str, str] = {
    "_map_country_cn_us": "将多种写法统一映射到 CN/US/OTHER。",
    "_ensure_country2": "若无 country2 则从 country / country_code 推导。",
    "_detect_topic_col": "优先使用全局 TOPIC_COL，否则在候选列中自动检测。",
    "fit_topics_once": "用给定 UMAP/聚类/向量化参数训练一次 BERTopic 并返回主题标签。",
    "_compute_lead_lag_summary": "基于 roll5 与 detect_enter_year 汇总 lead–lag 的均值/中位数/有效主题数。",
    "compute_gap_metrics": "在给定主题赋值下重算 JS、覆盖、前沿、影响与 lead–lag 等缺口指标。",
    "summarize_runs": "对多次实验的指标做均值、标准差与分位数汇总。",
    "stratified_bootstrap_indices": "按分层（如年×国）有放回抽样索引，用于自助法。",
}


def patch_main_flow_cell8(source: str) -> str:
    """Add Chinese comments to non-def execution blocks in cell 8 (idempotent)."""
    if not source.strip():
        return source
    replacements = [
        (
            "# ── 3.1 Run two-stage auto-search and write back best params ─────────────────\n",
            "# ── 3.1 Run two-stage auto-search and write back best params ─────────────────\n"
            "# 中文：执行两阶段搜索并把最优 UMAP/Agglo 超参写回全局变量，供下文 BERTopic 使用。\n",
        ),
        (
            "UMAP_N_NEIGHBORS = int(best_agglom_params[\"umap_n_neighbors\"])\n",
            "UMAP_N_NEIGHBORS = int(best_agglom_params[\"umap_n_neighbors\"])  # 中文：以下为搜索得到的最优超参\n",
        ),
        (
            "# ── 3.2 Build UMAP model (best params) ──────────────────────────────────────\n",
            "# ── 3.2 Build UMAP model (best params) ──────────────────────────────────────\n"
            "# 中文：用最优参数构造 UMAP，用于 BERTopic 内部降维。\n",
        ),
        (
            "# ── 3.3 Build Agglomerative model (best params) ─────────────────────────────\n",
            "# ── 3.3 Build Agglomerative model (best params) ─────────────────────────────\n"
            "# 中文：层次聚类器传入 hdbscan_model 槽位（命名历史原因）。\n",
        ),
        (
            "# ── 3.4 Vectorizer (c-TF-IDF) ───────────────────────────────────────────────\n",
            "# ── 3.4 Vectorizer (c-TF-IDF) ───────────────────────────────────────────────\n"
            "# 中文：词袋+ngram 供 c-TF-IDF 提取主题词。\n",
        ),
        (
            "# ── 3.5 BERTopic ─────────────────────────────────────────────────────────────\n",
            "# ── 3.5 BERTopic ─────────────────────────────────────────────────────────────\n"
            "# 中文：组装完整 BERTopic 流水线；层次聚类无原生软概率故 calculate_probabilities=False。\n",
        ),
    ]
    out = source
    for old, new in replacements:
        if old in out and new not in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell3(source: str) -> str:
    if "中文：读取 WoS" in source:
        return source
    reps = [
        (
            "# ── 1.1 Read data ────────────────────────────────────────────────────────────\n",
            "# ── 1.1 Read data ────────────────────────────────────────────────────────────\n"
            "# 中文：读取 WoS 导出 CSV；low_memory=False 避免分块推断导致列类型不一致。\n",
        ),
        (
            "df = df_raw.rename(columns=cols_present).copy()\n",
            "df = df_raw.rename(columns=cols_present).copy()  # 中文：仅重命名存在的列，保留其余原始字段\n",
        ),
        (
            "df[\"text\"] = df[\"title\"].str.strip() + \". \" + df[\"abstract\"].str.strip()\n",
            "df[\"text\"] = df[\"title\"].str.strip() + \". \" + df[\"abstract\"].str.strip()  # 中文：与 SPECTER 输入格式一致（标题+摘要）\n",
        ),
        (
            "df[\"country_code\"] = df[\"country\"].apply(normalise_country)\n",
            "df[\"country_code\"] = df[\"country\"].apply(normalise_country)  # 中文：统一中美标签，其它国家保留原值\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell6(source: str) -> str:
    if "中文：命中缓存则跳过" in source:
        return source
    reps = [
        (
            "if os.path.exists(EMBEDDINGS_CACHE):\n",
            "if os.path.exists(EMBEDDINGS_CACHE):  # 中文：命中缓存则跳过模型加载与编码，加速重跑\n",
        ),
        (
            "    # ── 2.2 Encode ──────────────────────────────────────────────────────────\n",
            "    # ── 2.2 Encode ──────────────────────────────────────────────────────────\n"
            "    # 中文：SentenceTransformer 批量前向；batch_size 过大易 GPU OOM\n",
        ),
        (
            "    embeddings = normalize(embeddings, norm=\"l2\")\n",
            "    embeddings = normalize(embeddings, norm=\"l2\")  # 中文：L2 后余弦相似度=点积，便于质心与距离\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell16(source: str) -> str:
    if "中文：groupby 统计" in source:
        return source
    reps = [
        (
            "# ── 6.2 Topic × Country count matrix ────────────────────────────────────────\n",
            "# ── 6.2 Topic × Country count matrix ────────────────────────────────────────\n"
            "# 中文：groupby 统计每个 (主题, 国家) 的论文篇数，unstack 成宽表。\n",
        ),
        (
            "topic_share_row = topic_country_matrix[[\"CN\", \"US\"]].div(\n",
            "topic_share_row = topic_country_matrix[[\"CN\", \"US\"]].div(  # 中文：行归一化=主题内中美占比\n",
        ),
        (
            "topic_share_col = topic_country_matrix[[\"CN\", \"US\"]].div(country_totals, axis=1)\n",
            "topic_share_col = topic_country_matrix[[\"CN\", \"US\"]].div(country_totals, axis=1)  # 中文：列归一化=国家内主题分布\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell19(source: str) -> str:
    if "中文：集合差分" in source:
        return source
    reps = [
        (
            "cn_only = cn_topics - us_topics\n",
            "cn_only = cn_topics - us_topics  # 中文：集合差分，找出仅一国出现的主题\n",
        ),
        (
            "js_div = jensenshannon(p_cn, p_us)   # returns the JS *distance* (sqrt of divergence)\n",
            "js_div = jensenshannon(p_cn, p_us)   # 中文：SciPy 返回 JS 距离；平方得散度\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell29(source: str) -> str:
    if "中文：Frontier_B" in source:
        return source
    reps = [
        (
            "WINDOW_YEARS         = 5      # recent-year window for Frontier_B\n",
            "WINDOW_YEARS         = 5      # 中文：Frontier_B 用最近 WINDOW_YEARS 年定义「当前前沿」窗口\n",
        ),
        (
            "df[\"country2\"] = df[COUNTRY_COL].apply(_map_country2)\n",
            "df[\"country2\"] = df[COUNTRY_COL].apply(_map_country2)  # 中文：三元标签，OTHER 不参与中美缺口核心对比\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell47(source: str) -> str:
    if "中文：汇总 capability" in source:
        return source
    reps = [
        (
            "# ── Gather top findings ──────────────────────────────────────────────────────\n",
            "# ── Gather top findings ──────────────────────────────────────────────────────\n"
            "# 中文：汇总 capability_gap 各模块的「头部主题」供 JSON/Markdown 报告使用。\n",
        ),
        (
            "with open(os.path.join(CAP_DIR, \"capability_gap_summary.json\"), \"w\") as f:\n",
            "with open(os.path.join(CAP_DIR, \"capability_gap_summary.json\"), \"w\") as f:  # 中文：结构化结果供程序读取\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell87(source: str) -> str:
    if "中文：固定其它超参" in source:
        return source
    reps = [
        (
            "if RUN_SEED_REPEAT:\n",
            "if RUN_SEED_REPEAT:  # 中文：固定其它超参，仅改 UMAP/Agglo 随机种子，观察指标方差\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell88(source: str) -> str:
    if "中文：在邻域" in source:
        return source
    reps = [
        (
            "if RUN_GRID:\n",
            "if RUN_GRID:  # 中文：在邻域内扰动 UMAP×Agglo，检验结论对超参的敏感性\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell89(source: str) -> str:
    if "中文：自助法" in source:
        return source
    reps = [
        (
            "if RUN_BOOTSTRAP:\n",
            "if RUN_BOOTSTRAP:  # 中文：自助法估计 gap 指标置信区间（固定主题赋值，重抽样论文行）\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell90(source: str) -> str:
    if "中文：汇总种子" in source:
        return source
    reps = [
        (
            "from IPython.display import display, Markdown\n",
            "from IPython.display import display, Markdown  # 中文：在 Notebook 中渲染 Markdown 小结\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out:
            out = out.replace(old, new, 1)
    return out


def patch_cell10(source: str) -> str:
    reps = [
        (
            "# ── 4.1 Fit BERTopic with best params ────────────────────────────────────────\n",
            "# ── 4.1 Fit BERTopic with best params ────────────────────────────────────────\n"
            "# 中文：fit_transform 使用预计算 embeddings；topics 为每篇文献的整数主题 ID。\n",
        ),
        (
            "df[\"topic_prob\"] = compute_topic_confidence(embeddings, topics)\n",
            "df[\"topic_prob\"] = compute_topic_confidence(embeddings, topics)  # 中文：非模型原生概率，仅作诊断用\n",
        ),
        (
            "# ── 4.4 Hierarchical visualizations and exports ─────────────────────────────\n",
            "# ── 4.4 Hierarchical visualizations and exports ─────────────────────────────\n"
            "# 中文：导出最终 UMAP 散点、主题规模图与质心树状图。\n",
        ),
    ]
    out = source
    for old, new in reps:
        if old in out and new not in out:
            out = out.replace(old, new, 1)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_path = root / "cluster.ipynb"
    nb = load_notebook(nb_path)

    # (cell_idx, comment_dict, optional postprocessor)
    jobs: List[Tuple[int, Dict[str, str], Optional[Callable[[str], str]]]] = [
        (3, {}, patch_cell3),
        (6, {}, patch_cell6),
        (8, CELL8_COMMENTS, patch_main_flow_cell8),
        (10, {}, patch_cell10),
        (16, {}, patch_cell16),
        (19, {}, patch_cell19),
        (29, {}, patch_cell29),
        (47, {}, patch_cell47),
        (50, CELL50_COMMENTS, None),
        (51, CELL51_COMMENTS, None),
        (52, CELL52_COMMENTS, None),
        (53, CELL53_COMMENTS, None),
        (55, CELL55_COMMENTS, None),
        (56, CELL56_COMMENTS, None),
        (65, CELL65_COMMENTS, None),
        (67, CELL67_COMMENTS, None),
        (69, CELL69_COMMENTS, None),
        (71, CELL71_COMMENTS, None),
        (73, CELL73_COMMENTS, None),
        (80, CELL80_COMMENTS, None),
        (82, CELL82_COMMENTS, None),
        (86, CELL86_COMMENTS, None),
        (87, {}, patch_cell87),
        (88, {}, patch_cell88),
        (89, {}, patch_cell89),
        (90, {}, patch_cell90),
    ]

    for idx, cmap, post in jobs:
        cell = nb["cells"][idx]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        src = _strip_zh_annotation_lines(src)
        if cmap:
            src = inject_after_docstrings(src, cmap)
        if post:
            src = post(src)
        set_cell_source(nb, idx, src)

    save_notebook(nb_path, nb)
    print(f"Patched {nb_path} with inline function + flow comments.")


if __name__ == "__main__":
    main()
