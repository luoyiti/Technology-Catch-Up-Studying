#!/usr/bin/env python3
# Inject Chinese explanations into cluster.ipynb (idempotent).
import json
from pathlib import Path

MARKER_MD = "**【中文说明】**"
MARKER_CODE = "# 【单元格中文说明】"

# (cell_index, markdown_block) — appended after existing markdown source
MD_APPEND: list[tuple[int, str]] = [
    (
        0,
        """

**【中文说明】**
- 本节为全笔记本的环境准备：安装/导入依赖、固定随机种子、配置数据路径与 UMAP/层次聚类/向量化等超参数。
- 选用 `allenai/specter2_base` 作为科学文献语义向量模型，输出目录集中在 `output/cluster_results/` 便于复现与归档。
- 设备自动选择 CUDA → MPS → CPU，以便在 GPU 或 Apple Silicon 上加速编码。
""",
    ),
    (
        2,
        """

**【中文说明】**
- 从 WoS 导出的 SCIE CSV 读入后，将常用字段映射为 `title`/`abstract`/`year` 等统一列名。
- 清洗规则：标题与摘要不能同时缺失；拼接为 `text` 供嵌入；过短文本剔除；国家名规范为 CN/US/其他。
- 保留全样本做主题建模，同时生成 `country_code` 供后续中美对比分析。
""",
    ),
    (
        4,
        """

**【中文说明】**
- 使用 SPECTER2（SentenceTransformer 接口）将每篇论文的 `text` 编码为高维向量，并进行 L2 归一化，使余弦相似度等价于点积。
- 向量可缓存为 `.npy`，避免重复下载模型与重复编码，加快迭代。
""",
    ),
    (
        7,
        """

**【中文说明】**
- 本笔记本用 **层次聚类（AgglomerativeClustering）** 替代 BERTopic 默认的 HDBSCAN，使每篇文献都有明确主题（无默认噪声类）。
- 流程含两阶段超参搜索：粗网格在子样本上筛选，再在全量数据上精评；综合轮廓系数、DBI、主题规模均衡与中美覆盖等秩和指标。
- BERTopic 仍负责 c-TF-IDF 主题表示；`hdbscan_model` 参数名保留，但实际传入的是 sklearn 聚类器。
""",
    ),
    (
        9,
        """

**【中文说明】**
- 在最优 UMAP+层次聚类参数下拟合 BERTopic，将主题编号写回 `df`，并计算“代理主题置信度”（样本与主题质心余弦相似度映射到 0–1）。
- 输出 UMAP 空间聚类质量、主题规模分布与主题质心树状图，用于诊断主题是否过度碎片化或失衡。
""",
    ),
    (
        11,
        """

**【中文说明】**
- 利用 BERTopic 的 c-TF-IDF 结果解释每个主题：主题摘要表、关键词列表、代表性文献。
- 代表性文档用于人工核对主题标签是否符合领域语义。
""",
    ),
    (
        15,
        """

**【中文说明】**
- 在中美子样本上构建「主题×国家」计数矩阵，并计算行归一化（主题内中美占比）与列归一化（国家内主题分布）。
- 为后续“技术结构差异”与可视化（堆叠柱状图）提供统一输入。
""",
    ),
    (
        18,
        """

**【中文说明】**
- **覆盖度**：中美各自覆盖了多少个主题，是否存在仅一国出现的主题。
- **Jensen–Shannon**：将两国在主题上的论文分布看作概率分布，衡量结构差异（0 为完全一致）。
""",
    ),
    (
        20,
        """

**【中文说明】**
- 生成 BERTopic 自带的交互式主题图、UMAP 二维散点与主题层次/条形图（HTML），便于探索与汇报。
- Matplotlib 散点作为静态备选或论文插图。
""",
    ),
    (
        24,
        """

**【中文说明】**
- 若主题过细，可调用 `reduce_topics` 合并相似主题；需手动取消注释并按需求设定目标主题数或 `auto`。
- 合并后应重新导出主题信息与下游表格。
""",
    ),
    (
        26,
        """

**【中文说明】**
- 持久化训练好的 BERTopic 模型（含 safetensors 与 c-TF-IDF），并导出 `paper_topics.csv` 供其他笔记本或软件读取。
- 对 JSON 序列化做 numpy 类型补丁，避免保存时报错。
""",
    ),
    (
        28,
        """

**【中文说明】**
- **能力缺口分析**的前置配置：自动选择 `topic`/`topic_reduced` 等列；统一 `country2`；设定前沿窗口、分桶最小样本量等。
- 输出子目录 `capability_gap/` 存放归一化引用、前沿集合与各类缺口指标。
""",
    ),
    (
        30,
        """

**【中文说明】**
- 在进入引用与前沿计算前，核对中美样本量、年份与被引字段缺失、主题数与 outlier 等，避免后续静默偏差。
""",
    ),
    (
        32,
        """

**【中文说明】**
- **MNCS 风格归一化引用**：在 (主题, 年份) 桶内用期望被引平滑，得到可比强度指标 `nc`，削弱发表年份与主题热度差异。
""",
    ),
    (
        34,
        """

**【中文说明】**
- 在每个足够大的 (主题, 年份) 桶内定义 **前 10% 高被引** 阈值 `q90`，构造二值 `top10_flag`，近似顶刊常用的顶尖论文占比分析。
""",
    ),
    (
        36,
        """

**【中文说明】**
- **前沿 A**：截至排除近年后的所有 top10%；**前沿 B（默认）**：最近 `WINDOW_YEARS` 年窗口内的 top10%，更贴近“当前前沿”。
- 排除最近 `EXCLUDE_RECENT_YEARS` 年可减少引用尚未稳定的偏差。
""",
    ),
    (
        38,
        """

**【中文说明】**
- 在中美各自的前沿论文集合上比较 **主题分布**；报告 JS 距离/散度与逐主题差异 Δ（美国前沿占比 − 中国前沿占比）。
- 用于刻画“前沿活动”在主题空间上的结构性偏移。
""",
    ),
    (
        40,
        """

**【中文说明】**
- 以美国前沿质心为参照，在原始嵌入空间计算中国与全美样本相对该质心的语义距离，并得到 per-topic 的 **Gap = D_CN − D_US**。
- 质心基于 L2 归一化向量，几何上兼容余弦相似度；样本不足的主题会跳过并记录原因。
""",
    ),
    (
        42,
        """

**【中文说明】**
- **影响缺口**：在每主题上比较 MNCS（归一化引用均值）与 PP(top10%)（顶尖论文比例），缺口定义为 US − CN。
- 与结构类指标互补：一个偏“整体影响力”，一个偏“顶尖突破比例”。
""",
    ),
    (
        44,
        """

**【中文说明】**
- **四象限图**：横轴为两国在该主题上发文占比之差（量），纵轴为 MNCS 之差（质），用于同时观察“做大”与“做强”是否一致。
""",
    ),
    (
        46,
        """

**【中文说明】**
- 将前沿分布、语义缺口、影响缺口等结果汇总为 JSON 与 Markdown，便于插入报告并留痕分析配置。
""",
    ),
    (
        48,
        """

**【中文说明】**
- **主题随时间演化**：构建年度与滚动窗口下的中美主题份额序列，识别追赶/拉开/平稳等模式，并输出大量 per-topic 曲线图。
- 滚动窗口可平滑单年噪声，更适合观察趋势。
""",
    ),
    (
        54,
        """

**【中文说明】**
- **进入年份**：当一国在某主题上的份额持续高于基线时记为“进入”该主题领域；中美进入年之差为 **滞后/领先年数**。
- 辅以交叉相关验证时间对齐关系，减少单次阈值带来的偶然性。
""",
    ),
    (
        59,
        """

**【中文说明】**
- **国家–年份主题重叠 lead–lag**：把每个 (国家, 年份) 的主题分布看作向量，在 0–15 年滞后下比较中美相似度，用于“技术周期”层面的同步与追赶诊断。
""",
    ),
    (
        60,
        """

**【中文说明】**
- 检查国家、年份、主题列与最小样本阈值；配置 Jaccard/余弦等指标及滚动窗口；输出目录 `overlap_leadlag/`。
""",
    ),
    (
        62,
        """

**【中文说明】**
- 由文献级数据聚合为 (国家, 年份, 主题) 计数与份额，形成宽表矩阵，供后续向量相似度与滞后扫描使用。
""",
    ),
    (
        64,
        """

**【中文说明】**
- 将每年的主题份额转为集合或概率向量：可做二元 Jaccard、加权 Jaccard、余弦相似度等，以刻画结构重叠程度。
""",
    ),
    (
        66,
        """

**【中文说明】**
- 定义各类相似度度量及滞后对齐方式，使不同度量下的结论可对照。
""",
    ),
    (
        68,
        """

**【中文说明】**
- 对每个 CN 年、每个候选滞后 Δt，计算与对应 US 年的相似度曲线，是四宫格图的核心输入。
""",
    ),
    (
        70,
        """

**【中文说明】**
- 将相似度–滞后曲线画成四面板：多 lag 曲线、最佳滞后随时间、最佳相似度随时间等，对应论文中常见 Figure 11 风格。
""",
    ),
    (
        72,
        """

**【中文说明】**
- 从曲线中提取每个 CN 年的 **argmax 滞后** 与对应最高相似度，并保存为表，用于画趋势与写结论。
""",
    ),
    (
        74,
        """

**【中文说明】**
- 更换滚动窗口、仅前沿子样本等设定，检验 lead–lag 结论是否依赖特定平滑或样本定义。
""",
    ),
    (
        76,
        """

**【中文说明】**
- 导出首选设定下的曲线与 best-lag 表，并打印文件清单，便于与 `time_evolution/` 结果一并归档。
""",
    ),
    (
        78,
        """

**【中文说明】**
- 解读四宫格各子图坐标与“正滞后”含义：正值多表示中国该年分布更接近美国 **之后** 若干年的分布，即相对滞后。
""",
    ),
    (
        79,
        """

**【中文说明】**
- **技术版图**：在主题嵌入的 UMAP 平面上用颜色表示中美主导程度、点大小表示论文规模，将结构与国别结合起来看。
- **仅美国出现的主题**：生成诊断包，区分真实空白与聚类边界/拆分伪影。
""",
    ),
    (
        81,
        """

**【中文说明】**
- 对指定 `US_ONLY_TIDS` 主题导出 Markdown 报告：关键词、代表文献、期刊机构、时序与邻域相似度、主题置信度与局部重聚类稳定性。
- 用于质性解释与稳健性说明，避免仅凭“无 CN 论文”下结论。
""",
    ),
    (
        84,
        """

**【中文说明】**
- 对核心缺口指标做 **随机种子重复、超参网格、分层自助法**，报告区间与方差，提醒读者注意估计不确定性。
- 明确 SciPy 返回的是 JS **距离**，散度为其平方。
""",
    ),
    (
        91,
        """

**【中文说明】**
- 小结层次聚类相对 HDBSCAN 的建模取舍：无默认噪声类、全样本可赋值；列出关键输出文件路径便于检查。
""",
    ),
]

# (cell_index, prefix) — prepended to code source (must start with MARKER_CODE first line)
CODE_PREFIX: list[tuple[int, str]] = [
    (
        1,
        """# 【单元格中文说明】
# - 导入本分析所需的全部库；设置 SEED 保证 UMAP/抽样等可复现。
# - DATA_PATH / OUTPUT_DIR：主数据与结果根目录；SPECTER_MODEL：句子嵌入模型名。
# - UMAP_* / AGGLO_*：初始默认值会被后续“自动搜索”覆盖；AGGLO_SEARCH_* 控制搜索规模与 linkage×metric 网格。
# - BATCH_SIZE：编码批大小，显存不足时调小；最后根据 torch 检测 DEVICE 并创建输出子目录。
""",
    ),
    (
        3,
        """# 【单元格中文说明】
# - 读入 CSV 后重命名列；若无 UT 则生成 paper_id。
# - 丢弃标题与摘要同时缺失的行；用空字符串填充单侧缺失；text = title + abstract。
# - 国家名映射为 CN/US，其它保留原值；docs 为全量待嵌入文本列表。
""",
    ),
    (
        5,
        """# 【单元格中文说明】
# - 以下为可选路径：用 HuggingFace adapters 加载带 adapter 的 SPECTER2，当前整段注释保留作参考。
# - 若需与论文完全一致的可尝试取消注释；默认流程使用下一单元格的 SentenceTransformer 基座模型。
""",
    ),
    (
        6,
        """# 【单元格中文说明】
# - 若存在缓存则直接加载 embeddings，否则用 SentenceTransformer 批量 encode。
# - L2 归一化后缓存到 specter2_embeddings_cache.npy，便于重复运行 BERTopic 而无需重算向量。
""",
    ),
    (
        8,
        """# 【单元格中文说明】
# - 定义层次聚类封装（兼容 sklearn 新旧 metric/affinity 参数名）、主题置信度与规模统计、聚类评分与秩和聚合。
# - run_agglomerative_search：Stage A 在子样本上遍历 UMAP×Agglo 网格；Stage B 对 Top-K 在全量上复评；可选注入 compute_gap_metrics 作为辅助目标。
# - 搜索结束将最优 UMAP/Agglo 参数写回全局变量，并装配 umap_model、hdbscan_model（实为 Agglo）、CountVectorizer 与 BERTopic。
# - 细节：用 kneighbors_graph 提供 connectivity 以加速/正则化层次聚类；结果 CSV/图保存于 HIER_DIR。
""",
    ),
    (
        10,
        """# 【单元格中文说明】
# - fit_transform：以预计算 embeddings 训练 BERTopic；topics 写入 df['topic']；topic_prob 为主题级置信度代理。
# - 计算最终轮廓系数与 DBI（在 UMAP 嵌入空间）；绘制最终二维 UMAP 散点、主题规模直方图与主题质心层次树状图。
""",
    ),
    (
        12,
        """# 【单元格中文说明】
# - get_topic_info() 返回每个主题的文档数、自动名称等；供后续导出与核对。
""",
    ),
    (
        13,
        """# 【单元格中文说明】
# - 遍历主题 ID，拉取 c-TF-IDF Top 词并拼成字符串，导出 topic_info.csv。
""",
    ),
    (
        14,
        """# 【单元格中文说明】
# - get_representative_docs：每个主题取若干篇最具代表性的原文（依 c-TF-IDF 与向量距离）；保存为 JSON。
""",
    ),
    (
        16,
        """# 【单元格中文说明】
# - 仅保留中美样本（层次聚类下通常无 topic=-1，此处仍兼容旧模型噪声类）。
# - groupby 构建 topic×country 矩阵；导出 topic_country_matrix.csv 与 topic_share_country.csv（两种归一化）。
""",
    ),
    (
        17,
        """# 【单元格中文说明】
# - 取规模最大的 TOP_K_TOPICS 个主题画堆叠柱：红=中国、蓝=美国，直观对比结构重心差异。
""",
    ),
    (
        19,
        """# 【单元格中文说明】
# - 覆盖度：主题集合差集；JS：将两国在全主题上的论文数转为分布后计算 jensenshannon。
# - metrics.json 汇总上述指标及中美论文量，供报告引用。
""",
    ),
    (
        21,
        """# 【单元格中文说明】
# - BERTopic 内置 visualize_topics 生成交互 HTML，可在浏览器中缩放、框选主题。
""",
    ),
    (
        22,
        """# 【单元格中文说明】
# - 额外做二维 UMAP 静态散点图（全量 embeddings），用于论文插图或与 HTML 互证。
""",
    ),
    (
        23,
        """# 【单元格中文说明】
# - visualize_hierarchy / visualize_barchart：主题层级与关键词条形图 HTML；失败时仅打印警告不影响主流程。
""",
    ),
    (
        25,
        """# 【单元格中文说明】
# - 主题归并示例代码，默认注释掉；需要时设置 NR_TOPICS_REDUCED 并取消注释后运行。
""",
    ),
    (
        27,
        """# 【单元格中文说明】
# - save：序列化模型；paper_topics.csv 保留每篇文献的主题与置信度（及年份若可用）。
# - JSONEncoder 补丁：解决 numpy 标量无法直接写入 JSON 的问题，保存后恢复原实现。
""",
    ),
    (
        29,
        """# 【单元格中文说明】
# - 自动选择最终主题列；定义引用列、年份列、国家列与前沿/分桶参数；建立 country2 与 CAP_DIR。
""",
    ),
    (
        31,
        """# 【单元格中文说明】
# - 打印描述性统计：分国样本量、年份分布、被引缺失率、主题数与嵌入形状，作为能力缺口分析的前置质检。
""",
    ),
    (
        33,
        """# 【单元格中文说明】
# - 在 (topic,year) 桶内计算期望被引，构造 nc；导出 paper_norm_citations.csv。
""",
    ),
    (
        35,
        """# 【单元格中文说明】
# - 仅对足够大桶估计 q90；生成 top10_flag；分国输出顶尖论文占比；导出 paper_top10_flag.csv。
""",
    ),
    (
        37,
        """# 【单元格中文说明】
# - 基于 top10_flag 与时间窗口生成 frontier_A / frontier_B；导出 frontier_papers.csv 并标记仅 A、仅 B 或同时属于两者。
""",
    ),
    (
        39,
        """# 【单元格中文说明】
# - 在中美前沿子集上估计主题分布 P_USF、P_CNF，计算 JS 与逐主题 delta_F；保存 CSV/JSON 与柱状图。
""",
    ),
    (
        41,
        """# 【单元格中文说明】
# - 对美国前沿质心（按主题）与中美全体样本做余弦距离 1−cos；得 gap_frontier_semantic；样本不足主题跳过。
""",
    ),
    (
        43,
        """# 【单元格中文说明】
# - 按主题聚合 MNCS 与 PP(top10%)，形成 US−CN 的缺口并绘制横向条形图。
""",
    ),
    (
        45,
        """# 【单元格中文说明】
# - 横轴：两国总发文中的主题占比差；纵轴：MNCS 差（注意与上一节 US−CN 定义通过取负对齐）。
# - 标注离原点较远的主题以突出“量质错位”象限。
""",
    ),
    (
        47,
        """# 【单元格中文说明】
# - 汇总前沿/语义/影响缺口的关键主题清单，写入 capability_gap_summary.json 与 .md，并列目录。
""",
    ),
    (
        49,
        """# 【单元格中文说明】
# - 为时间演化准备：确定 topic 列与 country2；筛中美；清洗年份范围；输出到 time_evolution/。
""",
    ),
    (
        50,
        """# 【单元格中文说明】
# - 按年计算各国总发文为分母，得到每主题年度份额及 delta=share_CN−share_US；导出 topic_share_yearly.csv。
""",
    ),
    (
        51,
        """# 【单元格中文说明】
# - 对年度计数做滚动求和再归一化，得到 roll5 份额序列，降低单年波动；导出 topic_share_roll5.csv。
""",
    ),
    (
        52,
        """# 【单元格中文说明】
# - 对每个主题的 delta(t) 用 Theil–Sen 估计稳健斜率，划分 catching_up / pulling_away / stable；导出 topic_trend_summary.csv。
""",
    ),
    (
        53,
        """# 【单元格中文说明】
# - 选取 |斜率| 最大的若干主题绘制份额与 delta 曲线，并叠绘总览图；PNG 存于 time_evolution/figs/。
""",
    ),
    (
        55,
        """# 【单元格中文说明】
# - detect_enter_year：用基线均值+kσ 阈值检测份额“跃升”进入年份；对中美分别求 enter_CN、enter_US。
""",
    ),
    (
        56,
        """# 【单元格中文说明】
# - lag = enter_CN − enter_US；交叉相关扫描 ±max_lag 年对齐，记录最大相关的滞后 lag_ccf；合并导出 topic_lead_lag.csv。
""",
    ),
    (
        57,
        """# 【单元格中文说明】
# - 将滞后与 CCF 滞后以条形图展示，颜色区分中国相对美国更晚或更早进入。
""",
    ),
    (
        58,
        """# 【单元格中文说明】
# - 罗列 time_evolution/ 下生成的全部文件，便于与 overlap 结果对照。
""",
    ),
    (
        61,
        """# 【单元格中文说明】
# - §14 配置：国家/年份/主题列检测；LAG_RANGE 与最小单元格样本；选择 Jaccard/余弦开关与输出路径。
""",
    ),
    (
        63,
        """# 【单元格中文说明】
# - 聚合为 (国,年,主题) 长表与宽表，过滤极小样本单元；写出 cy_long / count_wide / share_wide。
""",
    ),
    (
        65,
        """# 【单元格中文说明】
# - 将长表转为每年每国的主题向量（集合或概率），供滞后扫描函数复用。
""",
    ),
    (
        67,
        """# 【单元格中文说明】
# - 实现加权 Jaccard、二元 Jaccard、余弦等相似度及与滞后对齐的例程（与后面曲线计算衔接）。
""",
    ),
    (
        69,
        """# 【单元格中文说明】
# - 对每个 CN 年、每个候选 lag，将 CN 向量与“US 在 t+lag 年”向量比对，形成 similarity–lag 曲线表 overlap_curves_all。
""",
    ),
    (
        71,
        """# 【单元格中文说明】
# - plot_overlap_four_panel：绘制四宫格综述图（多 lag 曲线、最佳滞后趋势等），并保存 PNG/PDF。
""",
    ),
    (
        73,
        """# 【单元格中文说明】
# - 从曲线中提取 argmax lag 作为 best_lag，并绘制随 CN 年变化的最佳滞后与最佳相似度。
""",
    ),
    (
        75,
        """# 【单元格中文说明】
# - 对多种 metric×window×样本（全样本/前沿）重复计算 best_lag，汇总为 robustness_overlap_summary.csv。
""",
    ),
    (
        77,
        """# 【单元格中文说明】
# - 导出“首选”设定下的曲线与 best-lag 子表；打印 overlap_leadlag 目录文件清单。
""",
    ),
    (
        80,
        """# 【单元格中文说明】
# - build_topic_stats：每主题统计中美篇数、占比、dominance=2*CN_share−1；make_topic_map：主题嵌入 UMAP 二维 + Plotly 交互散点。
# - 颜色表示中国在该主题内占比，点大小表示全球该主题论文数；粗边框表示中美一侧主导（>60%）。
""",
    ),
    (
        82,
        """# 【单元格中文说明】
# - export_us_only_topic_pack：对 US_ONLY_TIDS 逐个生成 Markdown+图表；邻域高相似主题提示“拆分伪影”风险；UMAP 重聚类检查局部稳定性。
""",
    ),
    (
        83,
        """# 【单元格中文说明】
# - 列出 §15–§16 期望产物清单并检查文件是否存在，作为交付清单。
""",
    ),
    (
        85,
        """# 【单元格中文说明】
# - 稳健性开关：重复随机种子、超参网格、分层自助法；ROBUST_METRIC_COLS 列出要汇总的不确定性指标。
""",
    ),
    (
        86,
        """# 【单元格中文说明】
# - 封装一次性 BERTopic 拟合、gap 指标聚合、自助抽样索引等，供后续多实验循环调用。
""",
    ),
    (
        87,
        """# 【单元格中文说明】
# - 在 SEED_LIST 上重复训练并计算 gap 指标，输出箱线图/误差条，观察种子敏感性。
""",
    ),
    (
        88,
        """# 【单元格中文说明】
# - 在 UMAP×Agglo 网格上系统扰动超参，观察 JS 距离、前沿 JS 与主题规模离散度之间的关系。
""",
    ),
    (
        89,
        """# 【单元格中文说明】
# - 固定已训练主题赋值，对论文行做分层自助重抽样，得到 gap 指标的置信区间与直方图；不重新拟合主题模型以控制算力。
""",
    ),
    (
        90,
        """# 【单元格中文说明】
# - 将种子标准差与自助区间整理为 Markdown 摘要，提示哪些指标更稳定、建议保留的基线超参组合。
""",
    ),
    (
        92,
        """# 【单元格中文说明】
# - 打印最终选用的层次聚类参数与聚类诊断指标；强调相对 HDBSCAN 的理论差异（无显式噪声类、全样本有主题）。
""",
    ),
]


def _append_md(src: str, block: str) -> str:
    if MARKER_MD in src:
        return src
    return src.rstrip() + block


def _prepend_code(src: str, prefix: str) -> str:
    if MARKER_CODE in src:
        return src
    return prefix.rstrip() + "\n\n" + src


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_path = root / "cluster.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    md_map = dict(MD_APPEND)
    code_map = dict(CODE_PREFIX)

    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            if i not in md_map:
                continue
            src = "".join(cell.get("source", []))
            cell["source"] = _append_md(src, md_map[i]).splitlines(keepends=True)
        elif cell["cell_type"] == "code":
            if i not in code_map:
                continue
            src = "".join(cell.get("source", []))
            cell["source"] = _prepend_code(src, code_map[i]).splitlines(keepends=True)

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {nb_path} ({len(cells)} cells).")


if __name__ == "__main__":
    main()
