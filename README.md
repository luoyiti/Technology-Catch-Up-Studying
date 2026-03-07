# Catch-Up: 技术追赶导向的科研分析项目

本项目是一个研究型分析仓库，而不是面向终端用户的产品应用。它围绕“技术追赶”这一核心问题展开，基于 Web of Science 样本论文数据，结合主题建模、语义相似网络、引文网络与时间演化分析，研究中国相对美国在核科学相关技术版图中的位置变化、前沿能力差距、知识流动关系，以及这些差距是否呈现出可识别的追赶机制。中日比较是项目中的补充视角，用于帮助判断某些主题差异是中美特有现象，还是更广义的国家间技术分化。

This repository is a research-oriented analysis project rather than a product application. It studies technological catch-up using a Web of Science paper corpus and combines topic modeling, semantic similarity graphs, citation graphs, and time-varying analysis to examine China's position relative to the United States in nuclear-science-related research. The China-Japan comparison is used as a secondary lens to validate and contextualize the main China-US narrative.

## 项目关注什么

项目试图回答几个彼此关联的问题：

- 中国和美国是否已经进入相近的技术主题空间，还是仍停留在不同的知识版图中。
- 即便主题覆盖接近，中国是否仍然在“前沿分布”或“高影响力主题”上落后于美国。
- 中国对美国的知识吸收，是通过语义相近的渐进式追赶完成，还是更依赖非对称的引文依赖关系。
- 技术追赶是否体现在时间维度上，例如更晚进入主题、更长的跟随滞后，或在某些主题上逐步缩小差距。

## 数据与研究对象

- 核心数据文件为 `data/dataCleanSCIE.csv`。
- 当前样本规模约为 25,794 篇论文。
- 数据字段来自 WoS 清洗结果，包含题录、摘要、关键词、作者、期刊、国家、年份、DOI 与 `CR` 引文串等信息。
- 项目中的国家口径会按分析需要压缩为 `CN / US / JP / Other`，用于比较中国、美国、日本与其他国家的技术位置。

从当前结果文件可见，项目主要基于核科学及其相关研究主题展开，包括反应堆、辐射暴露、福岛与切尔诺贝利后果、核能系统、探测与测量等方向，因此这里的“技术追赶”更接近一个科学研究能力与知识结构追赶问题，而不是产业产品追赶。

## 方法链路

项目采用的是一条由文本语义到网络结构、再到时间机制的分析链路。

### 1. 技术主题空间

`cluster.ipynb` 与 `cluster-China-Japan.ipynb` 使用：

- `allenai/specter2` 生成论文嵌入
- `BERTopic` 进行主题建模
- `UMAP + HDBSCAN` 进行降维与聚类

这一部分的目标是把论文集合映射为可比较的技术主题空间，并得到各国在主题上的覆盖、占比与代表性文本。当前结果显示，CN-US 之间的主题覆盖已经较为接近，例如现有导出结果里 US 覆盖全部 63 个主题，CN 覆盖 61 个主题，但这并不直接意味着前沿能力已经同步。

### 2. 语义相似网络

`paper_knn_graph.ipynb` 基于 SPECTER2 向量构建论文 KNN 图，形成一个语义相似网络，并进一步分析：

- 社区结构与模块化
- 跨国连边比例
- 国家同配性与主题同配性
- 结构洞与桥接边
- 语义相似网络与引文网络之间的重叠和差异

现有结果表明，语义图具有较强的模块化结构，图规模约为 `N=25,794`、`E=149,705`，Leiden 社区数为 912，模块度达到 0.8262。国家同配性存在，但主题同配性更强，说明语义图首先是按技术主题组织的，其次才体现国家边界。

### 3. 引文网络

`paper_citation_graph.ipynb` 与 `paper_pure_citation_net.ipynb` 解析 WoS 的 `CR` 字段，采用三层匹配逻辑构建内部引文图：

- DOI 匹配
- strict bibkey 匹配
- loose bibkey 匹配

该部分用于研究真实知识流动，而不是语义上的相近性。当前结果表明，内部引文图大约包含 49k 条边，CN->US 引文依赖显著高于 US->CN，支持“追赶过程中存在不对称知识吸收”这一解释方向。

### 4. 时间机制与追赶证据

`time_varying_cn_us_citation_trends.ipynb` 与 `cluster.ipynb` 中的时间演化章节继续追问：技术追赶到底如何发生。

这里主要包括：

- topic time evolution
- lead-lag 分析
- time-consistent similarity evaluation
- 按年份滚动的 logistic 回归拟合

这些分析用来判断谁先进入某类主题、谁在后续跟进、语义接近是否真的会转化为引用行为，以及这种关系是否随时间变化。现有结果显示，引入时间约束后，语义相似对真实引用的预测显著改善，例如 Top-2000 precision 从 0.2565 提升到 0.4130，说明“遵守时序的语义接近”更接近真实知识传递机制。

### 5. 能力差距与前沿位置

项目并不把“覆盖相似主题”直接等同于“完成追赶”，而是进一步构造能力差距指标，包括：

- frontier distribution gap
- impact gap
- `MNCS`
- `PP(top 10%)`
- quantity-quality quadrant

从当前导出的 `capability_gap_summary.*` 来看，CN-US 在 frontier distribution 上仍然存在明显差距，前沿主题分布的 JS distance 约为 0.3879，说明即使主题层面趋同，前沿位置与高影响力占比仍未完全收敛。

## 当前可以支持的主要发现

以下内容来自仓库中已经生成的结果文件，构成当前 README 的核心结论，而不是未来计划。

### 1. 中美主题覆盖接近，但不代表前沿收敛

- `output/cluster_results/metrics.json` 显示，US 覆盖 63 个主题，CN 覆盖 61 个主题。
- CN-US 主题分布存在差异，但更重要的是前沿分布差距仍然明显。
- `output/cluster_results/capability_gap/capability_gap_summary.json` 中，frontier distribution 的 JS distance 约为 0.3879。

这说明“进入同一技术版图”与“占据同等前沿位置”是两个不同问题，技术追赶不能只看主题出现与否。

### 2. 语义相似与真实引用部分重叠，但并不等价

- 相似网络与引文网络之间有交集，但重叠有限。
- 现有报告给出的无向比较中，`both=9,725`，Jaccard 约为 0.112。
- 大量 `only_sim` 说明存在“语义接近但未发生引用”的潜在关联空间。
- 大量 `only_cite` 说明真实引用行为还受到经典文献、方法依赖、学术规范和圈层结构等非纯语义因素影响。

因此，语义接近不能直接替代知识流动，但可以作为理解追赶机制的一个候选视角。

### 3. 时间约束显著提升了“语义相似预测引用”的解释力

- 在不考虑时序时，语义近邻与真实引文的匹配能力有限。
- 加入 time-consistent 约束后，Top-K precision 明显提升。
- 这意味着真正有解释力的不是“谁和谁相似”，而是“后发论文是否接近并吸收了先发论文的知识结构”。

这也是本项目将时间维度放在重要位置的原因，因为技术追赶本质上就是一个时序问题。

### 4. CN->US 与 US->CN 呈现不对称机制

- 在引文图结果里，CN->US dependency 高于 US->CN dependency。
- 在语义网络与引文网络的机制分桶中，`CN->US` 的 `only_cite` 占比高于 `US->CN`。
- 这支持一个重要解释：CN 对 US 的知识吸收并不只是语义上靠近，也可能包含更强的“追赶型引用”或“依赖型引用”特征。

换言之，技术追赶可能体现为一种结构性不对称，而不仅仅是主题统计上的相似。

### 5. 中日比较提供补充证据，而不是替代主线

`cluster-China-Japan.ipynb` 与 `output/cluster_results/jp_only_topics/` 展示了另一条比较路径。当前结果中出现了 `JP-only topic`，例如围绕福岛后果、甲状腺癌与放射生态影响等主题的区域集中现象。这些结果的意义在于：

- 它帮助识别哪些主题缺口是中国长期缺席，而非仅仅相对美国落后。
- 它为“技术追赶”提供一个额外参照系，避免所有差异都被简单解释为中美竞争结果。

## 仓库地图

这个仓库以 notebook 为主，不存在一个真正的统一应用入口。`main.py` 目前只是占位文件，不承载核心逻辑。

### 关键输入

- `data/dataCleanSCIE.csv`：主数据集
- `data/dataCleanSCIE_Cut.csv`：裁切样本

### 核心 notebooks

- `cluster.ipynb`：CN-US 主题建模、能力差距、时间演化、lead-lag、US-only topics、稳健性分析
- `cluster-China-Japan.ipynb`：CN-JP 对比分析
- `paper_knn_graph.ipynb`：语义 KNN 图、网络结构、与引文网络对比
- `paper_pure_citation_net.ipynb`：WoS `CR` 驱动的纯引文网络
- `paper_citation_graph.ipynb`：引文网络分析的另一版流程
- `time_varying_cn_us_citation_trends.ipynb`：逐年 logit 拟合与时变引文机制

### 主要输出目录

- `output/cluster_results/`：主题建模、能力差距、时间演化、JP-only / US-only topics
- `output/graph/`：语义相似网络结果
- `output/citation_graph/`：纯引文网络结果
- `output/citation_results/`：引文网络汇总结果
- `output/compare_networks/`：相似网络与引文网络的对比分析
- `output/timeaware_similarity/`：时间一致的语义相似评估结果

如果你的目标是快速理解项目内容，优先看 `output/cluster_results/`、`output/graph/summary.json`、`output/citation_graph/summary.json` 与 `output/compare_networks/paper_knn_graph_report.md`。

## 如何使用这个仓库

项目当前更适合“研究复现”而不是“一键运行”。

### 1. 安装依赖

```bash
uv sync
```

依赖已经在 [`pyproject.toml`](./pyproject.toml) 中声明，核心包括：

- `sentence-transformers`
- `bertopic`
- `umap-learn`
- `hdbscan`
- `faiss-cpu`
- `pandas`
- `scikit-learn`
- `statsmodels`
- `pyarrow`

### 2. 按研究工作流阅读或复现

建议顺序如下：

1. `cluster.ipynb`
2. `paper_knn_graph.ipynb`
3. `paper_pure_citation_net.ipynb`
4. `time_varying_cn_us_citation_trends.ipynb`
5. `cluster-China-Japan.ipynb`

如果只想快速了解结论，可以先查看已经预生成的 `output/` 目录，再决定是否重新跑 notebook。

### 3. 复现时需要注意

- 仓库当前以 notebook 为主，参数管理与流程封装仍较弱。
- 某些 notebook 会依赖已经生成好的中间文件，例如 embeddings cache、topic 结果或 parquet 输出。
- 不同 notebook 之间存在结果复用关系，因此更适合按既定顺序逐步执行。

## 适用场景与研究价值

这个仓库适合以下场景：

- 技术追赶研究
- 国家创新能力比较
- 科学计量学与知识流动分析
- 科技政策评估
- 主题建模与网络分析结合的研究设计参考

项目的主要价值不在于单个模型本身，而在于它把几个通常分散的问题放进了同一个框架里：

- 主题差距
- 结构差距
- 流动差距
- 时间差距

也就是说，它并不只问“谁发了什么”，而是进一步追问“谁占据了前沿、谁在吸收谁、谁是在追赶、追赶是否真的在发生”。

## 局限性

当前结果仍应谨慎解释，至少有以下局限：

- 样本受 WoS 数据来源和当前主题范围限制，不代表全部科研活动。
- 国家归属是简化映射，无法完整表达多国合作和机构层级差异。
- frontier、impact、top10% 等指标依赖当前窗口、阈值与定义口径。
- notebook 驱动很强，工程化封装不足，可重复执行性和参数透明度仍可继续提升。
- 项目目前更适合作为研究原型与分析证据链，而不是稳定的软件工具。

## 推荐阅读结果

如果只想快速掌握项目内容，建议优先查看以下文件：

- `output/compare_networks/paper_knn_graph_report.md`
- `output/cluster_results/capability_gap/capability_gap_summary.md`
- `output/graph/summary.json`
- `output/citation_graph/summary.json`
- `output/timeaware_similarity/summary.json`

这些文件基本覆盖了本项目关于技术追赶的主要证据链：技术主题空间、网络结构、知识流动、时间一致性与前沿差距。
