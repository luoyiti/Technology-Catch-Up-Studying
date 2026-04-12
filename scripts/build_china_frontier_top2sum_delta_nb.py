#!/usr/bin/env python3
"""Build china_frontier_top2sum_delta.ipynb."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
import nbformat.v4 as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "china_frontier_top2sum_delta.ipynb"


def md(text: str):
    return nbf.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # 中国相对前沿 Top2 份额差距 Delta

        本 notebook 复用 `cluster_keybert_from_cluster.ipynb` 的时间序列口径与滚动平滑方式，
        将原先 `CN − US` 的单主题 delta 改为：

        `delta = 中国主题份额 − 同主题同年份前K国主题份额均值`

        关键约定：
        - frontier topK **包含中国**
        - 平滑发生在 **count 层**，不是直接对 `delta` 平滑
        - 输出写入 `output/cluster_results/time_evolution_cn_frontier_topkmean/`
        """
    ),
    md(
        """
        ## 流程概览

        1. 优先读取 `output/cluster_results/paper_topics.csv`
        2. 按 `topic × year × country` 统计发文数，并按原 notebook 口径计算主题份额
        3. 对每个 `topic-country` 的 yearly count 做 `rolling(5, min_periods=3).sum()`
        4. 在每个 `year-country` 内重新归一化得到平滑份额
        5. 对每个 `topic-year` 计算 `share_CN`、`frontier_share_topk` 与 `delta`
        6. 用 `Theil-Sen` 拟合 `delta ~ year`，并为全部主题输出 PNG
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import os
        import random
        from pathlib import Path
        from typing import Any

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from sklearn.linear_model import TheilSenRegressor

        _CONFIG_PATH = Path("config/cluster_keybert_from_cluster.json")
        _cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        _paths = _cfg["paths"]
        _time = _cfg.get("time_evolution", {})
        SEED = int(_cfg["reproducibility"]["seed"])

        PAPER_TOPICS_PATH = Path("output/cluster_results/paper_topics.csv")
        FALLBACK_DATA_PATH = Path(_paths["data_csv"])
        TOPIC_INFO_PATH = Path("output/cluster_results/topic_info.csv")

        FRONTIER_TOP_K = 2
        CHINA_CODE = "CN"
        ROLLING_WINDOW = int(_time.get("rolling_window_size", 5))
        ROLLING_MIN_PERIODS = int(_time.get("rolling_min_periods", 3))
        EPS_TREND = 1e-4

        OUT_DIR = Path("output/cluster_results/time_evolution_cn_frontier_topkmean")
        TABLES_DIR = OUT_DIR / "tables"
        FIGS_DIR = OUT_DIR / "figs"
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

        plt.rcParams["figure.dpi"] = 120

        print("PAPER_TOPICS_PATH:", PAPER_TOPICS_PATH.resolve())
        print("OUT_DIR:", OUT_DIR.resolve())
        print("ROLLING_WINDOW:", ROLLING_WINDOW, "ROLLING_MIN_PERIODS:", ROLLING_MIN_PERIODS)
        print("FRONTIER_TOP_K:", FRONTIER_TOP_K)
        """
    ),
    code(
        """
        def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
            topic_col = next((c for c in ("topic", "Topic", "topic_id") if c in df.columns), None)
            year_col = next((c for c in ("year", "Year", "pub_year", "publication_year") if c in df.columns), None)
            country_col = "country_code" if "country_code" in df.columns else (
                "country" if "country" in df.columns else None
            )
            if not topic_col or not year_col or not country_col:
                raise KeyError(
                    f"Cannot detect required columns. topic={topic_col}, year={year_col}, country={country_col}"
                )
            return {"topic": topic_col, "year": year_col, "country": country_col}


        def _load_input_table() -> tuple[pd.DataFrame, str]:
            if PAPER_TOPICS_PATH.exists():
                df = pd.read_csv(PAPER_TOPICS_PATH, low_memory=False)
                return df, str(PAPER_TOPICS_PATH)
            if not FALLBACK_DATA_PATH.exists():
                raise FileNotFoundError(
                    f"Neither {PAPER_TOPICS_PATH} nor {FALLBACK_DATA_PATH} exists."
                )
            df = pd.read_csv(FALLBACK_DATA_PATH, low_memory=False)
            cols = _detect_columns(df)
            if cols["topic"] not in df.columns:
                raise ValueError(
                    f"No topic column in fallback table {FALLBACK_DATA_PATH}. "
                    "Run cluster_keybert_from_cluster.ipynb to export paper_topics.csv."
                )
            return df, str(FALLBACK_DATA_PATH)


        def _normalize_country_value(v: Any) -> str:
            if pd.isna(v):
                return "UNKNOWN"
            x = str(v).strip()
            if x in {"CN", "China", "CHINA", "china", "PRC", "People's Republic of China"}:
                return "CN"
            if x in {"US", "USA", "United States", "United States of America"}:
                return "US"
            return x or "UNKNOWN"


        def load_topic_labels(path: Path) -> dict[int, str]:
            if not path.exists():
                return {}
            info = pd.read_csv(path)
            if not {"topic_id", "name"}.issubset(info.columns):
                return {}
            return {
                int(r["topic_id"]): str(r["name"]).strip()
                for _, r in info[["topic_id", "name"]].dropna().iterrows()
            }


        def topic_label(tid: int, labels: dict[int, str]) -> str:
            name = str(labels.get(int(tid), "")).strip()
            return f"T{tid}: {name}" if name else f"T{tid}"


        raw_df, raw_source = _load_input_table()
        cols = _detect_columns(raw_df)
        topic_col, year_col, country_col = cols["topic"], cols["year"], cols["country"]

        papers = raw_df[[topic_col, year_col, country_col]].copy()
        papers.columns = ["topic", "year", "country"]
        papers["topic"] = pd.to_numeric(papers["topic"], errors="coerce")
        papers["year"] = pd.to_numeric(papers["year"], errors="coerce")
        papers = papers.dropna(subset=["topic", "year"]).copy()
        papers["topic"] = papers["topic"].astype(int)
        papers["year"] = papers["year"].astype(int)
        papers["country"] = papers["country"].map(_normalize_country_value)

        labels = load_topic_labels(TOPIC_INFO_PATH)

        print("Loaded from:", raw_source)
        print("papers:", papers.shape)
        print("topics:", papers["topic"].nunique(), "countries:", papers["country"].nunique())
        print("year span:", int(papers["year"].min()), "-", int(papers["year"].max()))
        papers.head()
        """
    ),
    code(
        """
        def build_yearly_country_share(
            df_in: pd.DataFrame,
            topic_col: str = "topic",
            year_col: str = "year",
            country_col: str = "country",
        ) -> pd.DataFrame:
            counts = (
                df_in.groupby([topic_col, year_col, country_col])
                .size()
                .reset_index(name="count")
            )
            country_year_total = (
                df_in.groupby([year_col, country_col])
                .size()
                .reset_index(name="country_total")
            )
            yearly = counts.merge(country_year_total, on=[year_col, country_col], how="left")
            yearly["share"] = yearly["count"] / yearly["country_total"].replace(0, np.nan)
            yearly["share"] = yearly["share"].fillna(0.0)
            return yearly.sort_values([topic_col, country_col, year_col]).reset_index(drop=True)


        def build_rolling_country_share(
            df_in: pd.DataFrame,
            topic_col: str = "topic",
            year_col: str = "year",
            country_col: str = "country",
            window: int = ROLLING_WINDOW,
            min_periods: int = ROLLING_MIN_PERIODS,
        ) -> pd.DataFrame:
            counts = (
                df_in.groupby([topic_col, year_col, country_col])
                .size()
                .reset_index(name="count")
            )
            all_topics = sorted(df_in[topic_col].unique())
            all_countries = sorted(df_in[country_col].unique())
            all_years = np.arange(int(df_in[year_col].min()), int(df_in[year_col].max()) + 1, dtype=int)

            full_idx = pd.MultiIndex.from_product(
                [all_topics, all_countries, all_years],
                names=[topic_col, country_col, year_col],
            )
            full = full_idx.to_frame(index=False)
            wide = full.merge(counts, on=[topic_col, country_col, year_col], how="left")
            wide["count"] = wide["count"].fillna(0).astype(int)
            wide = wide.sort_values([topic_col, country_col, year_col]).reset_index(drop=True)

            wide["roll_count"] = (
                wide.groupby([topic_col, country_col])["count"]
                .transform(lambda s: s.rolling(window=window, min_periods=min_periods).sum())
            )
            wide = wide.dropna(subset=["roll_count"]).copy()

            year_country_total = wide.groupby([year_col, country_col])["roll_count"].transform("sum")
            wide["share_roll"] = wide["roll_count"] / year_country_total.replace(0, np.nan)
            wide["share_roll"] = wide["share_roll"].fillna(0.0)
            return wide.sort_values([topic_col, year_col, country_col]).reset_index(drop=True)


        yearly_country = build_yearly_country_share(papers)
        rolling_country = build_rolling_country_share(papers)

        print("yearly_country:", yearly_country.shape)
        print("rolling_country:", rolling_country.shape)
        display(yearly_country.head())
        display(rolling_country.head())
        """
    ),
    code(
        """
        def _topk_frontier_from_group(grp: pd.DataFrame, k: int = FRONTIER_TOP_K) -> dict[str, Any]:
            ordered = grp.sort_values(["share_roll", "country"], ascending=[False, True]).reset_index(drop=True)
            positive = ordered[ordered["share_roll"] > 0].copy()
            top = positive.head(k) if not positive.empty else ordered.head(k)

            frontier_mean = float(top["share_roll"].mean()) if not top.empty else 0.0
            c1 = str(top.iloc[0]["country"]) if len(top) >= 1 else pd.NA
            c2 = str(top.iloc[1]["country"]) if len(top) >= 2 else pd.NA
            s1 = float(top.iloc[0]["share_roll"]) if len(top) >= 1 else 0.0
            s2 = float(top.iloc[1]["share_roll"]) if len(top) >= 2 else 0.0
            return {
                "frontier_share_topk": frontier_mean,
                "frontier_country_1": c1,
                "frontier_country_2": c2,
                "frontier_country_1_share": s1,
                "frontier_country_2_share": s2,
            }


        def aggregate_cn_frontier_delta(
            country_share_df: pd.DataFrame,
            share_col: str,
            source_name: str,
        ) -> pd.DataFrame:
            rows = []
            for (topic, year), grp in country_share_df.groupby(["topic", "year"], sort=True):
                grp = grp.copy()
                cn_share = grp.loc[grp["country"] == CHINA_CODE, share_col]
                cn_share_val = float(cn_share.iloc[0]) if len(cn_share) else 0.0

                if share_col != "share_roll":
                    grp = grp.rename(columns={share_col: "share_roll"})
                frontier = _topk_frontier_from_group(grp, k=FRONTIER_TOP_K)
                delta = cn_share_val - frontier["frontier_share_topk"]

                rows.append({
                    "topic": int(topic),
                    "year": int(year),
                    "share_CN": cn_share_val,
                    "frontier_share_topk": frontier["frontier_share_topk"],
                    "delta": float(delta),
                    "frontier_country_1": frontier["frontier_country_1"],
                    "frontier_country_2": frontier["frontier_country_2"],
                    "frontier_country_1_share": frontier["frontier_country_1_share"],
                    "frontier_country_2_share": frontier["frontier_country_2_share"],
                    "source": source_name,
                })

            result = pd.DataFrame(rows).sort_values(["topic", "year"]).reset_index(drop=True)
            return result


        yearly_delta = aggregate_cn_frontier_delta(yearly_country, share_col="share", source_name="yearly")
        roll5_delta = aggregate_cn_frontier_delta(rolling_country, share_col="share_roll", source_name="roll5")

        yearly_path = TABLES_DIR / "topic_share_yearly_cn_frontier_topkmean.csv"
        roll5_path = TABLES_DIR / "topic_share_roll5_cn_frontier_topkmean.csv"
        yearly_delta.to_csv(yearly_path, index=False)
        roll5_delta.to_csv(roll5_path, index=False)

        print("Saved:", yearly_path)
        print("Saved:", roll5_path)
        display(roll5_delta.head(10))
        """
    ),
    code(
        """
        def compute_cross_year(years: np.ndarray, deltas: np.ndarray) -> float:
            for i in range(1, len(deltas)):
                if deltas[i - 1] < 0 and deltas[i] >= 0:
                    return float(years[i])
            return np.nan


        def classify_trends(roll_df: pd.DataFrame, min_years: int = 6, eps: float = EPS_TREND) -> pd.DataFrame:
            records = []
            for t, grp in roll_df.groupby("topic"):
                grp = grp.sort_values("year").dropna(subset=["delta"])
                n = len(grp)
                if n < min_years:
                    records.append({
                        "topic": int(t),
                        "slope_delta": np.nan,
                        "delta_start": np.nan,
                        "delta_end": np.nan,
                        "cross_year": np.nan,
                        "trend_label": "insufficient",
                        "n_years": n,
                    })
                    continue

                years = grp["year"].to_numpy(dtype=float).reshape(-1, 1)
                deltas = grp["delta"].to_numpy(dtype=float)
                try:
                    ts = TheilSenRegressor(random_state=SEED, max_subpopulation=5000)
                    ts.fit(years, deltas)
                    slope = float(ts.coef_[0])
                except Exception:
                    slope = float(np.polyfit(years.ravel(), deltas, 1)[0])

                delta_start = float(deltas[0])
                delta_end = float(deltas[-1])
                cross_year = compute_cross_year(years.ravel(), deltas)

                if slope > eps and (delta_end > delta_start + eps):
                    label = "catching_up"
                elif slope < -eps:
                    label = "pulling_away"
                else:
                    label = "stable"

                records.append({
                    "topic": int(t),
                    "slope_delta": round(slope, 8),
                    "delta_start": round(delta_start, 6),
                    "delta_end": round(delta_end, 6),
                    "cross_year": cross_year,
                    "trend_label": label,
                    "n_years": n,
                })

            return pd.DataFrame(records).sort_values(["trend_label", "topic"]).reset_index(drop=True)


        trend_summary = classify_trends(roll5_delta, min_years=6, eps=EPS_TREND)
        trend_path = TABLES_DIR / "topic_trend_summary_cn_frontier_topkmean.csv"
        trend_summary.to_csv(trend_path, index=False)

        print("Saved:", trend_path)
        print(trend_summary["trend_label"].value_counts(dropna=False).to_string())
        display(trend_summary.head(10))
        """
    ),
    code(
        """
        def plot_delta_over_time(roll_df, tid, trend_row, out_dir, labels):
            sub = roll_df[roll_df["topic"] == tid].sort_values("year")
            if sub.empty:
                return

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sub["year"], sub["delta"], "o-", color="#6A0DAD", markersize=3)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.fill_between(sub["year"], sub["delta"], 0, alpha=0.15, color="#6A0DAD")

            cross = trend_row.get("cross_year", np.nan)
            if pd.notna(cross):
                ax.axvline(
                    cross,
                    color="green",
                    linewidth=1.5,
                    linestyle=":",
                    label=f"cross year={int(cross)}",
                )
                ax.legend(fontsize=9)

            label = trend_row.get("trend_label", "")
            slope = trend_row.get("slope_delta", np.nan)
            slope_str = f"{slope:.2e}" if pd.notna(slope) else "N/A"

            ax.set_xlabel("Year")
            ax.set_ylabel(f"Δ share (CN − frontier top{FRONTIER_TOP_K} mean)")
            ax.set_title(
                f"Delta over Time — {topic_label(tid, labels)}  [{label}, slope={slope_str}]"
            )
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_dir / f"delta_T{tid}.png", dpi=150)
            plt.close(fig)


        trend_dict = trend_summary.set_index("topic").to_dict("index")
        topic_ids = sorted(roll5_delta["topic"].unique())
        for tid in topic_ids:
            plot_delta_over_time(roll5_delta, int(tid), trend_dict.get(int(tid), {}), FIGS_DIR, labels)

        print(f"Saved {len(topic_ids)} delta figures to {FIGS_DIR}")
        """
    ),
    code(
        """
        # Validation 1: frontier topK mean must lie within the selected topK share range.
        top1_max = float(roll5_delta["frontier_country_1_share"].max())
        frontier_max = float(roll5_delta["frontier_share_topk"].max())
        frontier_min = float(roll5_delta["frontier_share_topk"].min())
        print("frontier_share_topk range:", (frontier_min, frontier_max), "top1 max:", top1_max)
        assert frontier_max <= top1_max + 1e-12

        # Validation 2: manually verify one topic-year against rolling_country.
        sample = roll5_delta.iloc[0]
        sample_topic = int(sample["topic"])
        sample_year = int(sample["year"])
        grp = rolling_country.query("topic == @sample_topic and year == @sample_year").copy()
        grp = grp.sort_values(["share_roll", "country"], ascending=[False, True]).reset_index(drop=True)
        sample_cn = float(grp.loc[grp["country"] == CHINA_CODE, "share_roll"].iloc[0]) if (grp["country"] == CHINA_CODE).any() else 0.0
        sample_frontier = float(grp.head(FRONTIER_TOP_K)["share_roll"].mean()) if not grp.empty else 0.0
        sample_delta = sample_cn - sample_frontier

        print("Sample validation:")
        print({
            "topic": sample_topic,
            "year": sample_year,
            "share_CN_roll5": sample_cn,
            "frontier_share_topk_roll5": sample_frontier,
            "delta_roll5": sample_delta,
        })

        assert np.isclose(sample_cn, sample["share_CN"])
        assert np.isclose(sample_frontier, sample["frontier_share_topk"])
        assert np.isclose(sample_delta, sample["delta"])

        manifest = {
            "tables": sorted(p.name for p in TABLES_DIR.glob("*.csv")),
            "n_tables": len(list(TABLES_DIR.glob("*.csv"))),
            "n_figs": len(list(FIGS_DIR.glob("delta_T*.png"))),
            "fig_dir": str(FIGS_DIR.resolve()),
            "table_dir": str(TABLES_DIR.resolve()),
        }
        print("\\nManifest:")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        """
    ),
]

nb = nbf.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}

OUT_NB.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Wrote {OUT_NB}")
