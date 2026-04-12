#!/usr/bin/env python3
"""Plot delta(t) = share_country - frontier_value per topic (default: CN)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_COLS = ("topic", "country", "year", "share", "frontier_value")
EPS_SLOPE = 1e-12


def load_topic_labels(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"topic labels file not found: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        raise ValueError("JSON topic labels must be an object mapping topic_id -> label")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "topic_id" in df.columns and "label" in df.columns:
            tid_col, lab_col = "topic_id", "label"
        elif "topic" in df.columns and "label" in df.columns:
            tid_col, lab_col = "topic", "label"
        else:
            raise ValueError("CSV topic labels need columns (topic_id|topic) and label")
        return {int(r[tid_col]): str(r[lab_col]) for _, r in df.iterrows()}
    raise ValueError(f"Unsupported topic labels format: {path.suffix}")


def topic_label(tid: int, labels: dict[int, str]) -> str:
    if tid in labels:
        return f"T{tid}: {labels[tid]}"
    return f"T{tid}"


def compute_cross_year(years: np.ndarray, deltas: np.ndarray) -> float:
    for i in range(1, len(deltas)):
        if deltas[i - 1] < 0 and deltas[i] >= 0:
            return float(years[i])
    return np.nan


def compute_trend_row(years: np.ndarray, deltas: np.ndarray) -> dict:
    mask = np.isfinite(years) & np.isfinite(deltas)
    y = years[mask].astype(float)
    d = deltas[mask].astype(float)
    out: dict = {
        "cross_year": compute_cross_year(y, d),
        "slope_delta": np.nan,
        "trend_label": "flat",
    }
    if len(y) < 2:
        return out
    slope, _, _, _, _ = stats.linregress(y, d)
    out["slope_delta"] = slope
    if slope > EPS_SLOPE:
        out["trend_label"] = "catching_up"
    elif slope < -EPS_SLOPE:
        out["trend_label"] = "lagging"
    else:
        out["trend_label"] = "flat"
    return out


def plot_delta_over_time(
    roll_df: pd.DataFrame,
    tid: int,
    trend_row: dict,
    out_dir: Path,
    labels: dict[int, str],
    country: str,
) -> None:
    """Plot delta(t) = share - frontier for a single topic."""
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
    ax.set_ylabel(f"Δ share ({country} − frontier)")
    ax.set_title(
        f"Delta over Time — {topic_label(tid, labels)}  [{label}, slope={slope_str}]"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"delta_T{tid}.png", dpi=150)
    plt.close(fig)


def build_roll_df(df: pd.DataFrame, country: str) -> pd.DataFrame:
    sub = df.loc[df["country"] == country].copy()
    sub["delta"] = sub["share"].astype(float) - sub["frontier_value"].astype(float)
    if "gap_to_frontier" in sub.columns:
        g = sub["gap_to_frontier"].astype(float)
        chk = (sub["delta"] + g).abs()
        if not np.allclose(sub["delta"] + g, 0.0, rtol=0.0, atol=1e-8):
            raise ValueError(
                "Inconsistent panel: expected delta == share - frontier == -gap_to_frontier"
            )
    return sub[["topic", "year", "delta"]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-topic delta = share - frontier_value over time."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to topic_country_year_panel CSV (with traj columns OK).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for delta_T{topic}.png files.",
    )
    p.add_argument("--country", type=str, default="CN", help="Country code (default CN).")
    p.add_argument(
        "--topic-labels",
        type=Path,
        default=None,
        help="Optional JSON object or CSV with topic id and label columns.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_topic_labels(args.topic_labels)

    path = args.input.expanduser().resolve()
    if not path.is_file():
        print(f"error: input not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"error: CSV missing columns: {missing}", file=sys.stderr)
        return 1

    roll_df = build_roll_df(df, args.country)
    out_dir = args.out_dir.expanduser().resolve()

    for tid in sorted(roll_df["topic"].unique()):
        sub = roll_df[roll_df["topic"] == tid].sort_values("year")
        y = sub["year"].to_numpy()
        d = sub["delta"].to_numpy()
        trend_row = compute_trend_row(y, d)
        plot_delta_over_time(
            roll_df, int(tid), trend_row, out_dir, labels, args.country
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
