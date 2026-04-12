from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

CN_LABELS = {
    "China",
    "china",
    "CN",
    "CHINA",
    "Peoples R China",
    "PRC",
}
US_LABELS = {
    "United States",
    "united states",
    "US",
    "USA",
    "United States of America",
}


DEFAULT_COL_MAP = {
    "TI": "title",
    "AB": "abstract",
    "country": "country",
    "PY": "year",
    "TC": "citation",
    "UT": "paper_id",
}


def normalize_country_label(value: Any) -> str:
    """Map raw country names into CN / US / original value."""
    if pd.isna(value):
        return "Other"
    text = str(value).strip()
    if text in CN_LABELS:
        return "CN"
    if text in US_LABELS:
        return "US"
    return text


def load_and_clean_papers(
    data_path: str,
    min_text_len: int = 20,
    col_map: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load WoS CSV and apply consistent cleaning pipeline used in notebooks."""
    mapping = col_map or DEFAULT_COL_MAP

    df_raw = pd.read_csv(data_path, low_memory=False)
    cols_present = {k: v for k, v in mapping.items() if k in df_raw.columns}
    df = df_raw.rename(columns=cols_present).copy()

    if "paper_id" not in df.columns:
        df["paper_id"] = [f"paper_{i}" for i in range(len(df))]

    for col in ("title", "abstract"):
        if col not in df.columns:
            df[col] = ""

    n_before = len(df)
    missing_title = int(df["title"].isna().sum())
    missing_abstract = int(df["abstract"].isna().sum())

    df = df.dropna(subset=["title", "abstract"], how="all").copy()
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["text"] = df["title"].astype(str).str.strip() + ". " + df["abstract"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_text_len].reset_index(drop=True)

    has_year = False
    if "year" in df.columns and df["year"].notna().any():
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        has_year = True

    if "country" not in df.columns:
        df["country"] = "Other"
    df["country_code"] = df["country"].apply(normalize_country_label)

    stats = {
        "raw_rows": int(n_before),
        "clean_rows": int(len(df)),
        "dropped_rows": int(n_before - len(df)),
        "missing_title": missing_title,
        "missing_abstract": missing_abstract,
        "cn_papers": int((df["country_code"] == "CN").sum()),
        "us_papers": int((df["country_code"] == "US").sum()),
        "other_papers": int((~df["country_code"].isin(["CN", "US"])).sum()),
        "has_year": has_year,
    }

    return df, stats
