"""DFS ownership/chalk interpretation for adjusted projections.

This module is deliberately downstream of the baseball projection engine.
It may rank and label projections but must never alter Drew Floor/Median/Ceiling.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HIGH_OWNERSHIP_SHARE = 0.50
LOW_OWNERSHIP_SHARE = 0.30
ELITE_RESEARCH_SHARE = 0.30
CHEAP_SALARY_SHARE = 0.25
MIN_SUPPORTED_CONFIDENCE = 60.0
FRAGILE_CONFIDENCE = 50.0
FRAGILE_AGREEMENT_RATIO = 0.50
OVERHEATED_RANK_GAP = 3.0


def _rank_desc(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(method="min", ascending=False, na_option="bottom").astype(int)


def _rank_asc(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(method="min", ascending=True, na_option="bottom").astype(int)


def _agreement_ratio(row: pd.Series) -> float:
    count = pd.to_numeric(pd.Series([row.get("research_agreement_count")]), errors="coerce").iloc[0]
    total = pd.to_numeric(pd.Series([row.get("research_agreement_total")]), errors="coerce").iloc[0]
    if pd.isna(count) or pd.isna(total) or total <= 0:
        return 0.0
    return float(count) / float(total)


def classify_projection_chalk(adjusted_df: pd.DataFrame) -> pd.DataFrame:
    """Add slate-relative ownership/research classifications without mutation.

    Rank 1 is strongest/highest for adjusted Median, adjusted Ceiling and
    ownership. ``research_rank_gap`` is positive when popularity outruns the
    average adjusted projection rank.
    """
    if adjusted_df is None:
        return pd.DataFrame()
    out = adjusted_df.copy().reset_index(drop=True)
    if out.empty:
        for column in (
            "adjusted_median_rank",
            "adjusted_ceiling_rank",
            "ownership_rank",
            "research_rank_gap",
            "chalk_classification",
            "chalk_reason",
        ):
            out[column] = pd.Series(dtype="object")
        return out

    required = ["drew_median", "drew_ceiling", "ownership"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Adjusted projection chalk missing required columns: {', '.join(missing)}")

    n = len(out)
    high_own_cut = max(1, math.ceil(n * HIGH_OWNERSHIP_SHARE))
    low_own_cut = max(1, math.floor(n * (1.0 - LOW_OWNERSHIP_SHARE)) + 1)
    elite_cut = max(1, math.ceil(n * ELITE_RESEARCH_SHARE))
    cheap_cut = max(1, math.ceil(n * CHEAP_SALARY_SHARE))

    out["adjusted_median_rank"] = _rank_desc(out["drew_median"])
    out["adjusted_ceiling_rank"] = _rank_desc(out["drew_ceiling"])
    out["ownership_rank"] = _rank_desc(out["ownership"])
    out["research_rank_gap"] = (
        (out["adjusted_median_rank"] + out["adjusted_ceiling_rank"]) / 2.0
        - out["ownership_rank"]
    )
    salary_rank = _rank_asc(out.get("salary", pd.Series(np.nan, index=out.index)))

    labels: list[str] = []
    reasons: list[str] = []
    for idx, row in out.iterrows():
        own_rank = int(row["ownership_rank"])
        med_rank = int(row["adjusted_median_rank"])
        ceil_rank = int(row["adjusted_ceiling_rank"])
        gap = float(row["research_rank_gap"])
        confidence = float(pd.to_numeric(pd.Series([row.get("research_confidence", 0)]), errors="coerce").fillna(0).iloc[0])
        agreement = _agreement_ratio(row)
        high_owned = own_rank <= high_own_cut
        low_owned = own_rank >= low_own_cut
        elite_median = med_rank <= elite_cut
        elite_ceiling = ceil_rank <= elite_cut
        cheap = int(salary_rank.iloc[idx]) <= cheap_cut

        if high_owned and (confidence < FRAGILE_CONFIDENCE or agreement < FRAGILE_AGREEMENT_RATIO):
            label = "FRAGILE CHALK"
            reason = "Popularity is supported by too little independent research agreement/confidence."
        elif high_owned and cheap and not elite_median and not elite_ceiling:
            label = "PRICE-DRIVEN CHALK"
            reason = "Low salary is attracting ownership without a matching adjusted Median/Ceiling rank."
        elif high_owned and elite_ceiling and not elite_median and confidence >= MIN_SUPPORTED_CONFIDENCE:
            label = "CEILING CHALK"
            reason = "Popularity is backed primarily by elite tournament ceiling rather than median strength."
        elif high_owned and elite_median and elite_ceiling and confidence >= MIN_SUPPORTED_CONFIDENCE:
            label = "SUPPORTED CHALK"
            reason = "High ownership is supported by strong adjusted Median/Ceiling ranks and research confidence."
        elif high_owned and gap >= OVERHEATED_RANK_GAP:
            label = "OVERHEATED CHALK"
            reason = "Ownership rank materially exceeds the research-adjusted Median/Ceiling outlook."
        elif low_owned and elite_median and elite_ceiling and confidence >= MIN_SUPPORTED_CONFIDENCE:
            label = "LEVERAGE TARGET"
            reason = "Strong adjusted Median/Ceiling ranks have not been matched by field ownership."
        else:
            label = "NEUTRAL"
            reason = "Ownership and the research-adjusted outlook are broadly aligned or not extreme enough to flag."

        labels.append(label)
        reasons.append(reason)

    out["chalk_classification"] = labels
    out["chalk_reason"] = reasons
    return out
