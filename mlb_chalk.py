from __future__ import annotations

import numpy as np
import pandas as pd

TONE = {"GOOD CHALK": "green", "BAD CHALK": "red", "LEVERAGE PIVOT": "cyan", "NEUTRAL": "yellow"}


def _slate_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if np.isclose(lo, hi):
        out = pd.Series(50.0, index=series.index, dtype=float)
        out[values.isna()] = np.nan
        return out
    return (values - lo) / (hi - lo) * 100.0


def _classify(research_pct: float, own_pct: float) -> tuple[str, str]:
    if pd.isna(research_pct) or pd.isna(own_pct):
        return "NEUTRAL", "Insufficient research or ownership data for a chalk call."
    gap = research_pct - own_pct
    if research_pct >= 70 and own_pct >= 70 and gap >= -15:
        return "GOOD CHALK", "Popularity is supported by strong slate-relative research."
    if own_pct >= 70 and (research_pct < 55 or gap <= -20):
        return "BAD CHALK", "Field ownership materially exceeds the supporting research profile."
    if research_pct >= 70 and own_pct <= 45 and gap >= 25:
        return "LEVERAGE PIVOT", "Strong research is not matched by field ownership."
    return "NEUTRAL", "Research and ownership are not far enough apart for a strong chalk classification."


def _apply(df: pd.DataFrame, research_col: str, ownership_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else [])
    out = df.copy()
    if research_col not in out:
        out[research_col] = np.nan
    if ownership_col not in out:
        out[ownership_col] = np.nan
    out["Research Percentile"] = _slate_percentile(out[research_col])
    out["Ownership Percentile"] = _slate_percentile(out[ownership_col])
    out["Chalk Gap"] = out["Research Percentile"] - out["Ownership Percentile"]
    labels, reasons = [], []
    for rp, op in zip(out["Research Percentile"], out["Ownership Percentile"]):
        label, reason = _classify(rp, op)
        labels.append(label); reasons.append(reason)
    out["Chalk Label"] = labels
    out["Chalk Tone"] = [TONE[x] for x in labels]
    out["Chalk Reason"] = reasons
    return out


def classify_stack_chalk(stack_edges: pd.DataFrame) -> pd.DataFrame:
    research_col = "Research Strength Score" if stack_edges is not None and "Research Strength Score" in stack_edges.columns else "Stack Edge Score"
    return _apply(stack_edges, research_col, "Team Ownership")


def classify_pitcher_chalk(pitchers: pd.DataFrame) -> pd.DataFrame:
    return _apply(pitchers, "Pitcher Quality", "Projected Ownership")
