from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from modules.mlb_pitcher_portfolio import HITTER_ALIASES


def _norm_col(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " ").replace("-", " "))


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    normalized = {_norm_col(c): c for c in out.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in HITTER_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            hit = normalized.get(_norm_col(alias))
            if hit is not None:
                rename[hit] = canonical
                break
    return out.rename(columns=rename)


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None", "N/A", "NA"]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _percent(series: pd.Series) -> pd.Series:
    out = _numeric(series)
    obs = out.dropna()
    if len(obs) and obs.max() <= 1.5 and obs.quantile(0.95) <= 1.5:
        out = out * 100.0
    return out


def _slate_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    s = _numeric(series)
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.empty:
        return out
    lo, hi = float(valid.min()), float(valid.max())
    if np.isclose(lo, hi):
        out.loc[s.notna()] = 50.0
    else:
        out = (s - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        out = 100.0 - out
    return out.clip(0, 100)


def _weighted_available(parts: list[tuple[pd.Series, float]], index: pd.Index) -> pd.Series:
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for series, weight in parts:
        s = pd.to_numeric(series, errors="coerce")
        mask = s.notna()
        num.loc[mask] += s.loc[mask] * weight
        den.loc[mask] += weight
    return (num / den.replace(0, np.nan)).clip(0, 100)


def build_hitter_board(
    player_roo_df: pd.DataFrame | None,
    stack_edges: pd.DataFrame,
    approved_threshold: float = 70.0,
) -> pd.DataFrame:
    if player_roo_df is None or player_roo_df.empty:
        return pd.DataFrame()
    hit = _canonicalize(player_roo_df)
    if "player" not in hit or "team" not in hit:
        return pd.DataFrame()
    if "position" in hit:
        pos = hit["position"].astype(str).str.upper().str.strip()
        hit = hit.loc[~pos.str.fullmatch(r"P|SP|SP1|SP2", na=False)].copy()
    hit = hit.reset_index(drop=True)
    hit["team"] = hit["team"].astype(str).str.upper().str.strip()

    projection = _numeric(hit["projection"]) if "projection" in hit else pd.Series(np.nan, index=hit.index)
    ceiling = _numeric(hit["ceiling"]) if "ceiling" in hit else pd.Series(np.nan, index=hit.index)
    salary = _numeric(hit["salary"]) if "salary" in hit else pd.Series(np.nan, index=hit.index)
    ownership = _percent(hit["projected_ownership"]) if "projected_ownership" in hit else pd.Series(np.nan, index=hit.index)

    projection_score = _slate_score(projection)
    ceiling_score = _slate_score(ceiling)
    value_raw = projection / salary.replace(0, np.nan) * 1000.0
    value_score = _slate_score(value_raw)
    leverage_score = _slate_score(ownership, higher_is_better=False)

    stack_map = {}
    score_col = "Research Strength Score" if stack_edges is not None and "Research Strength Score" in stack_edges.columns else "Stack Edge Score"
    if stack_edges is not None and not stack_edges.empty and "Team" in stack_edges and score_col in stack_edges:
        stack_map = dict(zip(stack_edges["Team"].astype(str).str.upper().str.strip(), pd.to_numeric(stack_edges[score_col], errors="coerce")))
    stack_fit = hit["team"].map(stack_map).astype(float)

    hitter_score = _weighted_available([
        (projection_score, 0.30),
        (ceiling_score, 0.30),
        (value_score, 0.20),
        (stack_fit, 0.20),
    ], hit.index)

    approved = stack_fit >= float(approved_threshold)
    one_off = (
        (hitter_score >= 75.0)
        & (ceiling_score >= 75.0)
        & (~approved.fillna(False))
        & ((leverage_score >= 65.0) | (value_score >= 70.0))
    )

    reasons: list[str] = []
    for i in hit.index:
        if bool(approved.iloc[i]):
            reasons.append("Approved stack hitter backed by team-level research.")
        elif bool(one_off.iloc[i]):
            reasons.append("Individual ceiling/value/leverage clears the higher one-off bar despite a non-core stack.")
        else:
            reasons.append("Research option; does not currently clear approved-stack or one-off target gates.")

    out = pd.DataFrame({
        "Player": hit["player"].astype(str),
        "Team": hit["team"],
        "Position": hit["position"] if "position" in hit else "",
        "Salary": salary,
        "Projection": projection,
        "Ceiling": ceiling,
        "Projected Ownership": ownership,
        "Projection Score": projection_score,
        "Hitter Score": hitter_score,
        "Ceiling Score": ceiling_score,
        "Value Score": value_score,
        "Leverage Score": leverage_score,
        "Stack Fit": stack_fit,
        "Approved Stack": approved.fillna(False).astype(bool),
        "One-Off Target": one_off.fillna(False).astype(bool),
        "Hitter Reason": reasons,
    })
    return out.sort_values(["Approved Stack", "One-Off Target", "Hitter Score"], ascending=[False, False, False], na_position="last").reset_index(drop=True)


def filter_hitter_mode(board: pd.DataFrame, mode: str) -> pd.DataFrame:
    if board is None or board.empty:
        return pd.DataFrame(columns=list(board.columns) if isinstance(board, pd.DataFrame) else [])
    if mode == "Approved Stacks":
        return board.loc[board["Approved Stack"].fillna(False)].copy()
    if mode == "One-Off Targets":
        return board.loc[board["One-Off Target"].fillna(False)].copy()
    if mode == "All":
        return board.copy()
    raise ValueError(f"Unknown hitters mode: {mode}")
