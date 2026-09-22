from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _norm_col(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " ").replace("-", " "))


def _norm_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def _detect_pitcher_columns(lineups: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in lineups.columns:
        norm = _norm_col(col)
        compact = re.sub(r"\s+", "", norm).upper()
        if "pitcher" in norm or re.fullmatch(r"P\d*", compact) or re.fullmatch(r"SP\d*", compact):
            cols.append(col)
    return cols


def _detect_stack_column(lineups: pd.DataFrame) -> str | None:
    preferred = ["primary stack", "team stack", "stack"]
    normalized = {_norm_col(c): c for c in lineups.columns}
    for key in preferred:
        if key in normalized:
            return normalized[key]
    return next((c for c in lineups.columns if "stack" in _norm_col(c)), None)


def _exposure_table(values: list[str], label: str, total_lineups: int) -> pd.DataFrame:
    if not values or total_lineups <= 0:
        return pd.DataFrame(columns=[label, "Lineups", "Exposure %"])
    # A player appearing twice in the same lineup should count once for lineup exposure.
    counts = pd.Series(values, dtype="object").value_counts()
    return pd.DataFrame({label: counts.index, "Lineups": counts.values, "Exposure %": counts.values / total_lineups * 100.0})


def build_portfolio_exposures(lineups: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if lineups is None or lineups.empty:
        return {
            "pitchers": pd.DataFrame(columns=["Pitcher", "Lineups", "Exposure %"]),
            "stacks": pd.DataFrame(columns=["Stack", "Lineups", "Exposure %"]),
        }
    pcols = _detect_pitcher_columns(lineups)
    pitcher_rows: list[str] = []
    for _, row in lineups.iterrows():
        seen: set[str] = set()
        for col in pcols:
            val = str(row.get(col, "")).strip()
            key = _norm_name(val)
            if key and key not in seen:
                pitcher_rows.append(val)
                seen.add(key)
    stack_col = _detect_stack_column(lineups)
    stack_values = lineups[stack_col].dropna().astype(str).tolist() if stack_col else []
    return {
        "pitchers": _exposure_table(pitcher_rows, "Pitcher", len(lineups)).sort_values("Exposure %", ascending=False).reset_index(drop=True),
        "stacks": _exposure_table(stack_values, "Stack", len(lineups)).sort_values("Exposure %", ascending=False).reset_index(drop=True),
    }


def _lookup_map(df: pd.DataFrame, key_col: str, value_col: str) -> dict[str, float]:
    if df is None or df.empty or key_col not in df or value_col not in df:
        return {}
    keys = df[key_col].map(_norm_name)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    return {k: float(v) for k, v in zip(keys, vals) if k and pd.notna(v)}


def _pitchers_for_row(row: pd.Series, pcols: list[str]) -> list[str]:
    vals: list[str] = []
    for col in pcols:
        name = str(row.get(col, "")).strip()
        if _norm_name(name):
            vals.append(name)
    return vals


def triage_lineups(
    lineups: pd.DataFrame,
    stack_edges: pd.DataFrame,
    pitcher_analysis: pd.DataFrame,
    stack_chalk: pd.DataFrame | None = None,
    *,
    keep_min: float = 65.0,
    review_min: float = 45.0,
) -> pd.DataFrame:
    if lineups is None or lineups.empty:
        return pd.DataFrame(columns=["_Lineup ID", "Triage Score", "Triage Status", "Triage Reason", "Triage Confidence"])

    source = lineups.copy(deep=True)
    triaged = source.copy(deep=True)
    triaged.insert(0, "_Lineup ID", np.arange(len(source), dtype=int))
    pcols = _detect_pitcher_columns(source)
    stack_col = _detect_stack_column(source)

    stack_map = _lookup_map(stack_edges, "Team", "Stack Edge Score")
    quality_map = _lookup_map(pitcher_analysis, "Pitcher", "Pitcher Quality")
    risk_map = _lookup_map(pitcher_analysis, "Pitcher", "Portfolio Risk")
    rec_map: dict[str, str] = {}
    if pitcher_analysis is not None and not pitcher_analysis.empty and "Pitcher" in pitcher_analysis and "Recommendation" in pitcher_analysis:
        rec_map = {_norm_name(k): str(v) for k, v in zip(pitcher_analysis["Pitcher"], pitcher_analysis["Recommendation"]) if _norm_name(k)}

    chalk_map: dict[str, str] = {}
    if stack_chalk is not None and not stack_chalk.empty and "Team" in stack_chalk and "Chalk Label" in stack_chalk:
        chalk_map = {_norm_name(k): str(v) for k, v in zip(stack_chalk["Team"], stack_chalk["Chalk Label"])}

    portfolio = build_portfolio_exposures(source)
    pitcher_exposure = {_norm_name(r["Pitcher"]): float(r["Exposure %"]) for _, r in portfolio["pitchers"].iterrows()}
    stack_exposure = {_norm_name(r["Stack"]): float(r["Exposure %"]) for _, r in portfolio["stacks"].iterrows()}

    scores: list[float] = []
    statuses: list[str] = []
    reasons: list[str] = []
    confidences: list[float] = []
    chalk_profiles: list[str] = []

    for _, row in source.iterrows():
        score = 50.0
        parts: list[str] = []
        available = 0
        expected = 3

        stack = str(row.get(stack_col, "")).strip() if stack_col else ""
        stack_key = _norm_name(stack)
        stack_edge = stack_map.get(stack_key, np.nan)
        if pd.notna(stack_edge):
            available += 1
            score += (stack_edge - 50.0) * 0.30
            parts.append(f"stack research {stack_edge:.0f}")

        pitchers = _pitchers_for_row(row, pcols)
        qvals = [quality_map.get(_norm_name(p), np.nan) for p in pitchers]
        qvals = [v for v in qvals if pd.notna(v)]
        rvals = [risk_map.get(_norm_name(p), np.nan) for p in pitchers]
        rvals = [v for v in rvals if pd.notna(v)]
        mean_quality = float(np.mean(qvals)) if qvals else np.nan
        mean_risk = float(np.mean(rvals)) if rvals else np.nan
        if pd.notna(mean_quality):
            available += 1
            score += (mean_quality - 50.0) * 0.25
            parts.append(f"pitcher quality {mean_quality:.0f}")
        if pd.notna(mean_risk):
            available += 1
            score -= max(mean_risk - 50.0, 0.0) * 0.20
            parts.append(f"pitcher risk {mean_risk:.0f}")

        chalk_label = chalk_map.get(stack_key, "NEUTRAL")
        chalk_adjustment = {"GOOD CHALK": 3.0, "BAD CHALK": -7.0, "LEVERAGE PIVOT": 7.0, "NEUTRAL": 0.0}.get(chalk_label, 0.0)
        score += chalk_adjustment
        chalk_profiles.append(chalk_label)
        if chalk_adjustment:
            parts.append(chalk_label.lower())

        # Portfolio concentration is intentionally bounded: it can refine triage, never override research.
        portfolio_adjustment = 0.0
        sx = stack_exposure.get(stack_key, np.nan)
        if pd.notna(sx) and sx >= 40:
            portfolio_adjustment -= min(10.0, (sx - 40.0) * 0.25)
        for pitcher in pitchers:
            px = pitcher_exposure.get(_norm_name(pitcher), np.nan)
            if pd.notna(px) and px >= 50:
                portfolio_adjustment -= min(5.0, (px - 50.0) * 0.10)
        portfolio_adjustment = float(np.clip(portfolio_adjustment, -10.0, 10.0))
        score += portfolio_adjustment
        if portfolio_adjustment < -0.1:
            parts.append("portfolio concentration trim")

        bad_recs = [rec_map.get(_norm_name(p), "") for p in pitchers]
        if any(r in {"REMOVE CANDIDATE", "AVOID"} for r in bad_recs):
            score -= 8.0
            parts.append("contains remove/avoid pitcher")

        score = float(np.clip(score, 0.0, 100.0))
        if score >= keep_min:
            status = "KEEP"
        elif score >= review_min:
            status = "REVIEW"
        else:
            status = "CUT CANDIDATE"
        confidence = available / expected * 100.0
        if not parts:
            parts.append("limited research coverage")
        scores.append(score)
        statuses.append(status)
        reasons.append("; ".join(parts))
        confidences.append(confidence)

    triaged["Triage Score"] = scores
    triaged["Triage Status"] = statuses
    triaged["Triage Reason"] = reasons
    triaged["Triage Confidence"] = confidences
    triaged["Chalk Profile"] = chalk_profiles
    return triaged


def apply_lineup_filter(lineups: pd.DataFrame, selected_ids: list[int]) -> pd.DataFrame:
    if lineups is None or lineups.empty or not selected_ids:
        return lineups.copy() if isinstance(lineups, pd.DataFrame) else pd.DataFrame()
    ids = set(int(x) for x in selected_ids)
    mask = pd.Series([i not in ids for i in range(len(lineups))], index=lineups.index)
    return lineups.loc[mask].copy()


def _merge_exposure(before: pd.DataFrame, after: pd.DataFrame, label: str) -> pd.DataFrame:
    left = before[[label, "Exposure %"]].rename(columns={"Exposure %": "Before %"}) if not before.empty else pd.DataFrame(columns=[label, "Before %"])
    right = after[[label, "Exposure %"]].rename(columns={"Exposure %": "After %"}) if not after.empty else pd.DataFrame(columns=[label, "After %"])
    merged = left.merge(right, on=label, how="outer").fillna(0.0)
    merged["Delta"] = merged["After %"] - merged["Before %"]
    return merged.sort_values("Delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def preview_lineup_filter(lineups: pd.DataFrame, triaged: pd.DataFrame, selected_ids: list[int]) -> dict[str, object]:
    remaining = apply_lineup_filter(lineups, selected_ids)
    before = build_portfolio_exposures(lineups)
    after = build_portfolio_exposures(remaining)
    return {
        "lineups_before": int(len(lineups)),
        "lineups_removed": int(len(lineups) - len(remaining)),
        "lineups_after": int(len(remaining)),
        "remaining_pool": remaining,
        "stack_exposure": _merge_exposure(before["stacks"], after["stacks"], "Stack"),
        "pitcher_exposure": _merge_exposure(before["pitchers"], after["pitchers"], "Pitcher"),
    }
