from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
import pandas as pd


@dataclass
class PitcherPortfolioConfig:
    overexposure_delta_pp: float = 8.0
    underexposure_delta_pp: float = 6.0
    overexposure_ratio: float = 1.50
    low_optimal_pct: float = 5.0
    weak_pitcher_percentile: float = 75.0
    strong_offense_percentile: float = 75.0
    low_projection_percentile: float = 25.0
    poor_value_score_max: float = 38.0
    reduce_risk_min: float = 55.0
    remove_risk_min: float = 70.0
    avoid_risk_min: float = 84.0
    remove_min_negative_signals: int = 3
    avoid_min_negative_signals: int = 4
    remove_min_signal_groups: int = 2
    avoid_min_signal_groups: int = 3
    core_quality_min: float = 72.0
    core_value_min: float = 62.0
    core_matchup_max: float = 45.0
    chalk_ownership_percentile: float = 70.0


ALIASES = {
    "pitcher": ["pitcher", "player", "name", "pitcher name", "player name", "sp"],
    "position": ["position", "pos", "roster position"],
    "team": ["team", "tm", "pitcher team"],
    "opponent": ["opponent", "opp", "opposing team"],
    "salary": ["salary", "sal", "dk salary", "draftkings salary"],
    "projection": ["projection", "proj", "median", "projected points", "fpts"],
    "ceiling": ["ceiling", "ceil", "upside", "p75 ceiling", "source ceiling"],
    "projected_ownership": ["projected ownership", "proj own", "projown", "ownership", "own%", "own %", "own"],
    "weighted_ownership": ["weighted ownership", "weighted own", "weightedown"],
    "portfolio_exposure": ["portfolio exposure", "exposure", "exposure%", "portfolio %"],
    "optimal_pct": ["optimal %", "optimal%", "optimal pct", "optimal rate", "optimizer optimal %"],
    "sim_win_pct": ["sim win %", "sim win", "win %", "win%", "top_finish", "top finish"],
    "value": ["value", "pts/$", "points per dollar", "salary value"],
    "lineup_count": ["lineup count", "lineups", "optimizer lineups", "number of lineups"],
    "weighted_availability": ["weighted availability", "weighted avail", "weightedavailability", "weighted true avg", "weightedtrueavg"],
    "weighted_hwsr": ["weighted hwsr", "weightedhwsr", "hwsr"],
    "hard_per_swing": ["hardperswing", "hard per swing", "hard_per_swing", "hardperswing%"],
    "opponent_stack_score": ["opposing stack score", "opponent stack score", "stack score"],
    "opponent_raw_baseball_score": ["opposing raw baseball score", "opponent raw baseball score", "raw baseball score"],
    "opponent_dfs_stack_score": ["opposing dfs stack score", "opponent dfs stack score", "dfs stack score"],
}

TEAM_RESEARCH_ALIASES = {
    "team": ["team", "tm"],
    "hard_per_swing": ["hardperswing", "hard per swing", "hard_per_swing", "hardperswing%"],
    "stack_score": ["stack score", "drew stack score", "opposing stack score"],
    "raw_baseball_score": ["raw baseball score", "raw score", "baseball score"],
    "dfs_stack_score": ["dfs stack score", "dfs score"],
}

HITTER_ALIASES = {
    "player": ["player", "name", "hitter"],
    "team": ["team", "tm"],
    "position": ["position", "pos", "roster position"],
    "projection": ["projection", "proj", "median", "fpts"],
    "ceiling": ["ceiling", "ceil", "upside", "p75 ceiling", "source ceiling"],
    "salary": ["salary", "sal", "dk salary"],
    "projected_ownership": ["projected ownership", "proj own", "projown", "ownership", "own%", "own %", "own"],
    "weighted_ownership": ["weighted ownership", "weighted own", "weightedown"],
    "optimal_pct": ["optimal %", "optimal%", "optimal rate"],
}


def _norm_col(value: Any) -> str:
    s = str(value).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _canonicalize(df: pd.DataFrame, aliases: Dict[str, Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    normalized = {_norm_col(c): c for c in out.columns}
    rename: Dict[str, str] = {}
    for canonical, options in aliases.items():
        if canonical in out.columns:
            continue
        for option in options:
            hit = normalized.get(_norm_col(option))
            if hit is not None:
                rename[hit] = canonical
                break
    return out.rename(columns=rename)


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None", "N/A", "NA"]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _percent(series: pd.Series) -> pd.Series:
    out = _numeric(series)
    obs = out.dropna()
    if len(obs) and obs.max() <= 1.5 and obs.quantile(0.95) <= 1.5:
        out = out * 100.0
    return out


def _rank(series: pd.Series) -> pd.Series:
    s = _numeric(series)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.sum() == 1:
        out.loc[mask] = 50.0
    elif mask.sum() > 1:
        out.loc[mask] = s.loc[mask].rank(method="average", pct=True) * 100.0
    return out


def _weighted_score(index: pd.Index, *parts: Tuple[pd.Series, float]) -> Tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=index)
    denominator = pd.Series(0.0, index=index)
    total_weight = sum(max(weight, 0.0) for _, weight in parts)
    for series, weight in parts:
        weight = max(weight, 0.0)
        if weight == 0:
            continue
        s = pd.to_numeric(series, errors="coerce")
        mask = s.notna()
        numerator.loc[mask] += s.loc[mask] * weight
        denominator.loc[mask] += weight
    score = numerator / denominator.replace(0, np.nan)
    coverage = denominator / total_weight if total_weight else pd.Series(0.0, index=index)
    return score.clip(0, 100), coverage.clip(0, 1)


def _first_notna(*values: float) -> float:
    for value in values:
        if pd.notna(value):
            return float(value)
    return np.nan


def _suggested_exposure_range(
    recommendation: str,
    projected_own: float,
    optimal: float,
    portfolio_risk: float,
    chalk_risk: str,
) -> Tuple[float, float, float]:
    anchors = [x for x in (projected_own, optimal) if pd.notna(x)]
    anchor = max(anchors) if anchors else np.nan
    if pd.isna(anchor):
        if recommendation == "AVOID":
            return 0.0, 0.0, 0.0
        return np.nan, np.nan, np.nan

    if recommendation == "CORE":
        target = min(55.0, max(anchor, anchor * 1.20))
        max_exp = min(70.0, max(target + 8.0, anchor * 1.50))
        min_exp = max(0.0, target - 12.0)
    elif recommendation == "KEEP":
        target = min(40.0, anchor)
        max_exp = min(50.0, max(anchor * 1.20, target + 5.0))
        min_exp = max(0.0, target - 10.0)
    elif recommendation == "REDUCE":
        severity = 0.60 if chalk_risk == "FRAGILE CHALK" else 0.75
        target = max(2.0, anchor * severity)
        max_exp = max(target, anchor * min(0.90, severity + 0.10))
        min_exp = 0.0
    elif recommendation == "REMOVE CANDIDATE":
        target = min(8.0, anchor * 0.35)
        max_exp = min(12.0, max(target, anchor * 0.50))
        min_exp = 0.0
    else:  # AVOID
        target = 0.0
        max_exp = 0.0
        min_exp = 0.0

    if pd.notna(portfolio_risk) and portfolio_risk >= 90 and recommendation in {"REDUCE", "REMOVE CANDIDATE"}:
        max_exp = min(max_exp, 8.0)
        target = min(target, max_exp)
    return round(min_exp, 1), round(target, 1), round(max_exp, 1)


def analyze_pitcher_portfolio(
    pitcher_df: pd.DataFrame,
    config: Optional[PitcherPortfolioConfig] = None,
) -> pd.DataFrame:
    cfg = config or PitcherPortfolioConfig()
    df = _canonicalize(pitcher_df, ALIASES).reset_index(drop=True)
    if "pitcher" not in df.columns:
        raise ValueError("Pitcher portfolio data must contain a pitcher/player name column.")

    for col in [
        "salary", "projection", "ceiling", "value", "lineup_count",
        "weighted_availability", "weighted_hwsr", "hard_per_swing",
        "opponent_stack_score", "opponent_raw_baseball_score", "opponent_dfs_stack_score",
    ]:
        if col in df.columns:
            df[col] = _numeric(df[col])
    for col in ["projected_ownership", "weighted_ownership", "portfolio_exposure", "optimal_pct", "sim_win_pct"]:
        if col in df.columns:
            df[col] = _percent(df[col])

    if "value" not in df.columns:
        df["value"] = np.nan
    if "projection" in df.columns and "salary" in df.columns:
        calculated_value = df["projection"] / (df["salary"] / 1000.0).replace(0, np.nan)
        df["value"] = df["value"].where(df["value"].notna(), calculated_value)

    idx = df.index
    nan = pd.Series(np.nan, index=idx, dtype=float)

    # User-confirmed directions: higher Weighted Availability/HWSr are worse.
    avail_bad = _rank(df["weighted_availability"]) if "weighted_availability" in df else nan.copy()
    hwsr_bad = _rank(df["weighted_hwsr"]) if "weighted_hwsr" in df else nan.copy()
    pitcher_quality, quality_cov = _weighted_score(idx, (100 - avail_bad, 0.50), (100 - hwsr_bad, 0.50))
    pitcher_weakness = 100 - pitcher_quality

    hard = _rank(df["hard_per_swing"]) if "hard_per_swing" in df else nan.copy()
    stack = _rank(df["opponent_stack_score"]) if "opponent_stack_score" in df else nan.copy()
    raw = _rank(df["opponent_raw_baseball_score"]) if "opponent_raw_baseball_score" in df else nan.copy()
    dfs_stack = _rank(df["opponent_dfs_stack_score"]) if "opponent_dfs_stack_score" in df else nan.copy()
    opponent_offense, offense_cov = _weighted_score(
        idx, (hard, 0.30), (stack, 0.25), (raw, 0.20), (dfs_stack, 0.25)
    )

    interaction = np.sqrt(pitcher_weakness * opponent_offense)
    matchup_risk, matchup_cov = _weighted_score(
        idx, (opponent_offense, 0.50), (pitcher_weakness, 0.35), (interaction, 0.15)
    )

    projection_rank = _rank(df["projection"]) if "projection" in df else nan.copy()
    ceiling_rank = _rank(df["ceiling"]) if "ceiling" in df else nan.copy()
    value_rank = _rank(df["value"]) if "value" in df else nan.copy()
    sim_win_rank = _rank(df["sim_win_pct"]) if "sim_win_pct" in df else nan.copy()
    dfs_value, value_cov = _weighted_score(
        idx,
        (projection_rank, 0.34),
        (ceiling_rank, 0.24),
        (value_rank, 0.30),
        (sim_win_rank, 0.12),
    )

    pay_up_justification, _ = _weighted_score(
        idx,
        (pitcher_quality, 0.35),
        (dfs_value, 0.35),
        (100 - matchup_risk, 0.30),
    )

    projected_own = df["projected_ownership"] if "projected_ownership" in df else nan.copy()
    weighted_own = df["weighted_ownership"] if "weighted_ownership" in df else nan.copy()
    exposure = df["portfolio_exposure"] if "portfolio_exposure" in df else nan.copy()
    optimal = df["optimal_pct"] if "optimal_pct" in df else nan.copy()

    ownership_score, own_cov = _weighted_score(
        idx,
        (_rank(projected_own), 0.45),
        (_rank(weighted_own), 0.20),
        (_rank(exposure), 0.35),
    )

    baseline = pd.concat([projected_own.rename("own"), optimal.rename("optimal")], axis=1).max(axis=1, skipna=True)
    baseline.loc[projected_own.isna() & optimal.isna()] = np.nan
    exposure_delta = exposure - baseline
    exposure_ratio = exposure / baseline.where(baseline > 0)
    overexposed = exposure.notna() & baseline.notna() & (
        (exposure_delta >= cfg.overexposure_delta_pp) | (exposure_ratio >= cfg.overexposure_ratio)
    )
    underexposed = exposure.notna() & baseline.notna() & (exposure_delta <= -cfg.underexposure_delta_pp)

    overexp_risk = pd.Series(np.nan, index=idx, dtype=float)
    compare = exposure.notna() & baseline.notna()
    if compare.any():
        overexp_risk.loc[compare] = (exposure_delta.loc[compare].clip(lower=0) / max(cfg.overexposure_delta_pp * 2, 1) * 100).clip(0, 100)
        ratio_component = ((exposure_ratio.loc[compare] - 1.0) / max(cfg.overexposure_ratio - 1.0, 0.1) * 60).clip(0, 100)
        overexp_risk.loc[compare] = np.maximum(overexp_risk.loc[compare].fillna(0), ratio_component.fillna(0))

    optimal_weakness = 100 - _rank(optimal)
    portfolio_risk, risk_cov = _weighted_score(
        idx,
        (matchup_risk, 0.35),
        (pitcher_weakness, 0.20),
        (100 - dfs_value, 0.15),
        (optimal_weakness, 0.10),
        (overexp_risk, 0.20),
    )

    ownership_rank = _rank(projected_own)
    chalk_risk_values: List[str] = []
    recommendations: List[str] = []
    reasons: List[str] = []
    neg_counts: List[int] = []
    group_counts: List[int] = []
    mins: List[float] = []
    targets: List[float] = []
    maxes: List[float] = []

    confidence = (
        quality_cov * 0.30
        + offense_cov * 0.25
        + value_cov * 0.25
        + own_cov * 0.10
        + optimal.notna().astype(float) * 0.10
    ) * 100.0

    for i in idx:
        negatives: List[str] = []
        positives: List[str] = []
        groups = set()

        if pd.notna(hwsr_bad[i]) and hwsr_bad[i] >= cfg.weak_pitcher_percentile:
            negatives.append(f"Weighted HWSr ranks in the worst {max(1, round(100-hwsr_bad[i]))}% of slate pitchers")
            groups.add("pitcher research")
        elif pd.notna(hwsr_bad[i]) and hwsr_bad[i] <= 30:
            positives.append("Weighted HWSr is strong relative to the slate")

        if pd.notna(avail_bad[i]) and avail_bad[i] >= cfg.weak_pitcher_percentile:
            negatives.append(f"Weighted Availability ranks in the worst {max(1, round(100-avail_bad[i]))}% of slate pitchers")
            groups.add("pitcher research")
        elif pd.notna(avail_bad[i]) and avail_bad[i] <= 30:
            positives.append("Weighted Availability is strong relative to the slate")

        if pd.notna(hard[i]) and hard[i] >= cfg.strong_offense_percentile:
            negatives.append(f"Opponent HardPerSwing ranks in the top {max(1, round(100-hard[i]))}% of the slate")
            groups.add("opponent offense")
        if pd.notna(opponent_offense[i]) and opponent_offense[i] >= cfg.strong_offense_percentile:
            negatives.append(f"Opponent Offense Score is dangerous ({opponent_offense[i]:.0f}/100)")
            groups.add("opponent offense")
        elif pd.notna(opponent_offense[i]) and opponent_offense[i] <= 35:
            positives.append(f"Opponent Offense Score is favorable ({opponent_offense[i]:.0f}/100)")

        if pd.notna(dfs_value[i]) and dfs_value[i] <= cfg.poor_value_score_max:
            negatives.append(f"DFS Value Score is weak ({dfs_value[i]:.0f}/100)")
            groups.add("dfs value")
        elif pd.notna(dfs_value[i]) and dfs_value[i] >= 65:
            positives.append(f"DFS Value Score is strong ({dfs_value[i]:.0f}/100)")

        if pd.notna(projection_rank[i]) and projection_rank[i] <= cfg.low_projection_percentile:
            negatives.append("Projection ranks in the bottom quarter of slate pitchers")
            groups.add("dfs value")

        if pd.notna(optimal[i]) and optimal[i] <= cfg.low_optimal_pct:
            negatives.append(f"Optimizer Optimal Rate is only {optimal[i]:.1f}%")
            groups.add("optimizer/exposure")

        if bool(overexposed[i]):
            own_txt = f"{projected_own[i]:.1f}%" if pd.notna(projected_own[i]) else "missing"
            opt_txt = f"{optimal[i]:.1f}%" if pd.notna(optimal[i]) else "missing"
            negatives.append(f"Portfolio exposure {exposure[i]:.1f}% is high versus {own_txt} projected ownership and {opt_txt} Optimal")
            groups.add("optimizer/exposure")

        chalk = "NOT CHALK"
        if pd.notna(ownership_rank[i]) and ownership_rank[i] >= cfg.chalk_ownership_percentile:
            weak_support = sum([
                pd.notna(pitcher_quality[i]) and pitcher_quality[i] < 55,
                pd.notna(matchup_risk[i]) and matchup_risk[i] > 60,
                pd.notna(dfs_value[i]) and dfs_value[i] < 50,
                pd.notna(optimal[i]) and pd.notna(projected_own[i]) and optimal[i] + 5 < projected_own[i],
            ])
            if weak_support >= 2:
                chalk = "FRAGILE CHALK"
                negatives.append("High ownership is not well supported by pitcher quality, matchup, value, or Optimal rate")
                groups.add("optimizer/exposure")
            elif weak_support == 1:
                chalk = "CHALK WATCH"
            else:
                chalk = "SUPPORTED CHALK"
        chalk_risk_values.append(chalk)

        neg_count = len(negatives)
        group_count = len(groups)
        rec = "KEEP"
        if (
            pd.notna(portfolio_risk[i])
            and portfolio_risk[i] >= cfg.avoid_risk_min
            and neg_count >= cfg.avoid_min_negative_signals
            and group_count >= cfg.avoid_min_signal_groups
            and confidence[i] >= 65
            and pd.notna(pitcher_quality[i])
            and pitcher_quality[i] <= 30
            and pd.notna(matchup_risk[i])
            and matchup_risk[i] >= 72
        ):
            rec = "AVOID"
        elif (
            pd.notna(portfolio_risk[i])
            and portfolio_risk[i] >= cfg.remove_risk_min
            and neg_count >= cfg.remove_min_negative_signals
            and group_count >= cfg.remove_min_signal_groups
            and confidence[i] >= 50
        ):
            rec = "REMOVE CANDIDATE"
        elif (
            (pd.notna(portfolio_risk[i]) and portfolio_risk[i] >= cfg.reduce_risk_min)
            or (bool(overexposed[i]) and neg_count >= 1)
            or chalk == "FRAGILE CHALK"
        ):
            rec = "REDUCE"
        elif (
            pd.notna(pitcher_quality[i]) and pitcher_quality[i] >= cfg.core_quality_min
            and pd.notna(dfs_value[i]) and dfs_value[i] >= cfg.core_value_min
            and pd.notna(matchup_risk[i]) and matchup_risk[i] <= cfg.core_matchup_max
            and not bool(overexposed[i])
            and confidence[i] >= 55
        ):
            rec = "CORE"

        min_exp, target_exp, max_exp = _suggested_exposure_range(
            rec, projected_own[i], optimal[i], portfolio_risk[i], chalk
        )
        mins.append(min_exp)
        targets.append(target_exp)
        maxes.append(max_exp)

        reason_parts = negatives[:6] if rec in {"REDUCE", "REMOVE CANDIDATE", "AVOID"} else positives[:5]
        if pd.notna(portfolio_risk[i]):
            reason_parts.append(f"Portfolio Risk {portfolio_risk[i]:.0f}/100")
        if bool(underexposed[i]):
            reason_parts.append(f"Portfolio is underexposed by {abs(exposure_delta[i]):.1f} percentage points versus the ownership/Optimal baseline")
        if confidence[i] < 80:
            reason_parts.append(f"Research is incomplete; Recommendation Confidence reduced to {confidence[i]:.0f}% and missing fields were not scored as zero")
        if not reason_parts:
            reason_parts.append("Available data does not show enough evidence for a stronger portfolio action")

        recommendations.append(rec)
        reasons.append("; ".join(reason_parts))
        neg_counts.append(neg_count)
        group_counts.append(group_count)

    out = pd.DataFrame({
        "Pitcher": df["pitcher"],
        "Team": df["team"] if "team" in df else np.nan,
        "Opponent": df["opponent"] if "opponent" in df else np.nan,
        "Salary": df["salary"] if "salary" in df else np.nan,
        "Projection": df["projection"] if "projection" in df else np.nan,
        "Ceiling": df["ceiling"] if "ceiling" in df else np.nan,
        "Projected Ownership": projected_own,
        "Weighted Ownership": weighted_own,
        "Portfolio Exposure": exposure,
        "Optimal %": optimal,
        "Exposure Delta": exposure_delta,
        "Pitcher Quality": pitcher_quality,
        "Opponent Offense Score": opponent_offense,
        "Matchup Risk": matchup_risk,
        "DFS Value": dfs_value,
        "Pay-Up Justification": pay_up_justification,
        "Ownership Score": ownership_score,
        "Portfolio Risk": portfolio_risk,
        "Chalk Risk": chalk_risk_values,
        "Suggested Min Exposure": mins,
        "Suggested Target Exposure": targets,
        "Suggested Max Exposure": maxes,
        "Recommendation Confidence": confidence,
        "Negative Signal Count": neg_counts,
        "Negative Signal Groups": group_counts,
        "Overexposed": overexposed,
        "Underexposed": underexposed,
        "Recommendation": recommendations,
        "Reason": reasons,
    })
    return out.sort_values(["Portfolio Risk", "Recommendation Confidence"], ascending=[False, False], na_position="last").reset_index(drop=True)


def build_stack_edge_table(
    team_research_df: pd.DataFrame,
    hitter_scoring_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    team = _canonicalize(team_research_df, TEAM_RESEARCH_ALIASES).copy()
    if "team" not in team.columns:
        raise ValueError("Team research must contain a Team column.")
    team["team"] = team["team"].astype(str).str.upper().str.strip()
    for c in ["hard_per_swing", "stack_score", "raw_baseball_score", "dfs_stack_score"]:
        if c in team.columns:
            team[c] = _numeric(team[c])

    agg = None
    if hitter_scoring_df is not None and not hitter_scoring_df.empty:
        hit = _canonicalize(hitter_scoring_df, HITTER_ALIASES).copy()
        if "team" in hit.columns:
            hit["team"] = hit["team"].astype(str).str.upper().str.strip()
            if "position" in hit.columns:
                pos = hit["position"].astype(str).str.upper().str.strip()
                pitcher_only = pos.str.fullmatch(r"P|SP|SP1|SP2", na=False)
                hit = hit.loc[~pitcher_only].copy()
            for c in ["projection", "ceiling", "salary"]:
                if c in hit.columns:
                    hit[c] = _numeric(hit[c])
            for c in ["projected_ownership", "weighted_ownership", "optimal_pct"]:
                if c in hit.columns:
                    hit[c] = _percent(hit[c])

            rows = []
            for team_name, group in hit.groupby("team", dropna=False):
                g = group.copy()
                if "projection" in g:
                    g = g.sort_values("projection", ascending=False)
                top = g.head(5)
                row: Dict[str, Any] = {"team": team_name}
                row["top5_projection"] = top["projection"].sum(min_count=1) if "projection" in top else np.nan
                row["top5_ceiling"] = top["ceiling"].sum(min_count=1) if "ceiling" in top else np.nan
                row["top5_salary"] = top["salary"].sum(min_count=1) if "salary" in top else np.nan
                row["avg_projected_ownership"] = top["projected_ownership"].mean() if "projected_ownership" in top else np.nan
                row["avg_weighted_ownership"] = top["weighted_ownership"].mean() if "weighted_ownership" in top else np.nan
                row["avg_optimal_pct"] = top["optimal_pct"].mean() if "optimal_pct" in top else np.nan
                rows.append(row)
            agg = pd.DataFrame(rows)
            team = team.merge(agg, on="team", how="outer")

    idx = team.index
    nan = pd.Series(np.nan, index=idx, dtype=float)
    raw_strength, raw_cov = _weighted_score(
        idx,
        (_rank(team["hard_per_swing"]) if "hard_per_swing" in team else nan.copy(), 0.25),
        (_rank(team["stack_score"]) if "stack_score" in team else nan.copy(), 0.25),
        (_rank(team["raw_baseball_score"]) if "raw_baseball_score" in team else nan.copy(), 0.25),
        (_rank(team["dfs_stack_score"]) if "dfs_stack_score" in team else nan.copy(), 0.25),
    )
    projection_score, proj_cov = _weighted_score(
        idx,
        (_rank(team["top5_projection"]) if "top5_projection" in team else nan.copy(), 0.55),
        (_rank(team["top5_ceiling"]) if "top5_ceiling" in team else nan.copy(), 0.45),
    )
    value_raw = nan.copy()
    if "top5_projection" in team and "top5_salary" in team:
        value_raw = team["top5_projection"] / (team["top5_salary"] / 1000.0).replace(0, np.nan)
    value_score = _rank(value_raw)

    ownership_raw = team["avg_projected_ownership"] if "avg_projected_ownership" in team else nan.copy()
    low_ownership_leverage = 100 - _rank(ownership_raw)
    stack_edge, edge_cov = _weighted_score(
        idx,
        (raw_strength, 0.35),
        (projection_score, 0.30),
        (value_score, 0.15),
        (low_ownership_leverage, 0.20),
    )

    recs = []
    for score, own in zip(stack_edge, ownership_raw):
        if pd.isna(score):
            recs.append("INSUFFICIENT DATA")
        elif score >= 80:
            recs.append("PRIORITY OVERWEIGHT")
        elif score >= 65:
            recs.append("OVERWEIGHT")
        elif score >= 42:
            recs.append("NEUTRAL")
        else:
            recs.append("UNDERWEIGHT")

    confidence = (raw_cov * 0.50 + proj_cov * 0.35 + value_score.notna().astype(float) * 0.10 + ownership_raw.notna().astype(float) * 0.05) * 100

    return pd.DataFrame({
        "Team": team["team"],
        "HardPerSwing": team["hard_per_swing"] if "hard_per_swing" in team else np.nan,
        "Stack Score": team["stack_score"] if "stack_score" in team else np.nan,
        "Raw Baseball Score": team["raw_baseball_score"] if "raw_baseball_score" in team else np.nan,
        "DFS Stack Score": team["dfs_stack_score"] if "dfs_stack_score" in team else np.nan,
        "Top-5 Projection": team["top5_projection"] if "top5_projection" in team else np.nan,
        "Top-5 Ceiling": team["top5_ceiling"] if "top5_ceiling" in team else np.nan,
        "Top-5 Salary": team["top5_salary"] if "top5_salary" in team else np.nan,
        "Avg Projected Ownership": ownership_raw,
        "Raw Hitting Strength": raw_strength,
        "Projection Score": projection_score,
        "Value Score": value_score,
        "Ownership Leverage": low_ownership_leverage,
        "Stack Edge Score": stack_edge,
        "Stack Confidence": confidence,
        "Stack Recommendation": recs,
    }).sort_values("Stack Edge Score", ascending=False, na_position="last").reset_index(drop=True)


def _detect_pitcher_columns(lineups: pd.DataFrame) -> List[str]:
    result: List[str] = []
    for col in lineups.columns:
        n = _norm_col(col)
        compact = re.sub(r"\s+", "", n).upper()
        if "pitcher" in n or re.fullmatch(r"P\d*", compact) or re.fullmatch(r"SP\d*", compact):
            result.append(col)
    return result


def _pitcher_mask(lineups: pd.DataFrame, pitchers: Sequence[str]) -> pd.Series:
    pcols = _detect_pitcher_columns(lineups)
    if not pcols or not pitchers:
        return pd.Series(False, index=lineups.index)
    keys = {_norm_name(p) for p in pitchers if _norm_name(p)}

    def hit(value: Any) -> bool:
        v = _norm_name(value)
        if not v:
            return False
        return any(v == k or k in v for k in keys)

    return lineups[pcols].apply(lambda c: c.map(hit)).any(axis=1)


def preview_pitcher_exclusions(lineups: pd.DataFrame, pitchers: Sequence[str]) -> Dict[str, Any]:
    mask = _pitcher_mask(lineups, pitchers)
    remaining = lineups.loc[~mask].copy()
    result: Dict[str, Any] = {
        "pitchers": list(pitchers),
        "lineups_affected": int(mask.sum()),
        "lineups_remaining": int(len(remaining)),
        "remaining_pool": remaining,
        "stack_changes": pd.DataFrame(),
    }
    stack_col = next((c for c in lineups.columns if "stack" in _norm_col(c)), None)
    if stack_col and len(lineups):
        before = lineups[stack_col].fillna("Unknown").astype(str).value_counts(normalize=True) * 100
        after = remaining[stack_col].fillna("Unknown").astype(str).value_counts(normalize=True) * 100 if len(remaining) else pd.Series(dtype=float)
        teams = sorted(set(before.index) | set(after.index))
        changes = pd.DataFrame({
            "Stack/Team": teams,
            "Before %": [before.get(t, 0.0) for t in teams],
            "After %": [after.get(t, 0.0) for t in teams],
        })
        changes["Change (pp)"] = changes["After %"] - changes["Before %"]
        result["stack_changes"] = changes.sort_values("Change (pp)", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return result


def apply_pitcher_exclusions(lineups: pd.DataFrame, pitchers: Sequence[str]) -> pd.DataFrame:
    mask = _pitcher_mask(lineups, pitchers)
    return lineups.loc[~mask].copy()



def _coalesce_column(df: pd.DataFrame, target: str, source: pd.Series) -> None:
    if target not in df.columns:
        df[target] = source
    else:
        df[target] = df[target].where(df[target].notna(), source)


def _derive_lineup_exposure(base: pd.DataFrame, lineup_pool: Optional[pd.DataFrame]) -> pd.DataFrame:
    if lineup_pool is None or lineup_pool.empty or "pitcher" not in base.columns:
        return base
    pcols = _detect_pitcher_columns(lineup_pool)
    if not pcols:
        return base

    total = len(lineup_pool)
    counts = []
    for pitcher in base["pitcher"]:
        key = _norm_name(pitcher)
        if not key:
            counts.append(np.nan)
            continue
        mask = lineup_pool[pcols].apply(
            lambda col: col.map(lambda value: _norm_name(value) == key)
        ).any(axis=1)
        counts.append(float(mask.sum()))
    derived_count = pd.Series(counts, index=base.index, dtype=float)
    derived_exposure = derived_count / total * 100.0

    if "lineup_count" not in base.columns:
        base["lineup_count"] = derived_count
    else:
        current = _numeric(base["lineup_count"])
        base["lineup_count"] = current.where(current.notna(), derived_count)

    if "portfolio_exposure" not in base.columns:
        base["portfolio_exposure"] = derived_exposure
    else:
        current = _percent(base["portfolio_exposure"])
        base["portfolio_exposure"] = current.where(current.notna(), derived_exposure)
    return base


def merge_pitcher_sources(
    portfolio_df: Optional[pd.DataFrame] = None,
    pitcher_research_df: Optional[pd.DataFrame] = None,
    team_research_df: Optional[pd.DataFrame] = None,
    scoring_df: Optional[pd.DataFrame] = None,
    lineup_pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge uploaded MLB research sources without fabricating unavailable values.

    Matching is normalized exact-name matching for pitchers and exact team-code matching
    for the opponent. Existing portfolio values win; research/scoring fields fill blanks.
    """
    sources = [x for x in (portfolio_df, pitcher_research_df, scoring_df) if x is not None and not x.empty]
    if not sources:
        raise ValueError("Upload at least one pitcher portfolio, pitcher research, or scoring sheet.")

    if portfolio_df is not None and not portfolio_df.empty:
        base = _canonicalize(portfolio_df, ALIASES).copy()
    elif pitcher_research_df is not None and not pitcher_research_df.empty:
        base = _canonicalize(pitcher_research_df, ALIASES).copy()
    else:
        base = _canonicalize(scoring_df, ALIASES).copy()
        if "position" in base.columns:
            pos = base["position"].astype(str).str.upper().str.strip()
            pitcher_mask = pos.str.contains(r"(?:^|/|,|\s)(?:P|SP|SP1|SP2)(?:$|/|,|\s)", regex=True)
            if pitcher_mask.any():
                base = base.loc[pitcher_mask].copy()

    if "pitcher" not in base.columns:
        raise ValueError("Could not identify a pitcher/player-name column in the uploaded pitcher sources.")

    base = base.drop_duplicates(subset=["pitcher"], keep="first").reset_index(drop=True)
    base["_pitcher_key"] = base["pitcher"].map(_norm_name)

    def merge_pitcher_level(source: Optional[pd.DataFrame]) -> None:
        nonlocal base
        if source is None or source.empty:
            return
        src = _canonicalize(source, ALIASES).copy()
        if "pitcher" not in src.columns:
            return
        src["_pitcher_key"] = src["pitcher"].map(_norm_name)
        src = src.drop_duplicates(subset=["_pitcher_key"], keep="first")
        for col in src.columns:
            if col in {"pitcher", "_pitcher_key"}:
                continue
            mapping = src.set_index("_pitcher_key")[col]
            values = base["_pitcher_key"].map(mapping)
            _coalesce_column(base, col, values)

    # Research then scoring fill missing fields. Base values remain authoritative.
    if pitcher_research_df is not base:
        merge_pitcher_level(pitcher_research_df)
    merge_pitcher_level(scoring_df)

    if team_research_df is not None and not team_research_df.empty and "opponent" in base.columns:
        team = _canonicalize(team_research_df, TEAM_RESEARCH_ALIASES).copy()
        if "team" in team.columns:
            team["_team_key"] = team["team"].astype(str).str.upper().str.strip()
            team = team.drop_duplicates(subset=["_team_key"], keep="first").set_index("_team_key")
            opp_key = base["opponent"].astype(str).str.upper().str.strip()
            mapping = {
                "hard_per_swing": "hard_per_swing",
                "stack_score": "opponent_stack_score",
                "raw_baseball_score": "opponent_raw_baseball_score",
                "dfs_stack_score": "opponent_dfs_stack_score",
            }
            for source_col, target_col in mapping.items():
                if source_col in team.columns:
                    values = opp_key.map(team[source_col])
                    _coalesce_column(base, target_col, values)

    base = _derive_lineup_exposure(base, lineup_pool)
    return base.drop(columns=["_pitcher_key"], errors="ignore")


def build_pitcher_stack_leverage(
    pitcher_analysis: pd.DataFrame,
    stack_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Cross the pitcher portfolio view with the opponent's stack edge.

    This is descriptive portfolio guidance: high scores mean the offense across from a
    popular/risky pitcher deserves extra attention. It does not alter lineups.
    """
    if pitcher_analysis is None or pitcher_analysis.empty or stack_edges is None or stack_edges.empty:
        return pd.DataFrame()

    p = pitcher_analysis.copy()
    s = stack_edges.copy()
    if "Opponent" not in p.columns or "Team" not in s.columns:
        return pd.DataFrame()

    p["_opp"] = p["Opponent"].astype(str).str.upper().str.strip()
    s["_team"] = s["Team"].astype(str).str.upper().str.strip()
    merged = p.merge(
        s[[c for c in s.columns if c in {
            "_team", "Team", "Stack Edge Score", "Stack Confidence", "Stack Recommendation",
            "Raw Hitting Strength", "Projection Score", "Ownership Leverage"
        }]],
        left_on="_opp",
        right_on="_team",
        how="inner",
        suffixes=("", "_stack"),
    )
    if merged.empty:
        return pd.DataFrame()

    own_rank = _rank(merged["Projected Ownership"]) if "Projected Ownership" in merged else pd.Series(np.nan, index=merged.index)
    leverage, _ = _weighted_score(
        merged.index,
        (merged["Stack Edge Score"], 0.55),
        (merged["Portfolio Risk"], 0.25),
        (own_rank, 0.20),
    )

    labels: List[str] = []
    reasons: List[str] = []
    for i, row in merged.iterrows():
        score = leverage.loc[i]
        chalk = str(row.get("Chalk Risk", ""))
        stack_rec = str(row.get("Stack Recommendation", ""))
        if pd.notna(score) and score >= 80 and chalk == "FRAGILE CHALK":
            label = "ATTACK FRAGILE CHALK"
        elif pd.notna(score) and score >= 70:
            label = "STACK LEVERAGE"
        elif pd.notna(score) and score >= 50:
            label = "WATCH"
        else:
            label = "NO STRONG ATTACK EDGE"
        labels.append(label)

        reason_bits = []
        if pd.notna(row.get("Stack Edge Score", np.nan)):
            reason_bits.append(f"Stack Edge {row['Stack Edge Score']:.0f}/100")
        if pd.notna(row.get("Portfolio Risk", np.nan)):
            reason_bits.append(f"Pitcher Portfolio Risk {row['Portfolio Risk']:.0f}/100")
        if pd.notna(row.get("Projected Ownership", np.nan)):
            reason_bits.append(f"Pitcher projected ownership {row['Projected Ownership']:.1f}%")
        if chalk:
            reason_bits.append(chalk.title())
        if stack_rec:
            reason_bits.append(f"Stack view: {stack_rec}")
        reasons.append("; ".join(reason_bits))

    stack_team_col = "Team_stack" if "Team_stack" in merged.columns else "Team"
    return pd.DataFrame({
        "Stack Team": merged[stack_team_col],
        "Opposing Pitcher": merged["Pitcher"],
        "Pitcher Projected Ownership": merged.get("Projected Ownership", np.nan),
        "Pitcher Portfolio Exposure": merged.get("Portfolio Exposure", np.nan),
        "Pitcher Quality": merged.get("Pitcher Quality", np.nan),
        "Pitcher Matchup Risk": merged.get("Matchup Risk", np.nan),
        "Pitcher Portfolio Risk": merged.get("Portfolio Risk", np.nan),
        "Pitcher Chalk Risk": merged.get("Chalk Risk", ""),
        "Pitcher Recommendation": merged.get("Recommendation", ""),
        "Stack Edge Score": merged.get("Stack Edge Score", np.nan),
        "Stack Confidence": merged.get("Stack Confidence", np.nan),
        "Stack Recommendation": merged.get("Stack Recommendation", ""),
        "Pitcher-Stack Leverage": leverage,
        "Leverage Recommendation": labels,
        "Reason": reasons,
    }).sort_values("Pitcher-Stack Leverage", ascending=False, na_position="last").reset_index(drop=True)
