from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from modules.mlb_pitcher_portfolio import _norm_name, _numeric, _percent, _rank
from modules.mlb_slate_files import (
    BULLPEN_ALIASES,
    HARDPERSWING_ALIASES,
    HWSR_ALIASES,
    ROO_ALIASES,
    SCORING_ALIASES,
    STARTER_WEAKNESS_ALIASES,
    _canonicalize,
    _scoring_strength,
    build_bullpen_vulnerability_table,
    build_team_projection_research,
)
from modules.projection_models.model_v1_0 import (
    DEFAULT_MODEL_CONFIG,
    HITTER_FACTOR_WEIGHTS,
    PITCHER_FACTOR_WEIGHTS,
    ProjectionModelConfig,
)

DISTRIBUTIONS = ("floor", "median", "ceiling")
FACTOR_LABELS = {
    "team_hitting": "team hitting environment",
    "starter": "starter vulnerability",
    "bullpen": "bullpen vulnerability",
    "hard_contact": "team hard-contact environment",
    "lineup_position": "lineup position",
    "pitcher_quality": "pitcher quality",
    "opponent_hitting": "opponent hitting environment",
    "opponent_hard_contact": "opponent hard-contact environment",
}


def centered_edge(percentile: float | int | None) -> float:
    if percentile is None or pd.isna(percentile):
        return np.nan
    return float(np.clip((float(percentile) - 50.0) / 50.0, -1.0, 1.0))


def _series_percentile(series: pd.Series) -> pd.Series:
    return _rank(series)


def _lineup_position_percentile(order: pd.Series, hitter_mask: pd.Series) -> pd.Series:
    values = _numeric(order)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = hitter_mask & values.notna()
    if not valid.any():
        return out
    subset = values.loc[valid]
    lo, hi = float(subset.min()), float(subset.max())
    if math.isclose(lo, hi):
        out.loc[valid] = 50.0
    else:
        out.loc[valid] = (hi - subset) / (hi - lo) * 100.0
    return out


def _team_hitting_map(scoring_df: pd.DataFrame | None) -> dict[str, float]:
    if scoring_df is None or scoring_df.empty:
        return {}
    scoring = _canonicalize(scoring_df, SCORING_ALIASES).copy()
    if "team" not in scoring:
        return {}
    scoring["team"] = scoring["team"].astype(str).str.upper().str.strip()
    score, _ = _scoring_strength(scoring)
    return dict(zip(scoring["team"], score))


def _hard_contact_maps(hardperswing_df: pd.DataFrame | None) -> tuple[dict[str, float], dict[str, float]]:
    if hardperswing_df is None or hardperswing_df.empty:
        return {}, {}
    hard = _canonicalize(hardperswing_df, HARDPERSWING_ALIASES).copy()
    if "team" not in hard or "hard_per_swing" not in hard:
        return {}, {}
    hard["team"] = hard["team"].astype(str).str.upper().str.strip()
    hard["hard_per_swing"] = _numeric(hard["hard_per_swing"])
    pct = _series_percentile(hard["hard_per_swing"])
    return dict(zip(hard["team"], pct)), dict(zip(hard["team"], hard["hard_per_swing"]))


def _starter_maps(
    weighted_hwsr_df: pd.DataFrame | None,
    starter_weakness_df: pd.DataFrame | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return offense-team vulnerability, pitcher vulnerability, and raw audit maps."""
    offense_parts: dict[str, list[float]] = {}
    pitcher_parts: dict[str, list[float]] = {}
    raw_hwsr: dict[str, float] = {}

    if weighted_hwsr_df is not None and not weighted_hwsr_df.empty:
        h = _canonicalize(weighted_hwsr_df, HWSR_ALIASES).copy()
        if "weighted_hwsr" in h:
            h["weighted_hwsr"] = _numeric(h["weighted_hwsr"])
            h["_pct"] = _series_percentile(h["weighted_hwsr"])
            for _, row in h.iterrows():
                if "opp" in h and pd.notna(row.get("opp")):
                    offense_parts.setdefault(str(row["opp"]).upper().strip(), []).append(row["_pct"])
                if "player" in h and _norm_name(row.get("player")):
                    key = _norm_name(row["player"])
                    pitcher_parts.setdefault(key, []).append(row["_pct"])
                    if pd.notna(row.get("weighted_hwsr")):
                        raw_hwsr[key] = float(row["weighted_hwsr"])

    if starter_weakness_df is not None and not starter_weakness_df.empty:
        w = _canonicalize(starter_weakness_df, STARTER_WEAKNESS_ALIASES).copy()
        if "weighted_true_avg" in w:
            w["weighted_true_avg"] = _numeric(w["weighted_true_avg"])
            w["_pct"] = _series_percentile(w["weighted_true_avg"])
            for _, row in w.iterrows():
                if "opp" in w and pd.notna(row.get("opp")):
                    offense_parts.setdefault(str(row["opp"]).upper().strip(), []).append(row["_pct"])
                if "player" in w and _norm_name(row.get("player")):
                    pitcher_parts.setdefault(_norm_name(row["player"]), []).append(row["_pct"])

    def avg_map(parts: dict[str, list[float]]) -> dict[str, float]:
        return {
            key: float(np.nanmean([v for v in vals if pd.notna(v)]))
            for key, vals in parts.items()
            if any(pd.notna(v) for v in vals)
        }

    return avg_map(offense_parts), avg_map(pitcher_parts), raw_hwsr


def _bullpen_map(bullpen_df: pd.DataFrame | None) -> tuple[dict[str, float], pd.DataFrame]:
    if bullpen_df is None or bullpen_df.empty:
        return {}, pd.DataFrame()
    try:
        bp = build_bullpen_vulnerability_table(bullpen_df)
    except Exception:
        return {}, pd.DataFrame()
    if bp.empty:
        return {}, bp
    return dict(zip(bp["Team"].astype(str).str.upper().str.strip(), bp["Bullpen Vulnerability Score"])), bp


def compute_confidence(
    factor_edges: dict[str, float],
    factor_weights: dict[str, float],
    signal_threshold: float,
) -> dict[str, float | int | str]:
    expected_weight = sum(max(float(w), 0.0) for w in factor_weights.values())
    present = {k: float(v) for k, v in factor_edges.items() if k in factor_weights and pd.notna(v)}
    present_weight = sum(max(float(factor_weights[k]), 0.0) for k in present)
    coverage = present_weight / expected_weight if expected_weight else 0.0

    active = {k: v for k, v in present.items() if abs(v) >= signal_threshold}
    positive_mass = sum(float(factor_weights[k]) * abs(v) for k, v in active.items() if v > 0)
    negative_mass = sum(float(factor_weights[k]) * abs(v) for k, v in active.items() if v < 0)
    directional_mass = positive_mass + negative_mass
    active_weight = sum(float(factor_weights[k]) for k in active)

    if directional_mass:
        agreement_ratio = max(positive_mass, negative_mass) / directional_mass
        conflict = 1.0 - abs(positive_mass - negative_mass) / directional_mass
        dominant_positive = positive_mass >= negative_mass
        dominant_direction = "BOOST" if dominant_positive else "DOWNGRADE"
        agreement_count = sum(1 for v in active.values() if (v > 0) == dominant_positive)
    else:
        agreement_ratio = 0.5
        conflict = 0.0
        dominant_direction = "NEUTRAL"
        agreement_count = 0

    directional_coverage = active_weight / present_weight if present_weight else 0.0
    confidence = coverage * (0.45 + 0.55 * directional_coverage) * (0.65 + 0.35 * agreement_ratio) * (1.0 - 0.35 * conflict)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return {
        "confidence": confidence,
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "agreement_ratio": float(agreement_ratio),
        "conflict": float(np.clip(conflict, 0.0, 1.0)),
        "agreement_count": int(agreement_count),
        "agreement_total": int(len(active)),
        "dominant_direction": dominant_direction,
    }


def _distribution_change(
    factor_edges: dict[str, float],
    factor_weights: dict[str, dict[str, float]],
    distribution: str,
    confidence: float,
    sensitivity: float,
) -> tuple[float, dict[str, float]]:
    available = [(factor, float(factor_edges[factor]), float(weights[distribution])) for factor, weights in factor_weights.items() if pd.notna(factor_edges.get(factor, np.nan))]
    denom = sum(weight for _, _, weight in available)
    if denom <= 0 or confidence <= 0:
        return 0.0, {factor: 0.0 for factor in factor_weights}
    contributions: dict[str, float] = {factor: 0.0 for factor in factor_weights}
    for factor, edge, weight in available:
        normalized_weight = weight / denom
        contributions[factor] = edge * normalized_weight * confidence * sensitivity * 100.0
    return float(sum(contributions.values())), contributions


def _explanation(
    changes: dict[str, float],
    contributions: dict[str, dict[str, float]],
    confidence: float,
    agreement_count: int,
    agreement_total: int,
) -> str:
    dist = max(DISTRIBUTIONS, key=lambda d: abs(changes.get(d, 0.0)))
    change = changes.get(dist, 0.0)
    if abs(change) < 0.05:
        return f"No meaningful research adjustment; confidence {confidence*100:.0f}% with {agreement_count}/{agreement_total} directional research families aligned."
    items = [(factor, value) for factor, value in contributions.get(dist, {}).items() if abs(value) >= 0.05]
    items.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    labels = [FACTOR_LABELS.get(factor, factor.replace("_", " ")) for factor, _ in items[:4]]
    direction = "boosted" if change > 0 else "downgraded"
    joined = ", ".join(labels[:-1]) + (" and " + labels[-1] if len(labels) > 1 else (labels[0] if labels else "available research"))
    return f"{dist.title()} {direction} by {joined}; confidence {confidence*100:.0f}% with {agreement_count}/{agreement_total} directional research families aligned."


def build_adjusted_projections(
    player_roo_df: pd.DataFrame,
    scoring_df: pd.DataFrame | None = None,
    bullpen_df: pd.DataFrame | None = None,
    weighted_hwsr_df: pd.DataFrame | None = None,
    hardperswing_df: pd.DataFrame | None = None,
    starter_weakness_df: pd.DataFrame | None = None,
    game_environment_df: pd.DataFrame | None = None,
    model_config: ProjectionModelConfig = DEFAULT_MODEL_CONFIG,
) -> pd.DataFrame:
    if player_roo_df is None or player_roo_df.empty:
        return pd.DataFrame()
    roo = _canonicalize(player_roo_df, ROO_ALIASES).copy()
    if "player" not in roo:
        raise ValueError("Player ROO must include a player/name column.")
    for col in ("floor", "median", "ceiling", "salary", "order"):
        if col in roo:
            roo[col] = _numeric(roo[col])
    if "own" in roo:
        roo["own"] = _percent(roo["own"])
    else:
        roo["own"] = np.nan
    if "team" not in roo:
        roo["team"] = ""
    if "opp" not in roo:
        roo["opp"] = ""
    if "position" not in roo:
        roo["position"] = ""
    if "salary" not in roo:
        roo["salary"] = np.nan
    for col in ("floor", "median", "ceiling"):
        if col not in roo:
            roo[col] = np.nan

    roo["team"] = roo["team"].astype(str).str.upper().str.strip()
    roo["opp"] = roo["opp"].astype(str).str.upper().str.strip()
    position = roo["position"].astype(str).str.upper().str.strip()
    pitcher_mask = position.str.fullmatch(r"P|SP|SP1|SP2", na=False)
    order_pct = _lineup_position_percentile(roo.get("order", pd.Series(np.nan, index=roo.index)), ~pitcher_mask)

    team_hitting = _team_hitting_map(scoring_df)
    hard_pct, hard_raw = _hard_contact_maps(hardperswing_df)
    starter_offense, pitcher_vulnerability, raw_hwsr = _starter_maps(weighted_hwsr_df, starter_weakness_df)
    bullpen_vulnerability, _ = _bullpen_map(bullpen_df)

    # Build the shared context when possible so the engine and audit layer reuse the
    # same pure-baseball transformations as the stack research module.
    context_by_team: dict[str, dict[str, Any]] = {}
    if scoring_df is not None and not scoring_df.empty:
        try:
            ctx = build_team_projection_research(scoring_df, bullpen_df, weighted_hwsr_df, hardperswing_df, starter_weakness_df)
            context_by_team = {str(r["team"]).upper().strip(): r.to_dict() for _, r in ctx.iterrows()}
        except Exception:
            context_by_team = {}

    environment_by_player: dict[str, dict[str, Any]] = {}
    if game_environment_df is not None and not game_environment_df.empty and "player_names" in game_environment_df:
        environment_by_player = {
            _norm_name(r.get("player_names")): r.to_dict()
            for _, r in game_environment_df.iterrows()
            if _norm_name(r.get("player_names"))
        }

    rows: list[dict[str, Any]] = []
    for i, source in roo.iterrows():
        is_pitcher = bool(pitcher_mask.loc[i])
        team = str(source.get("team", "")).upper().strip()
        opp = str(source.get("opp", "")).upper().strip()
        player_key = _norm_name(source.get("player"))
        ctx = context_by_team.get(team, {})

        if is_pitcher:
            vulnerability_pct = pitcher_vulnerability.get(player_key, np.nan)
            env = environment_by_player.get(player_key, {})
            env_score = env.get("game_environment_score", np.nan)
            factors = {
                "pitcher_quality": -centered_edge(vulnerability_pct) if pd.notna(vulnerability_pct) else np.nan,
                "opponent_hitting": -centered_edge(team_hitting.get(opp, np.nan)) if pd.notna(team_hitting.get(opp, np.nan)) else np.nan,
                "opponent_hard_contact": -centered_edge(hard_pct.get(opp, np.nan)) if pd.notna(hard_pct.get(opp, np.nan)) else np.nan,
                "game_environment": centered_edge(env_score),
            }
            factor_weights = PITCHER_FACTOR_WEIGHTS
        else:
            team_hit_pct = ctx.get("team_hitting_pct", team_hitting.get(team, np.nan))
            starter_pct = ctx.get("starter_vulnerability_pct", starter_offense.get(team, np.nan))
            bullpen_pct = ctx.get("bullpen_vulnerability_pct", bullpen_vulnerability.get(opp, np.nan))
            hard_contact_pct = ctx.get("hard_contact_pct", hard_pct.get(team, np.nan))
            env = environment_by_player.get(player_key, {})
            env_score = env.get("game_environment_score", np.nan)
            factors = {
                "team_hitting": centered_edge(team_hit_pct),
                "starter": centered_edge(starter_pct),
                "bullpen": centered_edge(bullpen_pct),
                "hard_contact": centered_edge(hard_contact_pct),
                "lineup_position": centered_edge(order_pct.loc[i]),
                "game_environment": centered_edge(env_score),
            }
            factor_weights = HITTER_FACTOR_WEIGHTS

        confidence_weights = {factor: weights["median"] for factor, weights in factor_weights.items()}
        confidence_info = compute_confidence(factors, confidence_weights, model_config.signal_threshold)
        confidence = float(confidence_info["confidence"])

        changes: dict[str, float] = {}
        contribs: dict[str, dict[str, float]] = {}
        sensitivities = {
            "floor": model_config.floor_sensitivity,
            "median": model_config.median_sensitivity,
            "ceiling": model_config.ceiling_sensitivity,
        }
        adjusted: dict[str, float] = {}
        for distribution in DISTRIBUTIONS:
            change_pct, distribution_contrib = _distribution_change(
                factors, factor_weights, distribution, confidence, sensitivities[distribution]
            )
            changes[distribution] = change_pct
            contribs[distribution] = distribution_contrib
            baseline = source.get(distribution, np.nan)
            adjusted[distribution] = float(baseline * (1.0 + change_pct / 100.0)) if pd.notna(baseline) else np.nan

        row: dict[str, Any] = {
            "player_names": source.get("player", ""),
            "position": source.get("position", ""),
            "team": team,
            "opponent": opp,
            "salary": source.get("salary", np.nan),
            "ownership": source.get("own", np.nan),
            "original_floor": source.get("floor", np.nan),
            "original_median": source.get("median", np.nan),
            "original_ceiling": source.get("ceiling", np.nan),
            "drew_floor": adjusted["floor"],
            "drew_median": adjusted["median"],
            "drew_ceiling": adjusted["ceiling"],
            "floor_change_pct": changes["floor"],
            "median_change_pct": changes["median"],
            "ceiling_change_pct": changes["ceiling"],
            "research_confidence": confidence * 100.0,
            "research_coverage": float(confidence_info["coverage"]) * 100.0,
            "research_agreement_count": int(confidence_info["agreement_count"]),
            "research_agreement_total": int(confidence_info["agreement_total"]),
            "dominant_direction": confidence_info["dominant_direction"],
            "adjustment_explanation": _explanation(
                changes, contribs, confidence,
                int(confidence_info["agreement_count"]), int(confidence_info["agreement_total"]),
            ),
            "model_version": model_config.model_version,
            "is_pitcher": is_pitcher,
            "raw_weighted_hwsr": raw_hwsr.get(player_key, np.nan),
            "raw_hard_per_swing": hard_raw.get(team, np.nan),
            "bats": env.get("bats", np.nan) if isinstance(env, dict) else np.nan,
            "throws": env.get("throws", np.nan) if isinstance(env, dict) else np.nan,
            "park_split": env.get("park_split", np.nan) if isinstance(env, dict) else np.nan,
            "game_environment_score": env.get("game_environment_score", np.nan) if isinstance(env, dict) else np.nan,
        }
        for factor, edge in factors.items():
            row[f"{factor}_edge"] = edge
        for distribution, dist_contrib in contribs.items():
            for factor, value in dist_contrib.items():
                row[f"{distribution}_{factor}_contribution_pct"] = value
        rows.append(row)

    return pd.DataFrame(rows)
