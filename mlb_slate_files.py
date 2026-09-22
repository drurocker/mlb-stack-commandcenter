from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd

from modules.mlb_pitcher_portfolio import _norm_col, _norm_name, _numeric, _percent, _rank, _weighted_score


FILE_TYPES = [
    "Lineup Portfolio",
    "Team HardPerSwing",
    "Starter Weakness",
    "Scoring Sheet",
    "Bullpen Research",
    "Player ROO",
    "Weighted HWSr",
    "Ballpark Research",
    "Needs classification",
]


def _cols(df: pd.DataFrame) -> set[str]:
    return {_norm_col(c).replace(" ", "") for c in df.columns}


def _signature_score(columns: set[str], required: Sequence[str], optional: Sequence[str] = ()) -> float:
    req = {_norm_col(x).replace(" ", "") for x in required}
    opt = {_norm_col(x).replace(" ", "") for x in optional}
    req_hits = len(req & columns)
    if not req:
        req_score = 0.0
    else:
        req_score = req_hits / len(req)
    if req_hits < max(1, int(np.ceil(len(req) * 0.60))):
        return req_score * 0.55
    opt_score = (len(opt & columns) / len(opt)) if opt else 1.0
    return min(1.0, req_score * 0.85 + opt_score * 0.15)


def classify_mlb_dataframe(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    """Classify an MLB slate CSV by column signature.

    Returns a high-confidence type when the schema is distinctive. Ambiguous or weak
    matches are deliberately returned as ``Needs classification`` so the UI can ask
    the user instead of silently guessing.
    """
    columns = _cols(df)

    lineup_slots = {
        "sp1", "sp2", "p", "p2", "c", "1b", "2b", "3b", "ss",
        "of", "of1", "of2", "of3", "util",
    }
    slot_hits = len(columns & lineup_slots)

    scores: Dict[str, float] = {
        "Scoring Sheet": _signature_score(
            columns,
            ["names", "oppSP", "avgScore", "eightPlusRuns", "topScore", "teamOwnPct"],
            ["winPercentage", "avgFirstInning", "avgFifthInning", "prio"],
        ),
        "Player ROO": _signature_score(
            columns,
            ["Player", "Position", "Team", "Salary", "Median", "Ceiling", "Own"],
            ["Floor", "Top_finish", "Top_5_finish", "Top_10_finish", "15+%", "2x%", "3x%", "4x%"],
        ),
        "Bullpen Research": _signature_score(
            columns,
            ["Names", "xwOBA", "xSLG", "HWS Ratio", "xBA", "AVG", "BABIP"],
            ["PA", "Hits", "Homeruns", "Strikeoutper", "Walkper"],
        ),
        "Weighted HWSr": _signature_score(
            columns,
            ["Player", "Team", "Opp", "Weighted HWSr"],
            ["Handedness", "HWSr (LHH)", "HWSr (RHH)", "HWSr (Overall)", "Opp LHH", "Opp RHH"],
        ),
        "Ballpark Research": _signature_score(
            columns,
            ["Stadium", "Split", "PA", "Homeruns_boost", "xSLG_avg_boost", "xwOBA_avg_boost"],
            ["Hits_boost", "Doubles_boost", "xBA_avg_boost", "BABIP_avg_boost", "AVG_avg_boost"],
        ),
        "Team HardPerSwing": 0.98 if "hardperswing" in columns and ("team" in columns or "names" in columns) else 0.0,
        "Starter Weakness": 0.98 if "weightedtrueavg" in columns and ("player" in columns or "pitcher" in columns) else 0.0,
        "Lineup Portfolio": min(1.0, 0.70 + slot_hits * 0.04) if slot_hits >= 6 else (slot_hits / 10.0),
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Filename is only a small tie-breaker; schema remains authoritative.
    fname = filename.lower()
    hints = {
        "Scoring Sheet": ("scoring",),
        "Player ROO": ("roo", "range"),
        "Bullpen Research": ("bullpen", "team_research"),
        "Weighted HWSr": ("hwsr", "pitcher_research"),
        "Lineup Portfolio": ("optimal", "portfolio", "lineup"),
        "Ballpark Research": ("ballpark", "park", "stadium"),
    }
    if any(token in fname for token in hints.get(best_type, ())):
        best_score = min(1.0, best_score + 0.02)

    ambiguous = best_score < 0.65 or (best_score - second_score < 0.08 and second_score >= 0.65)
    detected = "Needs classification" if ambiguous else best_type

    return {
        "type": detected,
        "confidence": float(best_score),
        "candidates": ranked[:3],
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
    }


BULLPEN_ALIASES = {
    "team": ["Names", "Team", "Tm"],
    "xwoba": ["xwOBA"],
    "xslg": ["xSLG"],
    "hws_ratio": ["HWS Ratio", "HWSRatio"],
    "xba": ["xBA"],
    "avg": ["AVG"],
    "babip": ["BABIP"],
}


def _canonicalize(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
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


def _bullpen_label(score: float) -> str:
    if pd.isna(score):
        return "UNAVAILABLE"
    if score >= 90:
        return "ELITE BULLPEN TO ATTACK"
    if score >= 80:
        return "VERY BAD BULLPEN"
    if score >= 70:
        return "STRONG STACK BOOST"
    if score >= 60:
        return "MILD STACK BOOST"
    if score >= 40:
        return "NEUTRAL"
    if score >= 25:
        return "ABOVE-AVERAGE BULLPEN"
    return "BULLPEN DOWNGRADE"


def build_bullpen_vulnerability_table(bullpen_df: pd.DataFrame) -> pd.DataFrame:
    """Build the user-locked Bullpen Vulnerability Score (BVS).

    All inputs are converted to 0-100 percentile ranks first, with 100 meaning
    most vulnerable. Exact weights:
      xwOBA 30%, xSLG 22%, HWS Ratio 20%, xBA 13%, AVG 8%, BABIP 7%.
    """
    bp = _canonicalize(bullpen_df, BULLPEN_ALIASES).copy()
    required = ["team", "xwoba", "xslg", "hws_ratio", "xba", "avg", "babip"]
    missing = [c for c in required if c not in bp.columns]
    if missing:
        raise ValueError(f"Bullpen Research is missing required columns: {', '.join(missing)}")

    bp["team"] = bp["team"].astype(str).str.upper().str.strip()
    for c in required[1:]:
        bp[c] = _numeric(bp[c])

    components = {
        "xwOBA Percentile": (_rank(bp["xwoba"]), 0.30),
        "xSLG Percentile": (_rank(bp["xslg"]), 0.22),
        "HWS Ratio Percentile": (_rank(bp["hws_ratio"]), 0.20),
        "xBA Percentile": (_rank(bp["xba"]), 0.13),
        "AVG Percentile": (_rank(bp["avg"]), 0.08),
        "BABIP Percentile": (_rank(bp["babip"]), 0.07),
    }
    bvs, coverage = _weighted_score(bp.index, *components.values())

    out = pd.DataFrame({
        "Team": bp["team"],
        "xwOBA": bp["xwoba"],
        "xSLG": bp["xslg"],
        "HWS Ratio": bp["hws_ratio"],
        "xBA": bp["xba"],
        "AVG": bp["avg"],
        "BABIP": bp["babip"],
        **{name: series for name, (series, _) in components.items()},
        "Bullpen Vulnerability Score": bvs,
        "Bullpen Confidence": coverage * 100.0,
    })
    out["Bullpen Classification"] = out["Bullpen Vulnerability Score"].map(_bullpen_label)
    out = out.sort_values("Bullpen Vulnerability Score", ascending=False, na_position="last").reset_index(drop=True)
    out["Bullpen Rank"] = np.arange(1, len(out) + 1)
    cols = ["Bullpen Rank"] + [c for c in out.columns if c != "Bullpen Rank"]
    return out[cols]


SCORING_ALIASES = {
    "team": ["names", "name", "team"],
    "priority": ["prio", "priority"],
    "opp_sp": ["oppSP", "opposing pitcher", "opponent pitcher"],
    "avg_score": ["avgScore", "average score"],
    "eight_plus": ["eightPlusRuns", "8+ runs", "eight plus runs"],
    "top_score": ["topScore", "top score"],
    "team_own": ["teamOwnPct", "team ownership", "team own"],
    "win_pct": ["winPercentage", "win percentage"],
    "avg_first": ["avgFirstInning"],
    "first_lead": ["firstInningLeadPct"],
    "avg_fifth": ["avgFifthInning"],
    "fifth_lead": ["fifthInningLeadPct"],
}

ROO_ALIASES = {
    "player": ["Player", "Name"],
    "position": ["Position", "Pos"],
    "order": ["Order", "Batting Order"],
    "team": ["Team", "Tm"],
    "opp": ["Opp", "Opponent"],
    "salary": ["Salary"],
    "floor": ["Floor"],
    "median": ["Median", "Projection"],
    "ceiling": ["Ceiling"],
    "top_finish": ["Top_finish", "Top Finish"],
    "top5_finish": ["Top_5_finish", "Top 5 Finish"],
    "top10_finish": ["Top_10_finish", "Top 10 Finish"],
    "fifteen_plus": ["15+%", "15+"],
    "own": ["Own", "Projected Ownership"],
    "small_own": ["Small Own"],
    "large_own": ["Large Own"],
    "cash_own": ["Cash Own"],
}

HWSR_ALIASES = {
    "player": ["Player", "Pitcher"],
    "team": ["Team", "Tm"],
    "opp": ["Opp", "Opponent"],
    "weighted_hwsr": ["Weighted HWSr"],
}


def _clean_pitcher_label(value: Any) -> str:
    key = _norm_name(value)
    key = re.sub(r"\b(rhp|lhp)\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def _scoring_strength(scoring: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    idx = scoring.index
    nan = pd.Series(np.nan, index=idx, dtype=float)
    parts = [
        (_rank(scoring["avg_score"]) if "avg_score" in scoring else nan.copy(), 0.35),
        (_rank(scoring["eight_plus"]) if "eight_plus" in scoring else nan.copy(), 0.22),
        (_rank(scoring["top_score"]) if "top_score" in scoring else nan.copy(), 0.18),
        (_rank(scoring["win_pct"]) if "win_pct" in scoring else nan.copy(), 0.08),
        (_rank(scoring["avg_first"]) if "avg_first" in scoring else nan.copy(), 0.05),
        (_rank(scoring["first_lead"]) if "first_lead" in scoring else nan.copy(), 0.04),
        (_rank(scoring["avg_fifth"]) if "avg_fifth" in scoring else nan.copy(), 0.05),
        (_rank(scoring["fifth_lead"]) if "fifth_lead" in scoring else nan.copy(), 0.03),
    ]
    return _weighted_score(idx, *parts)


def _aggregate_roo(roo_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if roo_df is None or roo_df.empty:
        return pd.DataFrame()
    roo = _canonicalize(roo_df, ROO_ALIASES).copy()
    if "team" not in roo.columns:
        return pd.DataFrame()
    roo["team"] = roo["team"].astype(str).str.upper().str.strip()
    if "opp" in roo.columns:
        roo["opp"] = roo["opp"].astype(str).str.upper().str.strip()
    if "position" in roo.columns:
        pos = roo["position"].astype(str).str.upper().str.strip()
        roo = roo.loc[~pos.str.fullmatch(r"P|SP|SP1|SP2", na=False)].copy()
    for c in ["salary", "floor", "median", "ceiling", "top_finish", "top5_finish", "top10_finish", "fifteen_plus"]:
        if c in roo.columns:
            roo[c] = _numeric(roo[c])
    for c in ["own", "small_own", "large_own", "cash_own"]:
        if c in roo.columns:
            roo[c] = _percent(roo[c])

    rows: List[Dict[str, Any]] = []
    for team_name, group in roo.groupby("team"):
        g = group.sort_values("median", ascending=False) if "median" in group.columns else group.copy()
        top = g.head(5)
        row: Dict[str, Any] = {"team": team_name}
        if "opp" in g.columns and g["opp"].notna().any():
            mode = g["opp"].dropna().mode()
            row["opponent_team"] = mode.iloc[0] if not mode.empty else g["opp"].dropna().iloc[0]
        for col, out_name, agg in [
            ("median", "top5_median", "sum"),
            ("ceiling", "top5_ceiling", "sum"),
            ("salary", "top5_salary", "sum"),
            ("own", "top5_avg_own", "mean"),
            ("fifteen_plus", "top5_avg_15plus", "mean"),
            ("top_finish", "top5_avg_top_finish", "mean"),
        ]:
            if col in top.columns:
                row[out_name] = top[col].sum(min_count=1) if agg == "sum" else top[col].mean()
            else:
                row[out_name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _roo_strength(agg: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    idx = agg.index
    nan = pd.Series(np.nan, index=idx, dtype=float)
    value = nan.copy()
    if "top5_median" in agg.columns and "top5_salary" in agg.columns:
        value = agg["top5_median"] / (agg["top5_salary"] / 1000.0).replace(0, np.nan)
    return _weighted_score(
        idx,
        (_rank(agg["top5_median"]) if "top5_median" in agg else nan.copy(), 0.35),
        (_rank(agg["top5_ceiling"]) if "top5_ceiling" in agg else nan.copy(), 0.30),
        (_rank(agg["top5_avg_15plus"]) if "top5_avg_15plus" in agg else nan.copy(), 0.20),
        (_rank(value), 0.15),
    )


def _starter_vulnerability(scoring: pd.DataFrame, hwsr_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if hwsr_df is None or hwsr_df.empty:
        return pd.DataFrame(columns=["team", "starter_name", "weighted_hwsr", "starter_vulnerability", "starter_team"])
    h = _canonicalize(hwsr_df, HWSR_ALIASES).copy()
    if "player" not in h.columns or "weighted_hwsr" not in h.columns:
        return pd.DataFrame(columns=["team", "starter_name", "weighted_hwsr", "starter_vulnerability", "starter_team"])
    h["weighted_hwsr"] = _numeric(h["weighted_hwsr"])
    h["_player_key"] = h["player"].map(_clean_pitcher_label)
    h["_vuln"] = _rank(h["weighted_hwsr"])
    if "team" in h.columns:
        h["team"] = h["team"].astype(str).str.upper().str.strip()
    if "opp" in h.columns:
        h["opp"] = h["opp"].astype(str).str.upper().str.strip()

    rows = []
    for _, srow in scoring.iterrows():
        offense = str(srow.get("team", "")).upper().strip()
        target = _clean_pitcher_label(srow.get("opp_sp", ""))
        match = h[h["_player_key"] == target] if target else h.iloc[0:0]
        if match.empty and "opp" in h.columns:
            match = h[h["opp"] == offense]
            if len(match) > 1:
                match = match.sort_values("weighted_hwsr", ascending=False).head(1)
        if match.empty:
            rows.append({"team": offense, "starter_name": np.nan, "weighted_hwsr": np.nan, "starter_vulnerability": np.nan, "starter_team": np.nan})
        else:
            m = match.iloc[0]
            rows.append({
                "team": offense,
                "starter_name": m.get("player", np.nan),
                "weighted_hwsr": m.get("weighted_hwsr", np.nan),
                "starter_vulnerability": m.get("_vuln", np.nan),
                "starter_team": m.get("team", np.nan),
            })
    return pd.DataFrame(rows)




HARDPERSWING_ALIASES = {
    "team": ["Team", "Names", "Tm"],
    "hard_per_swing": ["HardperSwing", "HardPerSwing", "Hard per Swing"],
}

STARTER_WEAKNESS_ALIASES = {
    "player": ["Player", "Pitcher", "Name"],
    "team": ["Team", "Tm"],
    "opp": ["Opp", "Opponent"],
    "weighted_true_avg": ["Weighted True AVG", "WeightedTrueAVG", "Weighted Availability"],
}


def _hardperswing_score(scoring: pd.DataFrame, hardperswing_df: Optional[pd.DataFrame]) -> pd.Series:
    out = pd.Series(np.nan, index=scoring.index, dtype=float)
    if hardperswing_df is None or hardperswing_df.empty:
        return out
    hard = _canonicalize(hardperswing_df, HARDPERSWING_ALIASES).copy()
    if "team" not in hard.columns or "hard_per_swing" not in hard.columns:
        return out
    hard["team"] = hard["team"].astype(str).str.upper().str.strip()
    hard["hard_per_swing"] = _numeric(hard["hard_per_swing"])
    hard["score"] = _rank(hard["hard_per_swing"])
    mapping = hard.drop_duplicates("team").set_index("team")["score"]
    return scoring["team"].map(mapping)


def _starter_true_avg_vulnerability(scoring: pd.DataFrame, weakness_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["team", "weighted_true_avg", "true_avg_vulnerability", "true_avg_starter_team"]
    if weakness_df is None or weakness_df.empty:
        return pd.DataFrame(columns=cols)
    w = _canonicalize(weakness_df, STARTER_WEAKNESS_ALIASES).copy()
    if "weighted_true_avg" not in w.columns:
        return pd.DataFrame(columns=cols)
    w["weighted_true_avg"] = _numeric(w["weighted_true_avg"])
    w["_vuln"] = _rank(w["weighted_true_avg"])
    if "player" in w.columns:
        w["_player_key"] = w["player"].map(_clean_pitcher_label)
    if "team" in w.columns:
        w["team"] = w["team"].astype(str).str.upper().str.strip()
    if "opp" in w.columns:
        w["opp"] = w["opp"].astype(str).str.upper().str.strip()

    rows = []
    for _, srow in scoring.iterrows():
        offense = str(srow.get("team", "")).upper().strip()
        target = _clean_pitcher_label(srow.get("opp_sp", ""))
        match = w.iloc[0:0]
        if target and "_player_key" in w.columns:
            match = w[w["_player_key"] == target]
        if match.empty and "opp" in w.columns:
            match = w[w["opp"] == offense]
            if len(match) > 1:
                match = match.sort_values("weighted_true_avg", ascending=False).head(1)
        if match.empty:
            rows.append({"team": offense, "weighted_true_avg": np.nan, "true_avg_vulnerability": np.nan, "true_avg_starter_team": np.nan})
        else:
            m = match.iloc[0]
            rows.append({
                "team": offense,
                "weighted_true_avg": m.get("weighted_true_avg", np.nan),
                "true_avg_vulnerability": m.get("_vuln", np.nan),
                "true_avg_starter_team": m.get("team", np.nan),
            })
    return pd.DataFrame(rows)



def build_team_projection_research(
    scoring_df: pd.DataFrame,
    bullpen_df: Optional[pd.DataFrame] = None,
    weighted_hwsr_df: Optional[pd.DataFrame] = None,
    hardperswing_df: Optional[pd.DataFrame] = None,
    starter_weakness_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return pure baseball-only team research for projection adjustment.

    This intentionally excludes Player ROO, ownership, Stack Edge, and every score
    derived from the baseline projections. Positive percentiles always mean a more
    favorable hitting environment for the offense in ``team``.
    """
    if scoring_df is None or scoring_df.empty:
        return pd.DataFrame(columns=[
            "team", "opponent_team", "team_hitting_pct", "starter_vulnerability_pct",
            "bullpen_vulnerability_pct", "hard_contact_pct",
        ])
    scoring = _canonicalize(scoring_df, SCORING_ALIASES).copy()
    if "team" not in scoring.columns:
        raise ValueError("Scoring Sheet must contain a team/name column for projection research.")
    scoring["team"] = scoring["team"].astype(str).str.upper().str.strip()

    team_hitting, team_coverage = _scoring_strength(scoring)
    starter_hwsr = _starter_vulnerability(scoring, weighted_hwsr_df)
    starter_true = _starter_true_avg_vulnerability(scoring, starter_weakness_df)
    hard_pct = _hardperswing_score(scoring, hardperswing_df)

    out = pd.DataFrame({
        "team": scoring["team"],
        "team_hitting_pct": team_hitting,
        "team_hitting_coverage": team_coverage * 100.0,
        "hard_contact_pct": hard_pct,
    })

    if not starter_hwsr.empty:
        out = out.merge(
            starter_hwsr[["team", "starter_name", "weighted_hwsr", "starter_vulnerability", "starter_team"]],
            on="team", how="left",
        )
    else:
        for col in ["starter_name", "weighted_hwsr", "starter_vulnerability", "starter_team"]:
            out[col] = np.nan
    if not starter_true.empty:
        out = out.merge(
            starter_true[["team", "weighted_true_avg", "true_avg_vulnerability", "true_avg_starter_team"]],
            on="team", how="left",
        )
    else:
        for col in ["weighted_true_avg", "true_avg_vulnerability", "true_avg_starter_team"]:
            out[col] = np.nan

    starter_components = out[[c for c in ["starter_vulnerability", "true_avg_vulnerability"] if c in out.columns]]
    out["starter_vulnerability_pct"] = starter_components.mean(axis=1, skipna=True) if not starter_components.empty else np.nan
    out["opponent_team"] = out.get("starter_team", pd.Series(np.nan, index=out.index)).where(
        out.get("starter_team", pd.Series(np.nan, index=out.index)).notna(),
        out.get("true_avg_starter_team", pd.Series(np.nan, index=out.index)),
    )

    out["bullpen_vulnerability_pct"] = np.nan
    out["bullpen_bvs_raw"] = np.nan
    if bullpen_df is not None and not bullpen_df.empty:
        bullpen = build_bullpen_vulnerability_table(bullpen_df)
        if not bullpen.empty:
            bvs_map = bullpen.drop_duplicates("Team").set_index("Team")["Bullpen Vulnerability Score"]
            opp = out["opponent_team"].astype(str).str.upper().str.strip()
            mapped = opp.map(bvs_map)
            out["bullpen_vulnerability_pct"] = mapped
            out["bullpen_bvs_raw"] = mapped

    out["hard_per_swing_raw"] = np.nan
    if hardperswing_df is not None and not hardperswing_df.empty:
        hard = _canonicalize(hardperswing_df, HARDPERSWING_ALIASES).copy()
        if "team" in hard.columns and "hard_per_swing" in hard.columns:
            hard["team"] = hard["team"].astype(str).str.upper().str.strip()
            hard["hard_per_swing"] = _numeric(hard["hard_per_swing"])
            raw_map = hard.drop_duplicates("team").set_index("team")["hard_per_swing"]
            out["hard_per_swing_raw"] = out["team"].map(raw_map)

    return out

def _stack_recommendation(score: float) -> str:
    if pd.isna(score):
        return "INSUFFICIENT DATA"
    if score >= 85:
        return "CORE STACK"
    if score >= 72:
        return "OVERWEIGHT"
    if score >= 55:
        return "PLAY / MATCH FIELD"
    if score >= 40:
        return "UNDERWEIGHT"
    return "FADE"


def build_full_stack_edge_table(
    scoring_df: pd.DataFrame,
    player_roo_df: Optional[pd.DataFrame] = None,
    bullpen_df: Optional[pd.DataFrame] = None,
    weighted_hwsr_df: Optional[pd.DataFrame] = None,
    hardperswing_df: Optional[pd.DataFrame] = None,
    starter_weakness_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the slate's primary Stack Edge table from the user's current MLB sheets.

    The Scoring Sheet is the anchor. Player ROO, Weighted HWSr and BVS provide
    matchup/upside context, while team ownership creates the leverage adjustment.
    Missing optional research is excluded rather than filled with zero.
    """
    scoring = _canonicalize(scoring_df, SCORING_ALIASES).copy()
    if "team" not in scoring.columns:
        raise ValueError("Scoring Sheet must contain a names/team column.")
    scoring["team"] = scoring["team"].astype(str).str.upper().str.strip()
    for c in ["avg_score", "eight_plus", "top_score", "team_own", "win_pct", "avg_first", "first_lead", "avg_fifth", "fifth_lead"]:
        if c in scoring.columns:
            scoring[c] = _numeric(scoring[c])

    scoring_strength, scoring_cov = _scoring_strength(scoring)
    scoring["scoring_strength"] = scoring_strength
    scoring["scoring_coverage"] = scoring_cov
    scoring["ownership_leverage"] = 100.0 - _rank(scoring["team_own"]) if "team_own" in scoring.columns else np.nan

    roo_agg = _aggregate_roo(player_roo_df)
    if not roo_agg.empty:
        roo_strength, roo_cov = _roo_strength(roo_agg)
        roo_agg["roo_strength"] = roo_strength
        roo_agg["roo_coverage"] = roo_cov
        scoring = scoring.merge(roo_agg, on="team", how="left")
    else:
        scoring["roo_strength"] = np.nan
        scoring["roo_coverage"] = 0.0
        scoring["opponent_team"] = np.nan

    starter = _starter_vulnerability(scoring, weighted_hwsr_df)
    scoring = scoring.merge(starter, on="team", how="left")

    true_avg = _starter_true_avg_vulnerability(scoring, starter_weakness_df)
    if not true_avg.empty:
        scoring = scoring.merge(true_avg, on="team", how="left")
    else:
        scoring["weighted_true_avg"] = np.nan
        scoring["true_avg_vulnerability"] = np.nan
        scoring["true_avg_starter_team"] = np.nan

    combined_starter, starter_cov = _weighted_score(
        scoring.index,
        (scoring["starter_vulnerability"], 0.50),
        (scoring["true_avg_vulnerability"], 0.50),
    )
    scoring["starter_vulnerability"] = combined_starter
    scoring["hardperswing_score"] = _hardperswing_score(scoring, hardperswing_df)

    bvs_table = build_bullpen_vulnerability_table(bullpen_df) if bullpen_df is not None and not bullpen_df.empty else pd.DataFrame()
    bvs_map = bvs_table.set_index("Team")["Bullpen Vulnerability Score"] if not bvs_table.empty else pd.Series(dtype=float)
    bvs_label_map = bvs_table.set_index("Team")["Bullpen Classification"] if not bvs_table.empty else pd.Series(dtype=object)

    # The opponent bullpen can come from ROO's Opp team or, if ROO is absent, the
    # matched starter's team from the Weighted HWSr sheet.
    scoring["opponent_team_final"] = scoring.get("opponent_team", pd.Series(np.nan, index=scoring.index))
    if "starter_team" in scoring.columns:
        scoring["opponent_team_final"] = scoring["opponent_team_final"].where(
            scoring["opponent_team_final"].notna(), scoring["starter_team"]
        )
    scoring["opponent_team_final"] = scoring["opponent_team_final"].astype(object)
    scoring["bullpen_vulnerability"] = scoring["opponent_team_final"].map(bvs_map)
    scoring["bullpen_classification"] = scoring["opponent_team_final"].map(bvs_label_map)

    # Primary weighting: Scoring Sheet remains the anchor. Supporting research is
    # only used when uploaded; missing signals are renormalized rather than scored zero.
    stack_edge, edge_cov = _weighted_score(
        scoring.index,
        (scoring["scoring_strength"], 0.40),
        (scoring["roo_strength"], 0.20),
        (scoring["starter_vulnerability"], 0.15),
        (scoring["bullpen_vulnerability"], 0.15),
        (scoring["hardperswing_score"], 0.05),
        (scoring["ownership_leverage"], 0.05),
    )

    recs = [_stack_recommendation(v) for v in stack_edge]
    reasons = []
    for i, row in scoring.iterrows():
        bits = []
        if pd.notna(row.get("scoring_strength", np.nan)):
            bits.append(f"Scoring Sheet {row['scoring_strength']:.0f}/100")
        if pd.notna(row.get("roo_strength", np.nan)):
            bits.append(f"Hitter ROO {row['roo_strength']:.0f}/100")
        if pd.notna(row.get("starter_vulnerability", np.nan)):
            bits.append(f"Starter vulnerability {row['starter_vulnerability']:.0f}/100")
        if pd.notna(row.get("bullpen_vulnerability", np.nan)):
            bits.append(f"Bullpen BVS {row['bullpen_vulnerability']:.0f}/100")
        if pd.notna(row.get("hardperswing_score", np.nan)):
            bits.append(f"HardPerSwing {row['hardperswing_score']:.0f}/100")
        if pd.notna(row.get("team_own", np.nan)):
            bits.append(f"Team ownership {row['team_own']:.1f}")
        reasons.append("; ".join(bits))

    out = pd.DataFrame({
        "Team": scoring["team"],
        "Priority": scoring["priority"] if "priority" in scoring.columns else np.nan,
        "Opposing SP": scoring["opp_sp"] if "opp_sp" in scoring.columns else np.nan,
        "Opponent Team": scoring["opponent_team_final"],
        "Avg Score": scoring["avg_score"] if "avg_score" in scoring.columns else np.nan,
        "8+ Runs": scoring["eight_plus"] if "eight_plus" in scoring.columns else np.nan,
        "Top Score": scoring["top_score"] if "top_score" in scoring.columns else np.nan,
        "Team Ownership": scoring["team_own"] if "team_own" in scoring.columns else np.nan,
        "Scoring Sheet Strength": scoring["scoring_strength"],
        "Top-5 Median": scoring["top5_median"] if "top5_median" in scoring.columns else np.nan,
        "Top-5 Ceiling": scoring["top5_ceiling"] if "top5_ceiling" in scoring.columns else np.nan,
        "Top-5 Avg Own": scoring["top5_avg_own"] if "top5_avg_own" in scoring.columns else np.nan,
        "Hitter ROO Strength": scoring["roo_strength"],
        "Weighted HWSr": scoring["weighted_hwsr"],
        "Weighted True AVG": scoring["weighted_true_avg"],
        "Starter Vulnerability": scoring["starter_vulnerability"],
        "HardPerSwing Score": scoring["hardperswing_score"],
        "Bullpen Vulnerability Score": scoring["bullpen_vulnerability"],
        "Bullpen Classification": scoring["bullpen_classification"],
        "Ownership Leverage": scoring["ownership_leverage"],
        "Stack Edge Score": stack_edge,
        "Stack Confidence": edge_cov * 100.0,
        "Stack Recommendation": recs,
        "Reason": reasons,
    })
    return out.sort_values("Stack Edge Score", ascending=False, na_position="last").reset_index(drop=True)
