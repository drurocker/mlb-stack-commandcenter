from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

STYLE_STRENGTH = {"Conservative": 0.08, "Balanced": 0.16, "Aggressive": 0.25}
LABEL_BIAS = {"GOOD CHALK": 0.20, "BAD CHALK": -0.55, "LEVERAGE PIVOT": 0.55, "NEUTRAL": 0.0}


@dataclass(frozen=True)
class AllocationConfig:
    lineups: int
    leverage_style: str = "Balanced"
    max_team_exposure: float = 30.0

    def __post_init__(self):
        if self.lineups < 1:
            raise ValueError("Lineups must be at least 1.")
        if self.leverage_style not in STYLE_STRENGTH:
            raise ValueError(f"Unknown leverage style: {self.leverage_style}")
        if not 0 < self.max_team_exposure <= 100:
            raise ValueError("Max Team Exposure must be between 0 and 100.")


def _normalize_weights(weights: pd.Series) -> pd.Series:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(1.0 / len(w), index=w.index) if len(w) else w
    return w / total


def _capped_percentages(weights: pd.Series, cap_pct: float) -> pd.Series:
    norm = _normalize_weights(weights)
    n = len(norm)
    if n == 0:
        return norm
    result = pd.Series(0.0, index=norm.index, dtype=float)
    remaining = set(norm.index)
    remaining_pct = 100.0
    while remaining:
        rem_idx = list(remaining)
        rem_weights = norm.loc[rem_idx]
        if float(rem_weights.sum()) <= 0:
            share = remaining_pct / len(rem_idx)
            if share > cap_pct + 1e-12:
                raise ValueError("Max Team Exposure is too low to allocate all lineups across the available teams.")
            result.loc[rem_idx] = share
            break
        proposed = rem_weights / rem_weights.sum() * remaining_pct
        over = proposed[proposed > cap_pct + 1e-12]
        if over.empty:
            result.loc[rem_idx] = proposed
            break
        for idx in over.index:
            result.loc[idx] = cap_pct
            remaining.remove(idx)
            remaining_pct -= cap_pct
        if remaining_pct < -1e-9:
            raise ValueError("Max Team Exposure is too low to allocate all lineups across the available teams.")
    return result


def _largest_remainder_counts(exposure_pct: pd.Series, lineups: int, score: pd.Series, max_count: int | None = None) -> pd.Series:
    targets = exposure_pct / 100.0 * lineups
    counts = np.floor(targets).astype(int)
    counts = pd.Series(counts, index=exposure_pct.index, dtype=int)
    if max_count is not None:
        counts = counts.clip(upper=max_count)
    remaining = int(lineups - counts.sum())
    frac = targets - np.floor(targets)
    # deterministic: largest fractional remainder, then research score, then index string.
    order = sorted(exposure_pct.index, key=lambda i: (-float(frac.loc[i]), -float(score.loc[i]), str(i)))
    while remaining > 0:
        placed = False
        for idx in order:
            if max_count is not None and counts.loc[idx] >= max_count:
                continue
            counts.loc[idx] += 1
            remaining -= 1
            placed = True
            if remaining == 0:
                break
        if not placed:
            raise ValueError("Max Team Exposure is too low to allocate all requested lineups.")
    return counts


def allocate_stack_exposure(stack_chalk: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    if stack_chalk is None or stack_chalk.empty:
        return pd.DataFrame(columns=[
            "Team", "Research Weight", "Chalk Modifier", "Adjusted Weight", "Base Exposure",
            "Final Exposure", "Base Lineups", "Final Lineups", "Cap Adjustment", "Allocation Reason",
        ])
    if "Team" not in stack_chalk or ("Stack Edge Score" not in stack_chalk and "Research Strength Score" not in stack_chalk):
        raise ValueError("Stack allocation requires Team and a research score column.")

    out = stack_chalk.copy().reset_index(drop=True)
    score_col = "Research Strength Score" if "Research Strength Score" in out.columns else "Stack Edge Score"
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).clip(0, 100)
    # Research remains the foundation. Very weak stacks keep only a tiny neutral floor.
    research_weight = (score / 100.0) ** 2
    research_weight = research_weight.where(score >= 45.0, 0.0025)

    labels = out.get("Chalk Label", pd.Series("NEUTRAL", index=out.index)).fillna("NEUTRAL").astype(str)
    strength = STYLE_STRENGTH[config.leverage_style]
    bias = labels.map(LABEL_BIAS).fillna(0.0)
    chalk_modifier = 1.0 + strength * bias
    adjusted_weight = research_weight * chalk_modifier

    base_norm = _normalize_weights(adjusted_weight)
    base_exposure = base_norm * 100.0

    max_count = math.floor(config.lineups * config.max_team_exposure / 100.0 + 1e-12)
    if len(out) * max_count < config.lineups:
        raise ValueError(
            f"Max Team Exposure of {config.max_team_exposure:.1f}% is too low for {config.lineups} lineups and {len(out)} available teams."
        )

    final_target_pct = _capped_percentages(adjusted_weight, config.max_team_exposure)
    base_lineups = _largest_remainder_counts(base_exposure, config.lineups, score)
    final_lineups = _largest_remainder_counts(final_target_pct, config.lineups, score, max_count=max_count)
    final_exposure = final_lineups / config.lineups * 100.0

    out["Research Weight"] = research_weight
    out["Chalk Modifier"] = chalk_modifier
    out["Adjusted Weight"] = adjusted_weight
    out["Base Exposure"] = base_exposure
    out["Final Exposure"] = final_exposure
    out["Base Lineups"] = base_lineups
    out["Final Lineups"] = final_lineups
    out["Cap Adjustment"] = out["Final Exposure"] - out["Base Exposure"]

    reasons = []
    for label, cap_delta in zip(labels, out["Cap Adjustment"]):
        parts = ["Research strength is the allocation foundation"]
        if label == "LEVERAGE PIVOT":
            parts.append(f"{config.leverage_style} leverage boost")
        elif label == "BAD CHALK":
            parts.append(f"{config.leverage_style} bad-chalk trim")
        elif label == "GOOD CHALK":
            parts.append("supported chalk retained")
        if cap_delta < -0.05:
            parts.append("max exposure cap redistributed excess")
        reasons.append("; ".join(parts))
    out["Allocation Reason"] = reasons

    out["Allocation Research Score"] = score
    return out.sort_values(["Final Lineups", "Allocation Research Score"], ascending=[False, False]).reset_index(drop=True)
