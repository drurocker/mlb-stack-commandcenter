"""Exports for the Research-Adjusted Projection Engine.

The Portfolio Manager export is intentionally narrow and stable. The audit
export is intentionally wide so model behavior can be evaluated later.
"""
from __future__ import annotations

import pandas as pd

PORTFOLIO_MANAGER_COLUMNS = [
    "player_names",
    "position",
    "team",
    "salary",
    "median",
    "ownership",
    "captain ownership",
]


def build_portfolio_manager_export(adjusted_df: pd.DataFrame) -> pd.DataFrame:
    """Return the exact external Portfolio Manager schema.

    ``captain ownership`` is a template-only field and is always blank by
    design. Only ``median`` is replaced, with the canonical Drew-adjusted
    median. All other fields are passthrough values from the canonical result.
    """
    if adjusted_df is None or adjusted_df.empty:
        return pd.DataFrame(columns=PORTFOLIO_MANAGER_COLUMNS)

    required = ["player_names", "position", "team", "salary", "drew_median", "ownership"]
    missing = [column for column in required if column not in adjusted_df.columns]
    if missing:
        raise ValueError(f"Adjusted projections missing required export columns: {', '.join(missing)}")

    return pd.DataFrame(
        {
            "player_names": adjusted_df["player_names"],
            "position": adjusted_df["position"],
            "team": adjusted_df["team"],
            "salary": adjusted_df["salary"],
            "median": adjusted_df["drew_median"],
            "ownership": adjusted_df["ownership"],
            "captain ownership": [""] * len(adjusted_df),
        },
        index=adjusted_df.index,
    ).reset_index(drop=True)


def build_model_audit_export(
    adjusted_df: pd.DataFrame,
    chalk_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the canonical adjusted result plus optional DFS chalk diagnostics.

    All engine diagnostics are preserved, including raw research fields, factor
    edges and contribution columns. Chalk fields are merged downstream by
    player identity and never recompute or mutate projections.
    """
    if adjusted_df is None:
        return pd.DataFrame()
    audit = adjusted_df.copy().reset_index(drop=True)
    if audit.empty or chalk_df is None or chalk_df.empty or "player_names" not in chalk_df.columns:
        return audit

    chalk_columns = [
        column
        for column in (
            "adjusted_median_rank",
            "adjusted_ceiling_rank",
            "ownership_rank",
            "research_rank_gap",
            "chalk_classification",
            "chalk_reason",
        )
        if column in chalk_df.columns
    ]
    if not chalk_columns:
        return audit

    # Avoid suffixing fields that may already have been carried from a
    # downstream classified frame; chalk is an interpretation layer only.
    for column in chalk_columns:
        if column in audit.columns:
            audit = audit.drop(columns=column)
    chalk_unique = chalk_df[["player_names", *chalk_columns]].drop_duplicates("player_names", keep="first")
    return audit.merge(chalk_unique, on="player_names", how="left", validate="many_to_one")


def csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe as UTF-8 CSV with no pandas index column."""
    return df.to_csv(index=False).encode("utf-8")
