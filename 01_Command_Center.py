from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from config import APP_NAME, APP_VERSION
from modules.mlb_chalk import classify_pitcher_chalk, classify_stack_chalk
from modules.mlb_hitters import build_hitter_board
from modules.mlb_game_environment import (
    add_weather_to_matchups, build_ballpark_profiles, build_player_environment,
    fetch_mlb_handedness_map, fetch_mlb_schedule,
)
from modules.mlb_projection_engine import build_adjusted_projections
from modules.mlb_projection_chalk import classify_projection_chalk
from modules.mlb_pitcher_portfolio import (
    PitcherPortfolioConfig,
    analyze_pitcher_portfolio,
    apply_pitcher_exclusions,
    build_pitcher_stack_leverage,
    merge_pitcher_sources,
    preview_pitcher_exclusions,
)
from modules.mlb_slate_files import (
    FILE_TYPES,
    build_bullpen_vulnerability_table,
    build_full_stack_edge_table,
    classify_mlb_dataframe,
)
from utils.ui import apply_global_styles
from views.setup import render_setup
import views.overview as overview_view
import views.stacks as stacks_view
import views.chalk_board as chalk_board_view
import views.projections as projections_view
import views.pitchers as pitchers_view
import views.bullpens as bullpens_view
import views.hitters as hitters_view
import views.lineups as lineups_view
import views.settings as settings_view
import views.game_environment as game_environment_view

BUILD_LABEL = "v13.50 · SINGLE-UPLOAD BUILD · GAME ENVIRONMENT INTELLIGENCE"
SOURCE_BASELINE_LABEL = "v13.48.2 baseline"
UPLOAD_SECTION_LABEL = "Upload MLB Slate Files"
ANALYZE_ACTION_LABEL = "Analyze Sheets"
SAFETY_ACTION_LABEL = "Apply Suggested Pitcher Exclusions"
NAV_ITEMS = ["Overview", "Stacks", "Chalk Board", "Projections", "Game Environments", "Pitchers", "Bullpens", "Hitters", "Lineups", "Settings"]

st.set_page_config(
    page_title=f"Drew MLB Command Center · {APP_VERSION}",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_global_styles()


def read_upload(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError(f"Unsupported file type: {uploaded.name}")


def _team_research_from_stack_edges(stack_edges: pd.DataFrame) -> pd.DataFrame:
    if stack_edges is None or stack_edges.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Team": stack_edges["Team"]})
    for source, target in [
        ("HardPerSwing Score", "HardPerSwing"),
        ("Stack Edge Score", "Stack Score"),
        ("Scoring Sheet Strength", "Raw Baseball Score"),
        ("Hitter ROO Strength", "DFS Stack Score"),
    ]:
        if source in stack_edges.columns:
            out[target] = stack_edges[source]
    return out


def _analysis_config() -> PitcherPortfolioConfig:
    defaults = PitcherPortfolioConfig()
    return PitcherPortfolioConfig(
        overexposure_delta_pp=float(st.session_state.get("overexposure_delta_pp", defaults.overexposure_delta_pp)),
        overexposure_ratio=float(st.session_state.get("overexposure_ratio", defaults.overexposure_ratio)),
        underexposure_delta_pp=float(st.session_state.get("underexposure_delta_pp", defaults.underexposure_delta_pp)),
        reduce_risk_min=float(st.session_state.get("reduce_risk_min", defaults.reduce_risk_min)),
        remove_risk_min=float(st.session_state.get("remove_risk_min", defaults.remove_risk_min)),
        remove_min_negative_signals=int(st.session_state.get("remove_min_negative_signals", defaults.remove_min_negative_signals)),
    )


def build_analysis(source_map: dict[str, pd.DataFrame | None], slate_date: str | None = None) -> dict[str, Any]:
    scoring_df = source_map.get("Scoring Sheet")
    roo_df = source_map.get("Player ROO")
    bullpen_df = source_map.get("Bullpen Research")
    hwsr_df = source_map.get("Weighted HWSr")
    hard_df = source_map.get("Team HardPerSwing")
    starter_weakness_df = source_map.get("Starter Weakness")
    lineup_pool = source_map.get("Lineup Portfolio")
    ballpark_df = source_map.get("Ballpark Research")
    errors: list[str] = []

    matchups = pd.DataFrame()
    ballpark_profiles = pd.DataFrame()
    player_environment = pd.DataFrame()
    team_environment = pd.DataFrame()
    if slate_date:
        try:
            matchups = fetch_mlb_schedule(slate_date)
            if not matchups.empty:
                matchups = add_weather_to_matchups(matchups)
        except Exception as exc:
            errors.append(f"MLB matchup/weather lookup unavailable: {exc}")
    if ballpark_df is not None and not ballpark_df.empty:
        try:
            ballpark_profiles = build_ballpark_profiles(ballpark_df)
            if roo_df is not None and not roo_df.empty and not matchups.empty:
                season = int(str(slate_date)[:4]) if slate_date else date.today().year
                handedness = fetch_mlb_handedness_map(season)
                player_environment = build_player_environment(roo_df, matchups, ballpark_profiles, handedness=handedness)
                if not player_environment.empty:
                    hitters_env = player_environment.loc[~player_environment["is_pitcher"].fillna(False)].copy()
                    if not hitters_env.empty:
                        team_environment = hitters_env.groupby("team", as_index=False).agg(
                            **{
                                "Game Environment Score": ("game_environment_score", "mean"),
                                "Environment Confidence": ("environment_confidence", "mean"),
                                "Contact Environment": ("contact_environment", "mean"),
                                "Power Environment": ("power_environment", "mean"),
                                "Weather Score": ("weather_score", "mean"),
                            }
                        ).rename(columns={"team":"Team"})
        except Exception as exc:
            errors.append(f"Game Environment could not be created: {exc}")

    bullpen_table = pd.DataFrame()
    if bullpen_df is not None and not bullpen_df.empty:
        try:
            bullpen_table = build_bullpen_vulnerability_table(bullpen_df)
        except Exception as exc:
            errors.append(f"Bullpen BVS could not be created: {exc}")

    stack_edges = pd.DataFrame()
    if scoring_df is not None and not scoring_df.empty:
        try:
            stack_edges = build_full_stack_edge_table(
                scoring_df=scoring_df,
                player_roo_df=roo_df,
                bullpen_df=bullpen_df,
                weighted_hwsr_df=hwsr_df,
                hardperswing_df=hard_df,
                starter_weakness_df=starter_weakness_df,
            )
        except Exception as exc:
            errors.append(f"Stack Edge could not be created: {exc}")

    if not stack_edges.empty:
        if not team_environment.empty:
            stack_edges = stack_edges.merge(team_environment, on="Team", how="left")
        else:
            stack_edges["Game Environment Score"] = np.nan
        # Pure research score for allocation/chalk: ownership is intentionally excluded.
        components = [
            ("Scoring Sheet Strength", .36), ("Hitter ROO Strength", .18),
            ("Starter Vulnerability", .14), ("Bullpen Vulnerability Score", .14),
            ("HardPerSwing Score", .08), ("Game Environment Score", .10),
        ]
        numer = pd.Series(0.0, index=stack_edges.index)
        denom = pd.Series(0.0, index=stack_edges.index)
        for col, weight in components:
            if col not in stack_edges:
                continue
            vals = pd.to_numeric(stack_edges[col], errors="coerce")
            mask = vals.notna()
            numer.loc[mask] += vals.loc[mask] * weight
            denom.loc[mask] += weight
        stack_edges["Research Strength Score"] = numer.div(denom.where(denom > 0)).clip(0, 100)

    pitcher_analysis = pd.DataFrame()
    if any(df is not None and not df.empty for df in [roo_df, hwsr_df, starter_weakness_df]):
        try:
            team_for_pitchers = _team_research_from_stack_edges(stack_edges)
            primary_research = hwsr_df if hwsr_df is not None and not hwsr_df.empty else starter_weakness_df
            merged_pitchers = merge_pitcher_sources(
                pitcher_research_df=primary_research,
                team_research_df=team_for_pitchers if not team_for_pitchers.empty else None,
                scoring_df=roo_df,
                lineup_pool=lineup_pool,
            )
            if hwsr_df is not None and not hwsr_df.empty and starter_weakness_df is not None and not starter_weakness_df.empty:
                merged_pitchers = merge_pitcher_sources(
                    portfolio_df=merged_pitchers,
                    pitcher_research_df=starter_weakness_df,
                    team_research_df=team_for_pitchers if not team_for_pitchers.empty else None,
                    scoring_df=roo_df,
                    lineup_pool=lineup_pool,
                )
            pitcher_analysis = analyze_pitcher_portfolio(merged_pitchers, config=_analysis_config())
            if not pitcher_analysis.empty and not player_environment.empty and "Pitcher" in pitcher_analysis:
                p_env = player_environment.loc[player_environment["is_pitcher"].fillna(False)].copy()
                if not p_env.empty:
                    env_cols = [c for c in ["player_names","throws","game_environment_score","environment_confidence","venue","weather_score","temperature","wind_speed","roof_type"] if c in p_env]
                    pitcher_analysis = pitcher_analysis.merge(p_env[env_cols].drop_duplicates("player_names"), left_on="Pitcher", right_on="player_names", how="left").drop(columns=["player_names"], errors="ignore")
        except Exception as exc:
            errors.append(f"Pitcher Portfolio Analysis could not be created: {exc}")

    pitcher_stack = (
        build_pitcher_stack_leverage(pitcher_analysis, stack_edges)
        if not pitcher_analysis.empty and not stack_edges.empty
        else pd.DataFrame()
    )
    stack_chalk = classify_stack_chalk(stack_edges) if not stack_edges.empty else pd.DataFrame()
    pitcher_chalk = classify_pitcher_chalk(pitcher_analysis) if not pitcher_analysis.empty else pd.DataFrame()
    hitter_board = build_hitter_board(roo_df, stack_edges) if roo_df is not None and not roo_df.empty else pd.DataFrame()
    if not hitter_board.empty and not player_environment.empty:
        env_cols = [c for c in ["player_names","bats","opposing_pitcher_throws","park_split","contact_environment","power_environment","weather_score","game_environment_score","environment_confidence"] if c in player_environment]
        hitter_board = hitter_board.merge(player_environment[env_cols].drop_duplicates("player_names"), left_on="Player", right_on="player_names", how="left").drop(columns=["player_names"], errors="ignore")

    adjusted_projections = pd.DataFrame()
    projection_chalk = pd.DataFrame()
    projection_model_version = ""
    if roo_df is not None and not roo_df.empty:
        try:
            adjusted_projections = build_adjusted_projections(
                player_roo_df=roo_df,
                scoring_df=scoring_df,
                bullpen_df=bullpen_df,
                weighted_hwsr_df=hwsr_df,
                hardperswing_df=hard_df,
                starter_weakness_df=starter_weakness_df,
                game_environment_df=player_environment,
            )
            projection_chalk = classify_projection_chalk(adjusted_projections)
            if not adjusted_projections.empty and "model_version" in adjusted_projections:
                projection_model_version = str(adjusted_projections["model_version"].iloc[0])
        except Exception as exc:
            errors.append(f"Projection Intelligence could not be created: {exc}")

    return {
        "source_map": source_map,
        "stack_edges": stack_edges,
        "bullpen_table": bullpen_table,
        "pitcher_analysis": pitcher_analysis,
        "pitcher_stack": pitcher_stack,
        "stack_chalk": stack_chalk,
        "pitcher_chalk": pitcher_chalk,
        "hitter_board": hitter_board,
        "adjusted_projections": adjusted_projections,
        "projection_chalk": projection_chalk,
        "projection_model_version": projection_model_version,
        "lineup_pool": lineup_pool,
        "matchups": matchups,
        "ballpark_profiles": ballpark_profiles,
        "player_environment": player_environment,
        "team_environment": team_environment,
        "errors": errors,
    }


if "mlb_run_analysis" not in st.session_state:
    st.session_state["mlb_run_analysis"] = False
if "mlb_analysis" not in st.session_state:
    st.session_state["mlb_analysis"] = {}

# Keep exactly one uploader in the page. Everything else is presentation/state.
bulk_files = st.file_uploader(
    "Drop all CSV / Excel files here",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="mlb_bulk_upload",
)

slate_date_value = st.date_input("Slate date", value=st.session_state.get("mlb_slate_date", date.today()), key="mlb_slate_date")

records: List[Dict[str, Any]] = []
read_errors: List[str] = []
for idx, uploaded in enumerate(bulk_files or []):
    try:
        frame = read_upload(uploaded)
        detected = classify_mlb_dataframe(frame, uploaded.name)
        records.append({
            "id": idx,
            "name": uploaded.name,
            "df": frame,
            "detected": detected["type"],
            "confidence": detected["confidence"],
            "candidates": detected["candidates"],
            "rows": len(frame),
        })
    except Exception as exc:
        read_errors.append(f"{uploaded.name}: {exc}")

if not st.session_state["mlb_run_analysis"]:
    clicked, source_map, source_name = render_setup(
        records, FILE_TYPES, read_errors, upload_label=UPLOAD_SECTION_LABEL, analyze_label=ANALYZE_ACTION_LABEL
    )
    if clicked:
        analyzed = build_analysis(source_map, slate_date=str(slate_date_value))
        st.session_state["mlb_analysis"] = analyzed
        st.session_state["mlb_adjusted_projections"] = analyzed["adjusted_projections"]
        st.session_state["mlb_projection_chalk"] = analyzed["projection_chalk"]
        st.session_state["mlb_projection_model_version"] = analyzed["projection_model_version"]
        st.session_state["mlb_source_name"] = source_name
        st.session_state["mlb_run_analysis"] = True
        st.session_state["mlb_active_nav"] = "Overview"
        st.rerun()
    st.stop()

analysis = st.session_state.get("mlb_analysis", {})
for message in analysis.get("errors", []):
    st.warning(message)

header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown("## ⚾ MLB DFS COMMAND CENTER")
    st.caption(f"{APP_NAME} · {BUILD_LABEL} · upgraded from {SOURCE_BASELINE_LABEL} · Research → Analyze → Build")
with header_right:
    if st.button("Change Slate", use_container_width=True):
        st.session_state["mlb_run_analysis"] = False
        st.session_state["mlb_analysis"] = {}
        st.session_state["mlb_adjusted_projections"] = pd.DataFrame()
        st.session_state["mlb_projection_chalk"] = pd.DataFrame()
        st.session_state["mlb_projection_model_version"] = ""
        st.session_state["mlb_active_nav"] = "Overview"
        st.rerun()

nav = st.segmented_control(
    "Dashboard navigation",
    NAV_ITEMS,
    default=st.session_state.get("mlb_active_nav", "Overview"),
    key="mlb_active_nav",
    label_visibility="collapsed",
)
active = nav or "Overview"

game_environment_view.render_matchup_strip(analysis)

# These references keep manual-exclusion safety part of the orchestrated dashboard contract.
_ = (preview_pitcher_exclusions, apply_pitcher_exclusions, SAFETY_ACTION_LABEL)

if active == "Overview":
    overview_view.render_overview(analysis)
elif active == "Stacks":
    stacks_view.render_stacks(analysis)
elif active == "Chalk Board":
    chalk_board_view.render_chalk_board(analysis)
elif active == "Projections":
    projections_view.render_projections(analysis)
elif active == "Game Environments":
    game_environment_view.render_game_environments(analysis)
elif active == "Pitchers":
    pitchers_view.render_pitchers(analysis)
elif active == "Bullpens":
    bullpens_view.render_bullpens(analysis)
elif active == "Hitters":
    hitters_view.render_hitters(analysis)
elif active == "Lineups":
    lineups_view.render_lineups(analysis, settings={})
elif active == "Settings":
    settings_view.render_settings(analysis, settings={})
