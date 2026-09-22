from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_dfs_plotly_theme(fig, *, height: int = 430):
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,.35)",
        font={"color": "#dbe8f5", "family": "Arial"},
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
        height=height,
        hoverlabel={"bgcolor": "#101f31", "font_color": "#f5f9ff"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor="rgba(142,164,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(142,164,184,.12)", zeroline=False)
    return fig


def opportunity_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str | None = None,
    title: str = "Opportunity Map",
    size: str | None = None,
    hover_data: Sequence[str] | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
):
    data = df.copy() if df is not None else pd.DataFrame()
    if data.empty or x not in data or y not in data:
        fig = go.Figure()
        fig.update_layout(title=title)
        return apply_dfs_plotly_theme(fig)
    kwargs = {"x": x, "y": y, "title": title}
    if label and label in data:
        kwargs["text"] = label
    if size and size in data:
        kwargs["size"] = size
    if hover_data:
        kwargs["hover_data"] = [c for c in hover_data if c in data]
    fig = px.scatter(data, **kwargs)
    if label and label in data:
        fig.update_traces(textposition="top center", marker={"line": {"width": 1, "color": "#46d8ff"}})
    fig.update_xaxes(title=x_title or x)
    fig.update_yaxes(title=y_title or y)
    return apply_dfs_plotly_theme(fig)


def heatmap_figure(
    df: pd.DataFrame,
    *,
    index: str,
    columns: Sequence[str],
    title: str = "Heatmap",
    reverse_scale: bool = False,
):
    data = df.copy() if df is not None else pd.DataFrame()
    cols = [c for c in columns if c in data]
    if data.empty or index not in data or not cols:
        fig = go.Figure()
        fig.update_layout(title=title)
        return apply_dfs_plotly_theme(fig)
    z = data[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scale = [[0.0, "#37d67a"], [0.45, "#f2cf5b"], [0.72, "#ff5d6c"], [1.0, "#ff9f43"]]
    if reverse_scale:
        scale = [[p, c] for p, c in [(0.0, "#ff9f43"), (0.28, "#ff5d6c"), (0.55, "#f2cf5b"), (1.0, "#37d67a")]]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=data[index].astype(str), colorscale=scale, colorbar={"title": "Score"}, hoverongaps=False))
    fig.update_layout(title=title)
    return apply_dfs_plotly_theme(fig, height=max(360, min(900, 50 + 24 * len(data))))


def horizontal_bar(df: pd.DataFrame, *, label: str, value: str, title: str, ascending: bool = True):
    data = df.copy() if df is not None else pd.DataFrame()
    if data.empty or label not in data or value not in data:
        fig = go.Figure(); fig.update_layout(title=title); return apply_dfs_plotly_theme(fig)
    data = data.sort_values(value, ascending=ascending)
    fig = px.bar(data, x=value, y=label, orientation="h", title=title, text_auto=".1f")
    return apply_dfs_plotly_theme(fig, height=max(360, min(760, 80 + 28 * len(data))))


def exposure_bar(df: pd.DataFrame, *, label: str, before: str, after: str, title: str):
    data = df.copy() if df is not None else pd.DataFrame()
    fig = go.Figure()
    if not data.empty and label in data:
        if before in data:
            fig.add_bar(name="Before", x=data[label].astype(str), y=data[before])
        if after in data:
            fig.add_bar(name="After", x=data[label].astype(str), y=data[after])
    fig.update_layout(title=title, barmode="group")
    return apply_dfs_plotly_theme(fig)
