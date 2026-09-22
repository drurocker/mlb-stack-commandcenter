from __future__ import annotations

from html import escape
from typing import Iterable


TONE_CLASS = {name: f"dfs-tone-{name}" for name in ("cyan", "green", "yellow", "orange", "red")}


def status_badge(label: str, tone: str = "cyan") -> str:
    cls = TONE_CLASS.get(tone, TONE_CLASS["cyan"])
    return f'<span class="dfs-badge {cls}">{escape(str(label))}</span>'


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge: str = "",
    tone: str = "cyan",
    image_url: str | None = None,
    image_fallback_url: str | None = None,
) -> str:
    cls = TONE_CLASS.get(tone, TONE_CLASS["cyan"])
    image = ""
    if image_url:
        primary = escape(str(image_url), quote=True)
        if image_fallback_url and str(image_fallback_url) != str(image_url):
            fallback = escape(str(image_fallback_url), quote=True)
            onerror = f"this.onerror=null;this.src='{fallback}'"
        else:
            onerror = "this.style.display='none'"
        image = f'<img class="dfs-card-image" src="{primary}" alt="" onerror="{onerror}">'
    badge_html = status_badge(badge, tone) if badge else ""
    return (
        f'<div class="dfs-card {cls}">{image}'
        f'<div class="dfs-kicker">{escape(str(title))}</div>'
        f'<div class="dfs-score">{escape(str(value))}</div>'
        f'<div>{badge_html}</div>'
        f'<div class="dfs-muted">{escape(str(subtitle))}</div></div>'
    )


def entity_card(
    title: str,
    score: float | None,
    metrics: dict[str, object],
    reason: str,
    badge: str = "",
    tone: str = "cyan",
    image_url: str | None = None,
    image_fallback_url: str | None = None,
) -> str:
    score_text = "—" if score is None else f"{float(score):.1f}"
    metric_html = "".join(
        f'<div><span class="dfs-kicker">{escape(str(k))}</span><br>{escape(str(v))}</div>'
        for k, v in metrics.items()
    )
    base = metric_card(title, score_text, reason, badge, tone, image_url, image_fallback_url)
    return base[:-6] + f'<div class="dfs-metric-strip">{metric_html}</div></div>'


def checklist_row(label: str, detail: str, state: str) -> str:
    tone = {"READY": "green", "REVIEW": "yellow", "MISSING": "red", "ERROR": "red"}.get(state, "cyan")
    return (
        '<div class="dfs-check-row">'
        f'<span>{escape(str(label))}</span><span class="dfs-muted">{escape(str(detail))}</span>'
        f'{status_badge(state, tone)}</div>'
    )


def section_header(title: str, subtitle: str = "") -> None:
    import streamlit as st
    st.markdown(
        f'<div class="dfs-section-title">{escape(str(title))}</div><div class="dfs-muted">{escape(str(subtitle))}</div>',
        unsafe_allow_html=True,
    )


def render_html_grid(cards: Iterable[str], columns: int = 3, css_class: str = "dfs-grid") -> None:
    import streamlit as st
    cards = list(cards)
    if not cards:
        return
    st.markdown(
        f'<div class="{escape(css_class)}" style="--dfs-cols:{max(1, int(columns))}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def dashboard_header(title: str, subtitle: str = "", status: str = "") -> None:
    import streamlit as st
    status_html = status_badge(status, "green") if status else ""
    st.markdown(
        '<div class="dfs-header">'
        f'<div><div class="dfs-title">{escape(title)}</div><div class="dfs-subtitle">{escape(subtitle)}</div></div>'
        f'<div>{status_html}</div></div>',
        unsafe_allow_html=True,
    )
