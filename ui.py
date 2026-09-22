from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
  --dfs-bg:#07111f; --dfs-panel:#0c1828; --dfs-panel-2:#101f31;
  --dfs-border:#1b4c68; --dfs-cyan:#46d8ff; --dfs-text:#f5f9ff;
  --dfs-muted:#8ea4b8; --dfs-green:#37d67a; --dfs-yellow:#f2cf5b;
  --dfs-orange:#ff9f43; --dfs-red:#ff5d6c;
}
.stApp { background: radial-gradient(circle at 10% 0%, #10243b 0%, var(--dfs-bg) 34%); color:var(--dfs-text); }
.block-container { padding-top:1rem; padding-bottom:3rem; max-width:1680px; }
[data-testid="stSidebar"] { background:#081522; }
[data-testid="stHeader"] { background:rgba(7,17,31,.78); }
.dfs-card { background:linear-gradient(180deg,rgba(16,31,49,.98),rgba(9,20,34,.98)); border:1px solid var(--dfs-border); border-radius:16px; padding:16px; box-shadow:0 12px 32px rgba(0,0,0,.24); min-height:140px; }
.dfs-card-image { width:58px; height:58px; object-fit:contain; float:right; margin-left:12px; border-radius:10px; }
.dfs-kicker { color:var(--dfs-muted); font-size:.74rem; letter-spacing:.09em; font-weight:700; text-transform:uppercase; }
.dfs-score { color:var(--dfs-text); font-size:2.25rem; line-height:1; font-weight:850; margin:.35rem 0 .45rem; }
.dfs-muted { color:var(--dfs-muted); }
.dfs-badge { display:inline-flex; align-items:center; padding:.18rem .5rem; border-radius:999px; font-size:.72rem; font-weight:800; letter-spacing:.04em; margin:.15rem 0 .45rem; border:1px solid currentColor; }
.dfs-tone-cyan { color:var(--dfs-cyan); }
.dfs-tone-green { color:var(--dfs-green); }
.dfs-tone-yellow { color:var(--dfs-yellow); }
.dfs-tone-orange { color:var(--dfs-orange); }
.dfs-tone-red { color:var(--dfs-red); }
.dfs-card.dfs-tone-green { border-color:rgba(55,214,122,.48); }
.dfs-card.dfs-tone-yellow { border-color:rgba(242,207,91,.48); }
.dfs-card.dfs-tone-orange { border-color:rgba(255,159,67,.55); }
.dfs-card.dfs-tone-red { border-color:rgba(255,93,108,.52); }
.dfs-grid { display:grid; grid-template-columns:repeat(var(--dfs-cols),minmax(0,1fr)); gap:14px; margin:.6rem 0 1.1rem; }
.dfs-grid-6 { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; }
.dfs-metric-strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(90px,1fr)); gap:8px; margin:.65rem 0; clear:both; }
.dfs-metric-strip>div { background:rgba(255,255,255,.025); border:1px solid rgba(142,164,184,.12); border-radius:10px; padding:8px; }
.dfs-section-title { color:var(--dfs-text); font-size:1.15rem; font-weight:850; letter-spacing:.02em; margin-top:.7rem; }
.dfs-check-row { display:grid; grid-template-columns:1.4fr 1.6fr auto; gap:12px; align-items:center; padding:10px 12px; margin:6px 0; border:1px solid rgba(142,164,184,.16); border-radius:11px; background:rgba(16,31,49,.66); }
.dfs-note { padding:.8rem 1rem; border:1px solid rgba(70,216,255,.22); border-radius:.7rem; margin:.4rem 0 1rem 0; background:rgba(16,31,49,.55); }
.drew-note { padding:.8rem 1rem; border:1px solid rgba(128,128,128,.25); border-radius:.6rem; margin:.4rem 0 1rem 0; }
.dfs-chalkboard { background:linear-gradient(180deg,#10251f,#0a1916); border:1px solid rgba(148,180,166,.28); border-radius:18px; padding:18px; box-shadow:inset 0 0 35px rgba(0,0,0,.24); }
.dfs-header { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; border-bottom:1px solid rgba(70,216,255,.18); padding:.3rem 0 .8rem; margin-bottom:.75rem; }
.dfs-title { font-size:1.7rem; line-height:1; font-weight:900; letter-spacing:.02em; }
.dfs-subtitle { color:var(--dfs-muted); font-size:.82rem; margin-top:.3rem; }
@media (max-width:1100px) { .dfs-grid-6 { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width:760px) {
  .block-container { padding-top:.55rem; padding-left:.7rem; padding-right:.7rem; padding-bottom:2rem; }
  .dfs-grid,.dfs-grid-6 { grid-template-columns:1fr !important; gap:10px; }
  .dfs-card { padding:13px; min-height:0; border-radius:13px; }
  .dfs-card-image { width:48px; height:48px; }
  .dfs-score { font-size:1.8rem; }
  .dfs-check-row { grid-template-columns:1fr; gap:6px; }
  .dfs-header { align-items:flex-start; flex-direction:column; gap:8px; }
  .dfs-title { font-size:1.35rem; }
  .dfs-section-title { font-size:1.05rem; }
  h1 { font-size:1.65rem !important; }
  h2 { font-size:1.35rem !important; }
  h3 { font-size:1.15rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.55rem !important; }
  [data-testid="stColumn"] { min-width:100% !important; flex:1 1 100% !important; }
  [data-testid="stSegmentedControl"] [role="radiogroup"] { flex-wrap:wrap !important; gap:.28rem !important; }
  [data-testid="stSegmentedControl"] button { min-height:40px; }
  [data-testid="stFileUploaderDropzone"] { padding:.7rem !important; }
  [data-testid="stDataFrame"] { font-size:.78rem; }
  .stButton > button, .stDownloadButton > button { min-height:44px; }
}
</style>
"""


def apply_global_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
