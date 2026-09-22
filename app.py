from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_VERSION
from utils.ui import apply_global_styles

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_global_styles()

st.title(f"🏆 {APP_NAME}")
st.caption(f"{APP_VERSION} · MLB Slate Auto-Detection + Stack/Pitcher Intelligence")

st.success(
    "Open **Command Center** and drop all MLB slate CSVs into one uploader. "
    "The app classifies the files, scores stacks and bullpens, analyzes pitcher chalk/caps, "
    "and keeps pitcher exclusions manual."
)
st.page_link("pages/01_Command_Center.py", label="Open Command Center", icon="🚀", use_container_width=True)

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("One-shot slate upload")
    st.write("Scoring Sheet, Player ROO, Bullpen Research, Weighted HWSr, HardPerSwing, Starter Weakness, and lineup pools can be auto-classified by columns.")
with c2:
    st.subheader("Stack Edge + BVS")
    st.write("Scoring Sheet remains the anchor, with hitter ROO, starter vulnerability, the locked Bullpen Vulnerability Score, and ownership leverage layered in.")
with c3:
    st.subheader("Pitcher portfolio")
    st.write("Find fragile chalk, questionable pay-ups, exposure caps, opposing-stack leverage, and manual remove candidates.")

st.info("Missing research is excluded from scoring and lowers confidence. No pitcher is automatically removed.")
