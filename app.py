"""
Drift Auditor — Entry Point
============================
Thin dispatcher (~80 lines). All logic lives in ui/modes/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st

from ui.theme import THEMES, _build_css
from ui.sidebar import render_sidebar
from ui.modes.file_analysis import render_file_analysis_mode
from ui.modes.live_analysis import render_live_analysis_mode
from ui.modes.regression import render_regression_mode

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme
if "theme_name" not in st.session_state:
    st.session_state["theme_name"] = "Ember"
T = THEMES[st.session_state["theme_name"]]
st.markdown(_build_css(T), unsafe_allow_html=True)

# Sidebar → returns config dict
config = render_sidebar()

# Mode dispatch
mode = config["mode"]
if mode == "⚡ Live Analysis":
    render_live_analysis_mode(config)
elif mode == "📊 Regression":
    render_regression_mode()
else:
    render_file_analysis_mode(config)
