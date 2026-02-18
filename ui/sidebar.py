"""
Drift Auditor — Sidebar rendering.
Extracted from app.py. Returns a config dict consumed by mode renderers.
"""

import os
import streamlit as st

from ui.theme import THEMES


def render_sidebar() -> dict:
    """
    Render the full sidebar and return a config dict.

    Returns keys:
        mode          — one of "📁 File Analysis", "⚡ Live Analysis", "📊 Regression"
        uploaded_file — UploadedFile or None
        system_prompt — str
        preferences   — str
        window_size   — int
        overlap       — int
        theme         — theme dict (THEMES[selected_name])
    """
    with st.sidebar:
        st.markdown("## 🧪 Drift Auditor")
        st.caption("Multi-turn drift diagnostic tool")

        analysis_mode = st.radio(
            "Mode",
            ["📁 File Analysis", "⚡ Live Analysis", "📊 Regression"],
            index=0,
            help="File: upload a conversation. Live: paste-as-you-go. Regression: batch analytics.",
        )

        if analysis_mode == "📁 File Analysis":
            st.markdown("---")
            uploaded = st.file_uploader(
                "Upload conversation",
                type=["txt", "json"],
                help="Supports Claude.ai exports (.json) and plain text transcripts (.txt)",
            )

            # Load sample button
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
            sample_conv_path = os.path.join(sample_dir, "sample_conversation.txt")
            sample_prompt_path = os.path.join(sample_dir, "sample_system_prompt.txt")

            if "sample_bytes" not in st.session_state:
                st.session_state.sample_bytes = None
            if "sample_prompt" not in st.session_state:
                st.session_state.sample_prompt = ""

            if st.button("📋 Load Sample Conversation", use_container_width=True):
                try:
                    with open(sample_conv_path, "r", encoding="utf-8") as f:
                        st.session_state.sample_bytes = f.read().encode("utf-8")
                    with open(sample_prompt_path, "r", encoding="utf-8") as f:
                        st.session_state.sample_prompt = f.read()
                except FileNotFoundError:
                    st.error("Sample files not found in examples/ directory.")

            st.markdown("---")
            st.markdown("**Configuration**")

            default_prompt = (
                st.session_state.sample_prompt
                if st.session_state.sample_bytes and not uploaded
                else ""
            )
            system_prompt = st.text_area(
                "System Prompt",
                value=default_prompt,
                height=100,
                placeholder="Optional: paste system prompt here…",
            )
            preferences = st.text_area(
                "User Preferences",
                height=70,
                placeholder="Optional: user-stated preferences…",
            )

            st.markdown("**Window Parameters**")
            window_size = st.slider("Window Size", 10, 100, 50, step=5)
            overlap = st.slider("Overlap", 0, 25, 10, step=1)
        else:
            uploaded = None
            system_prompt = ""
            preferences = ""
            window_size = 50
            overlap = 10

        # Theme — tucked at bottom in expander
        st.markdown("---")
        with st.expander("Theme", expanded=False):
            _theme_choice = st.radio(
                "Pick theme",
                list(THEMES.keys()),
                index=list(THEMES.keys()).index(st.session_state["theme_name"]),
                horizontal=True,
                key="_theme_radio",
                label_visibility="collapsed",
            )
            if _theme_choice != st.session_state["theme_name"]:
                st.session_state["theme_name"] = _theme_choice
                st.rerun()

    return {
        "mode": analysis_mode,
        "uploaded_file": uploaded,
        "system_prompt": system_prompt,
        "preferences": preferences,
        "window_size": window_size,
        "overlap": overlap,
        "theme": THEMES[st.session_state["theme_name"]],
    }
