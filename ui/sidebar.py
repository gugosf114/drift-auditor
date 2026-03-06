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
        mode          — one of "📁 File Analysis", "⚡ Live Analysis", "📊 Regression", "🕸️ Mesh Runtime"
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
            ["📁 File Analysis", "⚡ Live Analysis", "📊 Regression", "🕸️ Mesh Runtime"],
            index=0,
            help="File: upload a conversation. Live: paste-as-you-go. Regression: batch analytics. Mesh: lossless nonlinear persistence runtime.",
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

            # Auto-load sample on first visit so new users see results instantly
            if "first_visit_loaded" not in st.session_state:
                st.session_state.first_visit_loaded = True
                if not uploaded:
                    try:
                        with open(sample_conv_path, "r", encoding="utf-8") as f:
                            st.session_state.sample_bytes = f.read().encode("utf-8")
                        with open(sample_prompt_path, "r", encoding="utf-8") as f:
                            st.session_state.sample_prompt = f.read()
                    except FileNotFoundError:
                        pass

            if st.button("Load Sample Conversation", use_container_width=True):
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
        elif analysis_mode == "🕸️ Mesh Runtime":
            st.markdown("---")
            st.markdown("**Mesh Inputs**")
            mesh_uploaded_files = st.file_uploader(
                "Upload transcripts",
                type=["txt", "json"],
                accept_multiple_files=True,
                help="Upload one or many raw transcript files to compile a relationship mesh.",
            )
            mesh_index_upload = st.file_uploader(
                "Or load existing mesh index JSON",
                type=["json"],
                accept_multiple_files=False,
                help="Loads a previously exported relationship mesh index.",
            )

            st.markdown("**Mesh Compile Options**")
            mesh_include_audit = st.checkbox(
                "Include drift audit during compile",
                value=True,
                help="Runs drift audit per transcript during mesh compile. Turn off for faster compile.",
            )
            mesh_normalize_voice = st.checkbox(
                "Normalize voice-to-text noise",
                value=True,
                help="Applies typo/noise normalization while preserving raw source hash.",
            )

            st.markdown("**Mesh Query**")
            mesh_query = st.text_area(
                "Runtime query",
                height=90,
                placeholder="e.g., connect checkpoint memory failures to trust-repair loops",
            )
            mesh_top_k = st.slider("Top-K retrieval hits", 1, 20, 8, step=1)

            st.markdown("**Relationship State**")
            mesh_trust = st.slider("Trust", 0.0, 1.0, 0.7, 0.05)
            mesh_urgency = st.slider("Urgency", 0.0, 1.0, 0.5, 0.05)
            mesh_autonomy = st.slider("Autonomy preference", 0.0, 1.0, 0.7, 0.05)
            mesh_verbosity = st.slider("Verbosity tolerance", 0.0, 1.0, 0.3, 0.05)
            mesh_evidence = st.slider("Evidence strictness", 0.0, 1.0, 0.8, 0.05)
            mesh_debt = st.slider("Correction debt", 0.0, 1.0, 0.0, 0.05)
            mesh_unresolved = st.slider("Unresolved threads", 0, 50, 0, 1)
            mesh_risk = st.slider("Risk posture", 0.0, 1.0, 0.8, 0.05)
            mesh_continuity = st.slider("Continuity confidence", 0.0, 1.0, 0.8, 0.05)

            uploaded = None
            system_prompt = ""
            preferences = ""
            window_size = 50
            overlap = 10
        else:
            uploaded = None
            system_prompt = ""
            preferences = ""
            window_size = 50
            overlap = 10
            mesh_uploaded_files = []
            mesh_index_upload = None
            mesh_include_audit = True
            mesh_normalize_voice = True
            mesh_query = ""
            mesh_top_k = 8
            mesh_trust = 0.7
            mesh_urgency = 0.5
            mesh_autonomy = 0.7
            mesh_verbosity = 0.3
            mesh_evidence = 0.8
            mesh_debt = 0.0
            mesh_unresolved = 0
            mesh_risk = 0.8
            mesh_continuity = 0.8

        if analysis_mode != "🕸️ Mesh Runtime":
            mesh_uploaded_files = []
            mesh_index_upload = None
            mesh_include_audit = True
            mesh_normalize_voice = True
            mesh_query = ""
            mesh_top_k = 8
            mesh_trust = 0.7
            mesh_urgency = 0.5
            mesh_autonomy = 0.7
            mesh_verbosity = 0.3
            mesh_evidence = 0.8
            mesh_debt = 0.0
            mesh_unresolved = 0
            mesh_risk = 0.8
            mesh_continuity = 0.8

        # Metrics glossary
        st.markdown("---")
        with st.expander("Explain Metrics", expanded=False):
            st.markdown("""
**Drift Score** (1-10): How much the model strayed from your instructions. 1 = clean, 10 = severe.

**OLI** (Operator Load Index): How often you had to intervene. Lower = the model stays on track by itself.

**Alignment Tax**: What percentage of the conversation was spent correcting the model instead of doing productive work.

**Instruction Survival Rate**: Of all instructions you gave, what percentage were still being followed at the end.

**Correction Efficiency**: When you corrected the model, did the fix stick? 100% = every correction held.

**Self-Sufficiency**: Can this model run unsupervised? 100 = no hand-holding needed. 0 = constant babysitting.

**Frustration Index**: Experimental sentiment proxy measuring operator frustration over time.

---
**Detection Layers**
- **L1 Commission**: Model said something wrong (sycophancy, hallucination, false confidence)
- **L2 Omission**: Model stopped doing something right (dropped instructions, semantic dilution)
- **L3 Persistence**: Model acknowledged a correction but reverted later
- **L4 Barometer**: Structural posture shifts (hedging, certainty swings)
""")

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
        "mesh_uploaded_files": mesh_uploaded_files,
        "mesh_index_upload": mesh_index_upload,
        "mesh_include_audit": mesh_include_audit,
        "mesh_normalize_voice": mesh_normalize_voice,
        "mesh_query": mesh_query,
        "mesh_top_k": mesh_top_k,
        "mesh_trust": mesh_trust,
        "mesh_urgency": mesh_urgency,
        "mesh_autonomy": mesh_autonomy,
        "mesh_verbosity": mesh_verbosity,
        "mesh_evidence": mesh_evidence,
        "mesh_debt": mesh_debt,
        "mesh_unresolved": mesh_unresolved,
        "mesh_risk": mesh_risk,
        "mesh_continuity": mesh_continuity,
        "theme": THEMES[st.session_state["theme_name"]],
    }
