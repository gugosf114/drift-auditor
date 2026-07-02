"""
Drift Auditor — Reusable UI components.
Streamlit-dependent functions live here alongside the pure-HTML helpers
(_audit_progress_html, calibration_scale_html) that are unit-testable.
"""

import streamlit as st
import plotly.graph_objects as go

from ui.theme import score_color, FONT_STACK_UI, FONT_STACK_DATA


# ---------------------------------------------------------------------------
# Metric Card
# ---------------------------------------------------------------------------

def render_metric_card(label: str, score: int, subtitle: str,
                       t: dict, hero: bool = False) -> None:
    """Render a panel readout with a severity-colored value."""
    color = score_color(score, t)
    card_cls = "glass-card-hero" if hero else "glass-card"
    val_cls = "metric-value-hero" if hero else "metric-value"
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="metric-label">{label}</div>
        <div class="{val_cls}" style="color: {color}">{score}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Calibration scale — signature element
# ---------------------------------------------------------------------------

def calibration_scale_html(score: float) -> str:
    """The 1-10 severity ramp with this conversation's reading marked on it.

    Pure HTML — relies on .cal-scale/.cal-marker/.cal-ticks theme CSS.
    """
    pct = max(0.0, min(100.0, (score - 1) / 9 * 100))
    return f"""
    <div class="cal-scale">
        <div class="cal-marker" style="left: calc({pct:.1f}% - 1px)"></div>
    </div>
    <div class="cal-ticks">
        <span>1</span><span>Clean</span><span>Moderate</span><span>Severe</span><span>10</span>
    </div>
    """


def render_hero_readout(label: str, score: int, subtitle: str, t: dict) -> None:
    """Hero instrument readout: big value + the calibration scale beneath it."""
    color = score_color(score, t)
    st.markdown(f"""
    <div class="glass-card-hero">
        <div class="metric-label">{label}</div>
        <div class="metric-value-hero" style="color: {color}">{score}</div>
        <div class="metric-sub">{subtitle}</div>
        {calibration_scale_html(score)}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart Export
# ---------------------------------------------------------------------------

def chart_export_png(fig: go.Figure, filename: str,
                     label: str = "Download chart (PNG)") -> None:
    """Render a Plotly figure to PNG bytes and offer a Streamlit download button."""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception:
        # kaleido not installed — degrade gracefully
        pass


# ---------------------------------------------------------------------------
# Audit progress (pure HTML, no Streamlit)
# ---------------------------------------------------------------------------

_AUDIT_STAGES = [
    "Parsing transcript",
    "Extracting instructions",
    "Scanning commission markers",
    "Tracking omissions across turns",
    "Checking correction persistence",
    "Scoring",
]


def _audit_progress_html(t: dict) -> str:
    """Quiet staged progress readout shown while the audit runs.

    A signal line sweeps beneath the stage list. All stages are shown up
    front — the audit is fast enough that per-stage animation would lie.
    """
    rows = "".join(
        f"""<div style="display:flex; align-items:baseline; gap:10px; padding:3px 0;">
            <span style="font-family:{FONT_STACK_DATA}; font-size:0.7rem; color:{t["muted"]};">{i + 1:02d}</span>
            <span style="font-family:{FONT_STACK_UI}; font-size:0.88rem; color:{t["chart_text"]};">{stage}</span>
        </div>"""
        for i, stage in enumerate(_AUDIT_STAGES)
    )
    return f"""
    <div style="max-width: 420px; margin: 48px auto; padding: 24px 28px;
                background: {t["surface"]}; border: 1px solid {t["border"]};
                border-radius: 2px;">
        <div style="font-family:{FONT_STACK_UI}; font-size:0.68rem; font-weight:500;
                    text-transform:uppercase; letter-spacing:0.12em;
                    color:{t["muted"]}; margin-bottom:12px;">
            Running audit
        </div>
        {rows}
        <div style="position:relative; height:2px; background:{t["border"]};
                    margin-top:16px; overflow:hidden; border-radius:1px;">
            <div style="position:absolute; top:0; left:0; height:100%; width:30%;
                        background:{t["accent"]};
                        animation: audit-sweep 1.4s ease-in-out infinite;"></div>
        </div>
        <style>
            @keyframes audit-sweep {{
                0%   {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(430%); }}
            }}
            @media (prefers-reduced-motion: reduce) {{
                [style*="audit-sweep"] {{ animation: none; }}
            }}
        </style>
    </div>
    """


# Backward-compatible alias — older modes imported the previous loader name.
_racing_stickmen_html = _audit_progress_html
