"""
Drift Auditor — Reusable UI components.
Extracted from app.py. Streamlit-dependent functions live here alongside
the one pure-HTML helper (_racing_stickmen_html) that is unit-testable.
"""

import random
import streamlit as st
import plotly.graph_objects as go

from ui.theme import score_color


# ---------------------------------------------------------------------------
# Metric Card
# ---------------------------------------------------------------------------

def render_metric_card(label: str, score: int, subtitle: str,
                       t: dict, hero: bool = False) -> None:
    """Render a glass metric card with a colored score value."""
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
# Chart Export
# ---------------------------------------------------------------------------

def chart_export_png(fig: go.Figure, filename: str,
                     label: str = "Download Chart PNG") -> None:
    """Render a Plotly figure to PNG bytes and offer a Streamlit download button."""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label=f"\U0001f4f7 {label}",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception:
        # kaleido not installed — degrade gracefully
        pass


# ---------------------------------------------------------------------------
# Racing Stickmen (pure HTML, no Streamlit)
# ---------------------------------------------------------------------------

def _racing_stickmen_html(t: dict) -> str:
    """SVG stickmen with animated limbs racing left-to-right. Different winner each time."""
    speeds = [random.uniform(4.0, 5.5), random.uniform(4.0, 5.5), random.uniform(4.0, 5.5)]
    winner = random.randint(0, 2)
    speeds[winner] = random.uniform(3.0, 3.8)  # winner is fastest

    colors = [t["red"], t["accent"], t["green"]]
    names = ["Runner", "Swimmer", "Cyclist"]

    runner_svg = """<g>
        <circle cx="15" cy="6" r="5" fill="{c}" />
        <line x1="15" y1="11" x2="15" y2="26" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="15" y1="16" x2="8" y2="22" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="8;22;8" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="15" y1="16" x2="22" y2="22" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="22;8;22" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="15" y1="26" x2="8" y2="36" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="8;22;8" dur="0.35s" repeatCount="indefinite"/></line>
        <line x1="15" y1="26" x2="22" y2="36" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="22;8;22" dur="0.35s" repeatCount="indefinite"/></line>
    </g>"""

    swimmer_svg = """<g>
        <circle cx="15" cy="18" r="5" fill="{c}" />
        <line x1="15" y1="23" x2="15" y2="30" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="15" y1="25" x2="6" y2="20" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="6;24;6" dur="0.6s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="20;16;20" dur="0.6s" repeatCount="indefinite"/></line>
        <line x1="15" y1="25" x2="24" y2="20" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="24;6;24" dur="0.6s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="20;16;20" dur="0.6s" repeatCount="indefinite"/></line>
        <line x1="15" y1="30" x2="10" y2="36" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="10;20;10" dur="0.5s" repeatCount="indefinite"/></line>
        <line x1="15" y1="30" x2="20" y2="36" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="20;10;20" dur="0.5s" repeatCount="indefinite"/></line>
        <ellipse cx="15" cy="32" rx="12" ry="2" fill="{c}" opacity="0.15">
            <animate attributeName="rx" values="12;8;12" dur="0.6s" repeatCount="indefinite"/></ellipse>
    </g>"""

    cyclist_svg = """<g>
        <circle cx="15" cy="8" r="5" fill="{c}" />
        <line x1="15" y1="13" x2="13" y2="24" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="13" y1="18" x2="6" y2="15" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
        <line x1="13" y1="18" x2="22" y2="16" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
        <circle cx="13" cy="30" r="7" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.3"/>
        <line x1="13" y1="24" x2="8" y2="32" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="8;18;8" dur="0.4s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="32;28;32" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="13" y1="24" x2="18" y2="28" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="18;8;18" dur="0.4s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="28;32;28" dur="0.4s" repeatCount="indefinite"/></line>
    </g>"""

    svgs = [runner_svg, swimmer_svg, cyclist_svg]

    lanes = ""
    for i in range(3):
        figure = svgs[i].replace("{c}", colors[i])
        lanes += f"""
        <div style="display: flex; align-items: center; margin: 6px 0; height: 50px;">
            <div style="width: 70px; font-family: 'JetBrains Mono', monospace;
                        font-size: 0.7rem; color: {colors[i]}; text-align: right;
                        padding-right: 10px; font-weight: 600;">{names[i]}</div>
            <div style="flex: 1; position: relative; background: {t["surface"]};
                        border: 1px solid {t["border"]}; border-radius: 4px; height: 46px;
                        overflow: hidden;">
                <div style="position: absolute; top: 2px;
                            animation: race-move {speeds[i]}s linear infinite;">
                    <svg viewBox="0 0 30 40" width="30" height="40">{figure}</svg>
                </div>
            </div>
        </div>"""

    return f"""
    <div style="padding: 30px 20px; text-align: center;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                    color: {t["accent"]}; letter-spacing: 2px; text-transform: uppercase;
                    margin-bottom: 16px;">
            Auditing conversation...
        </div>
        <div style="max-width: 600px; margin: 0 auto; position: relative;">
            {lanes}
            <div style="position: absolute; right: 0; top: 0; bottom: 0; width: 3px;
                        background: repeating-linear-gradient(
                            to bottom, {t["text"]} 0px, {t["text"]} 4px,
                            {t["bg"]} 4px, {t["bg"]} 8px);
                        opacity: 0.5;">
            </div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                    color: {t["muted"]}; margin-top: 10px;">
            Analyzing drift patterns...
        </div>
        <style>
            @keyframes race-move {{
                0% {{ left: -30px; }}
                100% {{ left: calc(100% - 4px); }}
            }}
        </style>
    </div>
    """
