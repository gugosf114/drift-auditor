"""
Drift Auditor — Theme System
=============================
5 themes, CSS builder, and severity color/label helpers.
All pure functions — no Streamlit dependency. Fully testable.
"""

THEMES = {
    "Ember": {
        "label": "Ember",
        "desc": "Warm dark with amber accents",
        "bg": "#0a0807", "surface": "#12100f", "border": "#2a2623",
        "border_accent": "#3a3530", "text": "#e8dfd0", "muted": "#8b7d6b",
        "chart_text": "#c4b8a3", "accent": "#f59e0b", "accent_glow": "rgba(245,158,11,0.08)",
        "accent_hover": "rgba(245,158,11,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Midnight": {
        "label": "Midnight",
        "desc": "Cool blue on deep navy",
        "bg": "#0b0e14", "surface": "#111720", "border": "#1e2a3a",
        "border_accent": "#2a3a4e", "text": "#d0dce8", "muted": "#6b7d8b",
        "chart_text": "#a3b8c4", "accent": "#3b82f6", "accent_glow": "rgba(59,130,246,0.08)",
        "accent_hover": "rgba(59,130,246,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Phosphor": {
        "label": "Phosphor",
        "desc": "Terminal green on black",
        "bg": "#050505", "surface": "#0a0f0a", "border": "#1a2e1a",
        "border_accent": "#2a4a2a", "text": "#b8e6b8", "muted": "#5a8a5a",
        "chart_text": "#8bc48b", "accent": "#22c55e", "accent_glow": "rgba(34,197,94,0.08)",
        "accent_hover": "rgba(34,197,94,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Infrared": {
        "label": "Infrared",
        "desc": "Dark with crimson edge",
        "bg": "#0a0506", "surface": "#120a0c", "border": "#2a1620",
        "border_accent": "#3a2030", "text": "#e8d0d8", "muted": "#8b6b75",
        "chart_text": "#c4a3b0", "accent": "#ef4444", "accent_glow": "rgba(239,68,68,0.08)",
        "accent_hover": "rgba(239,68,68,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Bone": {
        "label": "Bone",
        "desc": "Light mode — paper white",
        "bg": "#faf8f5", "surface": "#ffffff", "border": "#e5e0d8",
        "border_accent": "#d5d0c8", "text": "#1a1610", "muted": "#8b8578",
        "chart_text": "#5a5548", "accent": "#b45309", "accent_glow": "rgba(180,83,9,0.06)",
        "accent_hover": "rgba(180,83,9,0.12)",
        "green": "#16a34a", "red": "#dc2626", "deep_red": "#b91c1c",
    },
}

DEFAULT_THEME = "Ember"

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=30, t=40, b=40),
)


def get_plotly_layout(t: dict) -> dict:
    """Return a Plotly layout dict styled for the given theme."""
    return {
        **PLOTLY_LAYOUT_BASE,
        "font": dict(
            color=t["chart_text"],
            family="JetBrains Mono, DM Sans, sans-serif",
            size=12,
        ),
    }


def score_color(score: int, t: dict | None = None) -> str:
    """Map a 1-10 severity score to a themed hex color.

    Falls back to Ember theme if no theme dict provided.
    """
    if t is None:
        t = THEMES[DEFAULT_THEME]
    if score <= 3:
        return t["green"]
    elif score <= 5:
        return t["accent"]
    elif score <= 7:
        return t["red"]
    else:
        return t["deep_red"]


def score_label(score: int) -> str:
    """Human-readable label for a 1-10 severity score."""
    if score <= 2:
        return "Clean"
    elif score <= 4:
        return "Low Drift"
    elif score <= 6:
        return "Moderate"
    elif score <= 8:
        return "Elevated"
    else:
        return "Severe"


def _build_css(t: dict) -> str:
    """Generate full Streamlit CSS for the given theme dict."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp {{
    background: {t["bg"]};
    font-family: 'DM Sans', sans-serif;
    color: {t["text"]};
}}
.glass-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.15);
}}
.glass-card-hero {{
    background: {t["surface"]};
    border: 1px solid {t["border_accent"]};
    border-radius: 4px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 0 30px {t["accent_glow"]};
}}
.metric-value {{
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}}
.metric-value-hero {{
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}}
.metric-label {{
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: {t["muted"]};
    text-align: center;
    margin-bottom: 6px;
}}
.metric-sub {{
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: {t["muted"]};
    text-align: center;
    margin-top: 4px;
}}
.status-held {{
    color: {t["green"]};
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.status-failed {{
    color: {t["red"]};
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.event-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 10px;
}}
section[data-testid="stSidebar"] {{
    background: {t["bg"]};
    border-right: 1px solid {t["border"]};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {t["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {t["muted"]};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: {t["surface"]};
    color: {t["accent"]};
    border-bottom: 2px solid {t["accent"]};
}}
h1, h2, h3 {{
    font-family: 'JetBrains Mono', monospace;
    color: {t["text"]};
}}
.streamlit-expanderHeader {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {t["chart_text"]};
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {t["accent"]};
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {t["muted"]};
}}
.stButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    color: {t["text"]};
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.78rem;
    border-radius: 4px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    border-color: {t["accent"]};
    color: {t["accent"]};
    box-shadow: 0 0 12px {t["accent_hover"]};
}}
.stDownloadButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["accent"]};
    color: {t["accent"]};
    font-family: 'JetBrains Mono', monospace;
    border-radius: 4px;
}}
.stTextArea textarea, .stTextInput input {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    color: {t["text"]};
    font-family: 'DM Sans', sans-serif;
    border-radius: 4px;
}}
[data-testid="stFileUploader"] {{
    border: 1px dashed {t["border"]};
    border-radius: 4px;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {t["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["border"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {t["accent"]}; }}
.stAlert {{
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
}}
</style>
"""
