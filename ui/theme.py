"""
Drift Auditor — Theme System
=============================
One visual identity ("Ledger"), CSS builder, severity color/label helpers.
All pure functions — no Streamlit dependency. Fully testable.

Design rule: interface chrome is achromatic graphite. Color is reserved
for findings, on a calibrated four-step ramp (stable -> caution ->
elevated -> critical). Type: IBM Plex Serif for the wordmark, Plex Sans
for interface text, Plex Mono for readouts and data.
"""

LEDGER = {
    "label": "Ledger",
    "desc": "Graphite chrome; color reserved for findings",
    # chrome (achromatic)
    "bg": "#0f1216", "surface": "#151a20", "border": "#232b34",
    "border_accent": "#33404d", "text": "#dee4ea", "muted": "#7d8894",
    "chart_text": "#a9b4bf", "grid": "rgba(222,228,234,0.07)",
    # findings ramp (the only chroma in the interface)
    "green": "#4e9478",       # stable
    "accent": "#c79a3f",      # caution
    "red": "#c05b3c",         # elevated
    "deep_red": "#a63b2a",    # critical
    "accent_glow": "rgba(199,154,63,0.10)",
    "accent_hover": "rgba(199,154,63,0.18)",
}

THEMES = {"Ledger": LEDGER}

DEFAULT_THEME = "Ledger"

FONT_STACK_UI = "'IBM Plex Sans', -apple-system, sans-serif"
FONT_STACK_DATA = "'IBM Plex Mono', ui-monospace, monospace"
FONT_STACK_WORDMARK = "'IBM Plex Serif', Georgia, serif"

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
            family="IBM Plex Mono, IBM Plex Sans, monospace",
            size=12,
        ),
    }


def score_color(score: int, t: dict | None = None) -> str:
    """Map a 1-10 severity score to a ramp color.

    Falls back to the Ledger theme if no theme dict provided.
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,500;1,500&display=swap');

.stApp {{
    background: {t["bg"]};
    font-family: {FONT_STACK_UI};
    color: {t["text"]};
    font-size: 15px;
}}
h1, h2, h3 {{
    font-family: {FONT_STACK_UI};
    font-weight: 600;
    letter-spacing: -0.01em;
    color: {t["text"]};
}}
h2 {{ font-size: 1.35rem; }}
h3 {{ font-size: 1.05rem; }}

/* Wordmark — the one serif moment */
.wordmark {{
    font-family: {FONT_STACK_WORDMARK};
    font-style: italic;
    font-weight: 500;
    font-size: 1.5rem;
    color: {t["text"]};
    line-height: 1.2;
    margin-bottom: 2px;
}}
.wordmark-eyebrow {{
    font-family: {FONT_STACK_UI};
    font-size: 0.66rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {t["muted"]};
}}

/* Panels — flat, hairline-ruled. No glass, no glow. */
.glass-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 2px;
    padding: 18px 20px;
    margin-bottom: 8px;
}}
.glass-card-hero {{
    background: {t["surface"]};
    border: 1px solid {t["border_accent"]};
    border-radius: 2px;
    padding: 24px 20px 18px;
    text-align: center;
}}

/* Instrument readouts */
.metric-value {{
    font-size: 2.4rem;
    font-weight: 500;
    font-family: {FONT_STACK_DATA};
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    text-align: center;
}}
.metric-value-hero {{
    font-size: 3.6rem;
    font-weight: 500;
    font-family: {FONT_STACK_DATA};
    font-variant-numeric: tabular-nums;
    line-height: 1.05;
    text-align: center;
}}
.metric-label {{
    font-size: 0.68rem;
    font-family: {FONT_STACK_UI};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {t["muted"]};
    text-align: center;
    margin-bottom: 6px;
}}
.metric-sub {{
    font-size: 0.78rem;
    font-family: {FONT_STACK_UI};
    color: {t["muted"]};
    text-align: center;
    margin-top: 4px;
}}

/* Calibration scale — signature element. A fixed 1-10 ramp with the
   conversation's reading marked on it, like a gauge face. */
.cal-scale {{
    position: relative;
    height: 6px;
    border-radius: 3px;
    margin: 14px 8px 4px;
    background: linear-gradient(to right,
        {t["green"]} 0%, {t["green"]} 30%,
        {t["accent"]} 42%, {t["accent"]} 50%,
        {t["red"]} 62%, {t["red"]} 70%,
        {t["deep_red"]} 85%, {t["deep_red"]} 100%);
    opacity: 0.85;
}}
.cal-marker {{
    position: absolute;
    top: -5px;
    width: 2px;
    height: 16px;
    background: {t["text"]};
    box-shadow: 0 0 0 2px {t["bg"]};
}}
.cal-ticks {{
    display: flex;
    justify-content: space-between;
    font-family: {FONT_STACK_DATA};
    font-size: 0.62rem;
    color: {t["muted"]};
    margin: 2px 8px 0;
}}

.status-held {{
    color: {t["green"]};
    font-weight: 600;
    font-family: {FONT_STACK_DATA};
}}
.status-failed {{
    color: {t["red"]};
    font-weight: 600;
    font-family: {FONT_STACK_DATA};
}}
.event-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 2px;
    padding: 14px 16px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}}

section[data-testid="stSidebar"] {{
    background: {t["bg"]};
    border-right: 1px solid {t["border"]};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 1px solid {t["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 0;
    padding: 8px 14px;
    font-family: {FONT_STACK_UI};
    font-size: 0.85rem;
    font-weight: 500;
    color: {t["muted"]};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: transparent;
    color: {t["text"]};
    border-bottom: 2px solid {t["accent"]};
}}

.streamlit-expanderHeader {{
    font-family: {FONT_STACK_UI};
    font-size: 0.88rem;
    color: {t["chart_text"]};
}}
[data-testid="stMetricValue"] {{
    font-family: {FONT_STACK_DATA};
    font-variant-numeric: tabular-nums;
    color: {t["text"]};
}}
[data-testid="stMetricLabel"] {{
    font-family: {FONT_STACK_UI};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {t["muted"]};
}}
.stButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["border_accent"]};
    color: {t["text"]};
    font-family: {FONT_STACK_UI};
    font-weight: 500;
    font-size: 0.85rem;
    border-radius: 2px;
    transition: border-color 0.15s;
}}
.stButton > button:hover {{
    border-color: {t["muted"]};
    color: {t["text"]};
}}
.stDownloadButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["border_accent"]};
    color: {t["text"]};
    font-family: {FONT_STACK_UI};
    font-weight: 500;
    border-radius: 2px;
}}
.stTextArea textarea, .stTextInput input {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    color: {t["text"]};
    font-family: {FONT_STACK_UI};
    border-radius: 2px;
}}
[data-testid="stFileUploader"] {{
    border: 1px dashed {t["border_accent"]};
    border-radius: 2px;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {t["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["border_accent"]}; border-radius: 3px; }}
.stAlert {{
    border-radius: 2px;
    font-family: {FONT_STACK_UI};
}}
hr {{ border-color: {t["border"]}; }}
</style>
"""
