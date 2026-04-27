"""
Application-wide configuration — styles, colour palette, and constants.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1d3557",
    "secondary": "#457b9d",
    "accent": "#a8dadc",
    "danger": "#e63946",
    "warning": "#f4a261",
    "success": "#2a9d8f",
    "dark": "#0d1b2a",
    "light": "#f8f9fa",
    "muted": "#6c757d",
}

# ---------------------------------------------------------------------------
# Machine-learning defaults
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_SAMPLING_STRATEGY = 0.7
MAX_CLUSTERS = 10
DEFAULT_CLUSTERS = 4

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------
RISK_HIGH_THRESHOLD = 0.50
RISK_MEDIUM_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Poppins', sans-serif; }

    /* ── Header ─────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: #fff; font-weight: 700; margin: 0; font-size: 2.5rem; }
    .main-header p  { color: #a8dadc; margin-top: 0.5rem; font-size: 1.1rem; }

    /* ── Metric cards ───────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: transform 0.3s ease;
    }
    .metric-card:hover   { transform: translateY(-5px); }
    .metric-card.danger  { border-left-color: #e63946; }
    .metric-card.warning { border-left-color: #f4a261; }
    .metric-card.success { border-left-color: #2a9d8f; }
    .metric-card.info    { border-left-color: #457b9d; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #1d3557; }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Insight box ────────────────────────────────────── */
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #457b9d;
    }
    .insight-box h4 { color: #1d3557; margin-top: 0; }

    /* ── Risk badges ────────────────────────────────────── */
    .risk-high   { color: #e63946; font-weight: 600; }
    .risk-medium { color: #f4a261; font-weight: 600; }
    .risk-low    { color: #2a9d8f; font-weight: 600; }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background-color: #1d3557; color: white; }

    /* ── Section header ─────────────────────────────────── */
    .section-header {
        background: linear-gradient(90deg, #1d3557 0%, #457b9d 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Streamlit metric override ──────────────────────── */
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }

    /* ── Recommendation card ────────────────────────────── */
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #2a9d8f;
    }
</style>
"""
