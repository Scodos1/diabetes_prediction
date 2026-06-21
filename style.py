"""
style.py
--------
Shared visual identity for all pages: a medical blue + white theme —
clean, clinical, trustworthy. Import and call inject_style() at the
top of every page.

Dark mode: every custom background/text color pair below has an
explicit dark-mode variant (via prefers-color-scheme) so nothing ends
up as dark text on a dark background or light text on a light
background. The .stApp background is intentionally left unset so it
inherits Streamlit's own light/dark theme rather than fighting it.

Risk colors (red/green) keep the same hue family in both modes since
they carry clinical meaning, but are lightened/desaturated for dark
mode so they don't glow against a dark background and stay AA-contrast
against their card backgrounds.
"""

import streamlit as st

PALETTE = {
    "bg": "#FFFFFF",
    "panel_bg": "#F4F8FC",
    "ink": "#0B2545",
    "primary": "#0B5FA5",
    "primary_dark": "#08406F",
    "accent": "#2E8BC9",
    "muted": "#5A7184",
    "border": "#D6E4F0",
    "high_bg": "#FDEEEE",
    "high_border": "#E8A6A6",
    "high_text": "#B23A3A",
    "low_bg": "#EAF6EE",
    "low_border": "#A9D9B8",
    "low_text": "#246840",
}


def inject_style():
    st.markdown(
        """
        <style>
        /* ============== LIGHT MODE (default) ============== */
        .block-container { padding-top: 2rem; max-width: 880px; }

        h1 { font-family: 'Georgia', serif; color: #0B2545; letter-spacing: -0.02em; }
        h2, h3 { font-family: 'Georgia', serif; color: #0B2545; }
        .subtitle { color: #5A7184; font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.8rem; }

        section[data-testid="stSidebar"] {
            background-color: #F4F8FC;
            border-right: 1px solid #D6E4F0;
        }

        .result-card {
            border-radius: 14px;
            padding: 1.8rem 2rem;
            margin-top: 1.2rem;
            border: 1px solid;
        }
        .result-high { background-color: #FDEEEE; border-color: #E8A6A6; }
        .result-low { background-color: #EAF6EE; border-color: #A9D9B8; }
        .risk-label {
            font-family: 'Georgia', serif; font-size: 1.1rem; text-transform: uppercase;
            letter-spacing: 0.08em; color: #5A7184; margin-bottom: 0.2rem;
        }
        .risk-value-high { font-size: 2.4rem; font-weight: 700; color: #B23A3A; }
        .risk-value-low { font-size: 2.4rem; font-weight: 700; color: #246840; }
        .prob-value { font-size: 1.3rem; color: #0B2545; margin-top: 0.6rem; }

        .metric-card {
            background: #F4F8FC;
            border: 1px solid #D6E4F0;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            text-align: left;
        }
        .metric-label {
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: #5A7184; margin-bottom: 0.3rem;
        }
        .metric-value { font-family: 'Georgia', serif; font-size: 1.9rem; color: #0B5FA5; font-weight: 700; }
        .metric-value-high { color: #B23A3A; }
        .metric-value-low { color: #246840; }

        div.stButton > button, div.stDownloadButton > button {
            background-color: #0B5FA5;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            border: none;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: #08406F; }
        div.stButton > button p, div.stDownloadButton > button p { color: #FFFFFF; }

        div[data-baseweb="radio"] label, .stRadio label { color: #0B2545; }
        input:focus, textarea:focus { border-color: #2E8BC9 !important; }

        .pill {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
        }
        .pill-high { background-color: #FDEEEE; color: #B23A3A; }
        .pill-low { background-color: #EAF6EE; color: #246840; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #D6E4F0 !important;
            background-color: #FBFDFF;
        }

        .metrics-table th, .metrics-table td { color: #0B2545; }

        /* ============== DARK MODE ============== */
        @media (prefers-color-scheme: dark) {
            h1, h2, h3 { color: #EAF2FA; }
            .subtitle { color: #9FB3C8; }

            section[data-testid="stSidebar"] {
                background-color: #11202F;
                border-right: 1px solid #1F3447;
            }

            .result-high { background-color: #3A1F1F; border-color: #6B3535; }
            .result-low { background-color: #1B3A28; border-color: #356B49; }
            .risk-label { color: #9FB3C8; }
            .risk-value-high { color: #F0938E; }
            .risk-value-low { color: #7FD79B; }
            .prob-value { color: #EAF2FA; }

            .metric-card {
                background: #11202F;
                border: 1px solid #1F3447;
            }
            .metric-label { color: #9FB3C8; }
            .metric-value { color: #6FB3F0; }
            .metric-value-high { color: #F0938E; }
            .metric-value-low { color: #7FD79B; }

            div.stButton > button, div.stDownloadButton > button {
                background-color: #2E8BC9;
                color: #0B1620;
            }
            div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: #58A6E0; }
            div.stButton > button p, div.stDownloadButton > button p { color: #0B1620; }

            div[data-baseweb="radio"] label, .stRadio label { color: #EAF2FA; }
            input:focus, textarea:focus { border-color: #6FB3F0 !important; }

            .pill-high { background-color: #3A1F1F; color: #F0938E; }
            .pill-low { background-color: #1B3A28; color: #7FD79B; }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: #1F3447 !important;
                background-color: #0E1B27;
            }

            .metrics-table th, .metrics-table td { color: #EAF2FA; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_pill(label: str) -> str:
    cls = "pill-high" if label == "HIGH" else "pill-low"
    return f'<span class="pill {cls}">{label}</span>'


def metric_card(label: str, value, variant: str = "") -> str:
    value_cls = "metric-value"
    if variant == "high":
        value_cls += " metric-value-high"
    elif variant == "low":
        value_cls += " metric-value-low"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{value_cls}">{value}</div>
    </div>
    """
