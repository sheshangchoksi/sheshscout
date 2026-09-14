"""
app.py — Streamlit entry point for the NSE + BSE Stock Scanner.

Four modes, two timeframes x two directions: Intraday Long/Short share
their pipeline via intraday_data.py, Swing Long/Short share theirs via
swing_data.py, and all four share universe loading, rate limiting,
checkpointing, trade-levels math, and chart/trade-card rendering via
scanner_common.py -- only each mode_*.py file's actual scoring logic
differs.
"""

from __future__ import annotations

import importlib

import streamlit as st

st.set_page_config(
    page_title="Stock Scanner — NSE & BSE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem; }
    [data-testid="stSidebar"] { min-width: 350px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("📊 Stock Scanner")
st.sidebar.caption("NSE + BSE · yfinance data, exchange bhavcopy for daily reference")

_MODE_TO_MODULE = {
    "📈 Intraday Long (Buy)": "mode_intraday_long",
    "📉 Intraday Short (Sell)": "mode_intraday_short",
    "📈 Swing Long (Buy)": "mode_swing_long",
    "📉 Swing Short (Sell)": "mode_swing_short",
}
choice = st.sidebar.radio("Screener", list(_MODE_TO_MODULE.keys()), key="app_mode_choice")
st.sidebar.markdown("---")

try:
    _active_mode = importlib.import_module(_MODE_TO_MODULE[choice])
    _active_mode.render()
except Exception as e:
    # Last line of defense: a bug or an exchange/Yahoo outage in one
    # screener must never take down the whole app for the user.
    st.error("Something went wrong while rendering this screener.")
    with st.expander("Technical details"):
        st.exception(e)
    st.info("Try reloading the page, switching screener modes, or reducing the scan size in the sidebar "
            "(Quick Sample / Slot-wise use far fewer requests than Full Universe).")
