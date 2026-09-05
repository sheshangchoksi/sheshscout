"""
app.py — Streamlit entry point for the NSE + BSE Intraday Scanner.

Two modes share everything (universe loading, data fetch, rate limiting,
checkpointing) via scanner_common.py / intraday_data.py; only the
scoring logic in each mode_intraday_*.py file differs.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Intraday Scanner — NSE & BSE",
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

st.sidebar.title("📊 Intraday Scanner")
st.sidebar.caption("NSE + BSE · yfinance intraday, exchange bhavcopy for daily reference")

_MODE_TO_KEY = {
    "📈 Intraday Long (Buy)": "long",
    "📉 Intraday Short (Sell)": "short",
}
choice = st.sidebar.radio("Screener", list(_MODE_TO_KEY.keys()), key="app_mode_choice")
st.sidebar.markdown("---")

try:
    if _MODE_TO_KEY[choice] == "long":
        import mode_intraday_long as _active_mode
    else:
        import mode_intraday_short as _active_mode
    _active_mode.render()
except Exception as e:
    # Last line of defense: a bug or an exchange/Yahoo outage in one
    # screener must never take down the whole app for the user.
    st.error("Something went wrong while rendering this screener.")
    with st.expander("Technical details"):
        st.exception(e)
    st.info("Try reloading the page, switching screener modes, or reducing the scan size in the sidebar "
            "(Quick Sample / Slot-wise use far fewer requests than Full Universe).")
