"""
mode_intraday_short.py — Intraday short-selling screener.

CORE LOGIC that makes this mode distinct: score_short() below — the
condition checks and point weights that flag a stock as a short-selling
setup (down from open, near day high about to roll over, downtrend,
negative momentum, RSI overbought, etc). Everything else (exchange/scan-mode
selection, rate limiting, checkpointing, results shell, filters, CSV export,
detail-view charting) comes from scanner_common / intraday_data exactly like
every other mode.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import indicators
import intraday_data
import scanner_common as sc
from scanner_common import sskey, get_state, set_state

MODE_KEY = sc.MODE_SHORT

_TIMEFRAME_MAP = {
    "1 Day": ("1d", "1m"), "1 Week": ("5d", "15m"), "1 Month": ("1mo", "1h"),
    "3 Months": ("3mo", "1d"), "6 Months": ("6mo", "1d"), "1 Year": ("1y", "1d"),
    "3 Years": ("3y", "1wk"), "All Time": ("max", "1wk"),
}


# ── CORE LOGIC: short-setup scoring ─────────────────────────────────────
def score_short(snap, params):
    try:
        current_price = snap["intraday_close"][-1]
        open_price = snap["intraday_open"]
        high_price = snap["day_high"]
        volume = float(snap["intraday_volume"].sum())

        if current_price < params["min_price"] or volume < params["min_volume"]:
            return None

        price_change_pct = ((current_price - open_price) / open_price) * 100 if open_price else 0
        dist_from_high = ((high_price - current_price) / high_price) * 100 if high_price else 0

        daily_close = snap["daily_close"]
        recent_change = ((daily_close[-1] - daily_close[0]) / daily_close[0]) * 100 if len(daily_close) >= 2 and daily_close[0] else 0

        closes = snap["intraday_close"]
        window = params["momentum_window"]
        if len(closes) >= window * 2:
            last_n = closes[-window:].mean()
            prev_n = closes[-window * 2:-window].mean()
            momentum_change = ((last_n - prev_n) / prev_n) * 100 if prev_n else 0
        else:
            momentum_change = 0

        avg_volume_5d = snap["daily_volume"].mean() if len(snap["daily_volume"]) else 0
        volume_ratio = volume / avg_volume_5d if avg_volume_5d > 0 else 0

        rsi = indicators.rsi(closes, period=params["rsi_period"])
        atr = indicators.atr(snap["intraday_high"], snap["intraday_low"], closes, period=params["atr_period"])
        atr_pct = (atr / current_price) * 100 if current_price else 0

        conditions_met = []
        if price_change_pct < params["price_change_threshold"]:
            conditions_met.append("Down from open")
        elif price_change_pct < 0.5:
            conditions_met.append("Flat/weak")
        if dist_from_high < params["dist_from_high_threshold"]:
            conditions_met.append("Near day high")
        if recent_change < params["trend_threshold"]:
            conditions_met.append("5-day downtrend")
        if momentum_change < params["momentum_threshold"]:
            conditions_met.append("Negative momentum")
        if volume_ratio > params["volume_ratio_threshold"]:
            conditions_met.append("High volume")
        if rsi and rsi > params["rsi_threshold"]:
            conditions_met.append("RSI overbought")
        if atr_pct > params["atr_threshold"]:
            conditions_met.append("Good volatility")

        if len(conditions_met) < params["min_conditions"]:
            return None

        score = 0
        if price_change_pct < -2: score += 30
        elif price_change_pct < -1: score += 20
        elif price_change_pct < 0: score += 10

        if dist_from_high < 1: score += 20
        elif dist_from_high < 2: score += 10

        if recent_change < -5: score += 20
        elif recent_change < -2: score += 10

        if momentum_change < -1: score += 15
        elif momentum_change < -0.5: score += 8

        if volume_ratio > 1.5: score += 10
        elif volume_ratio > 1.2: score += 5

        if rsi and rsi > 70: score += 5
        elif rsi and rsi > 65: score += 3

        if score < params["min_score"]:
            return None

        return {
            "price": current_price, "open": open_price, "high": high_price, "low": snap["day_low"],
            "change_pct": price_change_pct, "volume": volume, "volume_ratio": volume_ratio,
            "dist_from_high": dist_from_high, "recent_trend": recent_change, "momentum": momentum_change,
            "rsi": rsi if rsi else 0, "atr_pct": atr_pct, "score": score,
            "conditions": ", ".join(conditions_met),
            "signal_strength": "STRONG" if score >= params["strong_score"] else "MODERATE" if score >= 50 else "WEAK",
        }
    except Exception:
        return None


# ── UI ───────────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown('<p class="main-header">📉 Intraday Short Selling Screener</p>', unsafe_allow_html=True)
    st.markdown("*Scan for stocks showing downward momentum for intraday shorting*")

    scan_nse, scan_bse, universe = sc.render_exchange_selector(MODE_KEY)
    stocks_to_scan = sc.render_scan_mode_selector(MODE_KEY, universe)
    rate_cfg = sc.render_rate_limit_controls(MODE_KEY)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Screening Parameters")
    with st.sidebar.expander("Basic Filters", expanded=True):
        min_price = st.number_input("Min Price (₹)", min_value=1, max_value=500, value=20, step=5,
                                     key=sskey(MODE_KEY, "min_price"))
        min_volume = st.number_input("Min Volume", min_value=10000, max_value=10000000, value=100000, step=10000,
                                      key=sskey(MODE_KEY, "min_volume"))
        min_conditions = st.slider("Min Conditions (out of 7)", 2, 7, 4, key=sskey(MODE_KEY, "min_conditions"))
        min_score = st.slider("Min Score (0-100)", 20, 90, 50, 5, key=sskey(MODE_KEY, "min_score"))

    with st.sidebar.expander("Advanced Thresholds"):
        price_change_threshold = st.slider("Price Change (%)", -5.0, 1.0, 0.0, 0.5, key=sskey(MODE_KEY, "price_chg_th"))
        momentum_threshold = st.slider("Momentum (%)", -5.0, 0.0, -0.5, 0.1, key=sskey(MODE_KEY, "momentum_th"))
        dist_from_high_threshold = st.slider("Dist from High (%)", 0.5, 5.0, 2.0, 0.5, key=sskey(MODE_KEY, "dist_high_th"))
        volume_ratio_threshold = st.slider("Volume Ratio", 1.0, 3.0, 1.2, 0.1, key=sskey(MODE_KEY, "vol_ratio_th"))
        trend_threshold = st.slider("5-Day Trend (%)", -10.0, 0.0, -2.0, 0.5, key=sskey(MODE_KEY, "trend_th"))
        rsi_threshold = st.slider("RSI Overbought", 50, 80, 65, 5, key=sskey(MODE_KEY, "rsi_th"))
        atr_threshold = st.slider("ATR % Threshold", 0.5, 5.0, 1.0, 0.1, key=sskey(MODE_KEY, "atr_th"))

    with st.sidebar.expander("Technical Indicators & Trading Settings"):
        rsi_period = st.number_input("RSI Period", 5, 50, 14, 1, key=sskey(MODE_KEY, "rsi_period"))
        atr_period = st.number_input("ATR Period", 5, 50, 14, 1, key=sskey(MODE_KEY, "atr_period"))
        momentum_window = st.number_input("Momentum Window (min)", 10, 120, 30, 5, key=sskey(MODE_KEY, "mom_window"))
        stop_loss_pct = st.number_input("Stop Loss % above Entry Price", 0.1, 5.0, 0.5, 0.1, key=sskey(MODE_KEY, "sl_pct"))
        target_pct = st.number_input("Target % below Entry Price", 0.5, 20.0, 2.0, 0.5, key=sskey(MODE_KEY, "tgt_pct"))
        strong_score = st.number_input("Strong Signal Score", 60, 90, 70, 5, key=sskey(MODE_KEY, "strong_score"))
        chart_height = st.number_input("Chart Height (px)", 200, 500, 250, 50, key=sskey(MODE_KEY, "chart_height"))

    params = {
        "min_price": min_price, "min_volume": min_volume, "min_conditions": min_conditions, "min_score": min_score,
        "price_change_threshold": price_change_threshold, "dist_from_high_threshold": dist_from_high_threshold,
        "trend_threshold": trend_threshold, "momentum_threshold": momentum_threshold,
        "volume_ratio_threshold": volume_ratio_threshold, "rsi_threshold": rsi_threshold,
        "atr_threshold": atr_threshold, "rsi_period": rsi_period, "atr_period": atr_period,
        "momentum_window": momentum_window, "strong_score": strong_score,
    }
    set_state(MODE_KEY, "trading_settings", {"stop_loss_pct": stop_loss_pct, "target_pct": target_pct, "chart_height": chart_height})

    def fetch_and_analyze(rec):
        snap = sc.bulletproof_fetch(intraday_data.fetch_intraday_snapshot, rec["yf_symbol"])
        if snap is None:
            return "failed", None
        analysis = score_short(snap, params)
        if analysis is None:
            return "filtered", None
        analysis.update({"symbol": rec["symbol"], "name": rec["name"], "yf_symbol": rec["yf_symbol"], "exchange": rec["exchange"]})
        return "ok", analysis

    do_scan, resume_scan, checkpoint, _sig = sc.render_scan_trigger(
        MODE_KEY, stocks_to_scan, f"🔍 SCAN {len(stocks_to_scan)} STOCKS FOR SHORT SETUPS")

    if do_scan:
        sc.run_scan(MODE_KEY, stocks_to_scan, fetch_and_analyze, rate_cfg, resume_scan, checkpoint)

    _render_results()

    with st.expander("📚 How to Use"):
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("""
            **Stock Selection:** same Exchange / Scan-Mode controls as every other mode —
            Quick, Full, Slot-wise, Range or Custom List.

            **Best Scan Times:** 10:00–11:30 AM (post-opening) · 1:30–2:30 PM (post-lunch)
            """)
        with h2:
            st.markdown("""
            **Signal Strength:** 🔴 STRONG (≥ Strong Signal Score) · 🟡 MODERATE (50+)

            **Risk Management:** Stop loss 0.5–1% above day high · Position size 1–2% of capital · Exit before 3:15 PM
            """)
    sc.footer("<strong>Intraday Short Selling Screener</strong> · Short selling is risky.")


def _render_results() -> None:
    results = get_state(MODE_KEY, "results")
    if not results:
        st.info("👈 Configure and click 'SCAN' to start")
        return

    if not results:
        st.warning("⚠️ No stocks found matching criteria")
        return

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    st.markdown("---")
    st.success(f"✅ Found {len(results)} potential short-selling opportunities!")

    df = pd.DataFrame([{
        "Symbol": r["symbol"], "Name": r.get("name", ""), "Exchange": r.get("exchange", ""),
        "Price (₹)": r["price"], "Change %": r["change_pct"], "Score": r["score"],
        "Signal": r["signal_strength"], "Volume Ratio": r["volume_ratio"],
        "Dist from High (%)": r["dist_from_high"], "5D Trend (%)": r["recent_trend"],
        "RSI": r["rsi"], "ATR %": r["atr_pct"], "Conditions": r["conditions"],
    } for r in results])

    st.markdown("#### Screener Results Summary")

    def color_signal(val):
        if val == "STRONG":
            return "background-color: #ffcccc"
        if val == "MODERATE":
            return "background-color: #fff3cd"
        return ""

    def color_change(val):
        try:
            return "background-color: #ffcccc" if val < 0 else "background-color: #d4edda" if val > 0 else ""
        except Exception:
            return ""

    styled = df.style.map(color_signal, subset=["Signal"]).map(color_change, subset=["Change %", "5D Trend (%)"]).format({
        "Price (₹)": "₹{:.2f}", "Change %": "{:+.2f}%", "Volume Ratio": "{:.2f}x",
        "Dist from High (%)": "{:.2f}%", "5D Trend (%)": "{:+.2f}%", "RSI": "{:.1f}", "ATR %": "{:.2f}%",
    })
    st.dataframe(styled, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("Select one result below to see its chart and trade levels — showing every result's charts "
                     "inline used to be what made this screener feel sluggish, so (like the other 2 modes) only "
                     "the selected stock renders.")
    with col2:
        chart_timeframe = st.selectbox("Chart Timeframe", list(_TIMEFRAME_MAP.keys()), index=0,
                                        key=sskey(MODE_KEY, "chart_tf"))

    options = [f"{r['symbol']} — {r['name']}" if r.get("name") else r["symbol"] for r in results]
    idx_by_option = {opt: i for i, opt in enumerate(options)}
    selected_option = st.selectbox("Select stock for details", options, key=sskey(MODE_KEY, "detail_select"))
    result = results[idx_by_option[selected_option]]

    st.markdown(f"##### {result['symbol']} — {result['signal_strength']} (Score: {result['score']})")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"₹{result['price']:.2f}", f"{result['change_pct']:.2f}%")
    m2.metric("High", f"₹{result['high']:.2f}")
    m3.metric("Dist from High", f"{result['dist_from_high']:.2f}%")
    m4.metric("Vol Ratio", f"{result['volume_ratio']:.2f}x")
    m5.metric("RSI", f"{result['rsi']:.1f}")
    m6.metric("5D Trend", f"{result['recent_trend']:.2f}%")

    period, interval = _TIMEFRAME_MAP[chart_timeframe]
    chart_data = intraday_data.fetch_chart_history(result["yf_symbol"], period, interval)
    trading = get_state(MODE_KEY, "trading_settings", {"stop_loss_pct": 0.5, "target_pct": 2.0, "chart_height": 250})

    if chart_data is not None and not chart_data.empty:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Price",
                                       line=dict(color="#dc3545", width=2)))
            fig1.add_hline(y=result["open"], line_dash="dash", line_color="gray", line_width=1, annotation_text="Open")
            fig1.update_layout(title=f"Price Chart ({chart_timeframe})", xaxis_title="Time", yaxis_title="Price (₹)",
                                height=trading["chart_height"], margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        with cc2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color="#17a2b8"))
            fig2.update_layout(title=f"Volume ({chart_timeframe})", xaxis_title="Time", yaxis_title="Volume",
                                height=trading["chart_height"], margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with cc3:
            closes = chart_data["Close"].values
            rsi_vals, rsi_idx = [], []
            for j in range(14, len(closes)):
                window = closes[max(0, j - 14):j]
                if len(window) > 1:
                    diffs = window[1:] - window[:-1]
                    gains = diffs[diffs > 0].sum() / len(window)
                    losses = -diffs[diffs < 0].sum() / len(window)
                    rs = gains / losses if losses else 0
                    rsi_vals.append(100 - (100 / (1 + rs)) if rs > 0 else 50)
                    rsi_idx.append(chart_data.index[j])
            fig3 = go.Figure()
            if rsi_vals:
                fig3.add_trace(go.Scatter(x=rsi_idx, y=rsi_vals, mode="lines", name="RSI", line=dict(color="#28a745", width=2)))
                fig3.add_hline(y=70, line_dash="dash", line_color="red", line_width=1)
                fig3.add_hline(y=30, line_dash="dash", line_color="green", line_width=1)
            fig3.update_layout(title=f"RSI ({chart_timeframe})", xaxis_title="Time", yaxis_title="RSI",
                                height=trading["chart_height"], margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning(f"No chart data available for {result['symbol']}")

    stop_loss = result["price"] * (1 + trading["stop_loss_pct"] / 100)
    target = result["price"] * (1 - trading["target_pct"] / 100)
    t1, t2, t3, t4 = st.columns(4)
    t1.info(f"💡 Entry: ₹{result['price']:.2f}")
    t2.error(f"🛑 Stop: ₹{stop_loss:.2f}")
    t3.success(f"🎯 Target: ₹{target:.2f}")
    risk_reward = abs((result["price"] - target) / (stop_loss - result["price"])) if stop_loss != result["price"] else 0
    t4.metric("R:R Ratio", f"1:{risk_reward:.2f}")
    st.caption(f"**Conditions:** {result['conditions']}")

    sc.download_buttons(MODE_KEY, df, df, "intraday_short_scan")
