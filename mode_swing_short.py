"""
mode_swing_short.py — Swing short/sell screener: the mirror image of
mode_swing_long.py's buy-setup logic (down over the last few days, near an
N-day high as a rejection zone, downtrend, negative momentum, RSI
overbought, near real weekly resistance, bearish candlestick confirmation,
weekly-downtrend agreement, index agreement, extended-down-move-with-a-
bounce confirmation). Everything else is shared with the swing long
screener via swing_data.py / scanner_common.py / streak_analysis.py /
candlesticks.py / indicators.py / intraday_data.py exactly as-is.

CORE LOGIC that makes this mode distinct: score_swing_short() below.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import candlesticks
import indicators
import intraday_data
import scanner_common as sc
import streak_analysis
import swing_data
from scanner_common import sskey, get_state, set_state

MODE_KEY = sc.MODE_SWING_SHORT

_TIMEFRAME_MAP = {
    "3 Months": ("3mo", "1d"), "6 Months": ("6mo", "1d"), "1 Year": ("1y", "1d"),
    "2 Years": ("2y", "1d"), "3 Years": ("3y", "1wk"), "5 Years": ("5y", "1wk"), "All Time": ("max", "1wk"),
}


# ── CORE LOGIC: sell-setup scoring (mirror of the swing long screener) ───
def score_swing_short(snap, params, weekly_ctx=None, index_change_pct=None):
    try:
        closes = snap["daily_close"]
        highs = snap["daily_high"]
        lows = snap["daily_low"]
        opens = snap["daily_open"]
        volumes = snap["daily_volume"]
        current_price = closes[-1]
        volume = volumes[-1]

        if current_price < params["min_price"] or volume < params["min_volume"]:
            return None

        lb = min(params["lookback_window"], len(closes) - 1)

        # N-day high/low -- the swing equivalent of "today's day high/low":
        # is this stock sitting near a recent rejection/breakdown zone?
        recent_low = float(lows[-lb:].min()) if lb > 0 else float(lows[-1])
        recent_high = float(highs[-lb:].max()) if lb > 0 else float(highs[-1])
        dist_from_high = ((recent_high - current_price) / recent_high) * 100 if recent_high else 0

        recent_change = ((closes[-1] - closes[-lb - 1]) / closes[-lb - 1]) * 100 if lb > 0 and closes[-lb - 1] else 0

        window = min(params["momentum_window"], len(closes) // 2)
        if window >= 2:
            last_n = closes[-window:].mean()
            prev_n = closes[-window * 2:-window].mean()
            momentum_change = ((last_n - prev_n) / prev_n) * 100 if prev_n else 0
        else:
            momentum_change = 0

        short_window = min(5, len(closes) - 1)
        price_change_pct = (((closes[-1] - closes[-short_window - 1]) / closes[-short_window - 1]) * 100
                             if short_window > 0 and closes[-short_window - 1] else 0)

        if len(volumes) > lb + 1:
            avg_volume = float(volumes[-lb - 1:-1].mean())
        else:
            avg_volume = float(volumes[:-1].mean()) if len(volumes) > 1 else float(volumes[-1])
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0

        rsi = indicators.rsi(closes, period=params["rsi_period"])
        atr = indicators.atr(highs, lows, closes, period=params["atr_period"])
        atr_pct = (atr / current_price) * 100 if current_price else 0

        # Real WEEKLY support/resistance + the weekly trend.
        dist_from_resistance = None
        weekly_trend_pct = None
        if weekly_ctx is not None and weekly_ctx.get("resistance"):
            dist_from_resistance = ((weekly_ctx["resistance"] - current_price) / weekly_ctx["resistance"]) * 100
            weekly_trend_pct = weekly_ctx.get("weekly_trend_pct")

        # Candlestick confirmation -- same candlesticks.detect_pattern()
        # the long screener uses, fed daily bars near weekly resistance.
        near_resistance = dist_from_resistance is not None and dist_from_resistance < params["dist_from_resistance_threshold"]
        candle = {"pattern": None, "bullish": False, "bearish": False}
        if near_resistance:
            candle = candlesticks.detect_pattern(opens, highs, lows, closes, lookback=params["candle_lookback"])
        # Bearish shapes are a straightforward confirmation; doji and
        # spinning top are direction-agnostic on their own but still count
        # as reversal/indecision candidates at resistance (same reasoning
        # as the intraday short screener).
        candle_confirmed = near_resistance and (candle["bearish"] or candle["pattern"] in ("doji", "spinning_top"))

        # Extension check, mirrored: a stock that fell hard over the last
        # `momentum_window` days with NOT ONE up day in that stretch is
        # extended downward and due a bounce (chase risk for a fresh
        # short); the same move WITH at least one up day along the way has
        # already "confirmed" itself with a bounce -- a continuation setup
        # rather than a chase.
        ext_window = min(params["momentum_window"], len(closes) - 1)
        extension_chase = extension_with_pullback = False
        if ext_window >= 2 and closes[-ext_window - 1]:
            ext_change = ((closes[-1] - closes[-ext_window - 1]) / closes[-ext_window - 1]) * 100
            daily_rets = closes[-ext_window:] - closes[-ext_window - 1:-1]
            had_bounce = bool((daily_rets > 0).any())
            is_extended_down = ext_change < -params["extension_threshold"]
            extension_chase = is_extended_down and not had_bounce
            extension_with_pullback = is_extended_down and had_bounce

        conditions_met = []
        warnings = []

        if price_change_pct < params["price_change_threshold"]:
            conditions_met.append("Down over last few days")
        elif price_change_pct <= 0.5:
            conditions_met.append("Flat/weak")
        if dist_from_high < params["dist_from_high_threshold"]:
            conditions_met.append(f"Near {lb}-day high / rejection zone")
        if recent_change < -params["trend_threshold"]:
            conditions_met.append(f"{lb}-day downtrend")
        if momentum_change < -params["momentum_threshold"]:
            conditions_met.append("Negative momentum")
        if volume_ratio > params["volume_ratio_threshold"]:
            conditions_met.append("High volume")
        if rsi and rsi > params["rsi_threshold"]:
            conditions_met.append("RSI overbought")
        if atr_pct > params["atr_threshold"]:
            conditions_met.append("Good volatility")

        if near_resistance:
            conditions_met.append("Near real resistance (weekly)")
        if candle_confirmed:
            conditions_met.append(f"{candlesticks.PATTERN_LABELS.get(candle['pattern'], candle['pattern'])} at resistance confirmed")
        if weekly_trend_pct is not None:
            if weekly_trend_pct < -params["weekly_trend_threshold"]:
                conditions_met.append("Weekly downtrend confirmed")
            elif weekly_trend_pct > params["weekly_trend_threshold"]:
                warnings.append("against weekly trend")
        if index_change_pct is not None:
            # See score_swing_long's identical comment: scaled to the
            # user's weekly-trend sensitivity, not a fixed same-day-sized
            # constant, since this is measured over the whole lookback
            # window (~1 month by default).
            if index_change_pct < params["weekly_trend_threshold"]:
                conditions_met.append("Market supportive (index)")
            elif index_change_pct > params["weekly_trend_threshold"] * 3:
                warnings.append("against index trend")
        if extension_with_pullback:
            conditions_met.append("Extended move with bounce confirmed")
        elif extension_chase:
            warnings.append("extended, no bounce yet (chase risk)")

        if len(conditions_met) < params["min_conditions"]:
            return None

        score = 0
        if price_change_pct < -3: score += 30
        elif price_change_pct < -1.5: score += 20
        elif price_change_pct < 0: score += 10

        if dist_from_high < 2: score += 20
        elif dist_from_high < 5: score += 10

        if recent_change < -8: score += 20
        elif recent_change < -4: score += 10

        if momentum_change < -2: score += 15
        elif momentum_change < -1: score += 8

        if volume_ratio > 1.5: score += 10
        elif volume_ratio > 1.2: score += 5

        if rsi and rsi > 70: score += 5
        elif rsi and rsi > 65: score += 3

        if dist_from_resistance is not None:
            if dist_from_resistance < 2: score += 15
            elif dist_from_resistance < 4: score += 8
        if candle_confirmed:
            score += 12
        if weekly_trend_pct is not None and weekly_trend_pct < -params["weekly_trend_threshold"]:
            score += 10
        if index_change_pct is not None and index_change_pct < params["weekly_trend_threshold"]:
            score += 5
        if extension_with_pullback:
            score += 8
        if extension_chase:
            score -= 15
        if "against weekly trend" in warnings:
            score -= 10
        if "against index trend" in warnings:
            score -= 8
        score = max(0, score)

        if score < params["min_score"]:
            return None

        conditions_text = ", ".join(conditions_met)
        if warnings:
            conditions_text += " | ⚠️ " + "; ".join(warnings)

        return {
            "price": current_price, "open": float(opens[-1]), "high": recent_high, "low": recent_low,
            "change_pct": price_change_pct, "volume": volume, "volume_ratio": volume_ratio,
            "avg_volume_lookback": avg_volume, "lookback_days": lb,
            "dist_from_high": dist_from_high, "recent_trend": recent_change, "momentum": momentum_change,
            "rsi": rsi if rsi else 0, "atr": atr, "atr_pct": atr_pct, "score": score,
            "dist_from_resistance": dist_from_resistance, "weekly_trend_pct": weekly_trend_pct,
            "index_change_pct": index_change_pct,
            "support_level": weekly_ctx.get("support") if weekly_ctx else None,
            "resistance_level": weekly_ctx.get("resistance") if weekly_ctx else None,
            "candle_pattern": candlesticks.PATTERN_LABELS.get(candle["pattern"]) if candle_confirmed else None,
            "conditions": conditions_text,
            "conditions_list": conditions_met, "warnings_list": warnings,
            "signal_strength": "STRONG" if score >= params["strong_score"] else "MODERATE" if score >= 50 else "WEAK",
        }
    except Exception:
        return None


# ── UI ───────────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown('<p class="main-header">📉 Swing Short (Sell) Screener</p>', unsafe_allow_html=True)
    st.markdown("*Scan for stocks showing a multi-day/week short setup — daily bars, weekly S/R confirmation*")

    scan_nse, scan_bse, universe = sc.render_exchange_selector(MODE_KEY)
    stocks_to_scan = sc.render_scan_mode_selector(MODE_KEY, universe)
    rate_cfg = sc.render_rate_limit_controls(MODE_KEY)
    strict_mode = sc.render_strict_mode_toggle(MODE_KEY)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Screening Parameters")
    with st.sidebar.expander("Basic Filters", expanded=True):
        daily_period_label = st.selectbox(
            "Daily History Period", list(streak_analysis.PERIOD_LABELS.keys()), index=1,
            key=sskey(MODE_KEY, "daily_period"),
            help="How much daily history to pull per symbol -- also what the RSI/ATR/trend/volume "
                 "windows below have to work with, and the same daily bars the Historical Streak "
                 "Pattern box in the detail view re-uses (fetched once, not twice).",
        )
        min_price = st.number_input("Min Price (₹)", min_value=1, max_value=500, value=20, step=5,
                                     key=sskey(MODE_KEY, "min_price"))
        min_volume = st.number_input("Min Daily Volume", min_value=10000, max_value=10000000, value=100000, step=10000,
                                      key=sskey(MODE_KEY, "min_volume"))
        min_conditions = st.slider(
            "Min Conditions (out of 12)", 2, 12, 4, key=sskey(MODE_KEY, "min_conditions"),
            help="12 possible: price/momentum/volume/RSI/ATR/trend basics, real weekly-S/R proximity, "
                 "candle confirmation, weekly-trend agreement, index agreement, and extension-with-"
                 "bounce confirmation.",
        )
        min_score = st.slider("Min Score (0-100)", 20, 90, 50, 5, key=sskey(MODE_KEY, "min_score"))
        min_market_cap_cr = st.number_input(
            "Min Market Cap (₹ Cr)", min_value=0, max_value=1000000, value=0, step=500,
            key=sskey(MODE_KEY, "min_mcap"),
            help="0 = no market-cap filter. Stocks whose market cap can't be determined are never filtered out.",
        )

    with st.sidebar.expander("Advanced Thresholds"):
        lookback_window = st.slider(
            "Swing Lookback (days)", 10, 60, 20, 5, key=sskey(MODE_KEY, "lookback_window"),
            help="Governs the N-day high/low (rejection zone), N-day trend %, and N-day volume "
                 "average below -- one shared window instead of three separate ones to configure.",
        )
        price_change_threshold = st.slider("Price Change, last 5 days (%)", -10.0, 2.0, 0.0, 0.5, key=sskey(MODE_KEY, "price_chg_th"))
        momentum_threshold = st.slider("Momentum (%)", 0.0, 10.0, 1.0, 0.5, key=sskey(MODE_KEY, "momentum_th"))
        dist_from_high_threshold = st.slider("Dist from N-Day High (%)", 1.0, 15.0, 5.0, 0.5, key=sskey(MODE_KEY, "dist_high_th"))
        volume_ratio_threshold = st.slider("Volume Ratio", 1.0, 3.0, 1.2, 0.1, key=sskey(MODE_KEY, "vol_ratio_th"))
        trend_threshold = st.slider("N-Day Trend (%)", 0.0, 20.0, 4.0, 0.5, key=sskey(MODE_KEY, "trend_th"))
        rsi_threshold = st.slider("RSI Overbought", 50, 80, 65, 5, key=sskey(MODE_KEY, "rsi_th"))
        atr_threshold = st.slider("ATR % Threshold", 0.5, 8.0, 2.0, 0.1, key=sskey(MODE_KEY, "atr_th"))
        dist_from_resistance_threshold = st.slider(
            "Dist from Real Resistance, Weekly (%)", 0.5, 10.0, 3.0, 0.5, key=sskey(MODE_KEY, "dist_resistance_th"),
            help="Support/resistance from ~6 months of weekly bars -- not this stock's own N-day high.",
        )
        weekly_trend_threshold = st.slider(
            "Weekly Trend Confirmation (%)", 0.0, 5.0, 1.0, 0.25, key=sskey(MODE_KEY, "weekly_trend_th"),
            help="Recent ~2 months of weekly closes vs the 2 months before. A signal fighting this is "
                 "flagged, not filtered out.",
        )
        extension_threshold = st.slider(
            "Extension Threshold (%)", 3.0, 30.0, 10.0, 1.0, key=sskey(MODE_KEY, "extension_th"),
            help="A move beyond this over the Momentum Window with NO up day along the way is "
                 "flagged as extended/chase risk; the same move WITH an up day in it is a confirmed "
                 "bounce-and-continue setup instead.",
        )
        candle_lookback = st.slider(
            "Candle Lookback (days)", 1, 10, 3, 1, key=sskey(MODE_KEY, "candle_lookback"),
            help="How many recent DAILY bars to scan for a shooting-star/doji/marubozu/engulfing "
                 "confirmation near resistance -- only checked when price is already near real "
                 "weekly resistance, and only ever a score bonus, never a filter.",
        )

    with st.sidebar.expander("Technical Indicators & Trading Settings"):
        rsi_period = st.number_input("RSI Period (days)", 5, 50, 14, 1, key=sskey(MODE_KEY, "rsi_period"))
        atr_period = st.number_input("ATR Period (days)", 5, 50, 14, 1, key=sskey(MODE_KEY, "atr_period"))
        momentum_window = st.number_input("Momentum Window (days)", 3, 30, 5, 1, key=sskey(MODE_KEY, "mom_window"))
        stop_loss_pct = st.number_input("Stop Loss % above Entry Price", 0.5, 15.0, 3.0, 0.5, key=sskey(MODE_KEY, "sl_pct"))
        target_pct = st.number_input("Target % below Entry Price", 1.0, 40.0, 8.0, 1.0, key=sskey(MODE_KEY, "tgt_pct"))
        risk_per_trade = st.number_input(
            "Risk per Trade (₹)", 100, 1000000, 1000, 100, key=sskey(MODE_KEY, "risk_per_trade"),
            help="Max ₹ you're willing to lose if the stop is hit -- used only to suggest a position "
                 "size in the trade card below, never affects scoring.",
        )
        strong_score = st.number_input("Strong Signal Score", 60, 90, 70, 5, key=sskey(MODE_KEY, "strong_score"))
        chart_height = st.number_input("Chart Height (px)", 200, 500, 250, 50, key=sskey(MODE_KEY, "chart_height"))

    params = {
        "min_price": min_price, "min_volume": min_volume, "min_conditions": min_conditions, "min_score": min_score,
        "price_change_threshold": price_change_threshold, "dist_from_high_threshold": dist_from_high_threshold,
        "trend_threshold": trend_threshold, "momentum_threshold": momentum_threshold,
        "volume_ratio_threshold": volume_ratio_threshold, "rsi_threshold": rsi_threshold,
        "atr_threshold": atr_threshold, "rsi_period": rsi_period, "atr_period": atr_period,
        "momentum_window": momentum_window, "strong_score": strong_score,
        "min_market_cap_cr": min_market_cap_cr,
        "dist_from_resistance_threshold": dist_from_resistance_threshold,
        "weekly_trend_threshold": weekly_trend_threshold, "extension_threshold": extension_threshold,
        "candle_lookback": candle_lookback, "lookback_window": lookback_window,
    }
    if strict_mode:
        params = sc.apply_strict_screening(params, "short")
    daily_period = streak_analysis.PERIOD_LABELS[daily_period_label]
    set_state(MODE_KEY, "trading_settings", {
        "stop_loss_pct": stop_loss_pct, "target_pct": target_pct,
        "risk_per_trade": risk_per_trade, "chart_height": chart_height,
    })

    def fetch_and_analyze(rec):
        snap = swing_data.fetch_swing_snapshot(rec["yf_symbol"], daily_period, retries=rate_cfg["retries"])
        if snap is None:
            return "failed", None

        weekly_ctx = swing_data.fetch_weekly_context(rec["yf_symbol"], retries=rate_cfg["retries"])
        index_change_pct = None
        index_symbol = sc.INDEX_FOR_EXCHANGE.get(rec["exchange"])
        if index_symbol:
            index_snap = swing_data.fetch_swing_snapshot(index_symbol, daily_period, retries=rate_cfg["retries"])
            if index_snap is not None:
                idx_closes = index_snap["daily_close"]
                idx_lb = min(params["lookback_window"], len(idx_closes) - 1)
                if idx_lb > 0 and idx_closes[-idx_lb - 1]:
                    index_change_pct = ((idx_closes[-1] - idx_closes[-idx_lb - 1]) / idx_closes[-idx_lb - 1]) * 100

        analysis = score_swing_short(snap, params, weekly_ctx, index_change_pct)
        if analysis is None:
            return "filtered", None

        shares_out = intraday_data.fetch_shares_outstanding(rec["yf_symbol"], retries=rate_cfg["retries"])
        market_cap_cr = (shares_out * analysis["price"] / 1e7) if shares_out else None
        if params["min_market_cap_cr"] > 0 and market_cap_cr is not None and market_cap_cr < params["min_market_cap_cr"]:
            return "filtered", None

        analysis["market_cap_cr"] = market_cap_cr
        analysis["market_cap_category"] = sc.market_cap_category(market_cap_cr)

        # Purely informational "unusual activity" flag -- see
        # sc.assess_operator_risk()'s docstring and score_swing_long's
        # identical comment. Never affects score, conditions, gating, or
        # sorting. change_pct here is the 5-day move; sideways_then_spike
        # reuses the SAME daily closes already fetched for scoring.
        sideways_then_spike = sc.detect_sideways_then_spike(snap["daily_close"], params["lookback_window"])
        op_risk = sc.assess_operator_risk(
            price=analysis["price"], market_cap_cr=market_cap_cr, volume_ratio=analysis["volume_ratio"],
            change_pct=analysis["change_pct"], change_pct_extreme=15.0, change_pct_moderate=8.0,
            sideways_then_spike=sideways_then_spike,
        )
        analysis["operated_flag"] = "Operated" if op_risk["flagged"] else None
        analysis["operated_reasons"] = op_risk["reasons"]

        analysis.update({"symbol": rec["symbol"], "name": rec["name"], "yf_symbol": rec["yf_symbol"], "exchange": rec["exchange"]})
        return "ok", analysis

    do_scan, resume_scan, checkpoint, _sig, resume_ph = sc.render_scan_trigger(
        MODE_KEY, stocks_to_scan, f"🔍 SCAN {len(stocks_to_scan)} STOCKS FOR SWING SHORT SETUPS")

    if do_scan or resume_scan:
        sc.run_scan(MODE_KEY, stocks_to_scan, fetch_and_analyze, rate_cfg, resume_scan, checkpoint, resume_ph)

    _render_results()

    with st.expander("📚 How to Use"):
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("""
            **Stock Selection:** same Exchange / Scan-Mode controls as every other mode —
            Quick, Full, Slot-wise, Range or Custom List.

            **Holding Period:** days to a few weeks — this is NOT an intraday screener, there's no
            same-day exit assumption anywhere in it.
            """)
        with h2:
            st.markdown("""
            **Signal Strength:** 🔴 STRONG (≥ Strong Signal Score) · 🟡 MODERATE (50+)

            **Risk Management:** Stop loss 2–5% above entry is typical for swing shorts · size
            positions to your own risk tolerance, not just the score

            **Sell Conditions Checked:** price falling over the last few days · near an N-day high
            (rejection zone) · N-day downtrend · negative momentum · high volume · RSI overbought ·
            good ATR volatility · near real weekly resistance · bearish candlestick confirmation ·
            weekly-downtrend agreement · index agreement · extension-with-bounce confirmation
            """)
    sc.footer("<strong>Swing Short (Sell) Screener</strong> · Past patterns are not a prediction of what happens next.")


def _render_results() -> None:
    results = get_state(MODE_KEY, "results")
    if not results:
        st.info("👈 Configure and click 'SCAN' to start")
        return

    st.markdown("---")
    st.success(f"✅ Found {len(results)} potential swing short opportunities!")

    trading_defaults = get_state(MODE_KEY, "trading_settings",
                                  {"stop_loss_pct": 3.0, "target_pct": 8.0, "risk_per_trade": 1000, "chart_height": 250})

    def _rr(r):
        return sc.trade_levels(r["price"], "short", r.get("support_level"), r.get("resistance_level"),
                                trading_defaults["stop_loss_pct"], trading_defaults["target_pct"])["risk_reward"]

    st.markdown("#### Screener Results Summary")

    cap_options = ["Large Cap", "Mid Cap", "Small Cap", "Unknown"]
    f1, f2 = st.columns([1, 1])
    with f1:
        cap_filter = st.multiselect("Market Cap", cap_options, default=cap_options, key=sskey(MODE_KEY, "cap_filter"))
    with f2:
        sort_by = st.selectbox(
            "Sort by", ["Score", "Market Cap", "Change %", "Volume Ratio", "N-Day Trend", "R:R Ratio"],
            key=sskey(MODE_KEY, "sort_by"),
        )

    results = [r for r in results if r.get("market_cap_category", "Unknown") in cap_filter]
    if not results:
        st.warning("⚠️ No results match the current Market Cap filter.")
        return

    _sort_key = {
        "Score": lambda x: x["score"],
        "Market Cap": lambda x: x.get("market_cap_cr") if x.get("market_cap_cr") is not None else -1,
        "Change %": lambda x: -x["change_pct"],  # more negative change = a "stronger" short candidate
        "Volume Ratio": lambda x: x["volume_ratio"],
        "N-Day Trend": lambda x: -x["recent_trend"],
        "R:R Ratio": _rr,
    }[sort_by]
    results = sorted(results, key=_sort_key, reverse=True)

    df = pd.DataFrame([{
        "Symbol": r["symbol"], "Name": r.get("name", ""), "Exchange": r.get("exchange", ""),
        "Price (₹)": r["price"], "Change %": r["change_pct"], "Score": r["score"],
        "Signal": r["signal_strength"], "Market Cap (₹ Cr)": r.get("market_cap_cr"),
        "Cap": r.get("market_cap_category", "Unknown"), "Volume Ratio": r["volume_ratio"],
        "Flag": r.get("operated_flag"),
        "Dist from N-Day High (%)": r["dist_from_high"], "Dist from Resistance (%)": r.get("dist_from_resistance"),
        "Support (₹)": r.get("support_level"), "Resistance (₹)": r.get("resistance_level"),
        "R:R": _rr(r) or None,
        "N-Day Trend (%)": r["recent_trend"], "Weekly Trend (%)": r.get("weekly_trend_pct"),
        "RSI": r["rsi"], "ATR %": r["atr_pct"], "Candle Pattern": r.get("candle_pattern"),
        "Conditions": r["conditions"],
    } for r in results])

    def color_signal(val):
        if val == "STRONG":
            return "background-color: #f8d7da"
        if val == "MODERATE":
            return "background-color: #fff3cd"
        return ""

    def color_change(val):
        try:
            return "background-color: #d4edda" if val < 0 else "background-color: #ffcccc" if val > 0 else ""
        except Exception:
            return ""

    def color_flag(val):
        return "background-color: #f8d7da; font-weight: 600" if val == "Operated" else ""

    styled = df.style.map(color_signal, subset=["Signal"]).map(color_change, subset=["Change %", "N-Day Trend (%)"]) \
        .map(color_flag, subset=["Flag"]).format({
        "Price (₹)": "₹{:.2f}", "Change %": "{:+.2f}%", "Volume Ratio": "{:.2f}x",
        "Market Cap (₹ Cr)": "₹{:,.0f} Cr", "Dist from N-Day High (%)": "{:.2f}%", "Dist from Resistance (%)": "{:.2f}%",
        "Support (₹)": "₹{:.2f}", "Resistance (₹)": "₹{:.2f}", "R:R": "1:{:.2f}",
        "N-Day Trend (%)": "{:+.2f}%", "Weekly Trend (%)": "{:+.2f}%", "RSI": "{:.1f}", "ATR %": "{:.2f}%",
    }, na_rep="—")
    st.dataframe(styled, width="stretch", height=400)

    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("Select one result below to see its chart and trade levels.")
    with col2:
        chart_timeframe = st.selectbox("Chart Timeframe", list(_TIMEFRAME_MAP.keys()), index=2,
                                        key=sskey(MODE_KEY, "chart_tf"))

    options = [f"{r['symbol']} — {r['name']}" if r.get("name") else r["symbol"] for r in results]
    idx_by_option = {opt: i for i, opt in enumerate(options)}
    selected_option = st.selectbox("Select stock for details", options, key=sskey(MODE_KEY, "detail_select"))
    result = results[idx_by_option[selected_option]]

    st.markdown(f"##### {result['symbol']} — {result['signal_strength']} (Score: {result['score']})")
    sc.render_operator_flag_notice(result)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"₹{result['price']:.2f}", f"{result['change_pct']:.2f}%")
    m2.metric(f"{result.get('lookback_days', '?')}-Day Range", f"₹{result['low']:.2f} – ₹{result['high']:.2f}")
    m3.metric("Volume", f"{result['volume']:,.0f}",
              f"vs {result.get('lookback_days', '?')}D avg {result.get('avg_volume_lookback', 0):,.0f}", delta_color="off")
    m4.metric("Vol Ratio", f"{result['volume_ratio']:.2f}x")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("RSI", f"{result['rsi']:.1f}")
    m6.metric("ATR", f"₹{result['atr']:.2f}", f"{result['atr_pct']:.2f}% of price", delta_color="off")
    m7.metric(f"{result.get('lookback_days', '?')}-Day Trend", f"{result['recent_trend']:.2f}%")
    m8.metric("Weekly Trend", f"{result['weekly_trend_pct']:.2f}%" if result.get("weekly_trend_pct") is not None else "—")

    streak_analysis.render_streak_highlight(result, MODE_KEY)

    period, interval = _TIMEFRAME_MAP[chart_timeframe]
    chart_data = intraday_data.fetch_chart_history(result["yf_symbol"], period, interval)
    trading = get_state(MODE_KEY, "trading_settings", {"stop_loss_pct": 3.0, "target_pct": 8.0, "chart_height": 250})

    # Red/green here (not the Long screener's green/blue) is a deliberate
    # visual cue that this is a SHORT -- see sc.render_price_volume_rsi_charts()'s
    # docstring.
    sc.render_price_volume_rsi_charts(result, chart_data, chart_timeframe, trading["chart_height"],
                                       price_color="#dc3545", rsi_color="#28a745")

    # SHORT: stop loss is ABOVE entry, target is BELOW entry. same_session_exit
    # is False here (unlike the intraday screener) -- a real weekly-support
    # target several % away is the whole POINT of a multi-day/week swing
    # hold, not a red flag the way it is for a same-day exit.
    sc.render_trade_card(result, trading, "short", same_session_exit=False)
    sc.render_conditions_checklist(result)

    sc.download_buttons(MODE_KEY, df, df, "swing_short_scan")
