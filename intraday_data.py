"""
intraday_data.py — Shared raw-data fetch for the two intraday modes.

The Short and Long intraday screeners need exactly the same two Yahoo calls
per symbol (today's 1-minute bars + the last 5 daily bars) — only what they
do with that data (which direction counts as a "signal") differs. Fetching
it once, here, means:
  • one implementation to keep rate-limit-safe and dead-symbol-aware
    instead of two near-identical copies drifting apart
  • a single short-TTL cache shared by both modes, so running Short then
    Long back-to-back on an overlapping universe doesn't double Yahoo's
    load for data that's already sitting in memory
"""

from __future__ import annotations

import numpy as np

import bhavcopy
import scanner_common as sc
from scanner_common import yf

# Intraday data goes stale fast — a 45s TTL is long enough to dedupe the
# handful of repeat calls within one scan/rerun, short enough that a
# "refresh" never shows minutes-old prices ("no lagging").
_CACHE_TTL_S = 45


def fetch_intraday_snapshot(yf_symbol: str):
    """Returns a dict of numpy arrays / floats, or None if the symbol has no
    tradable data right now (pre-market, delisted, holiday, etc.)."""
    if sc.is_known_dead(yf_symbol):
        return None

    cache_key = f"intraday_snap:{yf_symbol}"
    cached = sc.cache_get(cache_key, _CACHE_TTL_S)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(yf_symbol)
        # 1-minute intraday bars have no free public source for NSE/BSE, so
        # this one call stays on yfinance no matter what — see bhavcopy.py's
        # module docstring for why.
        intraday = ticker.history(period="1d", interval="1m")

        # The plain 5-day *daily* reference bars ARE ordinary EOD data, so
        # try NSE/BSE's own bhavcopy first — zero Yahoo calls when it has
        # data — and only fall back to yfinance if it doesn't.
        bhav = bhavcopy.get_latest_daily(yf_symbol, n=5)
        if bhav is not None:
            daily_close = bhav["close"].values.astype(float)
            daily_volume = bhav["volume"].values.astype(float)
            daily_is_empty = False
        else:
            daily = ticker.history(period="5d", interval="1d")
            daily_close = daily["Close"].values.astype(float)
            daily_volume = daily["Volume"].values.astype(float)
            daily_is_empty = daily.empty

        if intraday.empty or daily_is_empty:
            # Empty daily history on a >5-day-old symbol is a real dead signal;
            # an empty *intraday* frame can just mean market's closed right
            # now, so only mark dead when the daily history is also empty.
            if daily_is_empty:
                sc.mark_dead_symbol(yf_symbol)
            return None

        snapshot = {
            "yf_symbol": yf_symbol,
            "intraday_open": float(intraday["Open"].iloc[0]),
            "intraday_close": intraday["Close"].values.astype(float),
            "intraday_high": intraday["High"].values.astype(float),
            "intraday_low": intraday["Low"].values.astype(float),
            "intraday_volume": intraday["Volume"].values.astype(float),
            "day_high": float(intraday["High"].max()),
            "day_low": float(intraday["Low"].min()),
            "daily_close": daily_close,
            "daily_volume": daily_volume,
        }
        sc.cache_set(cache_key, snapshot)
        return snapshot

    except Exception as e:
        if any(kw in str(e).lower() for kw in ("delisted", "not found", "no data found")):
            sc.mark_dead_symbol(yf_symbol)
        return None


def fetch_chart_history(yf_symbol: str, period: str, interval: str):
    """For the detail-view chart's user-selected timeframe. Cached briefly
    to survive re-renders (filter tweaks, etc.) without a fresh Yahoo hit."""
    cache_key = f"chart_hist:{yf_symbol}:{period}:{interval}"
    cached = sc.cache_get(cache_key, 60)
    if cached is not None:
        return cached
    try:
        data = sc.bulletproof_fetch(lambda: yf.Ticker(yf_symbol).history(period=period, interval=interval))
        if data is not None:
            sc.cache_set(cache_key, data)
        return data
    except Exception:
        return None
