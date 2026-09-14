"""
swing_data.py — Shared raw-data fetch for the two swing modes, mirroring
intraday_data.py's role for the intraday pair one level up in timeframe:
DAILY bars stand in for 1-minute bars, WEEKLY bars stand in for the hourly
context intraday_data.fetch_hourly_context() provides.

The daily pull itself is NOT reimplemented here -- it's exactly the same
"plain daily OHLCV for one symbol over one period" call streak_analysis.py
already makes for the per-stock streak box, so fetch_swing_snapshot()
below reuses streak_analysis.fetch_daily_history() instead of standing up
a second Yahoo daily-history fetcher with its own cache. A swing scan and
a later look at that same stock's streak box (or vice versa, in either
order) share one fetch and one cache entry per (symbol, period) rather
than paying for the daily history twice.
"""

from __future__ import annotations

import scanner_common as sc
import streak_analysis
from scanner_common import yf

# Weekly bars only change once a week, so a long TTL is safe -- same
# reasoning as intraday_data's hourly-context cache, one level up.
_WEEKLY_CACHE_TTL_S = 6 * 3600


def fetch_weekly_context(yf_symbol: str, retries: int = 3):
    """Real weekly support/resistance + the weekly trend direction, built
    from ~2 years of weekly bars -- the swing equivalent of
    intraday_data.fetch_hourly_context(): the higher-timeframe read a
    daily-bar signal needs to mean anything (a pullback to a daily support
    level means little if it isn't near a level anyone trading the weekly
    chart would recognise; a daily bounce that fights the weekly trend is
    a classic bull/bear-trap, not a confirmed setup).

    Returns {"resistance": float, "support": float, "weekly_trend_pct": float}
    or None if weekly history isn't available/long enough to be meaningful.
    Never raises.
    """
    if sc.is_known_dead(yf_symbol):
        return None
    cache_key = f"weekly_ctx:{yf_symbol}"
    cached = sc.cache_get(cache_key, _WEEKLY_CACHE_TTL_S)
    if cached is not None:
        return cached
    try:
        weekly = sc.bulletproof_fetch(lambda: yf.Ticker(yf_symbol).history(period="2y", interval="1wk"), retries=retries)
        if weekly is None or weekly.empty or len(weekly) < 6:
            return None
        highs = weekly["High"].values.astype(float)
        lows = weekly["Low"].values.astype(float)
        closes = weekly["Close"].values.astype(float)

        # Resistance/support from the last ~6 months of weekly bars (26
        # weeks) -- recent enough to matter to today's price, same spirit
        # as fetch_hourly_context's 5-day window one level down.
        window = min(26, len(highs))
        resistance = float(highs[-window:].max())
        support = float(lows[-window:].min())

        # Recent ~2 months of weekly closes vs the 2 months before that --
        # mirrors fetch_hourly_context's "recent half vs prior half" read,
        # just at weekly instead of hourly granularity.
        half = max(1, min(8, len(closes) // 2))
        recent = closes[-half:].mean()
        prior = closes[-2 * half:-half].mean() if len(closes) >= 2 * half else closes[0]
        weekly_trend_pct = ((recent - prior) / prior) * 100 if prior else 0.0

        ctx = {"resistance": resistance, "support": support, "weekly_trend_pct": weekly_trend_pct}
        sc.cache_set(cache_key, ctx)
        return ctx
    except Exception:
        return None


def fetch_swing_snapshot(yf_symbol: str, daily_period: str = "1y", retries: int = 3):
    """Returns a dict of numpy arrays built from daily bars (open/high/low/
    close/volume), or None if there isn't enough daily history to compute
    anything meaningful from (needs enough bars for a ~20-day moving
    average / volume average to mean something). Reuses
    streak_analysis.fetch_daily_history() for the actual Yahoo call --
    see this module's docstring for why."""
    daily = streak_analysis.fetch_daily_history(yf_symbol, daily_period, retries=retries)
    if daily is None or len(daily) < 30:
        return None
    try:
        return {
            "yf_symbol": yf_symbol,
            "daily_open": daily["Open"].values.astype(float),
            "daily_high": daily["High"].values.astype(float),
            "daily_low": daily["Low"].values.astype(float),
            "daily_close": daily["Close"].values.astype(float),
            "daily_volume": daily["Volume"].values.astype(float),
        }
    except Exception:
        return None
