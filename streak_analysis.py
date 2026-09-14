"""
streak_analysis.py — "This stock is currently on an N-day streak; here's
what happened historically after streaks like this" -- shown inline in
each mode's per-stock detail view, for the person's own reading only.

IMPORTANT: this is informational context, not a signal. Nothing here feeds
score_long()/score_short(), the conditions/warnings lists, sorting, or
min-conditions gating -- it's rendered straight from a symbol's own daily
close history, independently of whatever made that symbol show up in the
scan results in the first place.

Data cost: one plain daily-history fetch, and ONLY for whichever single
stock is currently selected in a detail view (same "only the selected
symbol pays for it" rule fetch_chart_history already follows) -- never
across the whole results table, so this doesn't multiply Yahoo calls per
scan the way anything in the per-symbol scan loop would.

fetch_daily_history() below is also reused by swing_data.py for the two
swing modes' actual scoring data (not just this file's own streak box) --
one fetch and one cache entry per (symbol, period) serves both, so
looking at a stock's streak box and then switching to a swing scan on the
same stock (or vice versa) doesn't pay for its daily history twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import scanner_common as sc
from scanner_common import yf

# Daily bars only change once a day (after close), so a long TTL is safe --
# unlike the 45s intraday cache, there's no staleness risk here, just a
# wasted re-fetch on every widget interaction (Streamlit reruns the whole
# script on each one).
_DAILY_HIST_CACHE_TTL_S = 6 * 3600

PERIOD_LABELS = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
_WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fetch_daily_history(yf_symbol: str, period: str, retries: int = 3):
    """Plain daily OHLC history for one symbol over one lookback period.
    A one-off single-symbol call (only for the selected detail-view stock,
    never the whole results table), so this doesn't need scanner_common's
    batch rate-limit/checkpoint machinery -- just the same bulletproof_fetch
    + cache pattern intraday_data.py uses for its own Yahoo calls."""
    if sc.is_known_dead(yf_symbol):
        return None
    cache_key = f"daily_hist_long:{yf_symbol}:{period}"
    cached = sc.cache_get(cache_key, _DAILY_HIST_CACHE_TTL_S)
    if cached is not None:
        return cached
    try:
        data = sc.bulletproof_fetch(lambda: yf.Ticker(yf_symbol).history(period=period, interval="1d"), retries=retries)
        if data is None or data.empty:
            return None
        sc.cache_set(cache_key, data)
        return data
    except Exception:
        return None


def current_streak(hist_df: pd.DataFrame):
    """How many days in a row, ending at the most recent close, moved the
    same direction. Returns (streak_len, direction, last_date) where
    direction is "up"/"down"/"flat" -- "flat" (streak_len 0) means the most
    recent close was unchanged from the one before it, so there's no
    active streak to look up. Never raises; returns (0, "flat", None) on
    bad/insufficient data."""
    try:
        closes = hist_df["Close"].values.astype(float)
        dates = hist_df.index
        if len(closes) < 2:
            return 0, "flat", (dates[-1] if len(dates) else None)

        daily_return = (closes[1:] - closes[:-1]) / closes[:-1]
        last_return = daily_return[-1]
        if last_return == 0:
            return 0, "flat", dates[-1]

        direction = "up" if last_return > 0 else "down"
        flags = daily_return > 0 if direction == "up" else daily_return < 0
        streak_len = 0
        for f in flags[::-1]:
            if not f:
                break
            streak_len += 1
        return streak_len, direction, dates[-1]
    except Exception:
        return 0, "flat", None


def find_streak_followthrough(hist_df: pd.DataFrame, streak_len: int, direction: str) -> dict | None:
    """Scans daily closes for every day that ends an N-day run of same-
    direction closes, and records what the very next trading day did.

    Note on overlapping streaks: a 7-day up run contains three separate
    5-in-a-row endings (days 5, 6, and 7 of that run) -- each is counted
    as its own occurrence with its own "next day", the same way a standard
    rolling-window backtest would, rather than only the first day a fresh
    streak reaches length N.

    Returns None if there isn't enough history for even one streak_len+1
    window, or if no occurrences exist; never raises.
    """
    try:
        closes = hist_df["Close"].values.astype(float)
        dates = hist_df.index
        n = len(closes)
        if n < streak_len + 1:
            return None

        daily_return_pct = np.full(n, np.nan)
        daily_return_pct[1:] = (closes[1:] - closes[:-1]) / closes[:-1] * 100
        is_up_day = daily_return_pct > 0
        is_down_day = daily_return_pct < 0
        day_flags = is_up_day if direction == "up" else is_down_day

        occurrences = []
        for i in range(streak_len, n - 1):  # need streak_len days ending at i, AND a next day at i+1
            if day_flags[i - streak_len + 1: i + 1].all():
                next_return = daily_return_pct[i + 1]
                occurrences.append({
                    "streak_end_date": dates[i],
                    "streak_end_dow": dates[i].day_name(),
                    "next_date": dates[i + 1],
                    "calendar_gap_days": (dates[i + 1] - dates[i]).days,
                    "next_return_pct": next_return,
                    "next_up": next_return > 0,
                    "next_down": next_return < 0,
                })

        if not occurrences:
            return None

        # Baseline: how often is ANY day (no streak condition) an up day in
        # this same period -- context for whether the streak actually
        # shifts the odds, or the stock just goes up most days regardless.
        baseline_up_pct = float(np.nanmean(is_up_day[1:])) * 100 if n > 1 else 0.0

        total = len(occurrences)
        next_up_count = sum(o["next_up"] for o in occurrences)
        next_down_count = sum(o["next_down"] for o in occurrences)
        avg_next_return = float(np.mean([o["next_return_pct"] for o in occurrences]))

        by_dow = {}
        for dow in sorted({o["streak_end_dow"] for o in occurrences}, key=_WEEKDAY_ORDER.index):
            rows = [o for o in occurrences if o["streak_end_dow"] == dow]
            by_dow[dow] = {
                "count": len(rows),
                "next_up_pct": sum(r["next_up"] for r in rows) / len(rows) * 100,
                "next_down_pct": sum(r["next_down"] for r in rows) / len(rows) * 100,
                "avg_next_return_pct": float(np.mean([r["next_return_pct"] for r in rows])),
                "avg_calendar_gap_days": float(np.mean([r["calendar_gap_days"] for r in rows])),
            }

        return {
            "total_occurrences": total, "next_up_count": next_up_count, "next_down_count": next_down_count,
            "next_up_pct": next_up_count / total * 100, "next_down_pct": next_down_count / total * 100,
            "avg_next_return_pct": avg_next_return, "baseline_up_pct": baseline_up_pct,
            "by_dow": by_dow, "occurrences": occurrences,
        }
    except Exception:
        return None


def render_streak_highlight(result: dict, mode_key: str) -> None:
    """The highlighted, inline "historical pattern" box for whichever stock
    is currently selected in a mode's detail view. Self-contained: reads
    only result["yf_symbol"]/result["symbol"], writes nothing back to the
    scan results, and is identical for the Long and Short screeners (a
    stock's own past-streak behavior doesn't depend on which screener
    happened to surface it today)."""
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown("##### 📅 Historical Streak Pattern *(info only — not part of the score)*")
        lookback_label = h2.selectbox(
            "Lookback", list(PERIOD_LABELS.keys()), index=1,
            key=sc.sskey(mode_key, f"streak_lookback_{result['yf_symbol']}"), label_visibility="collapsed",
        )

        hist = fetch_daily_history(result["yf_symbol"], PERIOD_LABELS[lookback_label])
        if hist is None:
            st.caption("Couldn't fetch daily history for this stock right now.")
            return

        streak_len, direction, last_date = current_streak(hist)
        if streak_len < 2 or last_date is None:
            st.caption(f"No active multi-day streak right now — the last close was "
                       f"{'flat' if direction == 'flat' else direction} vs. the day before.")
            return

        stats = find_streak_followthrough(hist, streak_len, direction)
        last_dow = last_date.day_name()
        st.markdown(f"**Currently on a {streak_len}-day {direction} streak** "
                    f"(as of {last_date.strftime('%d %b %Y')}, a {last_dow}).")

        if stats is None:
            st.caption(f"No matching {streak_len}-day {direction}-streaks found in the last "
                       f"{lookback_label.lower()} to compare against.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Same streak, last {lookback_label.lower()}", f"{stats['total_occurrences']}× before")
        c2.metric("Next day UP", f"{stats['next_up_pct']:.0f}%")
        c3.metric("Baseline (any day) UP", f"{stats['baseline_up_pct']:.0f}%",
                  help="How often ANY day in this period closed up, with no streak condition at all -- "
                       "compare to 'Next day UP' to see whether this streak actually shifts the odds.")

        # The specific, personalized version of the Friday→Monday question:
        # what happened after streaks that ended on the SAME weekday as the
        # one this stock is sitting on right now (a streak ending Friday
        # has its "next day" fall after a weekend gap; most other weekdays
        # don't).
        same_dow = stats["by_dow"].get(last_dow)
        if same_dow:
            st.info(f"📌 Specifically after a {last_dow}-ending streak like this one "
                    f"({same_dow['count']}× before): next trading day was UP "
                    f"{same_dow['next_up_pct']:.0f}% of the time, averaging "
                    f"{same_dow['avg_next_return_pct']:+.2f}%, after an average "
                    f"{same_dow['avg_calendar_gap_days']:.1f}-calendar-day gap.")
        else:
            st.caption(f"No historical occurrences specifically ending on a {last_dow} in this lookback window.")

        if stats["total_occurrences"] < 10:
            st.caption(f"⚠️ Only {stats['total_occurrences']} occurrences total — small sample, "
                       f"treat these percentages as indicative rather than reliable statistics.")

        with st.expander("Full breakdown by day the streak ended on"):
            dow_rows = [{
                "Streak ended on": dow, "Occurrences": s["count"], "Next day UP %": s["next_up_pct"],
                "Next day DOWN %": s["next_down_pct"], "Avg next-day return %": s["avg_next_return_pct"],
                "Avg calendar gap (days)": s["avg_calendar_gap_days"],
            } for dow, s in stats["by_dow"].items()]
            df_dow = pd.DataFrame(dow_rows)
            st.dataframe(df_dow.style.format({
                "Next day UP %": "{:.1f}%", "Next day DOWN %": "{:.1f}%",
                "Avg next-day return %": "{:+.2f}%", "Avg calendar gap (days)": "{:.1f}",
            }), width="stretch", height=min(260, 45 + 38 * len(df_dow)))
