"""
indicators.py — RSI and ATR, computed defensively.

Both mode files use these results directly in arithmetic without a
None-guard on the ATR path (`atr_pct = (atr / current_price) * 100`), so
atr() must always return a finite float. RSI is allowed to return None
(callers already check `if rsi and ...`) when there isn't enough history
to compute a meaningful value.
"""

from __future__ import annotations

import numpy as np


def rsi(closes, period: int = 14):
    """Wilder's RSI. Returns None if there isn't enough data (rather than
    guessing at a value from a too-short window)."""
    try:
        arr = np.asarray(closes, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < period + 1:
            return None

        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        value = 100.0 - (100.0 / (1.0 + rs))
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def atr(highs, lows, closes, period: int = 14) -> float:
    """Average True Range. ALWAYS returns a finite float (0.0 on
    insufficient/bad data) since callers use it without a None-check."""
    try:
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)
        n = min(len(h), len(l), len(c))
        if n < 2:
            return 0.0
        h, l, c = h[-n:], l[-n:], c[-n:]

        prev_close = np.roll(c, 1)
        prev_close[0] = c[0]

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
        tr = tr[np.isfinite(tr)]
        if tr.size == 0:
            return 0.0

        window = tr[-period:] if tr.size >= period else tr
        value = float(np.nanmean(window))
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0
