"""
candlesticks.py — Candlestick-pattern confirmation for a move near S/R.

The reference strategy material behind this screener treats a numeric
"near support/resistance" reading as necessary but not sufficient: it's
only a valid entry when the candle sitting at that zone actually shows a
reversal/indecision shape (doji, hammer, marubozu, a long wick, an
engulfing bar). Everything else in scoring is numeric (RSI/momentum/
volume/S-R distance) with no notion of candle shape at all, hence this
module.

Operates only on OHLC arrays already sitting in the intraday snapshot
(intraday_data.fetch_intraday_snapshot) -- no new Yahoo calls. Like
indicators.atr(), detect_pattern() is used by callers that build up a
conditions/score list without a None-guard on every field, so it ALWAYS
returns a well-formed dict (pattern=None, not-bullish, not-bearish) on
bad/insufficient data rather than raising or returning None -- the
"nothing detected" case is a normal, common outcome here (most candles,
most of the time, aren't a pattern), not an error.
"""

from __future__ import annotations

import numpy as np

# Human-readable labels for the results table -- keys match the "pattern"
# string returned by detect_pattern().
PATTERN_LABELS = {
    "marubozu": "Marubozu",
    "hammer": "Hammer",
    "shooting_star": "Shooting Star",
    "doji": "Doji",
    "spinning_top": "Spinning Top",
    "long_lower_wick": "Long Lower Wick",
    "long_upper_wick": "Long Upper Wick",
    "bullish_engulfing": "Bullish Engulfing",
    "bearish_engulfing": "Bearish Engulfing",
}


def _no_pattern() -> dict:
    """Fresh "nothing detected" result -- a literal dict per call so callers
    can't accidentally mutate a shared instance."""
    return {"pattern": None, "bullish": False, "bearish": False}


def _classify_bar(o: float, h: float, l: float, c: float) -> dict:
    """Classifies a single OHLC bar by body/wick shape. Checked in order
    from most specific/strongest shape to weakest, since a bar can satisfy
    more than one loose definition at once (e.g. a marubozu is trivially
    also "not a doji" but could also brush a long-wick threshold on a
    noisy 1-min bar -- the strongest read wins)."""
    if not all(np.isfinite(x) for x in (o, h, l, c)):
        return _no_pattern()

    rng = h - l
    if rng <= 0:
        return _no_pattern()

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_pct = body / rng
    upper_pct = upper_wick / rng
    lower_pct = lower_wick / rng

    # Marubozu: body dominates the whole range, wicks negligible on both
    # ends -- direction comes from which way the body closed.
    if body_pct >= 0.9 and upper_pct <= 0.05 and lower_pct <= 0.05:
        bullish = bool(c > o)
        return {"pattern": "marubozu", "bullish": bullish, "bearish": not bullish}

    # Hammer: small body sitting at the TOP of the range, long lower wick,
    # little/no upper wick -- bullish reversal shape, relevant near support.
    if body_pct <= 0.35 and lower_pct >= 0.5 and upper_pct <= 0.15:
        return {"pattern": "hammer", "bullish": True, "bearish": False}

    # Shooting star: the mirror image of a hammer -- small body at the
    # BOTTOM of the range, long upper wick -- bearish, relevant near
    # resistance.
    if body_pct <= 0.35 and upper_pct >= 0.5 and lower_pct <= 0.15:
        return {"pattern": "shooting_star", "bullish": False, "bearish": True}

    # Doji: body is negligible regardless of where the wicks sit --
    # indecision, direction-agnostic (its meaning comes from location:
    # bullish reversal candidate at support, bearish at resistance).
    if body_pct <= 0.1:
        return {"pattern": "doji", "bullish": False, "bearish": False}

    # Spinning top: small-ish body with BOTH wicks substantial -- also
    # indecision, same direction-agnostic treatment as doji.
    if body_pct <= 0.3 and upper_pct >= 0.25 and lower_pct >= 0.25:
        return {"pattern": "spinning_top", "bullish": False, "bearish": False}

    # Looser fallback: doesn't meet hammer/shooting-star's strict "small
    # body pinned to one end" shape, but one wick still clearly dominates
    # the bar -- the "candle with a long lower/upper tail" the reference
    # material calls out as its own (looser) confirmation case.
    if lower_pct >= 0.5 and body_pct <= 0.5:
        return {"pattern": "long_lower_wick", "bullish": True, "bearish": False}
    if upper_pct >= 0.5 and body_pct <= 0.5:
        return {"pattern": "long_upper_wick", "bullish": False, "bearish": True}

    return _no_pattern()


def _check_engulfing(prev_o: float, prev_c: float, cur_o: float, cur_c: float) -> dict:
    """Two-bar reversal pattern: the current bar's body fully engulfs the
    prior bar's body AND closes the opposite direction. Only meaningful on
    the freshest two bars -- an engulfing pattern from several bars ago
    isn't "the candle at the zone right now"."""
    if not all(np.isfinite(x) for x in (prev_o, prev_c, cur_o, cur_c)):
        return _no_pattern()

    prev_body = abs(prev_c - prev_o)
    cur_body = abs(cur_c - cur_o)
    if prev_body <= 0 or cur_body <= prev_body:
        return _no_pattern()

    prev_top, prev_bottom = max(prev_o, prev_c), min(prev_o, prev_c)
    cur_top, cur_bottom = max(cur_o, cur_c), min(cur_o, cur_c)
    engulfs = cur_top >= prev_top and cur_bottom <= prev_bottom

    if engulfs and prev_c < prev_o and cur_c > cur_o:
        return {"pattern": "bullish_engulfing", "bullish": True, "bearish": False}
    if engulfs and prev_c > prev_o and cur_c < cur_o:
        return {"pattern": "bearish_engulfing", "bullish": False, "bearish": True}
    return _no_pattern()


def detect_pattern(opens, highs, lows, closes, lookback: int = 5) -> dict:
    """Looks for a candlestick pattern in the most recent `lookback` bars.

    Checks the freshest two bars for an engulfing pattern first (a reversal
    signal that's stale by a few bars isn't the candle sitting at the S/R
    zone right now), then scans single-bar patterns from most recent
    backwards through the window and returns on the first match, so a
    fresh pattern is preferred over an older one further back in the
    window.

    Returns {"pattern": <name or None>, "bullish": bool, "bearish": bool}.
    Never raises -- returns the "nothing detected" default (pattern=None)
    on bad, missing, or insufficient data, same convention as indicators.atr().
    """
    try:
        o = np.asarray(opens, dtype=float) if opens is not None else np.array([])
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)
        n = min(len(o), len(h), len(l), len(c))
        if n < 1:
            return _no_pattern()
        o, h, l, c = o[-n:], h[-n:], l[-n:], c[-n:]

        window = min(lookback, n) if lookback and lookback > 0 else n

        if n >= 2:
            engulf = _check_engulfing(o[-2], c[-2], o[-1], c[-1])
            if engulf["pattern"]:
                return engulf

        for i in range(1, window + 1):
            bar = _classify_bar(o[-i], h[-i], l[-i], c[-i])
            if bar["pattern"]:
                return bar

        return _no_pattern()
    except Exception:
        return _no_pattern()
