"""
bhavcopy.py — Official NSE / BSE daily equity OHLCV, replacing repeated
yfinance "5d daily" calls with each exchange's own end-of-day bhavcopy file
wherever possible.

This mirrors the same NSE-vs-BSE split used by optvaluation's live_data.py
for option chains:
  - NSE publishes its "Common Bhavcopy" (UDiFF format) at a stable,
    unauthenticated archive URL -- no session/cookie handshake needed for
    the archive host itself (unlike NSE's option-chain JSON API).
  - BSE has no equally-documented public archive for this file. The
    template below is the commonly observed one; if BSE changes it,
    _download_bse_daily() simply fails to fetch (never guesses at a
    parsed value) and get_latest_daily() returns None so the caller
    (intraday_data.py) transparently falls back to a single yfinance
    ".history(period='5d')" call. A bhavcopy miss is never allowed to
    break a scan.

Both exchanges' files share the same column layout (TckrSymb, ClsPric,
TtlTradgVol, FinInstrmTp, ...), matching data_loaders.parse_udiff_bhavcopy()
in the valuation app, so anyone who has debugged that parser will
recognise this one.

Why bhavcopy instead of yfinance for the reference window at all: the
5-day trend calc in both screeners wants completed trading days, not
today's still-forming candle (which "5d daily" from yfinance often
includes intraday). Sourcing the prior full days from the exchange's own
settled EOD file removes that noise, and costs zero extra Yahoo calls
across an entire scan since one day's file is downloaded once and shared
(process-wide cache) across every symbol that asks for it.
"""

from __future__ import annotations

import io
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

_IST = timezone(timedelta(hours=5, minutes=30))

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

_NSE_HOME = "https://www.nseindia.com/"
_NSE_CM_URL_TMPL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ds}_F_0000.csv.zip"

# BSE's equity UDiFF bhavcopy hosting isn't as stably documented as NSE's
# archive host. Both extensions are tried; first one that returns a real
# file wins. If BSE reshapes this entirely, every attempt below just fails
# closed -- see module docstring.
_BSE_CM_URL_TMPLS = [
    "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{ds}_F_0000.CSV",
    "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{ds}_F_0000.csv",
]

_session = requests.Session()
_session.headers.update(_HEADERS)
_nse_warmed_at = 0.0
_warm_lock = threading.Lock()

# Process-wide (shared across users) day-level cache -- one download per
# exchange per calendar date no matter how many symbols/users ask for it.
_cache_lock = threading.Lock()
_day_cache: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
_no_data_days: dict[tuple[str, str], float] = {}  # holiday/weekend/not-yet-published


def _warm_nse() -> None:
    global _nse_warmed_at
    with _warm_lock:
        if (time.time() - _nse_warmed_at) < 300:
            return
        try:
            _session.get(_NSE_HOME, timeout=6)
        except Exception:
            pass
        _nse_warmed_at = time.time()


def _trading_days_back(n_calendar_days_buffer: int = 15) -> list:
    """Yesterday backwards, skipping Sat/Sun. Exchange holidays aren't
    tracked explicitly -- a holiday date just returns no file below and is
    silently skipped, same as a weekend."""
    out = []
    d = datetime.now(_IST).date() - timedelta(days=1)
    for _ in range(n_calendar_days_buffer):
        if d.weekday() < 5:  # Mon=0 ... Fri=4
            out.append(d)
        d -= timedelta(days=1)
    return out


def _download_nse_daily(d) -> Optional[pd.DataFrame]:
    ds = d.strftime("%Y%m%d")
    key = ("NSE", ds)
    with _cache_lock:
        if key in _day_cache:
            return _day_cache[key]
        if key in _no_data_days:
            return None
    try:
        _warm_nse()
        resp = _session.get(_NSE_CM_URL_TMPL.format(ds=ds), timeout=10)
        if resp.status_code != 200 or len(resp.content) < 200:
            raise ValueError(f"bad response ({resp.status_code})")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                raise ValueError("empty zip")
            with zf.open(names[0]) as fh:
                df = pd.read_csv(fh)
        df.columns = [c.strip() for c in df.columns]
        with _cache_lock:
            _day_cache[key] = df
        return df
    except Exception:
        with _cache_lock:
            _no_data_days[key] = time.time()
        return None


def _download_bse_daily(d) -> Optional[pd.DataFrame]:
    ds = d.strftime("%Y%m%d")
    key = ("BSE", ds)
    with _cache_lock:
        if key in _day_cache:
            return _day_cache[key]
        if key in _no_data_days:
            return None
    for tmpl in _BSE_CM_URL_TMPLS:
        try:
            resp = _session.get(tmpl.format(ds=ds), timeout=10,
                                 headers={**_HEADERS, "Referer": "https://www.bseindia.com/"})
            if resp.status_code != 200 or len(resp.content) < 200:
                continue
            content = resp.content
            if content[:2] == b"PK":  # zipped
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    names = zf.namelist()
                    if not names:
                        continue
                    with zf.open(names[0]) as fh:
                        df = pd.read_csv(fh)
            else:
                df = pd.read_csv(io.BytesIO(content))
            df.columns = [c.strip() for c in df.columns]
            with _cache_lock:
                _day_cache[key] = df
            return df
        except Exception:
            continue
    with _cache_lock:
        _no_data_days[key] = time.time()
    return None


def _split_yf_symbol(yf_symbol: str):
    su = yf_symbol.upper()
    if su.endswith(".NS"):
        return "NSE", yf_symbol[:-3].upper()
    if su.endswith(".BO"):
        return "BSE", yf_symbol[:-3]
    return None, yf_symbol


def _row_for_symbol(df: pd.DataFrame, exchange: str, raw_symbol: str):
    try:
        if df is None or df.empty:
            return None
        cols = set(df.columns)

        if exchange == "NSE":
            if "TckrSymb" not in cols:
                return None
            sub = df[df["TckrSymb"].astype(str).str.upper().str.strip() == raw_symbol.upper()]
        else:
            # BSE: joined on the numeric scrip code from bse_codes.csv.
            # The shared UDiFF template carries that code in FinInstrmId
            # for BSE files. If BSE's real layout differs, this simply
            # matches nothing and the caller falls back to yfinance.
            if "FinInstrmId" not in cols:
                return None
            try:
                code_num = int(str(raw_symbol).strip())
            except ValueError:
                return None
            sub = df[pd.to_numeric(df["FinInstrmId"], errors="coerce") == code_num]

        if sub.empty:
            return None

        if "FinInstrmTp" in sub.columns:
            eq = sub[sub["FinInstrmTp"].astype(str).str.upper().isin(["STK", "EQ", "IDX"])]
            if not eq.empty:
                sub = eq
        return sub.iloc[0]
    except Exception:
        return None


def get_latest_daily(yf_symbol: str, n: int = 5) -> Optional[pd.DataFrame]:
    """Best-effort last `n` COMPLETED trading days of (close, volume) for
    `yf_symbol`, chronologically ordered (oldest -> newest). Returns None
    on any uncertainty at all -- unmatched symbol, endpoint/format change,
    not enough history -- so the caller can fall back to yfinance without
    special-casing anything."""
    try:
        exchange, raw_symbol = _split_yf_symbol(yf_symbol)
        if exchange is None:
            return None
        downloader = _download_nse_daily if exchange == "NSE" else _download_bse_daily

        closes, volumes = [], []
        for d in _trading_days_back(15):
            if len(closes) >= n:
                break
            day_df = downloader(d)
            if day_df is None:
                continue
            row = _row_for_symbol(day_df, exchange, raw_symbol)
            if row is None:
                continue

            close_val = None
            for c in ("ClsPric", "ClosePrice", "Close"):
                if c in row.index and pd.notna(row[c]):
                    close_val = float(row[c])
                    break
            if close_val is None:
                continue

            vol_val = 0.0
            for c in ("TtlTradgVol", "TotalTradedQty", "Volume"):
                if c in row.index and pd.notna(row[c]):
                    vol_val = float(row[c])
                    break

            closes.append(close_val)
            volumes.append(vol_val)

        if len(closes) < 2:
            return None

        closes.reverse()
        volumes.reverse()
        return pd.DataFrame({"close": closes, "volume": volumes})
    except Exception:
        return None
