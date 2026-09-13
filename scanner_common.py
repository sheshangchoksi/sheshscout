"""
scanner_common.py — Everything the two intraday modes (Long / Short) share:
universe loading (NSE tickers + BSE codes -> yfinance symbols), the
exchange / scan-mode / rate-limit sidebar controls, a thread-safe
process-wide cache + dead-symbol registry, a retry wrapper around every
external call, checkpointed scanning, and small UI helpers (download
buttons, footer).

Deployment target is Streamlit Community Cloud's free tier: shared CPU,
~1GB RAM, one process that may serve more than one browser session. That
shapes several choices below:
  - the in-memory cache / dead-symbol registry are module-level (process-
    wide), not st.session_state, so concurrent users scanning overlapping
    universes share Yahoo/bhavcopy hits instead of each re-fetching
    identical market data.
  - per-user scan results / checkpoints DO use st.session_state, since
    those are naturally per-browser-tab.
  - scans run in small, capped batches with an explicit time budget so a
    big "Full Universe" scan degrades to a resumable checkpoint instead of
    hammering Yahoo, blowing the RAM ceiling, or hitting Streamlit Cloud's
    request timeout.
  - every external call is wrapped so a bad response, a network blip, or a
    changed endpoint returns None/[] instead of raising -- nothing here is
    allowed to take the whole app down mid-scan.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception as _e:  # pragma: no cover - surfaced clearly at import time
    yf = None
    _YF_IMPORT_ERROR = _e
else:
    _YF_IMPORT_ERROR = None

MODE_LONG = "long"
MODE_SHORT = "short"

# Rough, rule-of-thumb Indian market-cap bands (₹ Cr) shared by both modes
# so "Large/Mid/Small Cap" means the same thing regardless of screener.
_LARGE_CAP_CR = 20000
_MID_CAP_CR = 5000


def market_cap_category(market_cap_cr: Optional[float]) -> str:
    if market_cap_cr is None:
        return "Unknown"
    if market_cap_cr >= _LARGE_CAP_CR:
        return "Large Cap"
    if market_cap_cr >= _MID_CAP_CR:
        return "Mid Cap"
    return "Small Cap"

_HERE = Path(__file__).parent
_NSE_CSV = _HERE / "nse_tickers.csv"
_BSE_CSV = _HERE / "bse_codes.csv"


# --------------------------------------------------------------------- #
# Per-session state helpers (results/checkpoints are per browser tab)
# --------------------------------------------------------------------- #
def sskey(mode_key: str, name: str) -> str:
    return f"{mode_key}__{name}"


def get_state(mode_key: str, name: str, default: Any = None) -> Any:
    return st.session_state.get(f"{mode_key}::{name}", default)


def set_state(mode_key: str, name: str, value: Any) -> None:
    st.session_state[f"{mode_key}::{name}"] = value


# --------------------------------------------------------------------- #
# Process-wide cache (shared across users/sessions on purpose -- market
# data is identical for everyone hitting the same symbol in the same
# 45-second window, so sharing the cache cuts Yahoo calls under load).
# --------------------------------------------------------------------- #
_cache_lock = threading.Lock()
_cache_store: dict[str, tuple[float, Any]] = {}
_CACHE_HARD_CAP = 8000  # evict oldest entries past this to bound memory


def cache_get(key: str, ttl_seconds: float) -> Any:
    with _cache_lock:
        item = _cache_store.get(key)
    if item is None:
        return None
    ts, value = item
    if (time.time() - ts) > ttl_seconds:
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _cache_store[key] = (time.time(), value)
        if len(_cache_store) > _CACHE_HARD_CAP:
            oldest_keys = sorted(_cache_store, key=lambda k: _cache_store[k][0])[:1000]
            for k in oldest_keys:
                _cache_store.pop(k, None)


# --------------------------------------------------------------------- #
# Dead-symbol registry (process-wide, self-healing): a symbol that looked
# delisted / had zero data gets skipped for a few hours, not forever --
# in case it was a transient Yahoo/exchange hiccup rather than a real
# delisting.
# --------------------------------------------------------------------- #
_dead_lock = threading.Lock()
_dead_symbols: dict[str, float] = {}
_DEAD_TTL_S = 6 * 3600

# Separate, much shorter-lived registry for "no intraday trades today, but
# it does have daily history" -- i.e. genuinely listed, just illiquid right
# now (common across NSE's SME-platform tail). This is NOT the same thing
# as delisted: conflating the two into one 6h registry meant thin stocks
# got retried, and re-logged yfinance's own failure noise, on every single
# scan of the day. A short TTL here stops that within-day repeat hammering
# while still giving the stock a few more chances later in the session in
# case it starts trading.
_quiet_lock = threading.Lock()
_quiet_symbols: dict[str, float] = {}
_QUIET_TTL_S = 45 * 60


def is_known_dead(symbol: str) -> bool:
    with _dead_lock:
        ts = _dead_symbols.get(symbol)
    if ts is None:
        return False
    if (time.time() - ts) > _DEAD_TTL_S:
        with _dead_lock:
            _dead_symbols.pop(symbol, None)
        return False
    return True


def mark_dead_symbol(symbol: str) -> None:
    with _dead_lock:
        _dead_symbols[symbol] = time.time()


def is_quiet_today(symbol: str) -> bool:
    """True if this symbol had no intraday activity on a recent attempt
    (within the last ~45 min), even though it isn't considered dead."""
    with _quiet_lock:
        ts = _quiet_symbols.get(symbol)
    if ts is None:
        return False
    if (time.time() - ts) > _QUIET_TTL_S:
        with _quiet_lock:
            _quiet_symbols.pop(symbol, None)
        return False
    return True


def mark_quiet_today(symbol: str) -> None:
    with _quiet_lock:
        _quiet_symbols[symbol] = time.time()


# --------------------------------------------------------------------- #
# Retry wrapper -- the single choke point every yfinance / network call
# in this app goes through. Never raises; returns None on final failure.
# --------------------------------------------------------------------- #
def bulletproof_fetch(fn: Callable, *args, retries: int = 3, base_delay: float = 0.6,
                       max_delay: float = 6.0, **kwargs) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is the safety net
            last_exc = e
            if attempt == retries:
                break
            msg = str(e).lower()
            rate_limited = any(k in msg for k in ("429", "too many requests", "rate limit", "throttle"))
            sleep_s = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.4)
            if rate_limited:
                sleep_s = min(max_delay * 2, sleep_s * 2)
            time.sleep(sleep_s)
    return None


# --------------------------------------------------------------------- #
# Universe loading: NSE tickers + BSE codes -> yfinance symbols
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_nse_universe() -> list[dict]:
    try:
        df = pd.read_csv(_NSE_CSV, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "symbol", df.columns[1]: "name"})
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["name"] = df["name"].astype(str).str.strip()
        df = df[(df["symbol"] != "") & (df["symbol"].str.lower() != "nan")]
        df = df.drop_duplicates(subset="symbol")
        return [
            {"symbol": r.symbol, "name": r.name, "exchange": "NSE", "yf_symbol": f"{r.symbol}.NS"}
            for r in df.itertuples(index=False)
        ]
    except Exception as e:
        st.sidebar.error(f"Couldn't load nse_tickers.csv: {e}")
        return []


@st.cache_data(show_spinner=False)
def _load_bse_universe() -> list[dict]:
    try:
        df = pd.read_csv(_BSE_CSV, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "code", df.columns[1]: "name"})
        df["code"] = df["code"].astype(str).str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df = df[(df["code"] != "") & (df["code"].str.lower() != "nan")]
        df = df.drop_duplicates(subset="code")
        return [
            {"symbol": r.code, "name": r.name, "exchange": "BSE", "yf_symbol": f"{r.code}.BO"}
            for r in df.itertuples(index=False)
        ]
    except Exception as e:
        st.sidebar.error(f"Couldn't load bse_codes.csv: {e}")
        return []


def load_universe() -> list[dict]:
    return _load_nse_universe() + _load_bse_universe()


# --------------------------------------------------------------------- #
# Sidebar: exchange picker
# --------------------------------------------------------------------- #
def render_exchange_selector(mode_key: str):
    st.sidebar.subheader("🏦 Exchange")
    scan_nse = st.sidebar.checkbox("NSE", value=True, key=sskey(mode_key, "scan_nse"))
    scan_bse = st.sidebar.checkbox("BSE", value=False, key=sskey(mode_key, "scan_bse"))
    if not scan_nse and not scan_bse:
        st.sidebar.warning("Select at least one exchange — defaulting to NSE.")
        scan_nse = True

    universe = load_universe()
    filtered = [r for r in universe if (r["exchange"] == "NSE" and scan_nse) or (r["exchange"] == "BSE" and scan_bse)]
    st.sidebar.caption(f"{len(filtered):,} stocks available in selected universe")
    return scan_nse, scan_bse, filtered


# --------------------------------------------------------------------- #
# Sidebar: scan-mode picker (Quick / Full / Slot-wise / Range / Custom)
# --------------------------------------------------------------------- #
def render_scan_mode_selector(mode_key: str, universe: list[dict]) -> list[dict]:
    st.sidebar.subheader("🎯 Scan Mode")
    if not universe:
        st.sidebar.error("No stocks in the selected universe.")
        return []

    universe_sorted = sorted(universe, key=lambda r: (r["exchange"], r["symbol"]))
    mode = st.sidebar.radio(
        "Which stocks to scan",
        ["Quick Sample", "Full Universe", "Slot-wise", "Range", "Custom List"],
        key=sskey(mode_key, "scan_mode"),
    )

    if mode == "Quick Sample":
        n = st.sidebar.slider("Sample size", 10, min(300, len(universe_sorted)), min(50, len(universe_sorted)), 10,
                               key=sskey(mode_key, "quick_n"))
        # Seeded by today's date so the sample is stable across reruns
        # (slider tweaks etc.) within a day, but rotates day to day.
        seed = int(datetime.now().strftime("%Y%m%d")) + n + len(universe_sorted)
        rng = random.Random(seed)
        stocks = rng.sample(universe_sorted, min(n, len(universe_sorted)))

    elif mode == "Full Universe":
        stocks = universe_sorted
        if len(stocks) > 300:
            st.sidebar.warning(
                f"⚠️ {len(stocks):,} stocks selected. On Streamlit Cloud's free tier this will almost "
                "certainly need multiple 'Resume' clicks — Slot-wise mode is usually faster to get "
                "through in one sitting."
            )

    elif mode == "Slot-wise":
        n_slots = st.sidebar.number_input("Number of slots", 2, 40, 10, 1, key=sskey(mode_key, "n_slots"))
        slot_no = st.sidebar.number_input("Slot to scan (1-based)", 1, int(n_slots), 1, 1, key=sskey(mode_key, "slot_no"))
        chunks = np.array_split(universe_sorted, int(n_slots))
        idx = int(slot_no) - 1
        stocks = list(chunks[idx]) if 0 <= idx < len(chunks) else []
        st.sidebar.caption(f"Slot {slot_no}/{n_slots} → {len(stocks)} stocks")

    elif mode == "Range":
        max_idx = len(universe_sorted)
        start = st.sidebar.number_input("Start index", 0, max_idx - 1, 0, 1, key=sskey(mode_key, "range_start"))
        end = st.sidebar.number_input("End index (exclusive)", int(start) + 1, max_idx, min(int(start) + 100, max_idx), 1,
                                       key=sskey(mode_key, "range_end"))
        stocks = universe_sorted[int(start):int(end)]

    else:  # Custom List
        raw = st.sidebar.text_area(
            "Paste symbols (comma or newline separated)", key=sskey(mode_key, "custom_list"),
            placeholder="RELIANCE, TCS, INFY\n500325",
        )
        wanted = {tok.strip().upper() for tok in raw.replace("\n", ",").split(",") if tok.strip()}
        by_symbol = {r["symbol"].upper(): r for r in universe_sorted}
        stocks = [by_symbol[s] for s in wanted if s in by_symbol]
        missing = wanted - set(by_symbol)
        if missing:
            st.sidebar.warning(f"Not found in selected exchanges: {', '.join(sorted(missing)[:15])}"
                                + (" ..." if len(missing) > 15 else ""))

    if not stocks:
        st.sidebar.error("No stocks selected for this scan.")
    return stocks


# --------------------------------------------------------------------- #
# Sidebar: rate-limit / reliability controls
# --------------------------------------------------------------------- #
def render_rate_limit_controls(mode_key: str) -> dict:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚦 Rate Limit / Reliability")
    max_workers = st.sidebar.slider("Parallel requests", 1, 8, 3, 1, key=sskey(mode_key, "max_workers"),
                                     help="Kept low by default — Yahoo Finance throttles aggressive "
                                          "concurrent scraping, especially from shared cloud IPs.")
    delay = st.sidebar.slider("Delay between batches (s)", 0.0, 3.0, 0.4, 0.1, key=sskey(mode_key, "delay"))
    retries = st.sidebar.slider("Retries per symbol", 1, 5, 3, 1, key=sskey(mode_key, "retries"))
    time_budget = st.sidebar.slider("Time per leg (s)", 20, 120, 60, 10, key=sskey(mode_key, "time_budget"),
                                     help="The scan runs in short 'legs' rather than one long call — after "
                                          "each leg it saves progress and, if Auto-continue is on below, "
                                          "immediately keeps going on its own.")
    auto_continue = st.sidebar.checkbox("Auto-continue until done", value=True, key=sskey(mode_key, "auto_continue"),
                                         help="Off: you'll need to click Resume yourself after each leg. "
                                              "On: the scan keeps going leg after leg with no clicking, "
                                              "until it's fully done — uncheck mid-scan to pause it.")
    return {"max_workers": max_workers, "delay": delay, "retries": retries, "time_budget": time_budget,
            "auto_continue": auto_continue}


# --------------------------------------------------------------------- #
# Scan trigger + checkpointed runner
# --------------------------------------------------------------------- #
def _scan_signature(stocks_to_scan: list[dict]) -> str:
    return f"{len(stocks_to_scan)}:{hash(tuple(r['yf_symbol'] for r in stocks_to_scan))}"


def render_scan_trigger(mode_key: str, stocks_to_scan: list[dict], label: str):
    st.markdown("---")
    checkpoint = get_state(mode_key, "checkpoint")
    sig = _scan_signature(stocks_to_scan)

    if checkpoint and checkpoint.get("sig") != sig:
        # The scan universe/settings changed since the last paused/
        # incomplete run -- that checkpoint no longer applies to anything
        # resumable. Discard it instead of leaving it stranded in session
        # state forever (and clear any pending auto-continue flag with it,
        # so a stale flag can never silently fire against unrelated state).
        set_state(mode_key, "checkpoint", None)
        set_state(mode_key, "auto_pending", False)
        checkpoint = None

    resume_available = bool(
        checkpoint and checkpoint.get("next_index", 0) < len(stocks_to_scan)
    )

    # The main SCAN button and the Resume button are two independent rows,
    # not a shared st.columns() row. Splitting them into columns caused two
    # separate visual bugs: (1) the SCAN button was permanently ~25%
    # narrower than every other element on the page to reserve space for
    # a Resume button that mostly wasn't there, and (2) even after fixing
    # that for the no-checkpoint case, the *last* leg of a scan still
    # renders with resume_available=True (checkpoint from the previous leg
    # hasn't been cleared yet at render time), so the column split -- and
    # its leftover empty gap -- reappeared for exactly one frame right as
    # the scan finished. Two always-full-width, independent rows sidestep
    # both issues entirely.
    do_scan = st.button(label, type="primary", width="stretch",
                         disabled=(len(stocks_to_scan) == 0), key=sskey(mode_key, "scan_btn"))

    # Always create the placeholder (even when unused) so run_scan() can
    # unconditionally erase it later in this same pass if the scan
    # finishes inside this very call.
    resume_placeholder = st.empty()
    resume_scan = False
    if resume_available:
        resume_scan = resume_placeholder.button(
            f"▶ Resume ({checkpoint['next_index']}/{len(stocks_to_scan)})",
            width="stretch", key=sskey(mode_key, "resume_btn"),
        )

    # Auto-continue: a previous leg that paused mid-scan sets this flag and
    # triggers an immediate rerun. On that rerun nothing was clicked, so we
    # treat it as an implicit Resume rather than requiring the user to.
    if resume_available and get_state(mode_key, "auto_pending", False):
        set_state(mode_key, "auto_pending", False)
        resume_scan = True

    return do_scan, resume_scan, checkpoint, sig, resume_placeholder


def run_scan(mode_key: str, stocks_to_scan: list[dict], fetch_and_analyze: Callable[[dict], tuple],
             rate_cfg: dict, resume_scan: bool, checkpoint: Optional[dict],
             resume_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None) -> None:
    """Runs `fetch_and_analyze(rec) -> (status, analysis)` over
    `stocks_to_scan` in small parallel batches, checkpointing progress into
    per-session state after every batch so a time-budget stop, a Streamlit
    Cloud restart, or a page refresh never loses completed work."""
    if not stocks_to_scan:
        st.warning("No stocks to scan.")
        return

    sig = _scan_signature(stocks_to_scan)
    start_index = 0
    results: list = []
    filtered_count = 0
    failed_count = 0

    if resume_scan and checkpoint and checkpoint.get("sig") == sig:
        start_index = int(checkpoint.get("next_index", 0))
        results = list(checkpoint.get("results", []))
        filtered_count = int(checkpoint.get("filtered_count", 0))
        failed_count = int(checkpoint.get("failed_count", 0))
    else:
        set_state(mode_key, "checkpoint", None)

    total = len(stocks_to_scan)
    remaining = stocks_to_scan[start_index:]
    max_workers = max(1, min(8, int(rate_cfg.get("max_workers", 3))))
    delay = float(rate_cfg.get("delay", 0.4))
    time_budget = float(rate_cfg.get("time_budget", 180))

    progress = st.progress(min(1.0, start_index / total) if total else 0.0)
    status = st.empty()
    pos = start_index
    scan_start = time.time()
    stopped_early = False

    try:
        for batch_start in range(0, len(remaining), max_workers):
            if (time.time() - scan_start) > time_budget:
                stopped_early = True
                break
            batch = remaining[batch_start: batch_start + max_workers]
            try:
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    batch_results = list(executor.map(fetch_and_analyze, batch))
            except Exception:
                batch_results = [("failed", None) for _ in batch]

            for status_res, analysis in batch_results:
                if status_res == "ok" and analysis is not None:
                    results.append(analysis)
                elif status_res == "filtered":
                    filtered_count += 1
                else:
                    failed_count += 1

            pos += len(batch)
            processed = pos - start_index + start_index
            progress.progress(min(1.0, processed / total) if total else 1.0)
            status.caption(
                f"Processed {processed}/{total} · ✅ {len(results)} matches · "
                f"⏭ {filtered_count} filtered · ⚠️ {failed_count} failed"
            )
            set_state(mode_key, "checkpoint", {
                "sig": sig, "next_index": pos, "results": results,
                "filtered_count": filtered_count, "failed_count": failed_count,
            })
            if delay > 0:
                time.sleep(delay)
    except Exception as e:  # last-resort guard: never let a scan crash the app
        st.error(f"Scan hit an unexpected error and stopped safely at {pos}/{total}: {e}")
        stopped_early = True
        hard_error = True
    else:
        hard_error = False

    set_state(mode_key, "results", results)

    if hard_error:
        # Never auto-rerun on a genuine error -- that would just loop the
        # same failure. Leave the manual Resume button (still accurate --
        # checkpoint reflects real progress up to the error) for the user.
        pass
    elif stopped_early or pos < total:
        if rate_cfg.get("auto_continue", True):
            status.caption(
                f"Processed {pos}/{total} · ✅ {len(results)} matches · "
                f"⏭ {filtered_count} filtered · ⚠️ {failed_count} failed · continuing automatically…"
            )
            set_state(mode_key, "auto_pending", True)
            st.rerun()
        else:
            st.warning(f"⏸ Paused after {pos}/{total} stocks (time budget reached). "
                       "Click **Resume** above to continue, or turn on Auto-continue in the sidebar.")
    else:
        st.success(f"✅ Scan complete — {len(results)} match(es) out of {total} scanned.")
        set_state(mode_key, "checkpoint", None)
        set_state(mode_key, "auto_pending", False)
        if resume_placeholder is not None:
            # The Resume button above was drawn at the START of this leg
            # using the checkpoint as it stood THEN (e.g. "6/7"). The scan
            # has since finished inside this very call, so that button is
            # now stale and misleading -- erase it rather than leave a
            # "Resume" button sitting next to a "Scan complete" message.
            resume_placeholder.empty()


# --------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------- #
def download_buttons(mode_key: str, df_all: pd.DataFrame, df_filtered: pd.DataFrame, filename_prefix: str) -> None:
    if df_all is None or df_all.empty:
        return
    try:
        csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download results (CSV)",
            data=csv_bytes,
            file_name=f"{filename_prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            key=sskey(mode_key, "download_btn"),
            width="stretch",
        )
    except Exception as e:
        st.caption(f"(Download unavailable: {e})")


def footer(html_text: str) -> None:
    st.markdown("---")
    st.markdown(f"<small>{html_text}</small>", unsafe_allow_html=True)
