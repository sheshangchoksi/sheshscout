"""
scanner_common.py — Everything all four scoring modes (Intraday Long/Short,
Swing Long/Short) share: universe loading (NSE tickers + BSE codes ->
yfinance symbols), the exchange / scan-mode / rate-limit sidebar controls,
a thread-safe process-wide cache + dead-symbol registry, a retry wrapper
around every external call, checkpointed scanning, trade-levels math +
trade-card / conditions-checklist / price-volume-RSI-chart rendering, and
small UI helpers (download buttons, footer).

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
import plotly.graph_objects as go
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
MODE_SWING_LONG = "swing_long"
MODE_SWING_SHORT = "swing_short"

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


# NSE/BSE cash session is 09:15-15:30 IST = 375 minutes. Comparing a
# partial day's cumulative volume against a 5-day *full-day* average
# volume systematically understates the ratio in the morning and
# inflates it by afternoon -- a real volume burst at 10 AM can get
# filtered out purely because it's early. Scaling the average down by
# how much of the session has elapsed fixes that without any extra
# Yahoo calls (session length is a constant, elapsed bars are already
# in the snapshot we fetch anyway).
SESSION_MINUTES = 375


def session_elapsed_fraction(n_bars: int) -> float:
    """Fraction of the trading session elapsed, given the number of 1-min
    bars fetched so far today. Clamped away from 0 so a fresh-open snapshot
    (a handful of bars) doesn't blow up the ratio via division by ~0, and
    capped at 1.0 for anything at/after the close."""
    return min(1.0, max(0.05, n_bars / SESSION_MINUTES))


# Broad-market index to check for directional agreement, per exchange.
# Sector-index confirmation (checking e.g. CNXPHARMA/CNXAUTO specifically)
# would need a symbol->sector mapping this codebase doesn't have (the
# universe CSVs are just symbol+name), so this stays at the broad-market
# level for now rather than pretending to a sector check it can't do.
INDEX_FOR_EXCHANGE = {"NSE": "^NSEI", "BSE": "^BSESN"}

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
# Trade-levels math + trade-card / conditions-checklist rendering, shared
# by every scoring mode (Intraday Long/Short, Swing Long/Short) so this
# exists exactly once instead of drifting copies per mode.
# --------------------------------------------------------------------- #
def trade_levels(price: float, direction: str, support_level, resistance_level,
                  stop_loss_pct: float, target_pct: float) -> dict:
    """Stop/target/risk/reward for one result. direction "long": stop is
    BELOW entry (real support when it's actually below entry, else the
    fixed-% fallback), target is ABOVE entry (real resistance, same rule).
    direction "short" is the mirror image.

    Using a fixed % for both regardless of the stock makes the R:R ratio
    ALWAYS target_pct/stop_loss_pct -- literally identical for every
    result, since price cancels out of that division. Real support/
    resistance (already fetched for scoring) gives a per-stock stop/target
    instead, each only used when it's on the correct side of entry --
    otherwise this falls back to the fixed-% behavior."""
    if direction == "long":
        used_real_stop = support_level is not None and support_level < price
        used_real_target = resistance_level is not None and resistance_level > price
        stop_loss = support_level if used_real_stop else price * (1 - stop_loss_pct / 100)
        target = resistance_level if used_real_target else price * (1 + target_pct / 100)
    else:
        used_real_stop = resistance_level is not None and resistance_level > price
        used_real_target = support_level is not None and support_level < price
        stop_loss = resistance_level if used_real_stop else price * (1 + stop_loss_pct / 100)
        target = support_level if used_real_target else price * (1 - target_pct / 100)
    risk = abs(price - stop_loss)
    reward = abs(target - price)
    return {
        "stop_loss": stop_loss, "target": target,
        "used_real_stop": used_real_stop, "used_real_target": used_real_target,
        "risk": risk, "reward": reward,
        "risk_pct": (risk / price) * 100 if price else 0,
        "reward_pct": (reward / price) * 100 if price else 0,
        "risk_reward": reward / risk if risk > 0 else 0,
    }


def render_trade_card(result: dict, trading: dict, direction: str, *, same_session_exit: bool = True,
                       distant_target_threshold_pct: float = 3.0, atr_key: str = "atr") -> dict:
    """Entry/Stop/Target/R:R, risk-reward in ₹ and %, an ATR sanity check
    on the stop, and a position-size suggestion for the sidebar's risk
    budget -- identical across every mode, so it exists exactly once.
    Returns the underlying trade_levels() dict (the summary table's R:R
    column calls trade_levels() directly instead, to avoid re-rendering).

    same_session_exit controls the "target is unrealistically far away"
    caveat: True (intraday) assumes the position closes before the session
    ends, so a real S/R target more than distant_target_threshold_pct away
    is flagged as unlikely to be hit today. False (swing) skips that
    caveat entirely -- a real target several % away is the point of a
    multi-day/week hold, not a red flag.
    """
    lv = trade_levels(result["price"], direction, result.get("support_level"), result.get("resistance_level"),
                       trading["stop_loss_pct"], trading["target_pct"])
    stop_level_name = "support" if direction == "long" else "resistance"
    target_level_name = "resistance" if direction == "long" else "support"

    t1, t2, t3, t4 = st.columns(4)
    t1.info(f"💡 Entry: ₹{result['price']:.2f}")
    t2.error(f"🛑 Stop: ₹{lv['stop_loss']:.2f}" + (f" (real {stop_level_name})" if lv["used_real_stop"] else ""))
    t3.success(f"🎯 Target: ₹{lv['target']:.2f}" + (f" (real {target_level_name})" if lv["used_real_target"] else ""))
    t4.metric("R:R Ratio", f"1:{lv['risk_reward']:.2f}")

    risk_per_trade = trading.get("risk_per_trade", 1000)
    suggested_qty = int(risk_per_trade // lv["risk"]) if lv["risk"] > 0 else 0
    position_value = suggested_qty * result["price"]
    atr_value = result.get(atr_key, 0) or 0
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Risk / share", f"₹{lv['risk']:.2f}", f"{lv['risk_pct']:.2f}% of entry", delta_color="off")
    u2.metric("Reward / share", f"₹{lv['reward']:.2f}", f"{lv['reward_pct']:.2f}% of entry", delta_color="off")
    if atr_value <= 0:
        u3.metric("ATR", "—", "not enough price history yet", delta_color="off")
    elif lv["risk"] < atr_value * 0.5:
        u3.metric("ATR", f"₹{atr_value:.2f}", "stop tighter than typical noise ⚠️", delta_color="off")
    elif lv["risk"] > atr_value * 3:
        u3.metric("ATR", f"₹{atr_value:.2f}", "stop much wider than ATR", delta_color="off")
    else:
        u3.metric("ATR", f"₹{atr_value:.2f}", "stop is a reasonable multiple of ATR", delta_color="off")
    u4.metric(f"Qty for ₹{risk_per_trade:,.0f} risk", f"{suggested_qty:,} sh", f"≈ ₹{position_value:,.0f} position", delta_color="off")

    if same_session_exit and lv["used_real_target"] and lv["reward_pct"] > distant_target_threshold_pct:
        st.caption(f"🕒 Target is {lv['reward_pct']:.1f}% away — that's a large move for a single session; "
                   f"the R:R above assumes it gets hit today, which may not happen. Consider a nearer "
                   f"partial target or trailing the stop instead of holding for the full move.")

    return lv


def render_conditions_checklist(result: dict) -> None:
    """The conditions-met checklist + a separate warnings box, identical
    across every mode. Reads result["conditions_list"] / ["warnings_list"]
    -- kept alongside the single joined "conditions" string each score_*()
    also returns (that string is what the summary table's one-column
    display uses)."""
    st.markdown("**✅ Conditions met**")
    st.markdown("\n".join(f"- {c}" for c in result.get("conditions_list", [])) or "_none_")
    if result.get("warnings_list"):
        st.warning("⚠️ " + "; ".join(result["warnings_list"]))


# --------------------------------------------------------------------- #
# Strictest Screening + "Operated" activity flag -- both shared across
# every mode. Deliberately kept as two INDEPENDENT features: strict mode
# only tightens score/condition thresholds; the operator flag is purely
# informational and never gates, sorts, or filters anything, however
# strict the screening is set to.
# --------------------------------------------------------------------- #
def render_strict_mode_toggle(mode_key: str) -> bool:
    """A single checkbox that raises the floor on a curated set of
    well-known threshold keys at once, instead of a person having to hunt
    through every slider individually to ask for "only the best setups."
    Returns whether it's checked; the caller applies apply_strict_screening()
    to its own already-assembled params dict."""
    st.sidebar.markdown("---")
    return st.sidebar.checkbox(
        "🏆 Strictest Screening (cream only)", value=False, key=sskey(mode_key, "strict_mode"),
        help="Raises a curated set of thresholds at once instead of one slider at a time: "
             "near-full condition confluence, a much higher score bar, real S/R proximity "
             "within 1.5%, a bigger volume/trend/momentum requirement, tighter RSI, and a "
             "higher market-cap/liquidity floor -- small, thinly-traded names are exactly "
             "where the 'Operated' flag below concentrates, per the research this feature is "
             "based on. Every slider above still works underneath this; strict mode only "
             "raises the floor, it never loosens a stricter setting you've already made "
             "yourself. Does NOT touch the Operated flag -- that stays purely informational "
             "regardless of this setting.",
    )


def apply_strict_screening(params: dict, direction: str) -> dict:
    """Tightens a fixed set of well-known threshold keys -- only the ones
    actually present in the given params dict, so this one function works
    unmodified across Intraday and Swing, Long and Short, without needing
    to know which mode called it beyond the long/short RSI direction.
    Returns a NEW dict; never mutates the one passed in, since the
    original is what's still shown in the sidebar as what the person
    actually configured with their own sliders.

    Every floor below is a MAX/MIN against the person's own setting (via
    the tighten_min/tighten_max helpers), so a slider already stricter
    than the floor is left alone -- this can only raise the bar, never
    quietly loosen one the person set themselves.
    """
    p = dict(params)

    def tighten_min(key, floor):
        if key in p:
            p[key] = max(p[key], floor)

    def tighten_max(key, ceiling):
        if key in p:
            p[key] = min(p[key], ceiling)

    # Overall bar: near-full confluence of conditions, not just "enough".
    tighten_min("min_score", 75)
    tighten_min("strong_score", 80)
    tighten_min("min_conditions", 9)

    # Liquidity/size floor -- every source behind the Operated flag agrees
    # small, illiquid names are where manipulation concentrates; "cream
    # only" means leaning away from that end of the universe by
    # construction, not just scoring around it after the fact.
    tighten_min("min_market_cap_cr", 300)
    tighten_min("min_price", 50)
    tighten_min("min_volume", 300000)

    # Confirmation quality: real S/R proximity, higher-timeframe trend
    # agreement, and volume all need to clear a markedly higher bar than
    # the default "acceptable" one.
    tighten_max("dist_from_support_threshold", 1.5)
    tighten_max("dist_from_resistance_threshold", 1.5)
    tighten_max("dist_from_low_threshold", 2.0)
    tighten_max("dist_from_high_threshold", 2.0)
    tighten_min("volume_ratio_threshold", 1.8)
    tighten_min("trend_threshold", 6.0)
    tighten_min("momentum_threshold", 1.5)
    tighten_min("hourly_trend_threshold", 0.3)
    tighten_min("weekly_trend_threshold", 1.5)
    if "atr_threshold" in p and p["atr_threshold"] > 0:
        p["atr_threshold"] = max(p["atr_threshold"], p["atr_threshold"] * 1.3)

    # RSI is direction-aware: "long" wants a LOWER (more oversold) ceiling,
    # "short" wants a HIGHER (more overbought) floor.
    if "rsi_threshold" in p:
        p["rsi_threshold"] = min(p["rsi_threshold"], 28) if direction == "long" else max(p["rsi_threshold"], 72)

    return p


def detect_sideways_then_spike(closes, lookback_window: int) -> bool:
    """One of the most specifically-named operator patterns across the
    sources behind assess_operator_risk(): a stock trades in a tight range
    for weeks/months, then suddenly wakes up. Approximated here as "recent
    daily-return volatility is much higher than an unusually QUIET longer
    baseline period" -- a real breakout from genuine multi-week
    consolidation can look similar, which is exactly why this is one
    input into a multi-signal flag below, not a standalone verdict.
    Swing-only (needs real daily history; the intraday screener's daily
    reference window is too short -- a handful of days -- to judge "quiet
    for weeks"). Never raises; returns False on insufficient data."""
    try:
        closes = np.asarray(closes, dtype=float)
        baseline_len = lookback_window * 3
        if len(closes) < baseline_len + 5:
            return False
        daily_rets_pct = np.diff(closes) / closes[:-1] * 100
        recent_std = float(np.std(daily_rets_pct[-5:]))
        baseline_std = float(np.std(daily_rets_pct[-baseline_len:-5]))
        return bool(baseline_std > 0 and baseline_std < 1.5 and recent_std > baseline_std * 3)
    except Exception:
        return False


def assess_operator_risk(price, market_cap_cr, volume_ratio, change_pct,
                          change_pct_extreme: float, change_pct_moderate: float,
                          sideways_then_spike: bool = False) -> dict:
    """Heuristic-only 'unusual activity' flag, purely informational: it
    never filters, sorts, or scores a result, it only labels one for a
    person's own judgement (per the person's own framing: operator-driven
    stocks aren't automatically bad, they're just worth knowing about).

    Draws on patterns consistently named across public sources on Indian
    operator-driven stocks: abnormal volume spikes, outsized price moves
    without proportional justification, small/illiquid market cap (every
    source agrees large caps have too much depth for a single operator to
    move the way a small-cap can), and a long quiet period followed by a
    sudden spike. NOT a determination of manipulation or wrongdoing --
    many legitimate rallies (real news, a genuine re-rating) look
    identical from price/volume data alone, which is exactly why this
    requires two or more corroborating signals rather than firing on any
    one metric in isolation, and why small-cap status alone never
    triggers it (it only lowers how much OTHER evidence is required).

    change_pct_extreme/change_pct_moderate let each mode calibrate what
    counts as an outsized move on ITS OWN timeframe -- an intraday
    same-session move and a multi-day swing move aren't the same scale.

    (Sources: strike.money/stock-market/operators; Business Standard's
    "How to Identify an operator-driven stock"; mentoradityajain.com's
    "sideways then sudden spike" pattern.)
    """
    try:
        signals = 0
        reasons = []

        if volume_ratio is not None:
            if volume_ratio > 5:
                signals += 2
                reasons.append(f"volume {volume_ratio:.1f}x average")
            elif volume_ratio > 3:
                signals += 1
                reasons.append(f"volume {volume_ratio:.1f}x average")

        abs_change = abs(change_pct) if change_pct is not None else 0
        if abs_change > change_pct_extreme:
            signals += 2
            reasons.append(f"{abs_change:.1f}% move")
        elif abs_change > change_pct_moderate:
            signals += 1
            reasons.append(f"{abs_change:.1f}% move")

        if price is not None and price < 30 and volume_ratio is not None and volume_ratio > 2:
            signals += 1
            reasons.append("low price + volume spike")

        if sideways_then_spike:
            signals += 2
            reasons.append("quiet for weeks, then a sudden spike")

        # Market cap sets the BAR, not the score -- the same volume/price
        # anomaly is common and often perfectly legitimate in a large,
        # deep stock, precisely because a single operator can't move one
        # the way they can a small-cap. So the same behavior in a big
        # stock needs much more corroborating evidence before it's flagged.
        if market_cap_cr is None:
            threshold = 4
        elif market_cap_cr < 100:
            threshold = 2
        elif market_cap_cr < 500:
            threshold = 3
        else:
            threshold = 5

        return {"flagged": signals >= threshold, "signals": signals, "threshold": threshold, "reasons": reasons}
    except Exception:
        return {"flagged": False, "signals": 0, "threshold": 99, "reasons": []}


def render_operator_flag_notice(result: dict) -> None:
    """The detail-view explanation for a flagged result -- kept separate
    from the summary table's compact "Operated" label so the table stays
    scannable while the detail view gets the actual reasoning."""
    if not result.get("operated_flag"):
        return
    reasons = ", ".join(result.get("operated_reasons", [])) or "multiple signals"
    st.warning(
        f"🚩 **Flagged: possible unusual activity** ({reasons}). This is a heuristic based on "
        f"volume/price/market-cap patterns commonly associated with operator-driven stocks -- "
        f"**not proof of manipulation**. Many legitimate rallies look similar, and operator "
        f"activity isn't automatically something to avoid; it's just worth knowing about before "
        f"sizing the trade."
    )


def render_price_volume_rsi_charts(result: dict, chart_data: Optional[pd.DataFrame],
                                    chart_timeframe: str, chart_height: int,
                                    price_color: str = "#28a745", rsi_color: str = "#007bff") -> None:
    """The Price / Volume / RSI three-chart row shown in every mode's
    detail view -- identical across Intraday and Swing (Long/Short), so it
    exists exactly once. `chart_data` is whatever a mode's chart-history
    fetch returned (a plain OHLCV DataFrame); the RSI here is a lightweight
    from-scratch calc for the chart display only (NOT indicators.rsi(),
    which is Wilder-smoothed and used for actual scoring) -- see the
    branching below for why "no losses" and "no gains" need separate
    RSI-100 / RSI-0 cases rather than collapsing both into one "rs=0"
    branch.

    price_color/rsi_color default to the Long screeners' green/blue; the
    Short screeners pass red/green instead -- a deliberate visual cue
    ("this is a short") worth keeping distinct rather than flattening
    every mode to identical colors for the sake of sharing this function.
    """
    if chart_data is None or chart_data.empty:
        st.warning(f"No chart data available for {result['symbol']}")
        return

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Price",
                                   line=dict(color=price_color, width=2)))
        fig1.add_hline(y=result["open"], line_dash="dash", line_color="gray", line_width=1, annotation_text="Open")
        fig1.update_layout(title=f"Price Chart ({chart_timeframe})", xaxis_title="Time", yaxis_title="Price (₹)",
                            height=chart_height, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig1, width="stretch")
    with cc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color="#17a2b8"))
        fig2.update_layout(title=f"Volume ({chart_timeframe})", xaxis_title="Time", yaxis_title="Volume",
                            height=chart_height, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig2, width="stretch")
    with cc3:
        closes = chart_data["Close"].values
        rsi_vals, rsi_idx = [], []
        for j in range(14, len(closes)):
            window = closes[max(0, j - 14):j]
            if len(window) > 1:
                diffs = window[1:] - window[:-1]
                gains = diffs[diffs > 0].sum() / len(window)
                losses = -diffs[diffs < 0].sum() / len(window)
                # rs=0 legitimately means "no losses in the window" only
                # when there WERE gains (-> RSI should read 100, maximal
                # overbought); it's a different, opposite case when there
                # were also no gains (flat window -> RSI 50). Collapsing
                # both into one "rs=0" branch misreports a strongly
                # up-trending window (all gains, zero losses) as neutral
                # RSI 50 instead of 100 -- and symmetrically, an all-losses
                # window as 50 instead of 0.
                if losses == 0 and gains == 0:
                    rsi_val = 50.0
                elif losses == 0:
                    rsi_val = 100.0
                elif gains == 0:
                    rsi_val = 0.0
                else:
                    rsi_val = 100 - (100 / (1 + gains / losses))
                rsi_vals.append(rsi_val)
                rsi_idx.append(chart_data.index[j])
        fig3 = go.Figure()
        if rsi_vals:
            fig3.add_trace(go.Scatter(x=rsi_idx, y=rsi_vals, mode="lines", name="RSI", line=dict(color=rsi_color, width=2)))
            fig3.add_hline(y=70, line_dash="dash", line_color="red", line_width=1)
            fig3.add_hline(y=30, line_dash="dash", line_color="green", line_width=1)
        fig3.update_layout(title=f"RSI ({chart_timeframe})", xaxis_title="Time", yaxis_title="RSI",
                            height=chart_height, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig3, width="stretch")


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
