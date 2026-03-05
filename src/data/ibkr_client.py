"""
src/data/ibkr_client.py
=======================
Reusable IBKR client module for the options_algo scanner.

Provides a single shared connection pattern — callers should:
    1. Call connect_ibkr() ONCE at the start of a scan run.
    2. Pass the returned IB object to all fetch_* functions across all tickers.
    3. Call disconnect_ibkr() ONCE when the run is complete.

This avoids the overhead of connecting/disconnecting per-ticker (which would
hit IBKR's rate limits and slow down a 50–100 ticker scan considerably).

Example (nightly_scan.py integration):
    ib = connect_ibkr()
    for ticker in universe:
        chain = fetch_options_chain_ibkr(ib, ticker)
        ...
    disconnect_ibkr(ib)

Notes:
- clientId=11 (distinct from SPX algo which uses 10; avoid client ID conflicts)
- snapshot=True for all market data requests — no persistent subscriptions
- Paper trading account: market data may be delayed/unavailable outside hours
- Max 50 contracts per ticker to stay within IBKR concurrent data line limits
"""
from __future__ import annotations

import logging
import socket
import time
from datetime import date, timedelta
from typing import Optional

import math

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_int(val) -> int:
    """Convert a possibly-NaN or None float to int safely."""
    if val is None:
        return 0
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


def _safe_float(val) -> float:
    """Convert a possibly-NaN, None, or IBKR sentinel (-1) value to float safely."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        if f == -1.0:   # IBKR sentinel: "no data available"
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0

# Lazy imports — only pulled in when actually used so the module is safe to
# import even if ib_insync is not installed in a given environment.
try:
    from ib_insync import IB, Stock, Index, Option, util

    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False
    logger.warning("ib_insync not installed — IBKR data source unavailable")


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ─── Connection Management ────────────────────────────────────────────────────

def connect_ibkr(
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout: int | None = None,
) -> "IB | None":
    """
    Establish an IB Gateway connection and return the IB object.

    Args can be overridden (e.g. unique client IDs for ad-hoc tests). Any
    parameter left as None will fall back to config.settings defaults.

    Returns:
        Connected IB instance, or None if the gateway is unreachable / any
        error occurs.  Callers must check for None before using the result.
    """
    if not _IB_AVAILABLE:
        logger.warning("ib_insync not available — skipping IBKR connection")
        return None

    # Honour IBKR_ENABLED flag — allows callers to disable IBKR without
    # uninstalling ib_insync (e.g. IBKR_ENABLED=false in .env).
    try:
        from config.settings import IBKR_ENABLED as _IBKR_ENABLED
        if not _IBKR_ENABLED:
            logger.debug("IBKR_ENABLED=false — skipping connection")
            return None
    except Exception:
        pass  # settings import failed; attempt connection anyway

    # Pull defaults from settings only when needed (avoids circular import)
    if host is None or port is None or client_id is None or timeout is None:
        try:
            from config.settings import (
                IBKR_HOST,
                IBKR_PORT,
                IBKR_CLIENT_ID_OPTIONS,
                IBKR_TIMEOUT,
            )
            host = host or IBKR_HOST
            port = port or IBKR_PORT
            client_id = client_id or IBKR_CLIENT_ID_OPTIONS
            timeout = timeout or IBKR_TIMEOUT
        except Exception:
            host = host or "127.0.0.1"
            port = port or 4002
            client_id = client_id or 11
            timeout = timeout or 10
    else:
        # All overrides supplied; ensure they're usable
        host = host or "127.0.0.1"
        port = port or 4002
        client_id = client_id or 11
        timeout = timeout or 10

    # Fast-fail if IB Gateway port is not open
    if not _tcp_reachable(host, port):
        logger.warning(f"IB Gateway not reachable at {host}:{port} — skipping IBKR")
        return None

    try:
        util.patchAsyncio()
        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=True)
        logger.info(f"IBKR connected — {host}:{port} clientId={client_id}")
        return ib
    except Exception as exc:
        logger.warning(f"IBKR connect failed: {exc}")
        return None


def disconnect_ibkr(ib: "IB | None") -> None:
    """
    Cleanly disconnect from IB Gateway.

    Safe to call with None (no-op).
    """
    if ib is None:
        return
    try:
        ib.disconnect()
        logger.info("IBKR disconnected")
    except Exception as exc:
        logger.debug(f"IBKR disconnect error (non-fatal): {exc}")


# ─── Stock Snapshot ───────────────────────────────────────────────────────────

def fetch_stock_snapshot(ib: "IB", ticker: str) -> dict | None:
    """
    Fetch a real-time market snapshot for a stock.

    Args:
        ib:     Connected IB instance.
        ticker: Underlying stock ticker (e.g. "AAPL").

    Returns:
        Dict with keys: {last, bid, ask, volume, close}
        or None on any error / no data.
    """
    if ib is None:
        return None
    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(2)  # Delayed-Frozen data to avoid live session conflicts
        ticker_obj = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)

        # Wait up to 5 seconds for data to arrive
        def _has_price(val: float | None) -> bool:
            if val is None:
                return False
            try:
                f = float(val)
            except (TypeError, ValueError):
                return False
            return not math.isnan(f) and not math.isinf(f)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            ib.sleep(0.2)
            if any(_has_price(val) for val in (ticker_obj.last, ticker_obj.close, ticker_obj.bid, ticker_obj.ask)):
                break

        ib.cancelMktData(ticker_obj)

        last = _safe_float(ticker_obj.last or ticker_obj.close)
        bid = _safe_float(ticker_obj.bid)
        ask = _safe_float(ticker_obj.ask)
        volume = _safe_int(ticker_obj.volume)

        if last == 0.0:
            if bid > 0 and ask > 0:
                last = (bid + ask) / 2
            elif bid > 0:
                last = bid
            elif ask > 0:
                last = ask

        if last == 0.0 and bid == 0.0:
            logger.debug(f"{ticker}: IBKR stock snapshot returned no data")
            return None

        return {
            "last": last,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "close": _safe_float(ticker_obj.close if ticker_obj.close is not None else last),
        }
    except Exception as exc:
        logger.warning(f"{ticker}: IBKR stock snapshot error: {exc}")
        return None


# ─── Options Chain ────────────────────────────────────────────────────────────

def fetch_options_chain_ibkr(
    ib: "IB",
    ticker: str,
    dte_min: int = 7,
    dte_max: int = 60,
) -> pd.DataFrame:
    """
    Fetch an options chain from IBKR with real Greeks and live bid/ask.

    Returns a DataFrame with the same schema used throughout options_algo:
        ticker, contract_ticker, expiration, strike, type (call/put),
        bid, ask, mid, last, volume, open_interest,
        implied_volatility, delta, gamma, theta, vega,
        bid_ask_spread_pct, source

    Implementation strategy:
        1. reqSecDefOptParams — gets tradeable expirations and strikes.
        2. Filter expirations to dte_min..dte_max window.
        3. Fetch stock spot price (to filter strikes ±30% of ATM).
        4. Pick the 50 strikes closest to ATM (split evenly across expirations).
        5. reqMktData snapshot=True for each contract, collect Greeks.

    Rate-limit note:
        Max 50 contracts per ticker keeps us within IBKR's concurrent data
        line limits for paper accounts.  The IBKR connection should be
        established ONCE per scan run (see module docstring).

    Args:
        ib:      Connected IB instance.
        ticker:  Underlying stock ticker.
        dte_min: Minimum days to expiration (inclusive).
        dte_max: Maximum days to expiration (inclusive).

    Returns:
        DataFrame, possibly empty if IBKR has no data or an error occurs.
    """
    if ib is None:
        return pd.DataFrame()

    try:
        from config.settings import IBKR_MAX_CONTRACTS_PER_TICKER
        max_contracts = IBKR_MAX_CONTRACTS_PER_TICKER
    except Exception:
        max_contracts = 50

    try:
        # ── Step 1: qualify the underlying contract ───────────────────────────
        stock = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(stock)
        con_id = stock.conId
        if not con_id:
            logger.warning(f"{ticker}: could not qualify STK contract")
            return pd.DataFrame()

        # ── Step 2: get tradeable expirations + strikes ───────────────────────
        params = ib.reqSecDefOptParams(ticker, "", "STK", con_id)
        if not params:
            logger.warning(f"{ticker}: IBKR returned no option params")
            return pd.DataFrame()

        # Take the first (usually SMART or primary) exchange entry
        param = params[0]
        all_expirations: list[str] = sorted(param.expirations)
        all_strikes: list[float] = sorted(param.strikes)

        # ── Step 3: filter expirations by DTE ────────────────────────────────
        today = date.today()
        valid_exps = []
        for exp_str in all_expirations:
            try:
                exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
                dte = (exp_date - today).days
                if dte_min <= dte <= dte_max:
                    valid_exps.append((exp_str, exp_date, dte))
            except ValueError:
                continue

        if not valid_exps:
            logger.info(f"{ticker}: no IBKR expirations in {dte_min}–{dte_max} DTE window")
            return pd.DataFrame()

        # ── Step 4: get spot price to filter strikes ──────────────────────────
        snap = fetch_stock_snapshot(ib, ticker)
        spot = snap["last"] if snap else 0.0
        if spot <= 0:
            # Fall back: use mid-point of strike range as approximate spot
            spot = (all_strikes[0] + all_strikes[-1]) / 2
            logger.debug(f"{ticker}: using strike midpoint {spot:.2f} as spot proxy")

        # Filter strikes to ±30% of spot
        lo, hi = spot * 0.70, spot * 1.30
        near_strikes = [s for s in all_strikes if lo <= s <= hi]
        if not near_strikes:
            near_strikes = all_strikes  # Fallback: use all strikes

        # Sort by distance from spot, take closest N (split across expirations)
        near_strikes_sorted = sorted(near_strikes, key=lambda s: abs(s - spot))
        contracts_per_exp = max(1, max_contracts // (len(valid_exps) * 2))  # ÷2 for calls+puts
        chosen_strikes = near_strikes_sorted[:contracts_per_exp]

        # ── Step 5: build Option contracts to request ─────────────────────────
        option_contracts = []
        for exp_str, exp_date, dte in valid_exps:
            for strike in chosen_strikes:
                for right in ("C", "P"):
                    opt = Option(
                        symbol=ticker,
                        lastTradeDateOrContractMonth=exp_str,
                        strike=strike,
                        right=right,
                        exchange="SMART",
                        currency="USD",
                    )
                    option_contracts.append((opt, exp_str, exp_date, strike, right, dte))

        # Cap total to max_contracts
        option_contracts = option_contracts[:max_contracts]

        if not option_contracts:
            return pd.DataFrame()

        # ── Step 6: batch qualify + request market data ───────────────────────
        raw_contracts = [oc[0] for oc in option_contracts]

        # Qualify in bulk (suppresses warnings for non-tradeable contracts)
        try:
            ib.qualifyContracts(*raw_contracts)
        except Exception:
            pass  # Some contracts may not qualify; continue anyway

        # Request snapshots in batch
        # Note: do NOT pass generic ticks (e.g. "106") for option snapshots —
        # IBKR returns Error 321 "not applicable to generic ticks" in snapshot mode.
        tickers_list = []
        for opt_contract in raw_contracts:
            if opt_contract.conId:  # Only request qualified contracts
                t = ib.reqMktData(opt_contract, "", snapshot=True, regulatorySnapshot=False)
                tickers_list.append((t, opt_contract))

        # Wait up to 5 seconds for all snapshots
        ib.sleep(5.0)

        # Cancel all data subscriptions
        for t, opt_contract in tickers_list:
            try:
                ib.cancelMktData(opt_contract)
            except Exception:
                pass

        # ── Step 7: collect results ───────────────────────────────────────────
        rows: list[dict] = []
        # meta_by_contract removed — was dead code; metadata looked up inline below.

        for t, opt_contract in tickers_list:
            # Find matching metadata
            meta = None
            for oc in option_contracts:
                if (oc[0].lastTradeDateOrContractMonth == opt_contract.lastTradeDateOrContractMonth
                        and oc[0].strike == opt_contract.strike
                        and oc[0].right == opt_contract.right):
                    meta = oc
                    break
            if meta is None:
                continue

            _, exp_str, exp_date, strike, right, dte = meta

            bid = _safe_float(t.bid)
            ask = _safe_float(t.ask)
            last = _safe_float(t.last)

            # Skip contracts with no market data at all
            if bid == 0.0 and ask == 0.0 and last == 0.0:
                continue

            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999.0

            # Greeks come from modelGreeks (computed by IBKR) or undPrice tick
            greeks = t.modelGreeks
            iv = _safe_float(greeks.impliedVol if greeks else None)
            delta = _safe_float(greeks.delta if greeks else None)
            gamma = _safe_float(greeks.gamma if greeks else None)
            theta = _safe_float(greeks.theta if greeks else None)
            vega = _safe_float(greeks.vega if greeks else None)

            # Build contract ticker symbol (OCC format best-effort)
            exp_ymd = exp_str[:8] if len(exp_str) >= 8 else exp_str
            exp_display = (
                f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
                if len(exp_str) >= 8
                else exp_str
            )
            contract_ticker = opt_contract.localSymbol or (
                f"{ticker}{exp_ymd}{right}{int(strike * 1000):08d}"
            )

            rows.append({
                "ticker": ticker,
                "contract_ticker": contract_ticker,
                "expiration": exp_display,
                "strike": strike,
                "type": "call" if right == "C" else "put",
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": last,
                "volume": _safe_int(getattr(t, "volume", None)),
                "open_interest": _safe_int(getattr(t, "openInterest", None)),
                "implied_volatility": iv,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "bid_ask_spread_pct": round(spread_pct, 2),
                "source": "ibkr",
            })

        if not rows:
            logger.info(f"{ticker}: IBKR returned 0 contracts with data")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"{ticker}: {len(df)} contracts from IBKR")
        return df

    except Exception as exc:
        logger.warning(f"{ticker}: IBKR options chain error: {exc}")
        return pd.DataFrame()


# ─── VIX ─────────────────────────────────────────────────────────────────────

def fetch_vix_ibkr(ib: "IB") -> float | None:
    """
    Fetch the live VIX index level from IBKR.

    Contract: symbol='VIX', secType='IND', exchange='CBOE'

    Returns:
        Current VIX level as a float, or None if unavailable.
    """
    if ib is None:
        return None
    try:
        vix_contract = Index("VIX", "CBOE", "USD")
        ib.qualifyContracts(vix_contract)
        t = ib.reqMktData(vix_contract, "", snapshot=True, regulatorySnapshot=False)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            ib.sleep(0.2)
            val = t.last or t.close or t.bid or t.ask
            if val and val > 0:
                break

        ib.cancelMktData(vix_contract)

        val = t.last or t.close or t.bid or t.ask
        if val and val > 0:
            logger.info(f"IBKR VIX: {val:.2f}")
            return round(float(val), 2)

        logger.debug("IBKR VIX returned no data (outside market hours?)")
        return None
    except Exception as exc:
        logger.warning(f"IBKR VIX fetch error: {exc}")
        return None


# ─── Market Snapshot ──────────────────────────────────────────────────────────

def fetch_market_snapshot_ibkr(ib: "IB") -> dict:
    """
    Fetch a combined market snapshot: VIX + SPY prices.

    Returns:
        Dict with keys: {vix, spy_price, spy_bid, spy_ask}
        Values default to 0.0 on any failure.
    """
    result = {"vix": 0.0, "spy_price": 0.0, "spy_bid": 0.0, "spy_ask": 0.0}

    if ib is None:
        return result

    vix = fetch_vix_ibkr(ib)
    if vix is not None:
        result["vix"] = vix

    spy = fetch_stock_snapshot(ib, "SPY")
    if spy:
        result["spy_price"] = spy.get("last", 0.0)
        result["spy_bid"] = spy.get("bid", 0.0)
        result["spy_ask"] = spy.get("ask", 0.0)

    return result
