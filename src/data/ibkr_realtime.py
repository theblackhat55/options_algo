"""
src/data/ibkr_realtime.py
=========================
Real-time IBKR data enrichment at signal time.

Provides per-ticker enrichment functions for:
  - Real-time ATM implied volatility (from live Greeks)
  - Unusual options flow detection (volume vs open interest)
  - Intraday volume pace vs 20-day average

All functions accept an already-connected `ib` IB instance and
return gracefully (never raise) — enrichment is always optional.

Usage:
    from src.data.ibkr_client import connect_ibkr, disconnect_ibkr
    from src.data.ibkr_realtime import fetch_realtime_enrichment

    ib = connect_ibkr()
    rt = fetch_realtime_enrichment(ib, "AAPL")
    disconnect_ibkr(ib)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Real-time IV ─────────────────────────────────────────────────────────────

def fetch_realtime_iv(
    ib,
    ticker: str,
    chain_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Get real-time ATM IV for a ticker.

    If chain_df provided (from ibkr_client.fetch_options_chain_ibkr), extract
    ATM IV directly from the Greeks already in the DataFrame.
    Otherwise fetches a fresh ATM call+put snapshot and reads IV from Greeks.

    Returns:
        {
            'iv': float,        # ATM IV as decimal (e.g. 0.32 = 32%)
            'iv_pct': float,    # same as percentage (32.0)
            'call_iv': float,   # ATM call IV (%)
            'put_iv': float,    # ATM put IV (%)
            'skew': float,      # put_iv - call_iv (positive = puts expensive)
            'source': 'ibkr_realtime'
        }
        or None on failure.
    """
    try:
        if ib is None:
            return None

        # ── Use provided chain if available ───────────────────────────────────
        if chain_df is not None and not chain_df.empty:
            return _extract_iv_from_chain(ticker, chain_df)

        # ── Fetch fresh ATM snapshot ──────────────────────────────────────────
        try:
            from ib_insync import Option, Stock
        except ImportError:
            logger.warning("ib_insync not available for realtime IV fetch")
            return None

        from src.data.ibkr_client import fetch_stock_snapshot, _safe_float

        snap = fetch_stock_snapshot(ib, ticker)
        if not snap or snap.get("last", 0) <= 0:
            return None
        spot = snap["last"]

        # Qualify underlying to get conId
        stock = Stock(ticker, "SMART", "USD")
        qualified_list = ib.qualifyContracts(stock)
        if not qualified_list:
            return None
        con_id = qualified_list[0].conId

        # Get option params (expirations + strikes)
        params = ib.reqSecDefOptParams(ticker, "", "STK", con_id)
        if not params:
            return None

        param = params[0]
        today = date.today()
        target_exp = None
        for exp_str in sorted(param.expirations):
            try:
                exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
                dte = (exp_date - today).days
                if 25 <= dte <= 50:
                    target_exp = exp_str
                    break
            except ValueError:
                continue

        if not target_exp:
            # Fallback: try 15-60 DTE range
            for exp_str in sorted(param.expirations):
                try:
                    exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
                    dte = (exp_date - today).days
                    if 15 <= dte <= 60:
                        target_exp = exp_str
                        break
                except ValueError:
                    continue

        if not target_exp:
            return None

        # Find ATM strike (closest to spot)
        strikes = sorted(param.strikes, key=lambda s: abs(s - spot))
        atm_strike = strikes[0] if strikes else spot

        # Build and qualify ATM call + put
        call_c = Option(ticker, target_exp, atm_strike, "C", "SMART", currency="USD")
        put_c = Option(ticker, target_exp, atm_strike, "P", "SMART", currency="USD")
        try:
            ib.qualifyContracts(call_c, put_c)
        except Exception:
            pass

        call_t = ib.reqMktData(call_c, "", snapshot=True, regulatorySnapshot=False)
        put_t = ib.reqMktData(put_c, "", snapshot=True, regulatorySnapshot=False)
        ib.sleep(4.0)

        try:
            ib.cancelMktData(call_c)
            ib.cancelMktData(put_c)
        except Exception:
            pass

        call_iv_raw = _safe_float(call_t.modelGreeks.impliedVol if call_t.modelGreeks else None)
        put_iv_raw = _safe_float(put_t.modelGreeks.impliedVol if put_t.modelGreeks else None)

        # impliedVol from IBKR Greeks is a fraction (0.32 = 32%)
        call_iv = round(call_iv_raw * 100, 2) if call_iv_raw > 0 else 0.0
        put_iv = round(put_iv_raw * 100, 2) if put_iv_raw > 0 else 0.0

        if call_iv <= 0 and put_iv <= 0:
            return None

        atm_iv = (call_iv + put_iv) / 2 if call_iv > 0 and put_iv > 0 else max(call_iv, put_iv)
        skew = round(put_iv - call_iv, 2) if call_iv > 0 and put_iv > 0 else 0.0

        return {
            "iv": round(atm_iv / 100, 4),
            "iv_pct": round(atm_iv, 2),
            "call_iv": round(call_iv, 2),
            "put_iv": round(put_iv, 2),
            "skew": skew,
            "source": "ibkr_realtime",
        }

    except Exception as exc:
        logger.warning(f"{ticker}: fetch_realtime_iv failed: {exc}")
        return None


def _extract_iv_from_chain(ticker: str, chain_df: pd.DataFrame) -> Optional[dict]:
    """Extract ATM IV from an existing chain DataFrame (no new IBKR requests)."""
    try:
        calls = chain_df[chain_df["type"] == "call"]
        puts = chain_df[chain_df["type"] == "put"]

        if calls.empty or puts.empty:
            return None

        # Use median strike as ATM proxy (chain is already ±30% filtered)
        spot_proxy = float(chain_df["strike"].median())

        atm_call = calls.iloc[(calls["strike"] - spot_proxy).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts["strike"] - spot_proxy).abs().argsort()[:1]]

        if atm_call.empty or atm_put.empty:
            return None

        raw_call_iv = float(atm_call["implied_volatility"].iloc[0])
        raw_put_iv = float(atm_put["implied_volatility"].iloc[0])

        # IBKR returns implied_volatility as a fraction (0.32 = 32%)
        # Convert to percentage if needed
        call_iv = round(raw_call_iv * 100 if raw_call_iv < 5.0 and raw_call_iv > 0 else raw_call_iv, 2)
        put_iv = round(raw_put_iv * 100 if raw_put_iv < 5.0 and raw_put_iv > 0 else raw_put_iv, 2)

        if call_iv <= 0 and put_iv <= 0:
            return None

        atm_iv = (call_iv + put_iv) / 2 if call_iv > 0 and put_iv > 0 else max(call_iv, put_iv)
        skew = round(put_iv - call_iv, 2) if call_iv > 0 and put_iv > 0 else 0.0

        return {
            "iv": round(atm_iv / 100, 4),
            "iv_pct": round(atm_iv, 2),
            "call_iv": round(call_iv, 2),
            "put_iv": round(put_iv, 2),
            "skew": skew,
            "source": "ibkr_realtime",
        }
    except Exception as exc:
        logger.warning(f"{ticker}: _extract_iv_from_chain failed: {exc}")
        return None


# ─── Options Flow ─────────────────────────────────────────────────────────────

def fetch_options_flow(
    ib,
    ticker: str,
    chain_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Detect unusual options activity vs open interest.

    For each strike, compares today's volume to open interest.
    A volume/OI ratio > 0.5 on a strike is unusual (new positions opening).

    If chain_df provided, uses it. Otherwise fetches a fresh chain.

    Returns:
        {
            'flow_score': float,            # 0-100. 0=normal, 100=extreme
            'call_volume': int,             # total call volume today
            'put_volume': int,              # total put volume today
            'put_call_volume_ratio': float, # put vol / call vol
            'unusual_strikes': list[dict],  # strikes with vol/OI > 0.5, sorted by ratio
            'dominant_side': str,           # 'CALLS', 'PUTS', or 'NEUTRAL'
            'source': 'ibkr_realtime'
        }
        Default all-zeros dict if unavailable (never raises).
    """
    _default = {
        "flow_score": 0.0,
        "call_volume": 0,
        "put_volume": 0,
        "put_call_volume_ratio": 1.0,
        "unusual_strikes": [],
        "dominant_side": "NEUTRAL",
        "source": "ibkr_realtime",
    }
    try:
        if ib is None:
            return _default

        # Use provided chain or fetch fresh
        _chain = chain_df
        if _chain is None or _chain.empty:
            from src.data.ibkr_client import fetch_options_chain_ibkr
            _chain = fetch_options_chain_ibkr(ib, ticker)

        if _chain is None or _chain.empty:
            return _default

        return _compute_flow_from_chain(ticker, _chain, _default)

    except Exception as exc:
        logger.warning(f"{ticker}: fetch_options_flow failed: {exc}")
        return _default


def _compute_flow_from_chain(ticker: str, chain_df: pd.DataFrame, default: dict) -> dict:
    """Compute flow metrics from a chain DataFrame."""
    try:
        calls = chain_df[chain_df["type"] == "call"]
        puts = chain_df[chain_df["type"] == "put"]

        call_volume = int(calls["volume"].fillna(0).sum())
        put_volume = int(puts["volume"].fillna(0).sum())
        total_volume = call_volume + put_volume

        if total_volume == 0:
            return default

        # Put/call volume ratio
        pc_ratio = round(put_volume / call_volume, 3) if call_volume > 0 else 1.0

        # Unusual strikes: volume/OI > 0.5
        unusual = []
        for _, row in chain_df.iterrows():
            vol = float(row.get("volume", 0) or 0)
            oi = float(row.get("open_interest", 0) or 0)
            if oi > 0 and vol > 0:
                ratio = round(vol / oi, 3)
                if ratio > 0.5:
                    unusual.append({
                        "strike": float(row.get("strike", 0)),
                        "type": str(row.get("type", "")),
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "vol_oi_ratio": ratio,
                        "expiration": str(row.get("expiration", "")),
                    })

        # Sort by ratio descending
        unusual.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)

        # Flow score: proportion of volume in unusual strikes, scaled 0-100
        unusual_volume = sum(u["volume"] for u in unusual)
        vol_pct_score = (unusual_volume / total_volume * 100 * 2) if total_volume > 0 else 0.0
        # Breadth bonus: each unusual strike adds 2 points
        breadth_bonus = len(unusual) * 2.0
        flow_score = min(100.0, round(vol_pct_score + breadth_bonus, 1))

        # Dominant side from unusual activity
        unusual_calls = sum(u["volume"] for u in unusual if u["type"] == "call")
        unusual_puts = sum(u["volume"] for u in unusual if u["type"] == "put")

        if unusual_calls > 0 or unusual_puts > 0:
            if unusual_calls > unusual_puts * 1.5:
                dominant_side = "CALLS"
            elif unusual_puts > unusual_calls * 1.5:
                dominant_side = "PUTS"
            else:
                dominant_side = "NEUTRAL"
        else:
            # Fallback to overall call/put volume ratio
            if call_volume > put_volume * 1.3:
                dominant_side = "CALLS"
            elif put_volume > call_volume * 1.3:
                dominant_side = "PUTS"
            else:
                dominant_side = "NEUTRAL"

        return {
            "flow_score": flow_score,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_volume_ratio": pc_ratio,
            "unusual_strikes": unusual[:10],  # cap at top 10
            "dominant_side": dominant_side,
            "source": "ibkr_realtime",
        }

    except Exception as exc:
        logger.warning(f"{ticker}: _compute_flow_from_chain failed: {exc}")
        return default


# ─── Volume Pace ──────────────────────────────────────────────────────────────

def fetch_volume_pace(
    ib,
    ticker: str,
    stock_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Intraday volume vs 20-day average volume.

    stock_df: historical OHLCV DataFrame with 'volume' column (to get 20d avg).
    Gets current intraday volume from IBKR stock snapshot.

    Returns:
        {
            'current_volume': int,
            'avg_volume_20d': int,
            'volume_pace': float,   # current_volume / avg_volume_20d (1.0 = normal)
            'volume_signal': str,   # 'HIGH' (>1.5x), 'NORMAL' (0.7-1.5x), 'LOW' (<0.7x)
        }
    """
    _default = {
        "current_volume": 0,
        "avg_volume_20d": 0,
        "volume_pace": 1.0,
        "volume_signal": "NORMAL",
    }
    try:
        if ib is None:
            return _default

        from src.data.ibkr_client import fetch_stock_snapshot
        snap = fetch_stock_snapshot(ib, ticker)
        current_volume = int(snap.get("volume", 0)) if snap else 0

        # 20-day average from historical data
        avg_20d = 0
        if stock_df is not None and not stock_df.empty and "volume" in stock_df.columns:
            vol_series = stock_df["volume"].tail(20).dropna()
            if len(vol_series) > 0:
                avg_20d = int(vol_series.mean())

        if avg_20d <= 0 or current_volume <= 0:
            return _default

        pace = round(current_volume / avg_20d, 3)

        if pace >= 1.5:
            signal = "HIGH"
        elif pace < 0.7:
            signal = "LOW"
        else:
            signal = "NORMAL"

        return {
            "current_volume": current_volume,
            "avg_volume_20d": avg_20d,
            "volume_pace": pace,
            "volume_signal": signal,
        }

    except Exception as exc:
        logger.warning(f"{ticker}: fetch_volume_pace failed: {exc}")
        return _default


# ─── Fast Enrichment (morning brief) ─────────────────────────────────────────

def fetch_realtime_enrichment_fast(ib, ticker: str, stock_df=None) -> dict:
    """
    Lightweight enrichment for morning brief — ATM IV + volume pace only.
    Skips full chain download (~3s per ticker vs ~15s for full).
    flow_score is always 0 (no chain needed for IV/pace).
    """
    result: dict = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "data_quality": "NONE",
        "flow_score": 0,
        "dominant_side": "NEUTRAL",
        "put_call_volume_ratio": 1.0,
        "unusual_strikes": [],
        "call_volume": 0,
        "put_volume": 0,
        "current_volume": 0,
        "avg_volume_20d": 0,
        "volume_pace": 1.0,
        "volume_signal": "NORMAL",
        "iv": None,
        "iv_pct": None,
        "call_iv": None,
        "put_iv": None,
        "skew": None,
    }
    try:
        iv_data = fetch_realtime_iv(ib, ticker)
        if iv_data:
            result.update({k: v for k, v in iv_data.items() if k != "source"})
            result["data_quality"] = "PARTIAL"

        pace_data = fetch_volume_pace(ib, ticker, stock_df=stock_df)
        if pace_data:
            result.update({k: v for k, v in pace_data.items()})
            if result["data_quality"] == "PARTIAL":
                result["data_quality"] = "FULL"
    except Exception as exc:
        logger.warning(f"{ticker}: fast enrichment error — {exc}")

    return result


# ─── Master Enrichment ────────────────────────────────────────────────────────

def fetch_realtime_enrichment(
    ib,
    ticker: str,
    chain_df: Optional[pd.DataFrame] = None,
    stock_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Master function: fetches all real-time enrichment for one ticker in one call.

    Reuses the same chain_df for both IV and flow to avoid double-fetching.
    If chain_df is None, fetches it once and passes to both IV and flow functions.

    Returns combined dict with all keys from fetch_realtime_iv,
    fetch_options_flow, fetch_volume_pace plus:
        'ticker': str
        'timestamp': str (ISO)
        'data_quality': 'FULL' | 'PARTIAL' | 'NONE'
    """
    result: dict = {
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # IV defaults
        "iv": None,
        "iv_pct": None,
        "call_iv": None,
        "put_iv": None,
        "skew": 0.0,
        # Flow defaults
        "flow_score": 0.0,
        "call_volume": 0,
        "put_volume": 0,
        "put_call_volume_ratio": 1.0,
        "unusual_strikes": [],
        "dominant_side": "NEUTRAL",
        # Volume defaults
        "current_volume": 0,
        "avg_volume_20d": 0,
        "volume_pace": 1.0,
        "volume_signal": "NORMAL",
        # Quality
        "data_quality": "NONE",
        "source": "ibkr_realtime",
    }

    if ib is None:
        return result

    try:
        # ── Fetch chain once, reuse for IV + flow ─────────────────────────────
        _chain = chain_df
        if _chain is None or _chain.empty:
            try:
                from src.data.ibkr_client import fetch_options_chain_ibkr
                _chain = fetch_options_chain_ibkr(ib, ticker)
            except Exception as exc:
                logger.warning(f"{ticker}: chain fetch in enrichment failed: {exc}")
                _chain = pd.DataFrame()

        # ── IV enrichment ─────────────────────────────────────────────────────
        iv_data = fetch_realtime_iv(ib, ticker, chain_df=_chain)
        iv_ok = iv_data is not None
        if iv_ok:
            result.update(iv_data)

        # ── Flow enrichment ───────────────────────────────────────────────────
        flow_data = fetch_options_flow(ib, ticker, chain_df=_chain)
        flow_ok = (
            flow_data.get("call_volume", 0) > 0
            or flow_data.get("put_volume", 0) > 0
        )
        # Merge flow keys (skip 'source' to avoid overwriting)
        for k, v in flow_data.items():
            if k != "source":
                result[k] = v

        # ── Volume pace enrichment ────────────────────────────────────────────
        vol_data = fetch_volume_pace(ib, ticker, stock_df=stock_df)
        vol_ok = vol_data.get("current_volume", 0) > 0
        result.update(vol_data)

        # ── Data quality ──────────────────────────────────────────────────────
        parts_ok = sum([iv_ok, flow_ok, vol_ok])
        if parts_ok == 3:
            result["data_quality"] = "FULL"
        elif parts_ok > 0:
            result["data_quality"] = "PARTIAL"
        else:
            result["data_quality"] = "NONE"

        logger.debug(
            f"{ticker}: enrichment quality={result['data_quality']} "
            f"iv={result.get('iv_pct')} flow={result.get('flow_score')} "
            f"vol_pace={result.get('volume_pace')}"
        )

    except Exception as exc:
        logger.warning(f"{ticker}: fetch_realtime_enrichment failed: {exc}")

    return result
