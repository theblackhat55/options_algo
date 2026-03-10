from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from ib_insync import IB, Stock, Option
except Exception:  # pragma: no cover
    IB = None
    Stock = None
    Option = None


# ---------------------------------------------------------------------
# env bootstrap
# ---------------------------------------------------------------------

def _load_env() -> None:
    try:
        if load_dotenv is not None:
            load_dotenv()
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path, override=False)
    except Exception:
        pass


_load_env()


# ---------------------------------------------------------------------
# test / runtime guards
# ---------------------------------------------------------------------

def _is_test_env() -> bool:
    try:
        if "PYTEST_CURRENT_TEST" in os.environ:
            return True
        if "pytest" in sys.modules:
            return True
        if os.getenv("OPTIONS_FETCHER_DISABLE_NETWORK", "0") == "1":
            return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        out = str(value)
        return out if out else default
    except Exception:
        return default


def _mid_from_bid_ask(bid: Any, ask: Any) -> float:
    b = _safe_float(bid, 0.0)
    a = _safe_float(ask, 0.0)
    if b > 0 and a > 0 and a >= b:
        return (a + b) / 2.0
    return 0.0


def _valid_quote(bid: Any, ask: Any) -> bool:
    b = _safe_float(bid, 0.0)
    a = _safe_float(ask, 0.0)
    return b > 0 and a > 0 and a >= b


def _normalize_type(value: Any) -> str:
    s = _safe_str(value, "").strip().lower()
    if s in {"c", "call", "calls"}:
        return "call"
    if s in {"p", "put", "puts"}:
        return "put"
    return s


def _to_exp_yyyymmdd(value: Any) -> str:
    s = _safe_str(value, "")
    if not s:
        return ""
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y%m%d")
    except Exception:
        return ""


def _to_exp_iso(value: Any) -> str:
    s = _safe_str(value, "")
    if not s:
        return ""
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------
# public contract
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "ticker",
    "contract_ticker",
    "expiration",
    "strike",
    "type",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "bid_ask_spread_pct",
    "source",
]


# ---------------------------------------------------------------------
# massive / polygon snapshot fetch
# ---------------------------------------------------------------------

def _get_polygon_api_key() -> str:
    return (
        os.getenv("POLYGON_API_KEY")
        or os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_KEY")
        or ""
    ).strip()


def _massive_snapshot_url(ticker: str, api_key: str) -> List[str]:
    tk = _safe_str(ticker, "").upper()
    return [
        f"https://api.massive.com/v3/snapshot/options/{tk}?limit=250&apiKey={api_key}",
        f"https://api.polygon.io/v3/snapshot/options/{tk}?limit=250&apiKey={api_key}",
    ]


def _extract_snapshot_rows(payload: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
    results = payload.get("results") or []
    out: List[Dict[str, Any]] = []

    for item in results:
        details = item.get("details") or {}
        last_quote = item.get("last_quote") or {}
        last_trade = item.get("last_trade") or {}
        greeks = item.get("greeks") or {}
        day = item.get("day") or {}

        contract_ticker = (
            item.get("ticker")
            or details.get("ticker")
            or details.get("contract_ticker")
            or ""
        )

        expiration = (
            details.get("expiration_date")
            or details.get("expiration")
            or item.get("expiration_date")
            or ""
        )

        strike = details.get("strike_price", item.get("strike_price", 0.0))
        opt_type = details.get("contract_type", item.get("contract_type", ""))

        bid = last_quote.get("bid")
        ask = last_quote.get("ask")
        last = last_trade.get("price")
        volume = day.get("volume", item.get("volume"))
        open_interest = item.get("open_interest")
        iv = item.get("implied_volatility")
        delta = greeks.get("delta")
        gamma = greeks.get("gamma")
        theta = greeks.get("theta")
        vega = greeks.get("vega")

        bid_f = _safe_float(bid, 0.0)
        ask_f = _safe_float(ask, 0.0)
        mid_f = _mid_from_bid_ask(bid_f, ask_f)

        spread_pct = 999.0
        if mid_f > 0 and bid_f > 0 and ask_f > 0 and ask_f >= bid_f:
            spread_pct = (ask_f - bid_f) / mid_f

        out.append(
            {
                "ticker": _safe_str(ticker, "").upper(),
                "contract_ticker": _safe_str(contract_ticker, ""),
                "expiration": _to_exp_iso(expiration),
                "strike": _safe_float(strike, 0.0),
                "type": _normalize_type(opt_type),
                "bid": bid_f,
                "ask": ask_f,
                "mid": mid_f,
                "last": _safe_float(last, 0.0),
                "volume": _safe_int(volume, 0),
                "open_interest": _safe_int(open_interest, 0),
                "implied_volatility": _safe_float(iv, 0.0),
                "delta": _safe_float(delta, 0.0),
                "gamma": _safe_float(gamma, 0.0),
                "theta": _safe_float(theta, 0.0),
                "vega": _safe_float(vega, 0.0),
                "bid_ask_spread_pct": _safe_float(spread_pct, 999.0),
                "source": "massive_snapshot",
            }
        )
    return out


def _fetch_massive_chain(ticker: str) -> pd.DataFrame:
    if _is_test_env():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    api_key = _get_polygon_api_key()
    if not api_key or requests is None:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    last_err = None
    for url in _massive_snapshot_url(ticker, api_key):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                last_err = RuntimeError(f"{url} -> HTTP {resp.status_code}")
                continue
            payload = resp.json() or {}
            rows = _extract_snapshot_rows(payload, ticker)
            if rows:
                df = pd.DataFrame(rows)
                return _finalize_chain_df(df)
        except Exception as exc:
            last_err = exc
            continue

    if last_err:
        # swallow and return empty to preserve backward compatibility
        pass
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


# ---------------------------------------------------------------------
# IBKR enrichment
# ---------------------------------------------------------------------

@dataclass
class IBKREnricherConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 991
    readonly: bool = True
    market_data_type: int = 2  # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
    max_expiries: int = 2
    strikes_each_side: int = 5
    sleep_seconds: float = 2.0


def _get_ibkr_config() -> IBKREnricherConfig:
    return IBKREnricherConfig(
        enabled=_safe_str(os.getenv("IBKR_ENRICH_OPTIONS", "1"), "1") not in {"0", "false", "False"},
        host=_safe_str(os.getenv("IBKR_HOST", "127.0.0.1"), "127.0.0.1"),
        port=_safe_int(os.getenv("IBKR_PORT", 4002), 4002),
        client_id=_safe_int(os.getenv("IBKR_CLIENT_ID", 991), 991),
        readonly=True,
        market_data_type=_safe_int(os.getenv("IBKR_MARKET_DATA_TYPE", 2), 2),
        max_expiries=_safe_int(os.getenv("IBKR_MAX_EXPIRIES", 2), 2),
        strikes_each_side=_safe_int(os.getenv("IBKR_STRIKES_EACH_SIDE", 5), 5),
        sleep_seconds=_safe_float(os.getenv("IBKR_SLEEP_SECONDS", 2.0), 2.0),
    )


def _connect_ibkr(cfg: IBKREnricherConfig) -> Optional[Any]:
    if not cfg.enabled or IB is None:
        return None
    try:
        ib = IB()
        ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, readonly=cfg.readonly, timeout=10)
        if not ib.isConnected():
            return None
        ib.reqMarketDataType(cfg.market_data_type)
        return ib
    except Exception:
        return None


def _disconnect_ibkr(ib: Optional[Any]) -> None:
    try:
        if ib is not None and ib.isConnected():
            ib.disconnect()
    except Exception:
        pass


def _get_underlying_spot(ib: Any, ticker: str) -> float:
    try:
        stk = Stock(_safe_str(ticker, "").upper(), "SMART", "USD")
        ib.qualifyContracts(stk)
        t = ib.reqMktData(stk, "", False, False)
        ib.sleep(1.5)
        px = _safe_float(getattr(t, "marketPrice", lambda: float("nan"))(), 0.0)
        if px <= 0:
            px = _safe_float(getattr(t, "last", 0.0), 0.0)
        if px <= 0:
            px = _safe_float(getattr(t, "close", 0.0), 0.0)
        ib.cancelMktData(stk)
        return px
    except Exception:
        return 0.0


def _pick_subset_for_ibkr(df: pd.DataFrame, spot: float, cfg: IBKREnricherConfig) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    work["expiration_dt"] = pd.to_datetime(work["expiration"], errors="coerce")
    work = work.dropna(subset=["expiration_dt"])
    if work.empty:
        return work

    expiries = sorted(work["expiration_dt"].dt.strftime("%Y-%m-%d").unique().tolist())
    expiries = expiries[: max(1, cfg.max_expiries)]
    work = work[work["expiration_dt"].dt.strftime("%Y-%m-%d").isin(expiries)].copy()

    if spot > 0:
        strikes = sorted(pd.to_numeric(work["strike"], errors="coerce").dropna().unique().tolist())
        if strikes:
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
            lo = max(0, atm_idx - max(1, cfg.strikes_each_side))
            hi = min(len(strikes), atm_idx + max(1, cfg.strikes_each_side) + 1)
            chosen = set(strikes[lo:hi])
            work = work[work["strike"].isin(chosen)].copy()

    work = work.drop(columns=["expiration_dt"], errors="ignore")
    return work


def _ibkr_contract_from_row(ticker: str, row: pd.Series) -> Optional[Any]:
    if Option is None:
        return None
    try:
        expiry = _to_exp_yyyymmdd(row.get("expiration"))
        strike = _safe_float(row.get("strike"), 0.0)
        right = "C" if _normalize_type(row.get("type")) == "call" else "P"
        if not expiry or strike <= 0 or right not in {"C", "P"}:
            return None
        return Option(
            _safe_str(ticker, "").upper(),
            expiry,
            strike,
            right,
            "SMART",
            tradingClass=_safe_str(ticker, "").upper(),
            multiplier="100",
        )
    except Exception:
        return None


def _extract_ibkr_ticker_fields(ticker_obj: Any) -> Dict[str, Any]:
    bid = _safe_float(getattr(ticker_obj, "bid", 0.0), 0.0)
    ask = _safe_float(getattr(ticker_obj, "ask", 0.0), 0.0)
    last = _safe_float(getattr(ticker_obj, "last", 0.0), 0.0)
    close = _safe_float(getattr(ticker_obj, "close", 0.0), 0.0)
    volume = _safe_int(getattr(ticker_obj, "volume", 0), 0)

    iv = 0.0
    delta = 0.0
    gamma = 0.0
    theta = 0.0
    vega = 0.0

    mg = getattr(ticker_obj, "modelGreeks", None)
    if mg is not None:
        iv = _safe_float(getattr(mg, "impliedVol", 0.0), 0.0)
        delta = _safe_float(getattr(mg, "delta", 0.0), 0.0)
        gamma = _safe_float(getattr(mg, "gamma", 0.0), 0.0)
        theta = _safe_float(getattr(mg, "theta", 0.0), 0.0)
        vega = _safe_float(getattr(mg, "vega", 0.0), 0.0)

    mid = _mid_from_bid_ask(bid, ask)
    spread_pct = 999.0
    if _valid_quote(bid, ask) and mid > 0:
        spread_pct = (ask - bid) / mid

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last if last > 0 else close,
        "volume": volume,
        "implied_volatility": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "bid_ask_spread_pct": spread_pct,
        "source": "ibkr",
    }


def _contract_key(expiration: Any, strike: Any, opt_type: Any) -> Tuple[str, float, str]:
    return (
        _to_exp_iso(expiration),
        round(_safe_float(strike, 0.0), 6),
        _normalize_type(opt_type),
    )


def _enrich_chain_with_ibkr(ticker: str, df: pd.DataFrame, ib: Any | None = None) -> pd.DataFrame:
    if _is_test_env():
        return df
    cfg = _get_ibkr_config()
    if df.empty or not cfg.enabled:
        return df

    own_ib = ib is None
    if ib is None:
        ib = _connect_ibkr(cfg)
    if ib is None:
        return df

    try:
        spot = _get_underlying_spot(ib, ticker)
        subset = _pick_subset_for_ibkr(df, spot, cfg)
        if subset.empty:
            return df

        contracts: List[Any] = []
        key_by_conid: Dict[int, Tuple[str, float, str]] = {}

        for _, row in subset.iterrows():
            contract = _ibkr_contract_from_row(ticker, row)
            if contract is None:
                continue
            try:
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    continue
                qc = qualified[0]
                key = _contract_key(row.get("expiration"), row.get("strike"), row.get("type"))
                contracts.append(qc)
                key_by_conid[int(qc.conId)] = key
            except Exception:
                continue

        if not contracts:
            return df

        tickers = []
        for c in contracts:
            try:
                t = ib.reqMktData(c, "", False, False)
                tickers.append(t)
            except Exception:
                continue

        ib.sleep(max(1.0, cfg.sleep_seconds))

        updates: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
        for t in tickers:
            try:
                c = getattr(t, "contract", None)
                if c is None:
                    continue
                conid = int(getattr(c, "conId", 0))
                key = key_by_conid.get(conid)
                if key is None:
                    continue
                fields = _extract_ibkr_ticker_fields(t)
                updates[key] = fields
            except Exception:
                continue

        enriched = df.copy()
        if not updates:
            return enriched

        def _apply_row(row: pd.Series) -> pd.Series:
            key = _contract_key(row.get("expiration"), row.get("strike"), row.get("type"))
            upd = updates.get(key)
            if not upd:
                return row

            existing_bid = _safe_float(row.get("bid"), 0.0)
            existing_ask = _safe_float(row.get("ask"), 0.0)

            ib_valid = _valid_quote(upd.get("bid"), upd.get("ask"))
            base_valid = _valid_quote(existing_bid, existing_ask)

            if ib_valid:
                row["bid"] = upd["bid"]
                row["ask"] = upd["ask"]
                row["mid"] = upd["mid"]
                if _safe_float(upd.get("last"), 0.0) > 0:
                    row["last"] = upd["last"]
                if _safe_int(upd.get("volume"), 0) > 0:
                    row["volume"] = upd["volume"]
                if _safe_float(row.get("implied_volatility"), 0.0) <= 0 and _safe_float(upd.get("implied_volatility"), 0.0) > 0:
                    row["implied_volatility"] = upd["implied_volatility"]
                if _safe_float(row.get("delta"), 0.0) == 0.0 and _safe_float(upd.get("delta"), 0.0) != 0.0:
                    row["delta"] = upd["delta"]
                if _safe_float(row.get("gamma"), 0.0) == 0.0 and _safe_float(upd.get("gamma"), 0.0) != 0.0:
                    row["gamma"] = upd["gamma"]
                if _safe_float(row.get("theta"), 0.0) == 0.0 and _safe_float(upd.get("theta"), 0.0) != 0.0:
                    row["theta"] = upd["theta"]
                if _safe_float(row.get("vega"), 0.0) == 0.0 and _safe_float(upd.get("vega"), 0.0) != 0.0:
                    row["vega"] = upd["vega"]
                row["bid_ask_spread_pct"] = upd["bid_ask_spread_pct"]
                row["source"] = "ibkr_enriched"
            elif not base_valid:
                if _safe_float(upd.get("last"), 0.0) > 0:
                    row["last"] = upd["last"]
                if _safe_float(row.get("implied_volatility"), 0.0) <= 0 and _safe_float(upd.get("implied_volatility"), 0.0) > 0:
                    row["implied_volatility"] = upd["implied_volatility"]
                if _safe_float(row.get("delta"), 0.0) == 0.0 and _safe_float(upd.get("delta"), 0.0) != 0.0:
                    row["delta"] = upd["delta"]
                if _safe_float(row.get("gamma"), 0.0) == 0.0 and _safe_float(upd.get("gamma"), 0.0) != 0.0:
                    row["gamma"] = upd["gamma"]
                if _safe_float(row.get("theta"), 0.0) == 0.0 and _safe_float(upd.get("theta"), 0.0) != 0.0:
                    row["theta"] = upd["theta"]
                if _safe_float(row.get("vega"), 0.0) == 0.0 and _safe_float(upd.get("vega"), 0.0) != 0.0:
                    row["vega"] = upd["vega"]

            return row

        enriched = enriched.apply(_apply_row, axis=1)
        return _finalize_chain_df(enriched)

    finally:
        if own_ib:
            _disconnect_ibkr(ib)


# ---------------------------------------------------------------------
# final normalization
# ---------------------------------------------------------------------

def _finalize_chain_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    work = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in work.columns:
            if col in {"ticker", "contract_ticker", "expiration", "type", "source"}:
                work[col] = ""
            else:
                work[col] = 0.0

    work["ticker"] = work["ticker"].map(lambda x: _safe_str(x, "").upper())
    work["contract_ticker"] = work["contract_ticker"].map(lambda x: _safe_str(x, ""))
    work["expiration"] = work["expiration"].map(_to_exp_iso)
    work["type"] = work["type"].map(_normalize_type)
    work["source"] = work["source"].map(lambda x: _safe_str(x, ""))

    numeric_float_cols = [
        "strike",
        "bid",
        "ask",
        "mid",
        "last",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "bid_ask_spread_pct",
    ]
    for col in numeric_float_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    numeric_int_cols = ["volume", "open_interest"]
    for col in numeric_int_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    bad_mid = (work["mid"] <= 0) & (work["bid"] > 0) & (work["ask"] > 0) & (work["ask"] >= work["bid"])
    work.loc[bad_mid, "mid"] = (work.loc[bad_mid, "bid"] + work.loc[bad_mid, "ask"]) / 2.0

    valid_spread = (work["bid"] > 0) & (work["ask"] > 0) & (work["ask"] >= work["bid"]) & (work["mid"] > 0)
    work.loc[valid_spread, "bid_ask_spread_pct"] = (work.loc[valid_spread, "ask"] - work.loc[valid_spread, "bid"]) / work.loc[valid_spread, "mid"]
    work.loc[~valid_spread, "bid_ask_spread_pct"] = 999.0

    work = work[REQUIRED_COLUMNS].copy()
    work = work.drop_duplicates(subset=["expiration", "strike", "type"], keep="first").reset_index(drop=True)
    return work


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

def fetch_options_chain(ticker: str, ib: Any | None = None) -> pd.DataFrame:
    """
    Base chain comes from Massive/Polygon snapshot.
    A bounded IBKR enrichment pass improves bid/ask/mid/last/IV for near-ATM front expiries.
    Returns a normalized DataFrame with REQUIRED_COLUMNS.
    """
    if _is_test_env():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    base = _fetch_massive_chain(ticker)
    base = _finalize_chain_df(base)

    if base.empty:
        return base

    enriched = _enrich_chain_with_ibkr(ticker, base, ib=ib)
    enriched = _finalize_chain_df(enriched)
    return enriched


__all__ = ["fetch_options_chain", "REQUIRED_COLUMNS"]

# ---------------------------------------------------------------------
# backward-compatible legacy helpers
# ---------------------------------------------------------------------

def get_atm_iv(ticker: str, spot_price: float | None = None) -> float:
    """
    Backward-compatible helper used by volatility analysis.
    Returns the implied volatility of the nearest-ATM contract from the nearest expiry.
    Falls back to 0.0 on failure.
    """
    try:
        df = fetch_options_chain(ticker)
        if df is None or df.empty:
            return 0.0

        work = df.copy()
        work["expiration_dt"] = pd.to_datetime(work["expiration"], errors="coerce")
        work = work.dropna(subset=["expiration_dt"])
        if work.empty:
            return 0.0

        if not spot_price or _safe_float(spot_price, 0.0) <= 0:
            # try a simple ATM proxy from chain center if no spot passed
            strikes = sorted(pd.to_numeric(work["strike"], errors="coerce").dropna().unique().tolist())
            if not strikes:
                return 0.0
            spot_price = strikes[len(strikes) // 2]

        nearest_exp = work["expiration_dt"].min()
        front = work[work["expiration_dt"] == nearest_exp].copy()
        if front.empty:
            return 0.0

        front["dist"] = (pd.to_numeric(front["strike"], errors="coerce") - float(spot_price)).abs()
        front = front.sort_values(["dist", "open_interest", "volume"], ascending=[True, False, False])

        iv = _safe_float(front.iloc[0].get("implied_volatility"), 0.0)
        if iv > 0:
            return iv

        positive = front[pd.to_numeric(front["implied_volatility"], errors="coerce") > 0]
        if not positive.empty:
            return _safe_float(positive.iloc[0].get("implied_volatility"), 0.0)

        return 0.0
    except Exception:
        return 0.0


def get_options_chain(ticker: str) -> pd.DataFrame:
    """
    Legacy alias for compatibility.
    """
    return fetch_options_chain(ticker)


def get_option_chain(ticker: str) -> pd.DataFrame:
    """
    Legacy alias for compatibility.
    """
    return fetch_options_chain(ticker)


__all__ = [
    "fetch_options_chain",
    "get_options_chain",
    "get_option_chain",
    "get_atm_iv",
    "REQUIRED_COLUMNS",
]
