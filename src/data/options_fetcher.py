"""
src/data/options_fetcher.py
===========================
Fetch options chain data.
Primary:  Polygon.io v3 snapshot endpoint (requires Options Starter plan).
Fallback: yfinance (free — no Greeks, less reliable).
Backup:   Tradier API (if configured).

Usage:
    from src.data.options_fetcher import fetch_options_chain, filter_liquid_options

    chain = fetch_options_chain("AAPL")
    liquid = filter_liquid_options(chain)
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config.settings import (
    POLYGON_API_KEY, TRADIER_API_KEY, RAW_DIR, PROCESSED_DIR,
    MIN_OPTION_VOLUME, MIN_OPEN_INTEREST, MAX_BID_ASK_SPREAD_PCT,
)

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"
TRADIER_BASE = "https://api.tradier.com/v1"


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def fetch_options_chain(
    ticker: str,
    expiration_gte: Optional[str] = None,
    expiration_lte: Optional[str] = None,
    dte_min: int = 7,
    dte_max: int = 60,
) -> pd.DataFrame:
    """
    Fetch options chain, trying sources in order: Polygon → Tradier → yfinance.

    Returns DataFrame with columns:
        ticker, contract_ticker, expiration, strike, type,
        bid, ask, mid, last, volume, open_interest,
        implied_volatility, delta, gamma, theta, vega,
        bid_ask_spread_pct
    """
    today = date.today()
    if expiration_gte is None:
        expiration_gte = (today + timedelta(days=dte_min)).isoformat()
    if expiration_lte is None:
        expiration_lte = (today + timedelta(days=dte_max)).isoformat()

    # Try Polygon first
    if POLYGON_API_KEY:
        df = _fetch_polygon(ticker, expiration_gte, expiration_lte)
        if not df.empty:
            return df

    # Try Tradier
    if TRADIER_API_KEY:
        df = _fetch_tradier(ticker)
        if not df.empty:
            return df

    # Fall back to yfinance
    logger.info(f"{ticker}: falling back to yfinance options")
    return _fetch_yfinance(ticker, dte_min=dte_min, dte_max=dte_max)


# ─── Polygon.io ───────────────────────────────────────────────────────────────

def _fetch_polygon(
    ticker: str,
    expiration_gte: str,
    expiration_lte: str,
) -> pd.DataFrame:
    """Fetch options chain snapshot from Polygon.io v3."""
    url = f"{POLYGON_BASE}/v3/snapshot/options/{ticker}"
    params: dict = {
        "apiKey": POLYGON_API_KEY,
        "expiration_date.gte": expiration_gte,
        "expiration_date.lte": expiration_lte,
        "limit": 250,
    }

    all_contracts: list[dict] = []

    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            for result in payload.get("results", []):
                details = result.get("details", {})
                day = result.get("day", {})
                greeks = result.get("greeks", {})
                quote = result.get("last_quote", {})

                bid = float(quote.get("bid", 0) or 0)
                ask = float(quote.get("ask", 0) or 0)
                mid = (bid + ask) / 2

                contract = {
                    "ticker": ticker,
                    "contract_ticker": details.get("ticker", ""),
                    "expiration": details.get("expiration_date", ""),
                    "strike": float(details.get("strike_price", 0) or 0),
                    "type": (details.get("contract_type") or "").lower(),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": float(day.get("close", 0) or 0),
                    "volume": int(day.get("volume", 0) or 0),
                    "open_interest": int(result.get("open_interest", 0) or 0),
                    "implied_volatility": float(greeks.get("implied_volatility", 0) or 0),
                    "delta": float(greeks.get("delta", 0) or 0),
                    "gamma": float(greeks.get("gamma", 0) or 0),
                    "theta": float(greeks.get("theta", 0) or 0),
                    "vega": float(greeks.get("vega", 0) or 0),
                    "bid_ask_spread_pct": (
                        (ask - bid) / mid * 100 if mid > 0 else 999.0
                    ),
                    "source": "polygon",
                }
                all_contracts.append(contract)

            next_url = payload.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": POLYGON_API_KEY}
            else:
                url = None

            time.sleep(0.12)  # Respect Polygon rate limits

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 403:
                logger.warning(f"Polygon 403 for {ticker} — check API plan/key")
            else:
                logger.error(f"Polygon fetch failed for {ticker}: {exc}")
            break
        except Exception as exc:
            logger.error(f"Polygon fetch error for {ticker}: {exc}")
            break

    if not all_contracts:
        return pd.DataFrame()

    df = pd.DataFrame(all_contracts)
    logger.info(f"{ticker}: {len(df)} contracts from Polygon")
    return df


# ─── Tradier API ──────────────────────────────────────────────────────────────

def _fetch_tradier(ticker: str) -> pd.DataFrame:
    """Fetch options chain from Tradier (backup source)."""
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json",
    }

    # Get expirations
    try:
        exp_resp = requests.get(
            f"{TRADIER_BASE}/markets/options/expirations",
            params={"symbol": ticker},
            headers=headers,
            timeout=15,
        )
        exp_resp.raise_for_status()
        expirations = exp_resp.json().get("expirations", {}).get("date", []) or []
    except Exception as exc:
        logger.warning(f"Tradier expirations failed for {ticker}: {exc}")
        return pd.DataFrame()

    today = date.today()
    valid_exps = [
        e for e in expirations
        if 7 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 60
    ][:4]  # At most 4 expirations

    all_contracts: list[dict] = []

    for exp in valid_exps:
        try:
            chain_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={"symbol": ticker, "expiration": exp, "greeks": "true"},
                headers=headers,
                timeout=15,
            )
            chain_resp.raise_for_status()
            options = chain_resp.json().get("options", {}).get("option", []) or []

            for opt in options:
                greeks = opt.get("greeks") or {}
                bid = float(opt.get("bid", 0) or 0)
                ask = float(opt.get("ask", 0) or 0)
                mid = (bid + ask) / 2

                contract = {
                    "ticker": ticker,
                    "contract_ticker": opt.get("symbol", ""),
                    "expiration": exp,
                    "strike": float(opt.get("strike", 0) or 0),
                    "type": (opt.get("option_type") or "").lower(),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": float(opt.get("last", 0) or 0),
                    "volume": int(opt.get("volume", 0) or 0),
                    "open_interest": int(opt.get("open_interest", 0) or 0),
                    "implied_volatility": float(opt.get("bid_iv", 0) or 0),
                    "delta": float(greeks.get("delta", 0) or 0),
                    "gamma": float(greeks.get("gamma", 0) or 0),
                    "theta": float(greeks.get("theta", 0) or 0),
                    "vega": float(greeks.get("vega", 0) or 0),
                    "bid_ask_spread_pct": (
                        (ask - bid) / mid * 100 if mid > 0 else 999.0
                    ),
                    "source": "tradier",
                }
                all_contracts.append(contract)

            time.sleep(0.3)

        except Exception as exc:
            logger.warning(f"Tradier chain fetch failed for {ticker}/{exp}: {exc}")

    if not all_contracts:
        return pd.DataFrame()

    df = pd.DataFrame(all_contracts)
    logger.info(f"{ticker}: {len(df)} contracts from Tradier")
    return df


# ─── yfinance Fallback ────────────────────────────────────────────────────────

def _fetch_yfinance(
    ticker: str,
    dte_min: int = 7,
    dte_max: int = 60,
) -> pd.DataFrame:
    """Fetch options chain from yfinance (free, no Greeks)."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options or []
    except Exception as exc:
        logger.error(f"yfinance Ticker() failed for {ticker}: {exc}")
        return pd.DataFrame()

    today = date.today()
    valid_exps = [
        e for e in expirations
        if dte_min <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= dte_max
    ][:4]

    if not valid_exps:
        return pd.DataFrame()

    all_contracts: list[dict] = []

    for exp in valid_exps:
        try:
            chain = stock.option_chain(exp)

            for opt_type, df_raw in [("call", chain.calls), ("put", chain.puts)]:
                for _, row in df_raw.iterrows():
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    mid = (bid + ask) / 2

                    contract = {
                        "ticker": ticker,
                        "contract_ticker": row.get("contractSymbol", ""),
                        "expiration": exp,
                        "strike": float(row.get("strike", 0) or 0),
                        "type": opt_type,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": float(row.get("lastPrice", 0) or 0),
                        "volume": int(row.get("volume", 0) or 0),
                        "open_interest": int(row.get("openInterest", 0) or 0),
                        "implied_volatility": float(row.get("impliedVolatility", 0) or 0),
                        "delta": 0.0,
                        "gamma": 0.0,
                        "theta": 0.0,
                        "vega": 0.0,
                        "bid_ask_spread_pct": (
                            (ask - bid) / mid * 100 if mid > 0 else 999.0
                        ),
                        "source": "yfinance",
                    }
                    all_contracts.append(contract)

            time.sleep(0.8)  # Be polite to Yahoo

        except Exception as exc:
            logger.warning(f"yfinance chain failed for {ticker}/{exp}: {exc}")

    if not all_contracts:
        return pd.DataFrame()

    df = pd.DataFrame(all_contracts)
    logger.info(f"{ticker}: {len(df)} contracts from yfinance")
    return df


# ─── Filters ──────────────────────────────────────────────────────────────────

def filter_liquid_options(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Filter options chain to only liquid, tradeable contracts.
    Requires: adequate volume OR open interest, tight bid-ask spread, non-zero bid.
    """
    if chain.empty:
        return chain

    mask = (
        (
            (chain["volume"] >= MIN_OPTION_VOLUME) |
            (chain["open_interest"] >= MIN_OPEN_INTEREST)
        ) &
        (chain["bid_ask_spread_pct"] <= MAX_BID_ASK_SPREAD_PCT) &
        (chain["bid"] > 0)
    )
    filtered = chain[mask].copy()
    logger.debug(f"Liquidity filter: {len(filtered)}/{len(chain)} contracts passed")
    return filtered


def add_dte_column(chain: pd.DataFrame) -> pd.DataFrame:
    """Add 'dte' column (days to expiration) to options chain."""
    if chain.empty or "expiration" not in chain.columns:
        return chain
    chain = chain.copy()
    today = pd.Timestamp.now().normalize()
    chain["expiration_dt"] = pd.to_datetime(chain["expiration"])
    chain["dte"] = (chain["expiration_dt"] - today).dt.days
    return chain


def get_atm_iv(chain: pd.DataFrame, current_price: float) -> float:
    """Extract the ATM implied volatility (average of call and put)."""
    if chain.empty:
        return 0.0
    chain_with_dte = add_dte_column(chain)
    # Prefer ~30 DTE
    near = chain_with_dte[
        (chain_with_dte["dte"] >= 20) & (chain_with_dte["dte"] <= 40)
    ]
    if near.empty:
        near = chain_with_dte

    atm = near.iloc[(near["strike"] - current_price).abs().argsort()[:4]]
    iv = atm["implied_volatility"].replace(0, np.nan).dropna()
    return float(iv.mean()) if len(iv) > 0 else 0.0
