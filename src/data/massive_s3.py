"""
src/data/massive_s3.py
======================
Download and parse Polygon/Massive flat files from S3.

Handles:
  - us_stocks_sip/day_aggs_v1  → Stock daily OHLCV
  - us_options_opra/day_aggs_v1 → Options daily OHLCV (OPRA tickers)

Usage:
    from src.data.massive_s3 import download_stock_day, download_options_day, parse_opra_ticker
"""
from __future__ import annotations

import gzip
import io
import logging
import os
import re
from datetime import datetime, date
from typing import Optional

import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ─── S3 Client (lazy singleton) ──────────────────────────────────────────────

_s3_client = None


def _get_s3() -> boto3.client:
    """Return a cached boto3 S3 client configured for Massive."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com"),
            aws_access_key_id=os.getenv("MASSIVE_S3_ACCESS_KEY", ""),
            aws_secret_access_key=os.getenv("MASSIVE_S3_SECRET_KEY", ""),
            config=Config(signature_version="s3v4"),
        )
    return _s3_client


BUCKET = os.getenv("MASSIVE_S3_BUCKET", "flatfiles")


# ─── OPRA Ticker Parser ──────────────────────────────────────────────────────
# Format: O:AAPL250321C00170000
#   O:          prefix
#   AAPL        underlying (variable length, 1-6 chars)
#   250321      YYMMDD expiration
#   C           C=call, P=put
#   00170000    strike × 1000 (8 digits, zero-padded)

_OPRA_RE = re.compile(
    r"^O:(?P<underlying>[A-Z]+)"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])"
    r"(?P<strike>\d{8})$"
)


def parse_opra_ticker(opra: str) -> Optional[dict]:
    """
    Parse a Polygon OPRA option ticker into components.

    Example:
        parse_opra_ticker("O:AAPL250321C00170000")
        → {"underlying": "AAPL", "expiration": "2025-03-21",
           "type": "call", "strike": 170.0}
    """
    m = _OPRA_RE.match(opra)
    if not m:
        return None
    yy, mm, dd = m.group("yy"), m.group("mm"), m.group("dd")
    return {
        "underlying": m.group("underlying"),
        "expiration": f"20{yy}-{mm}-{dd}",
        "type": "call" if m.group("cp") == "C" else "put",
        "strike": int(m.group("strike")) / 1000.0,
    }


# ─── Stock Day Aggregates ────────────────────────────────────────────────────

def download_stock_day(
    trade_date: date,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download one day's stock OHLCV from the flat files.

    Args:
        trade_date: The trading day to fetch.
        tickers: If provided, filter to only these tickers.

    Returns:
        DataFrame with columns: ticker, open, high, low, close, volume, transactions
        Empty DataFrame if the file doesn't exist (weekend/holiday).
    """
    key = (
        f"us_stocks_sip/day_aggs_v1/"
        f"{trade_date.year}/{trade_date.month:02d}/"
        f"{trade_date.isoformat()}.csv.gz"
    )
    df = _download_and_parse_csv(key)
    if df.empty:
        return df

    # Rename columns to match algo conventions
    df.columns = [c.strip().lower() for c in df.columns]

    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        df = df[df["ticker"].isin(tickers_upper)]

    # Convert types
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    if "transactions" in df.columns:
        df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype(int)

    df["date"] = trade_date
    return df


# ─── Options Day Aggregates ──────────────────────────────────────────────────

def download_options_day(
    trade_date: date,
    underlyings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download one day's options OHLCV from the flat files.

    Args:
        trade_date: The trading day to fetch.
        underlyings: If provided, filter to only contracts on these underlyings.

    Returns:
        DataFrame with columns:
            opra_ticker, underlying, expiration, type, strike,
            open, high, low, close, volume, transactions
        Empty DataFrame if file doesn't exist.
    """
    key = (
        f"us_options_opra/day_aggs_v1/"
        f"{trade_date.year}/{trade_date.month:02d}/"
        f"{trade_date.isoformat()}.csv.gz"
    )
    df = _download_and_parse_csv(key)
    if df.empty:
        return df

    df.columns = [c.strip().lower() for c in df.columns]

    # Parse OPRA tickers to extract underlying, expiration, type, strike
    parsed = df["ticker"].apply(parse_opra_ticker)
    valid_mask = parsed.notna()
    df = df[valid_mask].copy()
    parsed = parsed[valid_mask]

    df["opra_ticker"] = df["ticker"]
    df["underlying"] = parsed.apply(lambda x: x["underlying"])
    df["expiration"] = parsed.apply(lambda x: x["expiration"])
    df["type"] = parsed.apply(lambda x: x["type"])
    df["strike"] = parsed.apply(lambda x: x["strike"])

    # Filter to requested underlyings
    if underlyings:
        underlyings_upper = {t.upper() for t in underlyings}
        df = df[df["underlying"].isin(underlyings_upper)]

    # Convert price/volume columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    df["trade_date"] = trade_date
    return df


# ─── List Available Dates ─────────────────────────────────────────────────────

def list_available_dates(
    data_type: str = "stocks",
    year: int | None = None,
    month: int | None = None,
) -> list[date]:
    """
    List trading dates available in the flat files.

    Args:
        data_type: "stocks" or "options"
        year: Optional year filter.
        month: Optional month filter (requires year).

    Returns:
        Sorted list of available dates.
    """
    prefix_map = {
        "stocks": "us_stocks_sip/day_aggs_v1/",
        "options": "us_options_opra/day_aggs_v1/",
    }
    prefix = prefix_map.get(data_type, prefix_map["stocks"])
    if year:
        prefix += f"{year}/"
        if month:
            prefix += f"{month:02d}/"

    s3 = _get_s3()
    dates = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Key ends with YYYY-MM-DD.csv.gz
            fname = obj["Key"].split("/")[-1]
            if fname.endswith(".csv.gz"):
                date_str = fname.replace(".csv.gz", "")
                try:
                    dates.append(date.fromisoformat(date_str))
                except ValueError:
                    pass
    return sorted(dates)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _download_and_parse_csv(key: str) -> pd.DataFrame:
    """Download a gzipped CSV from S3 and return as DataFrame."""
    s3 = _get_s3()
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        raw = obj["Body"].read()
        with gzip.open(io.BytesIO(raw), "rt") as f:
            df = pd.read_csv(f)
        logger.debug(f"Downloaded {key}: {len(df)} rows")
        return df
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404"):
            logger.debug(f"No flat file for {key} (weekend/holiday)")
        else:
            logger.warning(f"S3 download failed for {key}: {exc}")
        return pd.DataFrame()
    except Exception as exc:
        logger.warning(f"Failed to parse {key}: {exc}")
        return pd.DataFrame()
