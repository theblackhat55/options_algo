from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import DATA_DIR

FEATURES_DIR = DATA_DIR / "features"

STOCKS_DIR = FEATURES_DIR / "stocks"
OPTIONS_DIR = FEATURES_DIR / "options"
MARKET_DIR = FEATURES_DIR / "market"
CANDIDATES_DIR = FEATURES_DIR / "candidates"
RECOMMENDATIONS_DIR = FEATURES_DIR / "recommendations"
RUN_METADATA_DIR = FEATURES_DIR / "metadata" / "runs"

for d in [
    STOCKS_DIR,
    OPTIONS_DIR,
    MARKET_DIR,
    CANDIDATES_DIR,
    RECOMMENDATIONS_DIR,
    RUN_METADATA_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_stock_features(ticker: str, as_of_date: str, rows: list[dict]) -> Path | None:
    df = _safe_df(rows)
    if df.empty:
        return None
    out_dir = STOCKS_DIR / f"ticker={ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"date={as_of_date}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_options_features(ticker: str, as_of_date: str, rows: list[dict]) -> Path | None:
    df = _safe_df(rows)
    if df.empty:
        return None
    out_dir = OPTIONS_DIR / f"ticker={ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"date={as_of_date}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_market_features(as_of_date: str, rows: list[dict]) -> Path | None:
    df = _safe_df(rows)
    if df.empty:
        return None
    out_path = MARKET_DIR / f"date={as_of_date}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_candidate_features(as_of_date: str, ticker: str, rows: list[dict]) -> Path | None:
    df = _safe_df(rows)
    if df.empty:
        return None
    out_dir = CANDIDATES_DIR / f"scan_date={as_of_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ticker={ticker}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_recommendation_features(as_of_date: str, rows: list[dict]) -> Path | None:
    df = _safe_df(rows)
    if df.empty:
        return None
    out_path = RECOMMENDATIONS_DIR / f"date={as_of_date}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_run_metadata(as_of_date: str, payload: dict[str, Any]) -> Path:
    out_path = RUN_METADATA_DIR / f"scan_date={as_of_date}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path
