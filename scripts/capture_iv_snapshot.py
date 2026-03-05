#!/usr/bin/env python3
"""
scripts/capture_iv_snapshot.py
================================
Nightly IV snapshot collector.
Fetches ATM implied volatility + Greeks + open interest from Polygon
for every ticker in the universe and appends a row to per-ticker
Parquet files in data/processed/iv_snapshots/.

After 20+ trading days the snapshot history replaces the HV×1.15 proxy
in src/analysis/volatility.analyze_iv(), giving a more accurate IV rank.

Schedule (cron) — run after market close, e.g. 4:15 PM ET:
    15 16 * * 1-5 /path/to/venv/bin/python /path/to/scripts/capture_iv_snapshot.py >> /var/log/iv_snapshot.log 2>&1

Output:
    data/processed/iv_snapshots/{TICKER}_iv_history.parquet
    Each Parquet has columns: date (str, YYYY-MM-DD), atm_iv (float, %),
    call_iv (float, %), put_iv (float, %), skew (float),
    avg_oi (float), avg_volume (float), source (str: polygon/tradier/yfinance).

Usage:
    python scripts/capture_iv_snapshot.py
    python scripts/capture_iv_snapshot.py --tickers AAPL MSFT NVDA
    python scripts/capture_iv_snapshot.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ── Setup ──────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import IV_SNAPSHOT_DIR, LOG_LEVEL
from config.universe import get_universe

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("capture_iv_snapshot")

SNAPSHOT_DIR = Path(IV_SNAPSHOT_DIR)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# DTE window for ATM IV measurement
DTE_MIN = 20
DTE_MAX = 45
MAX_RETRIES = 2
RATE_LIMIT_PAUSE = 0.15   # seconds between Polygon calls


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly IV snapshot collector")
    parser.add_argument("--tickers", nargs="*", help="Override ticker list")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write Parquet files")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if today's row already exists")
    args = parser.parse_args()

    tickers = args.tickers or get_universe()
    today = date.today().isoformat()

    logger.info(f"IV snapshot capture started — {len(tickers)} tickers, date={today}")
    t0 = time.time()

    ok = skip = fail = 0
    for i, ticker in enumerate(tickers, 1):
        snap_file = SNAPSHOT_DIR / f"{ticker}_iv_history.parquet"

        # Skip if today's row already exists (unless --force)
        if not args.force and snap_file.exists():
            try:
                existing = pd.read_parquet(snap_file)
                if today in existing.index.astype(str).tolist():
                    logger.debug(f"  [{i}/{len(tickers)}] {ticker}: today already captured, skipping")
                    skip += 1
                    continue
            except Exception:
                pass  # File corrupt — re-fetch

        row = _fetch_iv_row(ticker, today)
        if row is None:
            logger.warning(f"  [{i}/{len(tickers)}] {ticker}: fetch failed — skipped")
            fail += 1
            continue

        if not args.dry_run:
            _append_row(snap_file, row)
            ok += 1
        else:
            logger.info(f"  [dry-run] {ticker}: atm_iv={row['atm_iv']:.1f}% skew={row['skew']:+.1f}")
            ok += 1

        time.sleep(RATE_LIMIT_PAUSE)

    elapsed = round(time.time() - t0, 1)
    logger.info(
        f"IV snapshot complete in {elapsed}s — "
        f"{ok} captured, {skip} skipped (already up-to-date), {fail} failed"
    )


# ─── IV Fetch ─────────────────────────────────────────────────────────────────

def _fetch_iv_row(ticker: str, today: str) -> dict | None:
    """
    Fetch ATM IV for a ticker.
    Tries: Polygon → Tradier → yfinance.
    Returns a dict with keys: date, atm_iv, call_iv, put_iv, skew, avg_oi, avg_volume, source.
    """
    for attempt in range(MAX_RETRIES):
        try:
            row = _from_polygon(ticker, today)
            if row:
                return row
        except Exception as exc:
            logger.debug(f"  {ticker} Polygon attempt {attempt+1} failed: {exc}")
            time.sleep(0.5)

    try:
        row = _from_tradier(ticker, today)
        if row:
            return row
    except Exception as exc:
        logger.debug(f"  {ticker} Tradier failed: {exc}")

    try:
        row = _from_yfinance(ticker, today)
        if row:
            return row
    except Exception as exc:
        logger.debug(f"  {ticker} yfinance failed: {exc}")

    return None


def _from_polygon(ticker: str, today: str) -> dict | None:
    """Fetch options chain from Polygon REST API and compute ATM IV."""
    from config.settings import POLYGON_API_KEY
    if not POLYGON_API_KEY:
        return None

    import requests

    exp_from = pd.Timestamp.today() + pd.Timedelta(days=DTE_MIN)
    exp_to   = pd.Timestamp.today() + pd.Timedelta(days=DTE_MAX)

    url = "https://api.polygon.io/v3/snapshot/options/" + ticker
    params = {
        "expiration_date.gte": exp_from.strftime("%Y-%m-%d"),
        "expiration_date.lte": exp_to.strftime("%Y-%m-%d"),
        "contract_type": "call",
        "limit": 50,
        "apiKey": POLYGON_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("results", [])

    if not data:
        return None

    # Get spot price
    spot_url = f"https://api.polygon.io/v2/last/trade/{ticker}"
    spot_resp = requests.get(spot_url, params={"apiKey": POLYGON_API_KEY}, timeout=10)
    spot_price = None
    if spot_resp.ok:
        spot_price = spot_resp.json().get("results", {}).get("p")

    calls, puts = [], []
    for c in data:
        details = c.get("details", {})
        greeks  = c.get("greeks", {})
        day     = c.get("day", {})
        iv      = c.get("implied_volatility", 0)
        if iv and iv > 0:
            row_iv = iv * 100 if iv < 1 else iv
            contract_type = details.get("contract_type", "call")
            oi = day.get("open_interest", 0) or c.get("open_interest", 0)
            vol = day.get("volume", 0) or 0
            calls.append({"iv": row_iv, "strike": details.get("strike_price", 0), "oi": oi, "vol": vol})

    # Fetch puts too
    params["contract_type"] = "put"
    time.sleep(0.12)
    resp2 = requests.get(url, params=params, timeout=15)
    if resp2.ok:
        for c in resp2.json().get("results", []):
            iv = c.get("implied_volatility", 0)
            if iv and iv > 0:
                details = c.get("details", {})
                day = c.get("day", {})
                row_iv = iv * 100 if iv < 1 else iv
                oi = day.get("open_interest", 0) or 0
                vol = day.get("volume", 0) or 0
                puts.append({"iv": row_iv, "strike": details.get("strike_price", 0), "oi": oi, "vol": vol})

    if not calls:
        return None

    call_ivs = [c["iv"] for c in calls]
    put_ivs  = [p["iv"] for p in puts] if puts else call_ivs
    all_oi   = [c["oi"] for c in calls + puts if c["oi"] > 0]
    all_vol  = [c["vol"] for c in calls + puts if c["vol"] > 0]

    atm_call_iv = _atm_iv(calls, spot_price)
    atm_put_iv  = _atm_iv(puts,  spot_price) if puts else atm_call_iv

    return {
        "date": today,
        "atm_iv": round((atm_call_iv + atm_put_iv) / 2, 2),
        "call_iv": round(atm_call_iv, 2),
        "put_iv": round(atm_put_iv, 2),
        "skew": round(atm_put_iv - atm_call_iv, 2),
        "avg_oi": round(sum(all_oi) / len(all_oi), 0) if all_oi else 0,
        "avg_volume": round(sum(all_vol) / len(all_vol), 0) if all_vol else 0,
        "source": "polygon",
    }


def _from_tradier(ticker: str, today: str) -> dict | None:
    """Fetch IV from Tradier options chain."""
    from config.settings import TRADIER_API_KEY
    if not TRADIER_API_KEY:
        return None

    import requests

    # Get expirations
    exp_url = "https://api.tradier.com/v1/markets/options/expirations"
    headers = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}
    resp = requests.get(exp_url, params={"symbol": ticker}, headers=headers, timeout=10)
    if not resp.ok:
        return None

    exps = resp.json().get("expirations", {}).get("date", [])
    if not exps:
        return None

    today_ts = pd.Timestamp.today()
    valid_exps = [
        e for e in exps
        if DTE_MIN <= (pd.Timestamp(e) - today_ts).days <= DTE_MAX
    ]
    if not valid_exps:
        return None

    exp = valid_exps[0]
    chain_url = "https://api.tradier.com/v1/markets/options/chains"
    time.sleep(0.3)
    cresp = requests.get(chain_url,
                         params={"symbol": ticker, "expiration": exp, "greeks": "true"},
                         headers=headers, timeout=15)
    if not cresp.ok:
        return None

    options = cresp.json().get("options", {}).get("option", [])
    if not options:
        return None

    calls = [o for o in options if o.get("option_type") == "call" and o.get("greeks", {}).get("smv_vol")]
    puts  = [o for o in options if o.get("option_type") == "put"  and o.get("greeks", {}).get("smv_vol")]

    # Get spot for ATM
    quote_url = "https://api.tradier.com/v1/markets/quotes"
    qresp = requests.get(quote_url, params={"symbols": ticker}, headers=headers, timeout=10)
    spot = None
    if qresp.ok:
        q = qresp.json().get("quotes", {}).get("quote", {})
        spot = q.get("last") or q.get("bid")

    def to_iv_list(lst, spot):
        return [{"iv": float(o["greeks"]["smv_vol"]) * 100, "strike": float(o.get("strike", 0))} for o in lst]

    call_list = to_iv_list(calls, spot)
    put_list  = to_iv_list(puts,  spot)
    if not call_list:
        return None

    atm_call = _atm_iv(call_list, spot)
    atm_put  = _atm_iv(put_list,  spot) if put_list else atm_call

    return {
        "date": today,
        "atm_iv": round((atm_call + atm_put) / 2, 2),
        "call_iv": round(atm_call, 2),
        "put_iv": round(atm_put, 2),
        "skew": round(atm_put - atm_call, 2),
        "avg_oi": 0,
        "avg_volume": 0,
        "source": "tradier",
    }


def _from_yfinance(ticker: str, today: str) -> dict | None:
    """Fallback: fetch IV from yfinance options chain."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        return None

    today_ts = pd.Timestamp.today()
    valid_exps = [
        e for e in exps
        if DTE_MIN <= (pd.Timestamp(e) - today_ts).days <= DTE_MAX
    ]
    if not valid_exps:
        return None

    chain = t.option_chain(valid_exps[0])
    calls = chain.calls
    puts  = chain.puts

    hist = t.history(period="1d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else None

    def df_to_iv_list(df):
        if "impliedVolatility" not in df.columns:
            return []
        return [
            {"iv": float(r["impliedVolatility"]) * 100, "strike": float(r["strike"])}
            for _, r in df.iterrows()
            if float(r.get("impliedVolatility", 0)) > 0
        ]

    call_list = df_to_iv_list(calls)
    put_list  = df_to_iv_list(puts)
    if not call_list:
        return None

    atm_call = _atm_iv(call_list, spot)
    atm_put  = _atm_iv(put_list,  spot) if put_list else atm_call

    return {
        "date": today,
        "atm_iv": round((atm_call + atm_put) / 2, 2),
        "call_iv": round(atm_call, 2),
        "put_iv": round(atm_put, 2),
        "skew": round(atm_put - atm_call, 2),
        "avg_oi": 0,
        "avg_volume": 0,
        "source": "yfinance",
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atm_iv(contracts: list[dict], spot: float | None) -> float:
    """Compute average IV for the 2 strikes nearest ATM."""
    if not contracts:
        return 0.0
    if spot is None:
        return round(sum(c["iv"] for c in contracts) / len(contracts), 2)

    sorted_c = sorted(contracts, key=lambda c: abs(c["strike"] - spot))
    nearest = sorted_c[:4]
    ivs = [c["iv"] for c in nearest if c["iv"] > 0]
    return round(sum(ivs) / len(ivs), 2) if ivs else 0.0


def _append_row(snap_file: Path, row: dict) -> None:
    """Append a new row to the per-ticker Parquet file."""
    new_df = pd.DataFrame([row]).set_index("date")
    new_df.index = pd.to_datetime(new_df.index)

    if snap_file.exists():
        try:
            existing = pd.read_parquet(snap_file)
            # Remove any existing row for today before appending
            existing = existing[~existing.index.astype(str).str.startswith(row["date"][:10])]
            combined = pd.concat([existing, new_df]).sort_index()
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(snap_file)
    logger.debug(f"  Saved {snap_file.name} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
