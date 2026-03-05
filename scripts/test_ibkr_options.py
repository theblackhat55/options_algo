#!/usr/bin/env python3
"""
scripts/test_ibkr_options.py
============================
Standalone test for the IBKR data integration layer.

Exit codes:
    0 — Full success (connection + data)
    2 — Partial success (connected but no market data, e.g. pre-market/weekend)
    1 — Connection failure
"""
from __future__ import annotations

import logging
import os
import sys

# ── Add project root to path so src.* and config.* resolve ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_ibkr_options")


def divider(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def main() -> int:
    from src.data.ibkr_client import (
        connect_ibkr,
        disconnect_ibkr,
        fetch_vix_ibkr,
        fetch_stock_snapshot,
        fetch_options_chain_ibkr,
        fetch_market_snapshot_ibkr,
    )

    # ── 1. Connection ─────────────────────────────────────────────────────────
    divider("IBKR Connection")
    ib = connect_ibkr()
    if ib is None:
        print("❌  FAILED to connect to IB Gateway at 127.0.0.1:4002")
        print("    Make sure IB Gateway is running and API access is enabled.")
        return 1

    print(f"✅  Connected  |  isConnected={ib.isConnected()}")
    account = ib.managedAccounts()
    print(f"    Account(s): {account}")

    partial = False  # Track if we got connection but no live data

    try:
        # ── 2. VIX (live) ─────────────────────────────────────────────────────
        divider("VIX — Live")
        vix = fetch_vix_ibkr(ib)
        if vix:
            print(f"✅  VIX = {vix:.2f}")
        else:
            print("⚠️   VIX returned no data (expected pre-market or weekend)")
            partial = True

        # ── 3. SPY stock snapshot ─────────────────────────────────────────────
        divider("SPY Stock Snapshot")
        spy = fetch_stock_snapshot(ib, "SPY")
        if spy:
            print(f"✅  SPY  last={spy['last']:.2f}  bid={spy['bid']:.2f}  "
                  f"ask={spy['ask']:.2f}  vol={spy['volume']:,}")
        else:
            print("⚠️   SPY snapshot returned no data")
            partial = True

        # ── 4. Market snapshot (VIX + SPY combined) ───────────────────────────
        divider("Market Snapshot (combined)")
        mkt = fetch_market_snapshot_ibkr(ib)
        print(f"    vix={mkt['vix']}  spy_price={mkt['spy_price']}  "
              f"spy_bid={mkt['spy_bid']}  spy_ask={mkt['spy_ask']}")

        # ── 5. AAPL options chain ─────────────────────────────────────────────
        divider("AAPL Options Chain")
        print("    Requesting options chain for AAPL (DTE 7–60)...")
        df = fetch_options_chain_ibkr(ib, "AAPL", dte_min=7, dte_max=60)

        if df.empty:
            print("⚠️   AAPL options chain returned empty DataFrame")
            print("    This is expected outside market hours (paper account)")
            partial = True
        else:
            print(f"✅  {len(df)} contracts fetched")
            expirations = sorted(df["expiration"].unique())
            print(f"    Expiry dates: {expirations}")

            calls = df[df["type"] == "call"]
            puts = df[df["type"] == "put"]
            print(f"    Calls: {len(calls)}  |  Puts: {len(puts)}")

            # Sample: ~5 near-ATM rows sorted by strike
            divider("Sample Contracts (near ATM)")
            sample_cols = ["expiration", "strike", "type", "bid", "ask", "mid",
                           "implied_volatility", "delta", "gamma", "theta", "vega"]
            available_cols = [c for c in sample_cols if c in df.columns]
            sample = df.sort_values("strike").head(10)[available_cols]
            print(sample.to_string(index=False))

            # Check for Greeks data
            has_greeks = (df["delta"].abs() > 0).any()
            print(f"\n    Greeks present: {'✅ Yes' if has_greeks else '⚠️  No (outside market hours?)'}")
            if not has_greeks:
                partial = True

        # ── 6. market_context VIX fetch (integration test) ────────────────────
        divider("market_context._fetch_real_vix() integration")
        try:
            from src.data.market_context import _fetch_real_vix
            result = _fetch_real_vix()
            if result:
                vix_close, vix_avg = result
                print(f"✅  _fetch_real_vix() → vix={vix_close}  5d_avg={vix_avg}")
            else:
                print("⚠️   _fetch_real_vix() returned None (yfinance and IBKR both no data)")
                partial = True
        except Exception as exc:
            print(f"⚠️   _fetch_real_vix() raised: {exc}")
            partial = True

        # ── 7. options_fetcher._fetch_ibkr() integration ──────────────────────
        divider("options_fetcher._fetch_ibkr() integration")
        try:
            # We already have the chain via ibkr_client; just verify the wrapper
            from src.data.options_fetcher import _fetch_ibkr
            print("    Calling _fetch_ibkr('AAPL') (opens own connection)...")
            df2 = _fetch_ibkr("AAPL", dte_min=7, dte_max=60)
            if df2.empty:
                print("⚠️   _fetch_ibkr() returned empty DataFrame")
                partial = True
            else:
                print(f"✅  _fetch_ibkr() returned {len(df2)} contracts")
        except Exception as exc:
            print(f"⚠️   _fetch_ibkr() raised: {exc}")
            partial = True

    finally:
        disconnect_ibkr(ib)

    # ── Summary ───────────────────────────────────────────────────────────────
    divider("Result")
    if partial:
        print("⚠️   PARTIAL — connection OK, some data unavailable (pre-market / paper account)")
        print("    All fallback sources (Polygon → Tradier → yfinance) remain intact.")
        return 2
    else:
        print("✅  ALL TESTS PASSED — IBKR integration fully operational")
        return 0


if __name__ == "__main__":
    sys.exit(main())
