#!/usr/bin/env python3
"""Integration test for IBKR real-time enrichment."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibkr_client import connect_ibkr, disconnect_ibkr
from src.data.ibkr_realtime import fetch_realtime_enrichment
from src.pipeline.nightly_scan import run_nightly_scan


def _print_enrichment(ticker: str, rt: dict) -> None:
    print(
        f"{ticker}: quality={rt.get('data_quality')} | "
        f"IV={rt.get('iv_pct')}% (call={rt.get('call_iv')}% / put={rt.get('put_iv')}%) | "
        f"Flow score={rt.get('flow_score')} dom={rt.get('dominant_side')} | "
        f"Vol pace={rt.get('volume_pace')}x"
    )
    unusual = rt.get("unusual_strikes", [])
    if unusual:
        top = unusual[:3]
        summary = ", ".join(
            [
                f"{u['type'].upper()} {u['strike']} ({u['vol_oi_ratio']:.2f}x)"
                for u in top
            ]
        )
        print(f"  Unusual: {summary}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    status = 0

    ib = connect_ibkr()
    if not ib:
        print("IBKR unavailable — skipping real-time sampling (partial pass)")
        status = 2
    else:
        try:
            for ticker in ("AAPL", "MSFT"):
                rt = fetch_realtime_enrichment(ib, ticker)
                if not rt or rt.get("data_quality") == "NONE":
                    status = 2
                    print(f"{ticker}: No real-time enrichment (likely outside market hours)")
                else:
                    _print_enrichment(ticker, rt)
        finally:
            disconnect_ibkr(ib)

    # Dry-run nightly scan to ensure options_flow context exists downstream
    print("\nRunning nightly scan dry-run (limited universe)...")
    signal = run_nightly_scan(universe_override=["AAPL", "MSFT", "SPY"], dry_run=True)
    picks = signal.get("top_picks", [])
    flow_context = sum(1 for p in picks if p.get("context", {}).get("options_flow"))
    print(f"Dry-run produced {len(picks)} picks; {flow_context} include options_flow context.")
    if picks and flow_context == 0:
        status = max(status, 2)

    if status == 0:
        print("Real-time enrichment tests passed ✅")
    elif status == 2:
        print("Partial pass — data limited (pre/post-market?)")

    return status


if __name__ == "__main__":
    sys.exit(main())
