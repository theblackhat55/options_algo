#!/bin/bash
# Daily pre-market scan
# Runs stock data update, IV update from flat files, then nightly scan

set -e
cd /root/options_algo
PYTHON=/root/options_algo/.venv/bin/python
LOG=/root/options_algo/logs/daily_run_$(date +%Y%m%d).log

echo "=== Daily run started: $(date) ===" >> "$LOG"

# Step 1: Update stock data
echo "Step 1: Updating stock data..." >> "$LOG"
$PYTHON -m src.data.stock_fetcher >> "$LOG" 2>&1

# Step 2: Update today's IV from flat files (yesterday's options data)
echo "Step 2: Updating IV..." >> "$LOG"
$PYTHON scripts/backfill_iv_from_flatfiles.py --start $(date -d "yesterday" +%Y-%m-%d) --end $(date +%Y-%m-%d) >> "$LOG" 2>&1

# Step 3: Run nightly scan
echo "Step 3: Running scan..." >> "$LOG"
$PYTHON -m src.pipeline.nightly_scan >> "$LOG" 2>&1

echo "=== Daily run complete: $(date) ===" >> "$LOG"
