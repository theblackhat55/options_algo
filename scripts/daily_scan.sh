#!/bin/bash
# scripts/daily_scan.sh
# Cron entry point for nightly scan.
# Run via OpenClaw cron at 02:30 UTC (9:30 PM ET) Tue-Sat

set -e

PROJECT_DIR="/root/options_algo"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python3"
LOG_FILE="${PROJECT_DIR}/logs/nightly_scan_$(date +%Y%m%d).log"

echo "[$(date -u)] Starting nightly scan" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

# Activate venv
source "${PROJECT_DIR}/.venv/bin/activate"

# Run nightly scan
"$VENV_PYTHON" -m src.pipeline.nightly_scan 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u)] Nightly scan complete" | tee -a "$LOG_FILE"
