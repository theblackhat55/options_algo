#!/bin/bash
# scripts/morning_report.sh
# Cron entry point for morning brief WhatsApp delivery.
# Run via OpenClaw at 14:00 UTC (9:00 AM ET) Mon-Fri

set -e

PROJECT_DIR="/root/options_algo"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python3"
LOG_FILE="${PROJECT_DIR}/logs/morning_brief_$(date +%Y%m%d).log"

echo "[$(date -u)] Generating morning brief" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"
source "${PROJECT_DIR}/.venv/bin/activate"

"$VENV_PYTHON" -m src.pipeline.morning_brief 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u)] Morning brief complete" | tee -a "$LOG_FILE"
