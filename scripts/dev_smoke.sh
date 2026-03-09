#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Running tests"
pytest tests/ -q

echo "[2/3] Running nightly scan dry-run"
python -m src.pipeline.nightly_scan --dry-run

echo "[3/3] Listing feature outputs"
find data/features -type f | sort | tail -50
