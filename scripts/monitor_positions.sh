#!/usr/bin/env bash
# ============================================================================
# scripts/monitor_positions.sh
# Cron wrapper for the position lifecycle monitor.
#
# Cron (every 30 min during market hours Mon-Fri):
#   */30 9-16 * * 1-5  /path/to/.venv/bin/bash /path/to/scripts/monitor_positions.sh \
#                          >> /var/log/monitor_positions.log 2>&1
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${REPO_ROOT}/.venv"

# Activate virtualenv if present
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

cd "$REPO_ROOT"

echo "=== position_monitor run: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
python3 -m src.pipeline.position_monitor "$@"
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: position_monitor exited with code $EXIT_CODE" >&2
fi

exit $EXIT_CODE
