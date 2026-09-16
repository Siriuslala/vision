#!/usr/bin/env bash
# Run: bash scripts/sft_data/viewer.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
CONFIG="$ROOT_DIR/scripts/sft_data/viewer_config.json"
HOST="127.0.0.1"
PORT=1043
# =======================================================
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
exec uv run --frozen python -m data.sft.viewer.server --config "$CONFIG" --host "$HOST" --port "$PORT"
