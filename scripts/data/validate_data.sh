#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/data/validate_data.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi

# ==================== Configuration ====================
BENCH="babyvision"
DATA_ROOT="$DATA_DIR"

exec uv run --frozen python -m data.validate_data --bench "$BENCH" --data-root "$DATA_ROOT"
