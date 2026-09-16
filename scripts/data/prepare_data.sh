#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/data/prepare_data.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi

# ==================== Configuration ====================
BENCH="babyvision"
CACHE_ROOT="$HF_HUB_CACHE"
OUTPUT_ROOT="$DATA_DIR"  # Data is already exported; choose a new directory to export again.

exec uv run --frozen python -m data.prepare_data --bench "$BENCH" --cache-root "$CACHE_ROOT" --output-root "$OUTPUT_ROOT"
