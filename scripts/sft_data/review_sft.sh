#!/usr/bin/env bash
# Run: bash scripts/sft_data/review_sft.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
STAGE="render" # render or admit
RECORDS="$WORK_DIR/sft/cosyn/candidates.jsonl"
OUTPUT="$WORK_DIR/sft/cosyn_review" # admit: use a fresh .jsonl file
LIMIT=5
DECISIONS="$WORK_DIR/sft/cosyn_review/decisions.jsonl"
BATCH_REVIEW="$WORK_DIR/sft/cosyn_batch_review.json"
# =======================================================
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
args=("$STAGE" --records "$RECORDS" --data-root "$DATA_DIR" --output "$OUTPUT" --limit "$LIMIT")
if [[ "$STAGE" == "admit" ]]; then args+=(--decisions "$DECISIONS" --batch-review "$BATCH_REVIEW"); fi
exec uv run --frozen python -m data.sft.tools.review "${args[@]}"
