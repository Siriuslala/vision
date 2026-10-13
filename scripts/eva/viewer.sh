#!/usr/bin/env bash
# Edit configuration, then run from the repository root:
# bash scripts/eva/viewer.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
# List exact result files, or retain the glob to show all evaluation JSONL files.
shopt -s nullglob
JSONL_FILES=(
    "$WORK_DIR"/eval/qwen35_9b_babyvision-thinkingoff-gpua800-random_seed42-do_sampletrue-temperature1.0-top_p1.0-top_k40-min_p0.0-repetition_penalty1.0.jsonl
    "$WORK_DIR"/eval/qwen35_9b_babyvision-thinkingon-gpua800-random_seed42-do_sampletrue-temperature1.0-top_p0.95-top_k20-min_p0.0-repetition_penalty1.0.jsonl
)
MANIFEST=""  # Optional: "$DATA_DIR/BabyVision/train.jsonl" fills missing reference fields.
HOST="127.0.0.1"
PORT=1045
# =======================================================
if (( $# > 0 )); then echo 'Edit the script configuration instead of passing arguments.' >&2; exit 2; fi
if (( ${#JSONL_FILES[@]} == 0 )); then echo 'No prediction JSONL files selected. Set JSONL_FILES in this script.' >&2; exit 2; fi
ARGS=(--jsonl "${JSONL_FILES[@]}" --host "$HOST" --port "$PORT")
[[ -z "$MANIFEST" ]] || ARGS+=(--manifest "$MANIFEST")
exec uv run --frozen python -m eva.viewer.server "${ARGS[@]}"
