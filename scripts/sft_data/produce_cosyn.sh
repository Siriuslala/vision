#!/usr/bin/env bash
# Run: bash scripts/sft_data/produce_cosyn.sh
set -euo pipefail
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

# Limit counts QA candidates, not images; zero scans all completed shards.
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/CoSyn-400K"
OUTPUT_ROOT="$WORK_DIR/sft/cosyn"
IMAGE_OUTPUT="$DATA_DIR/sft_images/cosyn"
ANNOTATIONS="$WORK_DIR/sft_annotations/cosyn.jsonl"
LIMIT=10

uv run --frozen python -m data.sft.tools.produce \
  --dataset cosyn --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" --data-root "$DATA_DIR" \
  --image-output "$IMAGE_OUTPUT" --annotations "$ANNOTATIONS" --limit "$LIMIT"
