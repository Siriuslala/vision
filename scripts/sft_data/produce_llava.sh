#!/usr/bin/env bash
# Run: bash scripts/sft_data/produce_llava.sh
set -euo pipefail
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

# Edit these values; zero scans all completed shards. Keep downloads immutable.
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/LLaVA-OneVision-Data"
OUTPUT_ROOT="$WORK_DIR/sft/llava"
IMAGE_OUTPUT="$DATA_DIR/sft_images/llava"
ANNOTATIONS="$WORK_DIR/sft/llava_annotations.jsonl"
LIMIT=10

uv run --frozen python -m data.sft.tools.produce \
  --dataset llava --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" --data-root "$DATA_DIR" \
  --image-output "$IMAGE_OUTPUT" --limit "$LIMIT" --annotations "$ANNOTATIONS"
