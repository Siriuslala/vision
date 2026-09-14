#!/usr/bin/env bash
# Run: bash scripts/sft_data/produce_visual_cot.sh
set -euo pipefail
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi

# Configure this block before each run. 0 processes all completed annotation rows.
LIMIT=10
RUN_NAME="visual_cot_pilot"
EXTRACT_IMAGES=false
EXTRACT_LIMIT=10

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT"
IMAGE_CACHE="$WORK_DIR/sft/visual_cot_extracted"
OUTPUT_ROOT="$WORK_DIR/sft/$RUN_NAME"
IMAGE_OUTPUT="$DATA_DIR/sft_images/$RUN_NAME"

if [[ "$EXTRACT_IMAGES" == true ]]; then
  uv run --frozen python -m data.sft.datasets.visual_cot \
    --source-root "$SOURCE_ROOT" --output-root "$IMAGE_CACHE" --limit "$EXTRACT_LIMIT"
fi
uv run --frozen python -m data.sft.tools.produce \
  --dataset visual_cot --source-root "$SOURCE_ROOT" --image-cache "$IMAGE_CACHE" \
  --output-root "$OUTPUT_ROOT" --data-root "$DATA_DIR" \
  --image-output "$IMAGE_OUTPUT" --limit "$LIMIT"
