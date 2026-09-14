#!/usr/bin/env bash
# Run: bash scripts/sft_data/produce_thyme.sh
set -euo pipefail
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

# Edit these values before a production run; zero scans all completed shards.
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/Thyme-SFT"
OUTPUT_ROOT="$WORK_DIR/sft/thyme"
IMAGE_OUTPUT="$DATA_DIR/sft_images/thyme"
LIMIT=10

uv run --frozen python -m data.sft.tools.produce \
  --dataset thyme --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" --data-root "$DATA_DIR" \
  --image-output "$IMAGE_OUTPUT" --limit "$LIMIT"
