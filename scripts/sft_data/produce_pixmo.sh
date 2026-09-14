#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
STAGE="prepare" # prepare: acquire/audit points; convert: consume reviewed boxes.
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/pixmo-count"
PREPARE_ROOT="$DATA_DIR/sft_annotations/pixmo_trial"
OUTPUT_ROOT="$WORK_DIR/sft/pixmo_trial"
AUDIT_ROOT="$DATA_DIR/sft_annotations/pixmo_candidates_trial" # Reviewed point-window output.
LIMIT=3 # 0 explicitly processes every available training record.
DOWNLOAD_IMAGES=true
# =======================================================
if (( $# > 0 )); then
  echo "Configure this script instead of passing arguments." >&2
  exit 2
fi
if [[ "$STAGE" == "prepare" ]]; then
  args=(--source-root "$SOURCE_ROOT" --output-root "$PREPARE_ROOT" --limit "$LIMIT")
  if [[ "$DOWNLOAD_IMAGES" == "true" ]]; then args+=(--download-images); fi
  exec uv run --frozen python -m data.sft.datasets.pixmo "${args[@]}"
elif [[ "$STAGE" == "convert" ]]; then
  export PIXMO_IMAGE_CACHE="$PREPARE_ROOT/images"
  export PIXMO_AUDIT_DIR="$AUDIT_ROOT"
  exec uv run --frozen python -m data.sft.tools.produce --dataset pixmo --source-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_ROOT" --data-root "$DATA_DIR" --limit "$LIMIT" \
    --image-output "$DATA_DIR/sft_images/pixmo"
else
  echo "Unknown STAGE: $STAGE" >&2
  exit 2
fi
