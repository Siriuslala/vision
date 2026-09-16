#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
AUDIT_DIR="$DATA_DIR/sft_annotations/pixmo_trial/audits"
IMAGE_CACHE="$DATA_DIR/sft_annotations/pixmo_trial/images"
OUTPUT_ROOT="$DATA_DIR/sft_annotations/pixmo_candidates_trial"
METHOD="sam3" # Natural images use SAM3; geometry requires regular_compact below.
REGULAR_COMPACT=false
CHECKPOINT="${SAM3_CHECKPOINT:-facebook/sam3}"
DEVICE="cpu"
LIMIT=3
# =======================================================
if (( $# > 0 )); then
  echo "Configure this script instead of passing arguments." >&2
  exit 2
fi
args=(--audit-dir "$AUDIT_DIR" --image-cache "$IMAGE_CACHE" --output-root "$OUTPUT_ROOT"
      --method "$METHOD" --device "$DEVICE" --limit "$LIMIT")
if [[ "$REGULAR_COMPACT" == "true" ]]; then args+=(--regular-compact); fi
if [[ "$METHOD" == "sam3" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
fi
exec uv run --frozen python -m data.sft.tools.point_windows "${args[@]}"
