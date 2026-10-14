#!/usr/bin/env bash
# Configure this section, then run this script without terminal parameters.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="6"
RUN_NAME="distance_register_v1"
RESUME_SYNTHETIC=false
: "${REFCOCO_DIR:?Set REFCOCO_DIR in .env}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
EXTRA_ARGS=()
if [[ "$RESUME_SYNTHETIC" == true ]]; then EXTRA_ARGS+=(--resume-synthetic); fi
uv run --frozen python -m analysis_tools.representation.distance_register \
  "${EXTRA_ARGS[@]}" --model "$MODEL" --refcoco-dir "$REFCOCO_DIR" --output "$WORK_DIR/grounding/$RUN_NAME"
