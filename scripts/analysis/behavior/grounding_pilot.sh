#!/usr/bin/env bash
# Configure below, then run: bash scripts/analysis/behavior/grounding_pilot.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="1"
RUN_NAME="qwen35_9b_pilot_v2"
RUN_REPRESENTATION=false
RUN_BEHAVIOR=true
MAX_NEW_TOKENS=512
OUTPUT_DIR="$WORK_DIR/grounding/$RUN_NAME"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$RUN_REPRESENTATION" == true ]]; then
  uv run --frozen python -m analysis_tools.behavior.grounding_pilot \
  --mode representation --model "$MODEL" --gpu-ids "$GPU_IDS" --output "$OUTPUT_DIR/representation"
elif [[ "$RUN_REPRESENTATION" != false ]]; then
  echo 'RUN_REPRESENTATION must be true or false.' >&2; exit 2
fi
if [[ "$RUN_BEHAVIOR" == true ]]; then
  uv run --frozen python -m analysis_tools.behavior.grounding_pilot \
  --mode behavior --model "$MODEL" --gpu-ids "$GPU_IDS" --max-new-tokens "$MAX_NEW_TOKENS" --output "$OUTPUT_DIR/behavior"
elif [[ "$RUN_BEHAVIOR" != false ]]; then
  echo 'RUN_BEHAVIOR must be true or false.' >&2; exit 2
fi
