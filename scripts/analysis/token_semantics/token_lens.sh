#!/usr/bin/env bash
# Configure below, then run: bash scripts/analysis/token_semantics/token_lens.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="7"
RUN_NAME="qwen35_9b_raw_lens_v1"
OUTPUT_DIR="$WORK_DIR/grounding/$RUN_NAME"
: "${REFCOCO_DIR:?Set REFCOCO_DIR in .env}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.token_semantics.token_lens --model "$MODEL" --output "$OUTPUT_DIR"
