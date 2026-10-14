#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
: "${REFCOCO_DIR:?Set REFCOCO_DIR in the private .env file.}"
MODEL="Qwen3.5-9B"
GPU_IDS="0"
IMAGES=360
TOKENS_PER_IMAGE=48
OUTPUT_DIR="$WORK_DIR/analysis/position/2026-10-01_refcoco_layers"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.position.position_real_probe --model "$MODEL" --refcoco "$REFCOCO_DIR" --images "$IMAGES" --tokens "$TOKENS_PER_IMAGE" --output "$OUTPUT_DIR"
uv run --frozen python -m analysis_tools.position.position_real_probe --evaluate-only --projection-dim 128 --output "$OUTPUT_DIR"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vision-matplotlib"
uv run --frozen python -m analysis_tools.position.position_figures --real "$OUTPUT_DIR" --output "$OUTPUT_DIR/figures"
mkdir -p "$ROOT_DIR/docs/analysis/position/figures"
cp "$OUTPUT_DIR"/figures/*.png "$ROOT_DIR/docs/analysis/position/figures/"
