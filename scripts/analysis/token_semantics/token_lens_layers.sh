#!/usr/bin/env bash
# Configure the layer list and run directory before launching.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="7"
LAYERS=(0 24 25 26 27 28 29 30 31 32)
OUTPUT_DIR="$WORK_DIR/analysis/token_semantics/2026-10-01_final_layers"
: "${REFCOCO_DIR:?Set REFCOCO_DIR in .env}"
: "${TOKEN_MAP_FONT:?Set TOKEN_MAP_FONT to a Chinese-supporting font in .env}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.token_semantics.token_lens --model "$MODEL" --output "$OUTPUT_DIR" --layers "${LAYERS[@]}"
uv run --frozen python -m analysis_tools.token_semantics.token_lens_figures --input "$OUTPUT_DIR" --output "$OUTPUT_DIR/figures"
