#!/usr/bin/env bash
# Compare a second frozen model on a fixed subset of identical images.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
MODEL="Qwen3-VL-8B-Instruct"
GPU_IDS="6"
OUTPUT_DIR="$WORK_DIR/analysis/representation/2026-10-01_cross_model_random_targets"
LAYOUTS=(grid6_cell64 grid20_cell64 circle16)
TARGETS_PER_CASE=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.representation_patterns --model "$MODEL" --output "$OUTPUT_DIR" --layouts "${LAYOUTS[@]}" --targets-per-case "$TARGETS_PER_CASE"
