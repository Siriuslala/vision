#!/usr/bin/env bash
# Configure the experiment here, then execute this file.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="6"
OUTPUT_DIR="$WORK_DIR/analysis/representation/2026-10-01_layouts_random_targets"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.representation.representation_patterns --model "$MODEL" --output "$OUTPUT_DIR"
