#!/usr/bin/env bash
# Configure below, then run: bash scripts/analysis/representation/visual_detail.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

# Experiment configuration. Existing output directories are rejected.
MODEL="Qwen3.5-9B"
GPU_IDS="0"
OUTPUT_DIR="$WORK_DIR/visual_detail/qwen35_9b_synthetic_fp32"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
exec uv run --frozen python -m analysis_tools.representation.visual_detail \
  --model "$MODEL" --output "$OUTPUT_DIR" --device cuda:0
