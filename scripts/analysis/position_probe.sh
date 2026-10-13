#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="0"
RUN_NAME="qwen35_9b_slot_probe_v2"
OUTPUT_DIR="$WORK_DIR/grounding/$RUN_NAME"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.position_probe --model "$MODEL" --output "$OUTPUT_DIR"
