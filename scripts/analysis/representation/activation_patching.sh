#!/usr/bin/env bash
# Experiment 2.5a activation patching (docs/analysis/representation/design_causal.md).
# Configure this section, then run this script without terminal parameters.
# STAGE: screen (baseline scores and pair qualification) -> scan (patching) -> detail (high-norm overlap,
# single text-token patches) -> analyze (bootstrap and figures).
# The float32 check reruns scan/analyze with DTYPE=float32, TAG=fp32 and MAX_PAIRS_PER_CONDITION set.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="1"
RUN_DATE="2026-10-13"
STAGE="screen"
CONDITIONS="red_rowcol red_box gap_rowcol gap_box gap_box_attr"
PAIRS_PER_CONDITION=64
MAX_PAIRS_PER_CONDITION=0       # 0 scans every qualified pair
BATCH_SIZE=16
VIT_LAYERS="3 9 18 27"
DECODER_LAYERS="$(seq -s ' ' 1 32)"
SANITY_LAYERS="1 16"
TOKEN_LAYERS="12 14 15 16 17 20 24 28"   # detail stage: single text-token patches
RANDOM_SEEDS=5
N_BOOT=2000
SEED=0
DTYPE="bfloat16"
TAG="bf16"
REFERENCE_TAG=""                # analyze: compare with another run, e.g. bf16 when TAG=fp32
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_activation_patching"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.activation_patching \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --conditions $CONDITIONS \
  --pairs-per-condition "$PAIRS_PER_CONDITION" --max-pairs-per-condition "$MAX_PAIRS_PER_CONDITION" \
  --batch-size "$BATCH_SIZE" --vit-layers $VIT_LAYERS --decoder-layers $DECODER_LAYERS \
  --sanity-layers $SANITY_LAYERS --token-layers $TOKEN_LAYERS --random-seeds "$RANDOM_SEEDS" --n-boot "$N_BOOT" --seed "$SEED" \
  --dtype "$DTYPE" --tag "$TAG" --reference-tag "$REFERENCE_TAG"
