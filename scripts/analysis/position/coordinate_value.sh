#!/usr/bin/env bash
# Experiment 3.6: coordinate-value representation and conversion (docs/analysis/position/design_coordinate_value.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="2026-10-15_coordinate_value"
STAGES="${STAGES:-tokenizer,gen,spotcheck,capture,force,steer,interp}"   # comma list, or "figures" (analysis + figures)
SEED=20261015
BATCH_SIZE=16                  # unpadded batches of equal-length prompts
MAX_NEW_TOKENS=48
N_SPOT=20                      # spot-check samples against the reference generate()
T6_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"   # about 5 GB of float16 shards
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_value"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"
[[ -e "$OUTPUT_DIR/features" ]] || ln -s "$FEATURE_DIR" "$OUTPUT_DIR/features"
if [[ "$STAGES" == "figures" ]]; then
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8   # the shared CPU is heavily loaded
  uv run --frozen python -m analysis_tools.position.coordinate_value_analysis --run "$OUTPUT_DIR" --features "$FEATURE_DIR"
  uv run --frozen python -m analysis_tools.position.coordinate_value_figures \
    --run "$OUTPUT_DIR" --features "$FEATURE_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.coordinate_value_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --features "$FEATURE_DIR" --t6-source "$T6_SOURCE" --stages "$STAGES" \
  --seed "$SEED" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" --n-spot "$N_SPOT" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
