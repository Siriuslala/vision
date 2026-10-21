#!/usr/bin/env bash
# Experiment 3.12: failure mechanism of coordinate-digit readout
# (docs/analysis/position/design_digit_failure.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-6}"          # check nvidia-smi and the GPU ledger first
RUN_NAME="2026-10-20_digit_failure"
ROUND1_NAME="2026-10-15_coordinate_value"
STAGES="${STAGES:-grid_gen,factorial,attn}"   # GPU stages; "analysis" = CPU analysis + figures
TOKEN_BUDGET=6000                # padded tokens per batched forward (rows = budget / prompt length)
MAX_NEW_TOKENS=48
MAX_MEMORY="${MAX_MEMORY:-}"        # optional weight split over the GPUs in GPU_IDS, e.g. "0:13GiB,1:13GiB"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_DIR="$WORK_DIR/analysis/position/$ROUND1_NAME"
ROUND2_DIR="$WORK_DIR/analysis/position/2026-10-16_coordinate_value_r2"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"      # residual shards (about 1 GB float16)
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/digit_failure"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"
[[ -e "$OUTPUT_DIR/features" ]] || ln -s "$FEATURE_DIR" "$OUTPUT_DIR/features"
if [[ "$STAGES" == "analysis" ]]; then
  export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
  uv run --frozen python -m analysis_tools.position.digit_failure_analysis \
    --run "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --round2 "$ROUND2_DIR" --features "$FEATURE_DIR"
  uv run --frozen python -m analysis_tools.position.digit_failure_figures \
    --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.digit_failure_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --features "$FEATURE_DIR" --round1 "$ROUND1_DIR" \
  --refcoco-cases "$REFCOCO_CASES" --stages "$STAGES" --token-budget "$TOKEN_BUDGET" \
  --max-new-tokens "$MAX_NEW_TOKENS" ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} 2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
