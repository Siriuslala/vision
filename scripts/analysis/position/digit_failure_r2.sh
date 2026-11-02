#!/usr/bin/env bash
# Experiment 3.12 round 2: refining the digit-readout mechanism
# (docs/analysis/position/results_2026-10-21_digit_failure_r2.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-6,7}"         # check nvidia-smi and the GPU ledger first
RUN_NAME="2026-10-21_digit_failure_r2"
ROUND1_RUN="2026-10-20_digit_failure"      # factorial, residual and DLA artifacts of round 1
STAGES="${STAGES:-weights,attn,patch,capture}"   # GPU stages; "select" and "analysis" run on the CPU
N_REFCOCO=40                      # RefCOCO units in the capture / ablation stages
MAX_MEMORY="${MAX_MEMORY:-0:12GiB,1:10GiB}"   # weight split over the GPUs in GPU_IDS ("" = one GPU)
LIMIT="${LIMIT:-0}"               # smoke test: units / pairs per stage (0 = all)
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_DIR="$WORK_DIR/analysis/position/$ROUND1_RUN"
COORD_R1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/digit_failure_r2"
ROUND1_FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/digit_failure"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"
[[ -e "$OUTPUT_DIR/features" ]] || ln -s "$FEATURE_DIR" "$OUTPUT_DIR/features"
if [[ "$STAGES" == "select" || "$STAGES" == "analysis" ]]; then
  export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
  uv run --frozen python -m analysis_tools.position.digit_failure_r2_analysis \
    --run "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --features "$FEATURE_DIR" --stage "$STAGES"
  if [[ "$STAGES" == "analysis" ]]; then
    uv run --frozen python -m analysis_tools.position.digit_failure_r2_figures \
      --run "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --figures "$FIGURE_DIR" --round1-figures "$ROUND1_FIGURE_DIR"
  fi
  exit 0
fi
uv run --frozen python -m analysis_tools.position.digit_failure_r2 \
  --model "$MODEL" --output "$OUTPUT_DIR" --features "$FEATURE_DIR" --round1 "$COORD_R1_DIR" \
  --refcoco-cases "$REFCOCO_CASES" --stages "$STAGES" --n-refcoco "$N_REFCOCO" --limit "$LIMIT" ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
