#!/usr/bin/env bash
# Experiment 7.4: located mixed-data fine-tuning for BabyVision row/column find-the-difference
# (docs/analysis/application/results_2026-11-11_babyvision_rowcol.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=prep bash scripts/analysis/application/bv_rowcol.sh                                  # CPU
#   CONDITION=base STAGES=spot,bv,rep,up2 GPU_IDS=<k> bash scripts/analysis/application/bv_rowcol.sh
#   CONDITION=main STAGES=train,bv,rep,coord,general,ordinal,ordinal_cot GPU_IDS=<k> bash ...
#   STAGES=analysis bash scripts/analysis/application/bv_rowcol.sh                              # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                     # an idle GPU claimed through the claim helper; required for GPU stages
RUN_NAME="2026-11-11_babyvision_rowcol"
STAGES="${STAGES:-prep}"
CONDITION="${CONDITION:-base}"
# Training (LoRA rank 8, alpha 32; per-part learning rates in analysis_tools/application/bv_rowcol.py: LR)
RANK=8
ALPHA=32
EPOCHS=1
STEP_SAMPLES=16
MICRO_TOKENS=4096
# Evaluation
TOKEN_BUDGET=16000
MAX_NEW_TOKENS=64
GENERAL_N=200
GENERAL_MAX_NEW=96
BATCH=24
COT_BATCH=16
BV_MAX_NEW=1024
WORKERS=8                                  # CPU processes for replica rendering (prep)

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
RUN72="$WORK_DIR/analysis/application/2026-11-05_located_finetune"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-11-03_targeted_finetune_r2"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
FEATURE_DIR="$DATA_DIR/scratch/analysis/application/$RUN_NAME"
BV_ANNOTATION="$DATA_DIR/BabyVision_annotations/find_the_different_grid_v1.json"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/babyvision_rowcol"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.bv_rowcol_analysis \
    --run "$OUTPUT_DIR" --run72 "$RUN72" --features "$FEATURE_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "prep" ]]; then
  [[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.bv_rowcol_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --condition "$CONDITION" --run72 "$RUN72" \
  --finetune-run "$FINETUNE_RUN" --round1 "$ROUND1_DIR" --features "$FEATURE_DIR" --data-dir "$DATA_DIR" \
  --bv-annotation "$BV_ANNOTATION" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" --rank "$RANK" --alpha "$ALPHA" \
  --epochs "$EPOCHS" --step-samples "$STEP_SAMPLES" --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" \
  --max-new-tokens "$MAX_NEW_TOKENS" --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" --batch "$BATCH" \
  --cot-batch "$COT_BATCH" --bv-max-new "$BV_MAX_NEW" --workers "$WORKERS" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${CONDITION//,/+}.log"
