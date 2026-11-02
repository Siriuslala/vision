#!/usr/bin/env bash
# Experiment 3.13: mechanism-guided fine-tuning of the coordinate readout
# (docs/analysis/position/design_targeted_finetune.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=prepare bash scripts/analysis/position/targeted_finetune.sh                 # CPU
#   CONDITION=mlp28_32 STAGES=train,coord,general,factorial GPU_IDS=2 bash ...          # one condition
#   CONDITION=base STAGES=coord,general,factorial,spotcheck GPU_IDS=2 bash ...          # base model
#   STAGES=analysis bash ...                                                            # CPU analysis + figures
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"                   # check nvidia-smi and the GPU ledger first
RUN_NAME="2026-10-21_targeted_finetune"
STAGES="${STAGES:-prepare}"
CONDITION="${CONDITION:-base}"            # base | all_linear | mlp28_32 | mixer16_24 | mlp10_14 | late25_32
CONDITIONS_ANALYSED="base all_linear mlp28_32 mlp28_32_r64 late25_32 mixer16_24 mlp10_14"
# Training (identical for every condition; ms-swift Qwen3.5 LoRA defaults: rank 8, alpha 32, lr 1e-4)
RANK=8
ALPHA=32
LR=1e-4
EPOCHS=1
STEP_SAMPLES=16                           # samples per optimizer step
MICRO_TOKENS=4096                         # padded tokens per micro-batch (gradient accumulation inside a step)
# Evaluation
TOKEN_BUDGET="${TOKEN_BUDGET:-16000}"      # padded tokens per decoding / factorial batch
MAX_NEW_TOKENS=64
GENERAL_N=200                             # POPE / TextVQA / MMVP / GQA questions (the 5.2 samples)
GENERAL_MAX_NEW=96
MAX_MEMORY="${MAX_MEMORY:-}"              # optional weight split, e.g. "0:12GiB,1:12GiB"

OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"
READOUT_TOOL_RUN="$WORK_DIR/analysis/position/2026-10-16_readout_tool"        # RefCOCO image split and images (3.9)
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"   # 3.12 RefCOCO units
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"          # 3.6 disc sweeps
BASE_3_12="$WORK_DIR/analysis/position/2026-10-20_digit_failure"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/targeted_finetune"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.position.targeted_finetune_analysis \
    --run "$OUTPUT_DIR" --conditions $CONDITIONS_ANALYSED --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.targeted_finetune_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --condition "$CONDITION" \
  --readout-tool-run "$READOUT_TOOL_RUN" --refcoco-cases "$REFCOCO_CASES" --round1 "$ROUND1_DIR" \
  --features "$FEATURE_DIR" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --epochs "$EPOCHS" --step-samples "$STEP_SAMPLES" \
  --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" --max-new-tokens "$MAX_NEW_TOKENS" \
  --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${CONDITION}.log"
