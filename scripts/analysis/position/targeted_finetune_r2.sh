#!/usr/bin/env bash
# Experiment 3.13 round 2: locating the readout defect, the change of the upstream code, scaled fine-tuning
# (docs/analysis/position/results_2026-11-03_targeted_finetune_r2.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=link bash scripts/analysis/position/targeted_finetune_r2.sh                                   # CPU
#   CONDITION=all_linear STAGES=splice,splicegen,capture,jacobian GPU_IDS=2 bash ...                       # analyses
#   CONDITION=fa_attn STAGES=train,coord,factorial GPU_IDS=2 bash ...                                       # a control
#   STAGES=scaledprep bash ...  ;  CONDITION=base STAGES=distill GPU_IDS=2 bash ...                         # scaled data
#   STAGES=analysis bash ...                                                                                # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                     # claim an idle GPU first (never 0 or 1); required for GPU stages
RUN_NAME="2026-11-03_targeted_finetune_r2"
ROUND1_NAME="2026-10-21_targeted_finetune"
STAGES="${STAGES:-link}"
CONDITION="${CONDITION:-base}"
# Training (identical to round 1: rank 8, alpha 32, lr 1e-4 unless a condition overrides lr)
RANK=8
ALPHA=32
LR=1e-4
EPOCHS=1
STEP_SAMPLES=16
MICRO_TOKENS=4096
# Evaluation / analysis
TOKEN_BUDGET="${TOKEN_BUDGET:-16000}"
MAX_NEW_TOKENS=64
GENERAL_N=200
GENERAL_MAX_NEW=96
SPLICE_BATCH=24
JACOBIAN_BATCH=4
SPLICE_CONFIG="${SPLICE_CONFIG:-}"          # capture with one splice configuration of the adapter (e.g. band1_4)
MAX_MEMORY="${MAX_MEMORY:-}"
CONDITIONS_ANALYSED="base all_linear early1_4 early1_4_img all_linear_img scaled_early1_4_img mlp28_32 mlp28_32_r64 late25_32 mixer16_24 mlp10_14 fa_attn dn_proj mlp_all fa_mlp all_but_mlp28_32 all_but_25_32 mlp28_32_lr1e3 scaled_all_linear scaled_all_but_25_32"

OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_RUN="$WORK_DIR/analysis/position/$ROUND1_NAME"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"
READOUT_TOOL_RUN="$WORK_DIR/analysis/position/2026-10-16_readout_tool"
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/targeted_finetune_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.position.targeted_finetune_r2_analysis \
    --run "$OUTPUT_DIR" --conditions $CONDITIONS_ANALYSED --figures "$FIGURE_DIR" --hf-cache "$HF_HUB_CACHE" --model "$MODEL"
  exit 0
fi
if [[ "$STAGES" != "link" && "$STAGES" != "scaledprep" ]]; then
  [[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.position.targeted_finetune_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --round1-run "$ROUND1_RUN" --stages "$STAGES" --condition "$CONDITION" \
  --readout-tool-run "$READOUT_TOOL_RUN" --refcoco-cases "$REFCOCO_CASES" --round1 "$ROUND1_DIR" \
  --features "$FEATURE_DIR" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --epochs "$EPOCHS" --step-samples "$STEP_SAMPLES" \
  --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" --max-new-tokens "$MAX_NEW_TOKENS" \
  --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" --splice-batch "$SPLICE_BATCH" \
  --jacobian-batch "$JACOBIAN_BATCH" ${SPLICE_CONFIG:+--splice-config "$SPLICE_CONFIG"} ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${CONDITION}${SPLICE_CONFIG:++$SPLICE_CONFIG}.log"
