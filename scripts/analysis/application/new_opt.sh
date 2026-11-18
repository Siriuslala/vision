#!/usr/bin/env bash
# Experiment 7.5: new interpretability-guided optimisation pilots
# (docs/analysis/application/results_2026-11-11_new_optimizations.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=link bash scripts/analysis/application/new_opt.sh                                   # CPU: reuse 7.3 data
#   STAGES=premask GPU_IDS=1 bash scripts/analysis/application/new_opt.sh                      # base-model masks
#   MODEL=Qwen3-VL-8B PREMASK=none,own_l1-6_all,... STAGES=premask GPU_IDS=1 bash ...         # mechanism check
#   NAME=fa8qk_s1 SCOPE=fa8qk SEED=1 STAGES=train,eval,attn GPU_IDS=1 bash ...                 # pilot A training
#   STAGES=format GPU_IDS=1 bash ...                                                           # pilot B formats
#   STAGES=analysis bash ...                                                                   # CPU tables + figures
# GPU: claim with the helper first (gpu_claim.sh claim 7.5) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"
GPU_IDS="${GPU_IDS:-}"                    # required for GPU stages; must be a claimed, idle GPU
RUN_NAME="2026-11-11_new_optimizations"
STAGES="${STAGES:-link}"
NAME="${NAME:-base}"                      # adapter name; 'base' = original model
SCOPE="${SCOPE:-}"                        # decoder LoRA scope (new_opt.SCOPES): fa8qk | fa12qk | fa24qk | mlp8_15
SEED="${SEED:-1}"
N_ODD=1200                                # 7.3 recipe: 1200 odd-entity items + 1200 TextVQA items
PREMASK="${PREMASK:-none,own_l8_h691,own_l8_all,own_l4_all,own_l12_all}"
SETS="${SETS:-probe,count,coord,general,babyvision}"
BATCH=12                                  # teacher-forced probe batch (premask / attn)
# Training (the 7.3 recipe; rank comes from the scope, alpha = 4 x rank)
LR=2e-4
STEPS=150
STEP_SAMPLES=16
MICRO_TOKENS=3072
# Evaluation
TOKEN_BUDGET=16000
GENERAL_N=200
GENERAL_MAX_NEW=96
BV_MAX_NEW=1536

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
VIT_RUN="$WORK_DIR/analysis/application/2026-11-05_located_vit_finetune"            # 7.3 (data, probe, base eval)
INTERP_RUN="$WORK_DIR/analysis/application/2026-11-03_interp_guided_optimization"   # 7.1 (count set, coord ids)
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"           # 3.13 coordinate test set
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/new_optimizations"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.new_opt_analysis \
    --run "$OUTPUT_DIR" --vit-run "$VIT_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "link" ]]; then
  if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.new_opt_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --name "$NAME" --scope "$SCOPE" --seed "$SEED" \
  --vit-run "$VIT_RUN" --premask "$PREMASK" --batch "$BATCH" --n-odd "$N_ODD" --sets "$SETS" \
  --interp-run "$INTERP_RUN" --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" \
  --mmvp-csv "$MMVP_CSV" --lr "$LR" --steps "$STEPS" --step-samples "$STEP_SAMPLES" --micro-tokens "$MICRO_TOKENS" \
  --token-budget "$TOKEN_BUDGET" --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" --bv-max-new "$BV_MAX_NEW" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${NAME}_${STAGES//,/_}.log"
