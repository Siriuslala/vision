#!/usr/bin/env bash
# Experiment 7.3: located ViT fine-tuning for features without detectors
# (docs/analysis/application/results_2026-11-05_located_vit_finetune.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=locate GPU_IDS=1 bash scripts/analysis/application/located_finetune.sh            # base-model localisation
#   STAGES=prepare bash scripts/analysis/application/located_finetune.sh                     # CPU data
#   NAME=loc_n1200_s1 SCOPE=vit_mlp:9-11 N_ODD=1200 SEED=1 STAGES=train,eval,detect GPU_IDS=1 bash ...
#   NAME=base STAGES=eval,detect GPU_IDS=1 bash ...
#   STAGES=analysis bash ...                                                                 # CPU tables + figures
# GPU: claim with the helper first (gpu_claim.sh claim 7.3) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                    # required for GPU stages; must be a claimed, idle GPU
RUN_NAME="2026-11-05_located_vit_finetune"
STAGES="${STAGES:-prepare}"
NAME="${NAME:-base}"                      # adapter name; 'base' = original model
SCOPE="${SCOPE:-}"                        # LoRA scope: vit_mlp:<blocks> | vit_attn:<blocks> | vit_all | model_all
N_ODD="${N_ODD:-1200}"                    # odd-entity training items (+ the same number of TextVQA items)
SEED="${SEED:-1}"
SETS="${SETS:-probe,count,coord,general,babyvision}"
# Training (identical for every condition; only SCOPE and N_ODD vary)
RANK="${RANK:-8}"
ALPHA="${ALPHA:-32}"
LR=2e-4
STEPS=150                                 # optimizer steps (150 x 16 = 2400 sample visits for every data size)
STEP_SAMPLES=16
MICRO_TOKENS=3072
# Evaluation
TOKEN_BUDGET=16000
GENERAL_N=200
GENERAL_MAX_NEW=96
BV_MAX_NEW=1536

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
INTERP_RUN="$WORK_DIR/analysis/application/2026-11-03_interp_guided_optimization"   # 7.1 (TextVQA items, count, coord ids)
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"           # 3.13 coordinate test set
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/located_vit_finetune"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.located_finetune_analysis \
    --run "$OUTPUT_DIR" --interp-run "$INTERP_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "prepare" ]]; then
  if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.located_finetune_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --name "$NAME" --scope "$SCOPE" --n-odd "$N_ODD" \
  --seed "$SEED" --sets "$SETS" --interp-run "$INTERP_RUN" --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" \
  --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --steps "$STEPS" \
  --step-samples "$STEP_SAMPLES" --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" \
  --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" --bv-max-new "$BV_MAX_NEW" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${NAME}_${STAGES//,/_}.log"
