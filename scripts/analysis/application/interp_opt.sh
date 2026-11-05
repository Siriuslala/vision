#!/usr/bin/env bash
# Experiment 7.1: interpretability-guided optimisation pilots
# (docs/analysis/application/design_interp_guided_optimization.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=prepare bash scripts/analysis/application/interp_opt.sh                          # CPU
#   CONDITION=lora_bidir STAGES=train,eval,mech GPU_IDS=4 bash ...                           # one condition
#   CONDITION=base STAGES=eval,mech,spotcheck GPU_IDS=4 bash ...                              # base model
#   STAGES=analysis bash ...                                                                  # CPU tables + figures
# Conditions: base | base_bidir | lora_causal | lora_bidir | lora_bidir_causaleval | lora_contrast | lora_vit | vcd_all | vcd_digit
# GPU: claim with the helper first (gpu_claim.sh claim 7.1) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                    # required for GPU stages; must be a claimed, idle GPU
RUN_NAME="2026-11-03_interp_guided_optimization"
STAGES="${STAGES:-prepare}"
CONDITION="${CONDITION:-base}"
SETS="${SETS:-probe,count,coord,general}"
CONDITIONS_ANALYSED="base base_bidir lora_causal lora_bidir lora_bidir_causaleval lora_contrast lora_vit lora_vit_bidireval lora_vit_bidir vcd_all vcd_digit"
# Training (identical for every LoRA condition; the 3.13 recipe: rank 8, alpha 32, lr 1e-4, all decoder linear layers)
RANK=8
ALPHA=32
LR=1e-4
STEPS=150                                 # optimizer steps
STEP_SAMPLES=16                           # samples per step (150 x 16 = 2400 = the whole mixture once)
MICRO_TOKENS=3072                         # padded tokens per micro-batch
# Evaluation
TOKEN_BUDGET="${TOKEN_BUDGET:-16000}"
VCD_ALPHA=1.0                             # prior-subtracted decoding strength
VCD_BETA=0.1                              # plausibility cut-off relative to the top real-image probability
GENERAL_N=200
GENERAL_MAX_NEW=96

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"     # 3.13 coordinate test set
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
ODDNESS_DIRS="$WORK_DIR/analysis/representation/2026-11-03_search_asymmetry_r2/directions.npz"   # 2.7 r2 oddness directions
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/interp_guided_optimization"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.interp_opt_analysis \
    --run "$OUTPUT_DIR" --conditions $CONDITIONS_ANALYSED --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "prepare" ]]; then
  if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.interp_opt_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --condition "$CONDITION" --sets "$SETS" \
  --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" --oddness-dirs "$ODDNESS_DIRS" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --steps "$STEPS" --step-samples "$STEP_SAMPLES" \
  --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" --vcd-alpha "$VCD_ALPHA" --vcd-beta "$VCD_BETA" \
  --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${CONDITION}.log"
