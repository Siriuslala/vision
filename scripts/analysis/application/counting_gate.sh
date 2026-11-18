#!/usr/bin/env bash
# Experiment 7.3 Part C: located fine-tuning of the Gated DeltaNet counting gates
# (docs/analysis/application/results_2026-11-06_counting_gate_finetune.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=sweep GPU_IDS=<claimed> bash scripts/analysis/application/counting_gate.sh        # base-model layer sweep
#   STAGES=prepare bash scripts/analysis/application/counting_gate.sh                        # CPU data
#   NAME=gate_sel_s1 SCOPE=gate:9,10 STAGES=train,eval,gatestat GPU_IDS=<claimed> bash ...
#   STAGES=analysis bash ...
# GPU: claim with the helper first (gpu_claim.sh claim 7.3C) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"
RUN_NAME="2026-11-06_counting_gate_finetune"
STAGES="${STAGES:-prepare}"
NAME="${NAME:-base}"
SCOPE="${SCOPE:-}"                        # gate:<DeltaNet layers> | dnall
SEED="${SEED:-1}"
SETS="${SETS:-ord,think,g8,probe,count,coord,general,babyvision}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-32}"
LR=2e-4
STEPS=150
STEP_SAMPLES=16
N_ODD=1200
MAX_NEW_THINK=768

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
ORDINAL_RUN="$WORK_DIR/analysis/behavior/2026-10-16_ordinal_mechanism"            # 1.5 slot-grid cases
GATE_STATS="$WORK_DIR/analysis/position/2026-10-16_sequence_order_r2/gates/stats.json"   # 3.8 head classes
INTERP_RUN="$WORK_DIR/analysis/application/2026-11-03_interp_guided_optimization"      # 7.1 TextVQA, count, coord ids
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/counting_gate_finetune"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.counting_gate_analysis \
    --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "prepare" ]]; then
  if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.counting_gate_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --name "$NAME" --scope "$SCOPE" --seed "$SEED" \
  --sets "$SETS" --ordinal-run "$ORDINAL_RUN" --gate-stats "$GATE_STATS" --interp-run "$INTERP_RUN" \
  --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --steps "$STEPS" --step-samples "$STEP_SAMPLES" --n-odd "$N_ODD" \
  --max-new-think "$MAX_NEW_THINK" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run_${NAME}_${STAGES//,/_}.log"
