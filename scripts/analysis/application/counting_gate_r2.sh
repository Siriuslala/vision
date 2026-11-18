#!/usr/bin/env bash
# Experiment 7.3 Part C round 2: gate modifications of the Gated DeltaNet counting layers
# (docs/analysis/application/results_2026-11-11_counting_gate_r2.md).
#
# Usage: edit the parameters below, then run, e.g.
#   NAME=base STAGES=eval SETS=ord,g8,gmulti,ansprobe GPU_IDS=<claimed> bash scripts/analysis/application/counting_gate_r2.sh
#   NAME=l15_nonint SPEC='fix:15:nonint:1@img' STAGES=eval GPU_IDS=<claimed> bash ...      # inference-time edit
#   NAME=full_L6 SPEC='full:6@img' LR=1e-4 STAGES=train,eval GPU_IDS=<claimed> bash ...    # trained gate modification
#   STAGES=headsweep SWEEP_LAYER=15 GPU_IDS=<claimed> bash ...
#   STAGES=analysis bash ...
# Spec grammar: analysis_tools/application/counting_gate_r2.py.  Round-1 adapters: NAME=r1:gate_sel_s1.
# GPU: claim with the helper first (gpu_claim.sh claim 7.3C2) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"
GPU_IDS="${GPU_IDS:-}"
RUN_NAME="2026-11-11_counting_gate_r2"
STAGES="${STAGES:-eval}"
NAME="${NAME:-base}"
SPEC="${SPEC:-}"
SEED="${SEED:-1}"
SETS="${SETS:-ord,g8,probe,ansprobe,gatestat}"
SWEEP_LAYER="${SWEEP_LAYER:-15}"
LR="${LR:-1e-4}"
STEPS="${STEPS:-150}"
STEP_SAMPLES=16
N_ODD=1200
MAX_NEW_THINK=768

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
R1_RUN="$WORK_DIR/analysis/application/2026-11-06_counting_gate_finetune"                # 7.3C data and adapters
ORDINAL_RUN="$WORK_DIR/analysis/behavior/2026-10-16_ordinal_mechanism"
GATE_STATS="$WORK_DIR/analysis/position/2026-10-16_sequence_order_r2/gates/stats.json"
INTERP_RUN="$WORK_DIR/analysis/application/2026-11-03_interp_guided_optimization"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/counting_gate_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR"
for f in train_odd.jsonl train_vqa.jsonl; do [[ -f "$OUTPUT_DIR/$f" ]] || cp "$R1_RUN/$f" "$OUTPUT_DIR/$f"; done

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.counting_gate_r2_analysis \
    --run "$OUTPUT_DIR" --r1-run "$R1_RUN" --ordinal-run "$ORDINAL_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.application.counting_gate_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --name "$NAME" --spec "$SPEC" --seed "$SEED" \
  --sets "$SETS" --sweep-layer "$SWEEP_LAYER" --r1-run "$R1_RUN" --ordinal-run "$ORDINAL_RUN" --gate-stats "$GATE_STATS" \
  --interp-run "$INTERP_RUN" --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" \
  --mmvp-csv "$MMVP_CSV" --lr "$LR" --steps "$STEPS" --step-samples "$STEP_SAMPLES" --n-odd "$N_ODD" \
  --max-new-think "$MAX_NEW_THINK" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run_${NAME//:/_}_${STAGES//,/_}.log"
