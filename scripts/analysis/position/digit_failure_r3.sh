#!/usr/bin/env bash
# Experiment 3.12 round 3: units-digit readout, relay routing and q2 source patching
# (docs/analysis/position/results_2026-10-21_digit_failure_r2.md, rewritten in round 3).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"            # one idle GPU (claim it first; GPUs 0 and 1 are excluded)
RUN_NAME="${RUN_NAME:-2026-11-03_digit_failure_r3}"   # override only for smoke tests
R2_RUN="2026-10-21_digit_failure_r2"       # weights.npz (digit unembedding and final-norm gain)
STAGES="${STAGES:-units,relay,patch3}"      # GPU stages; "analysis" runs on the CPU
LIMIT="${LIMIT:-0}"               # smoke test: units / pairs per stage (0 = all)
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
R2_DIR="$WORK_DIR/analysis/position/$R2_RUN"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-20_digit_failure"
COORD_R1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/digit_failure_r2"

mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "analysis" ]]; then
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  uv run --frozen python -m analysis_tools.position.digit_failure_r3_analysis \
    --run "$OUTPUT_DIR" --r2-run "$R2_DIR" --round1 "$ROUND1_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ -z "$GPU_IDS" ]]; then
  echo 'Set GPU_IDS to one idle GPU.' >&2
  exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run --frozen python -m analysis_tools.position.digit_failure_r3 \
  --model "$MODEL" --output "$OUTPUT_DIR" --round1 "$COORD_R1_DIR" --r2-run "$R2_DIR" \
  --refcoco-cases "$REFCOCO_CASES" --stages "$STAGES" --limit "$LIMIT" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
