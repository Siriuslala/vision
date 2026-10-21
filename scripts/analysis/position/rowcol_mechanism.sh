#!/usr/bin/env bash
# Experiment 3.10: one mechanism for row and column ordinals
# (docs/analysis/position/results_2026-10-20_rowcol_mechanism.md).
# COMMAND=run runs the GPU stages in STAGES; COMMAND=heads selects the boundary-reset heads from the gate
# stage (needed by gapgate); COMMAND=analyze writes results.json and the figures (CPU only).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
COMMAND="${COMMAND:-run}"                  # run | heads | analyze
GPU_IDS="${GPU_IDS:-2}"                    # check nvidia-smi and the GPU ledger first
STAGES="${STAGES:-spotcheck,gates,order,headclass,implicit,gapgate}"
RUN_NAME="2026-10-20_rowcol_mechanism"
CONDITIONS="H32,H64,V32,V64"               # experiment-1.5 slot-grid conditions
SEED=20261020
BATCH_SIZE=16                              # unpadded batches of equal-length prompts
PREFILL_BATCH=8                            # images per prefill forward in the gate stage
GAP_CHUNK=4                                # images per decoding round in gapgate / headclass
MAX_NEW_TOKENS=24                          # direct-answer prompts
MAX_NEW_THINK=768                          # think-then-answer prompt (thinking off)
MAX_NEW_THINKING_ON=2048                   # thinking-on comparison
N_THINKING_ON=8                            # 8x8 images in the thinking-on comparison
N_SPOT=20                                  # spot-check samples against the reference generate()
ORDINAL_RUN="$WORK_DIR/analysis/behavior/2026-10-16_ordinal_mechanism"
GATE_STATS="$WORK_DIR/analysis/position/2026-10-16_sequence_order_r2/gates/stats.json"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/rowcol_mechanism"

mkdir -p "$OUTPUT_DIR"
case "$COMMAND" in
  heads|analyze)
    uv run --frozen python -m analysis_tools.position.rowcol_mechanism_analysis --command "$COMMAND" \
      --run "$OUTPUT_DIR" --ordinal-run "$ORDINAL_RUN" --gate-stats "$GATE_STATS" --figures "$FIGURE_DIR"
    ;;
  run)
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    uv run --frozen python -m analysis_tools.position.rowcol_mechanism_run \
      --model "$MODEL" --output "$OUTPUT_DIR" --ordinal-run "$ORDINAL_RUN" --gate-stats "$GATE_STATS" \
      --stages "$STAGES" --conditions "$CONDITIONS" --seed "$SEED" --batch-size "$BATCH_SIZE" \
      --prefill-batch "$PREFILL_BATCH" --gap-chunk "$GAP_CHUNK" --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-new-think "$MAX_NEW_THINK" --max-new-thinking-on "$MAX_NEW_THINKING_ON" \
      --n-thinking-on "$N_THINKING_ON" --n-spot "$N_SPOT" \
      2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run.log"
    ;;
  *) echo "unknown COMMAND $COMMAND" >&2; exit 1 ;;
esac
