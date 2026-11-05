#!/usr/bin/env bash
# Experiment 3.15: row/column ordinals vs coordinate values (docs/analysis/position/results_2026-11-03_ordinal_vs_coordinate.md).
# COMMAND=run executes the GPU stages in STAGES (resumable, batch size 1 for forced answers);
# COMMAND=analyze writes summary.json and the figures (CPU only).
# GPU: claim an idle GPU first with the project claim helper and pass it as GPU_IDS; release it afterwards.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
COMMAND="${COMMAND:-run}"                     # run | analyze
GPU_IDS="${GPU_IDS:-}"                        # an idle GPU you have claimed
STAGES="${STAGES:-spotcheck,answer,gate,refcoco}"
N_REFCOCO=160                                 # RefCOCO test cases (direct and think)
BATCH_DIRECT=16                               # batch size of short (direct) generations
BATCH_THINK=16                                # batch size of think generations
RUN_NAME="2026-11-03_ordinal_vs_coordinate"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/ordinal_vs_coordinate"

mkdir -p "$OUTPUT_DIR" "$FIGURE_DIR"
case "$COMMAND" in
  run)
    if [ -z "$GPU_IDS" ]; then echo "Set GPU_IDS to a claimed idle GPU." >&2; exit 1; fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8
    uv run --frozen python -m analysis_tools.position.ordinal_vs_coordinate_run \
      --stages "$STAGES" --output "$OUTPUT_DIR" --work "$WORK_DIR/analysis" --model "$MODEL" \
      --n-refcoco "$N_REFCOCO" --batch-direct "$BATCH_DIRECT" --batch-think "$BATCH_THINK"
    ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.ordinal_vs_coordinate_analysis \
      --run-dir "$OUTPUT_DIR" --figures "$FIGURE_DIR"
    ;;
  *) echo "unknown COMMAND $COMMAND" >&2; exit 1 ;;
esac
