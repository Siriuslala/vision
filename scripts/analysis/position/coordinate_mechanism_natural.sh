#!/usr/bin/env bash
# Experiment 3.15 round 2: coordinate-value mechanism on natural vs synthetic images
# (docs/analysis/position/results_2026-11-05_coordinate_mechanism_natural.md).
# COMMAND=run executes the GPU stages in STAGES (resumable); COMMAND=analyze writes summary.json and figures (CPU).
# GPU: claim an idle GPU first with the project claim helper and pass it as GPU_IDS; release it afterwards.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
COMMAND="${COMMAND:-run}"                     # run | analyze
GPU_IDS="${GPU_IDS:-}"                        # an idle GPU you have claimed
STAGES="${STAGES:-spot,gen,think,flow,timeline}"
N_ITEMS=64                                    # cases per synthetic / thumbnail family (grid8 and thumbnails: 36 cells)
N_REFCOCO=160                                 # RefCOCO test cases (the 3.15 round-1 selection)
N_PAIRS=48                                    # flow pairs per family (bbox and CoT pairs: half)
N_TIMELINE=24                                 # teacher-forced timelines per family
BATCH=16                                      # batch size of batched greedy decoding
RUN_NAME="2026-11-05_coordinate_mechanism_natural"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_mechanism_natural"

mkdir -p "$OUTPUT_DIR" "$FIGURE_DIR"
case "$COMMAND" in
  run)
    if [ -z "$GPU_IDS" ]; then echo "Set GPU_IDS to a claimed idle GPU." >&2; exit 1; fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8
    uv run --frozen python -m analysis_tools.position.coordinate_mechanism_natural_run \
      --stages "$STAGES" --output "$OUTPUT_DIR" --work "$WORK_DIR/analysis" --data-dir "$DATA_DIR" --model "$MODEL" \
      --n-items "$N_ITEMS" --n-refcoco "$N_REFCOCO" --n-pairs "$N_PAIRS" --n-timeline "$N_TIMELINE" --batch "$BATCH"
    ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.coordinate_mechanism_natural_analysis \
      --run-dir "$OUTPUT_DIR" --figures "$FIGURE_DIR"
    ;;
  *) echo "unknown COMMAND $COMMAND" >&2; exit 1 ;;
esac
