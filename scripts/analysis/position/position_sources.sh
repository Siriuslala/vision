#!/usr/bin/env bash
# Experiment 3.15 round 3: sources of position information in Qwen3.5-9B and Qwen3-VL-8B
# (docs/analysis/position/results_2026-11-11_position_sources.md).
# COMMAND=run executes the GPU stages in STAGES for MODEL (resumable); COMMAND=analyze runs the CPU analysis
# (probes, variance partition, steering directions, figures) and writes summary.json.
# GPU: claim an idle GPU with the project claim helper, pass it as GPU_IDS, release it afterwards.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"                  # Qwen3.5-9B (capture, roll, steer) | Qwen3-VL-8B-Instruct (roll, vlflow)
COMMAND="${COMMAND:-run}"                     # run | analyze
GPU_IDS="${GPU_IDS:-}"                        # an idle GPU you have claimed
STAGES="${STAGES:-capture,roll}"
N_JGRID=160                                   # decorrelated grids (probe fit on the first, steering on the last)
N_SINGLE=64                                   # single red figure on a blank canvas
N_REFCOCO=96                                  # RefCOCO test photos (3.15 round-1 selection order)
N_ROLL=64                                     # images per family in the raster-order dissociation
N_STEER=48                                    # jgrid images in the steering stage (single / RefCOCO: half)
N_VLPAIRS=32                                  # Qwen3-VL patching pairs per family
BATCH=16                                      # batch size of batched greedy decoding
RUN_NAME="2026-11-11_position_sources"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/position_sources"

mkdir -p "$OUTPUT_DIR" "$FIGURE_DIR"
case "$COMMAND" in
  run)
    if [ -z "$GPU_IDS" ]; then echo "Set GPU_IDS to a claimed idle GPU." >&2; exit 1; fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8
    uv run --frozen python -m analysis_tools.position.position_sources_run \
      --stages "$STAGES" --model "$MODEL" --output "$OUTPUT_DIR" --work "$WORK_DIR/analysis" --data-dir "$DATA_DIR" \
      --n-jgrid "$N_JGRID" --n-single "$N_SINGLE" --n-refcoco "$N_REFCOCO" --n-roll "$N_ROLL" \
      --n-steer "$N_STEER" --n-vlpairs "$N_VLPAIRS" --batch "$BATCH"
    ;;
  analyze)
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
    uv run --frozen python -m analysis_tools.position.position_sources_analysis \
      --run-dir "$OUTPUT_DIR" --figures "$FIGURE_DIR"
    ;;
  *) echo "unknown COMMAND $COMMAND" >&2; exit 1 ;;
esac
