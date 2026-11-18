#!/usr/bin/env bash
# Experiment 3.14 round 4: origin of the row-start anchor of the y-reading heads in Qwen3-VL-8B and Qwen3.5-9B
# (docs/analysis/position/results_2026-11-11_row_anchor.md).
# Configure this section, then run this script without terminal parameters.
# JOB:
#   gpu      GPU: factor stimuli (attention / content scores), content-matched patching of column-0 states
#            along the ViT and the decoder, and answer-time value edits (models one after another)
#   analyze  CPU: statistics, probes and figures
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
JOB="analyze"
GPU_IDS="3"                                        # one idle GPU (claimed through the group's claim helper)
TORCH_DELTANET=""                                  # "--torch-deltanet" on A40 GPUs (FLA kernel needs more shared memory)
MODELS="Qwen3-VL-8B-Instruct Qwen3.5-9B"
STAGES="factors locate edits"
N_FACTOR=24                                        # cases per (variant, format) in the factors stage
N_LOCATE=32                                        # grid images in the locate stage
N_STATS=16                                         # locate images whose stage-wise states are stored for probes
N_EDITS=32                                         # images per variant (edge, left margin) in the edits stage
SEED=20261111
PHOTO_DIR="$WORK_DIR/analysis/position/2026-10-14_probe_refinement/images/natural512"
OUTPUT_DIR="$WORK_DIR/analysis/position/2026-11-11_row_anchor"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/row_anchor"
R3_RUN="$WORK_DIR/analysis/position/2026-11-05_row_attention"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
case "$JOB" in
  gpu)
    for MODEL in $MODELS; do
      # shellcheck disable=SC2086
      uv run --frozen python -m analysis_tools.position.row_anchor --model "$MODEL" --stages $STAGES \
        --output "$OUTPUT_DIR" --photo-dir "$PHOTO_DIR" --n-factor "$N_FACTOR" --n-locate "$N_LOCATE" \
        --n-stats "$N_STATS" --n-edits "$N_EDITS" --seed "$SEED" $TORCH_DELTANET
    done ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.row_anchor_analysis --run "$OUTPUT_DIR" --r3-run "$R3_RUN" \
      --figures "$FIGURE_DIR" ;;
  *) echo "Unknown JOB=$JOB" >&2; exit 1 ;;
esac
