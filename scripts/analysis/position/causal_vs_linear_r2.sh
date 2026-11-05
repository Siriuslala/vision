#!/usr/bin/env bash
# Experiment 3.14 round 2: answer-time reading heads of Qwen3.5-9B and Qwen3-VL-8B
# (docs/analysis/position/results_2026-11-03_causal_vs_linear_r2.md).
# Configure this section, then run this script without terminal parameters.
# JOB:
#   heads    GPU: per-head attention maps at the first-digit queries and single-head output patching
#            (decoder-only splice pairs) for both models, on answers already decoded in round 1
#   analyze  CPU: reading-head selection, attention mass on target / target row / target column, figures
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
JOB="analyze"
GPU_IDS="2"                                        # one idle GPU; both models run one after another
TORCH_DELTANET=""                                  # "--torch-deltanet" on A40 GPUs (FLA kernel needs more shared memory)
VL_MODEL="Qwen3-VL-8B-Instruct"
Q35_MODEL="Qwen3.5-9B"
CONDITIONS="t6g10_rowcol t6g10_point"
N_ANSWERS=16                                       # answers per condition (the round-1 kv answers)
HEAD_ROWS=32                                       # patched rows per forward
VL_SUBSET_TOP=192                                  # Qwen3-VL: single-head patching for the 192 heads with the largest image-attention share (1152 in total)
TOP_K=5                                            # reading heads per model and axis in the analysis
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-21_causal_vs_linear"
FLOW_RUN="$WORK_DIR/analysis/position/2026-10-20_coordinate_flow"
OUTPUT_DIR="$WORK_DIR/analysis/position/2026-11-03_causal_vs_linear_r2"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/causal_vs_linear_r2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$JOB" in
  heads)
    COMMON="--output $OUTPUT_DIR --round1 $ROUND1_DIR --flow-run $FLOW_RUN --n-answers $N_ANSWERS --head-rows $HEAD_ROWS"
    # shellcheck disable=SC2086
    uv run --frozen python -m analysis_tools.position.causal_vs_linear_heads --model "$Q35_MODEL" --mode none \
      --conditions $CONDITIONS $COMMON $TORCH_DELTANET
    # shellcheck disable=SC2086
    uv run --frozen python -m analysis_tools.position.causal_vs_linear_heads --model "$VL_MODEL" --mode ds_on --subset-top "$VL_SUBSET_TOP" \
      --conditions $CONDITIONS $COMMON ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.causal_vs_linear_heads_analysis --run "$OUTPUT_DIR" \
      --round1 "$ROUND1_DIR" --flow-run "$FLOW_RUN" --figures "$FIGURE_DIR" --top-k "$TOP_K" ;;
  *) echo "Unknown JOB=$JOB" >&2; exit 1 ;;
esac
