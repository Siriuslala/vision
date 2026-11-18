#!/usr/bin/env bash
# Experiment 3.14 round 3: row-directed attention of the y-reading heads in Qwen3-VL-8B and Qwen3.5-9B
# (docs/analysis/position/results_2026-11-05_row_attention.md).
# Configure this section, then run this script without terminal parameters.
# JOB:
#   gpu      GPU: attention maps and MRoPE section decomposition of the round-2 reading heads, then
#            answer-time value edits of token groups (both models, one after another)
#   analyze  CPU: statistics and figures
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
JOB="analyze"
GPU_IDS="3"                                        # one idle GPU (claimed through the group's claim helper)
TORCH_DELTANET=""                                  # "--torch-deltanet" on A40 GPUs (FLA kernel needs more shared memory)
MODELS="Qwen3-VL-8B-Instruct Qwen3.5-9B"
STAGES="maps edits"
TOP_HEADS=5                                        # round-2 reading heads per format and axis (union is analysed)
N_MAPS=40                                          # answers per condition in the maps stage
N_EDITS=40                                         # answers per condition in the edits stage
FLOW_RUN="$WORK_DIR/analysis/position/2026-10-20_coordinate_flow"
VL_GEN="$WORK_DIR/analysis/position/2026-10-21_causal_vs_linear/gen_vl_ds_on.jsonl"
R2_SUMMARY="$WORK_DIR/analysis/position/2026-11-03_causal_vs_linear_r2/summary.json"
R2_RUN="$WORK_DIR/analysis/position/2026-11-03_causal_vs_linear_r2"
OUTPUT_DIR="$WORK_DIR/analysis/position/2026-11-05_row_attention"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/row_attention"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$JOB" in
  gpu)
    for MODEL in $MODELS; do
      # shellcheck disable=SC2086
      uv run --frozen python -m analysis_tools.position.row_attention --model "$MODEL" --stages $STAGES \
        --output "$OUTPUT_DIR" --flow-run "$FLOW_RUN" --vl-gen "$VL_GEN" --r2-summary "$R2_SUMMARY" \
        --top-heads "$TOP_HEADS" --n-maps "$N_MAPS" --n-edits "$N_EDITS" $TORCH_DELTANET
    done ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.row_attention_analysis --run "$OUTPUT_DIR" \
      --flow-run "$FLOW_RUN" --r2-run "$R2_RUN" --figures "$FIGURE_DIR" ;;
  *) echo "Unknown JOB=$JOB" >&2; exit 1 ;;
esac
