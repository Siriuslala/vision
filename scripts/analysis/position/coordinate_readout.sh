#!/usr/bin/env bash
# Experiment 3.5: coordinate readout path (docs/analysis/position/design_coordinate_readout.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="2026-10-14_coordinate_readout"
STAGES="${STAGES:-c0,spotcheck}"       # comma list of: c0, spotcheck, c1, c1grad, c2, c4, fail, fail2; or "figures" (analysis + figures)
SEED=20261014
BATCH_SIZE=8                   # unpadded batches of equal-length prompts
MAX_NEW_TOKENS=96
N_SPOT=20                      # spot-check samples against the reference generate()
N_RANDOM=10                    # random head sets for the C4 ablation control
FAIL_ORDER="20:rowcol,10:point,20:point,10:rowcol,6:rowcol"   # T6 (grid:format) blocks for fail/fail2, in priority order
GAP=8                          # unused MRoPE positions before and after the image in the gap baseline
T6_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_readout"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.position.coordinate_readout_analysis --run "$OUTPUT_DIR"
  uv run --frozen python -m analysis_tools.position.coordinate_readout_figures \
    --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.coordinate_readout_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --t6-source "$T6_SOURCE" --stages "$STAGES" --seed "$SEED" \
  --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" --n-spot "$N_SPOT" --gap "$GAP" --n-random "$N_RANDOM" --fail-order "$FAIL_ORDER" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
