#!/usr/bin/env bash
# Experiment 3.7 round 2: ViT position on synthetic vs natural images, PE vs RoPE reliance
# (docs/analysis/position/design_vit_layout.md, section "第二轮").
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"           # Qwen3.5-9B or Qwen3-VL-8B-Instruct
TAG="${TAG:-q35}"                      # model tag in artifact names: q35 or q3vl
GPU_IDS="${GPU_IDS:-4}"                # never use GPU 1 (reserved)
RUN_NAME="2026-10-16_vit_layout_r2"
# comma list of: ladder, spot, edits, capture, erase, l20, smalldisc, layout; or "figures"
STAGES="${STAGES:-ladder,spot}"
SEED=20261016
LAYOUT_SEED=20261015                   # round-1 layout cases
BATCH_SIZE="${BATCH_SIZE:-32}"         # rows per left-padded generate_batch call
VIT_BATCH=16                           # images per vision-only forward
MAX_NEW_TOKENS=40
LADDER_N=120                           # RefCOCO test cases with valid displaced regions on every stimulus
PROBE_TOKENS=64                        # base merger tokens kept per image for the position probe
EDITS_N=300                            # RefCOCO test cases of the whole-image edit stage
CAPTURE_TRAIN=320                      # natural train / synthetic layout images for the subspace
CAPTURE_TEST=100                       # natural test / flipped / shuffled images for the subspace
ERASE_N=300                            # RefCOCO test cases of the erasure stage (plus 108 discs)
LAYOUT_N=1600
N_SPOT=20
NATURAL_RUN="$WORK_DIR/analysis/position/2026-10-14_probe_refinement"            # experiment 3.4 images
COORD_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"  # experiment 3.5
ROUND1_RUN="$WORK_DIR/analysis/position/2026-10-15_vit_layout"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/vit_layout_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.position.vit_layout_r2_analysis --run "$OUTPUT_DIR" --round1 "$ROUND1_RUN" \
    --coord-run "$WORK_DIR/analysis/position/2026-10-14_coordinate_readout"
  uv run --frozen python -m analysis_tools.position.vit_layout_r2_figures --run "$OUTPUT_DIR" --round1 "$ROUND1_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.vit_layout_r2_run \
  --model "$MODEL" --tag "$TAG" --output "$OUTPUT_DIR" --natural "$NATURAL_RUN" --coord-cases "$COORD_CASES" \
  --refcoco "$REFCOCO_DIR" --stages "$STAGES" --seed "$SEED" --layout-seed "$LAYOUT_SEED" --batch-size "$BATCH_SIZE" \
  --vit-batch "$VIT_BATCH" --max-new-tokens "$MAX_NEW_TOKENS" --ladder-n "$LADDER_N" --probe-tokens "$PROBE_TOKENS" \
  --edits-n "$EDITS_N" --capture-train "$CAPTURE_TRAIN" --capture-test "$CAPTURE_TEST" --erase-n "$ERASE_N" \
  --layout-n "$LAYOUT_N" --n-spot "$N_SPOT" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run_${TAG}_${STAGES//,/-}.log"
