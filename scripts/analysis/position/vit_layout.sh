#!/usr/bin/env bash
# Experiment 3.7: ViT position and layout representations (docs/analysis/position/design_vit_layout.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"           # Qwen3.5-9B or Qwen3-VL-8B-Instruct
TAG="${TAG:-q35}"                      # model tag in artifact names: q35 or q3vl
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="2026-10-15_vit_layout"
# comma list of: table, norms, coord, spot, t2, swap, probe, layout, leace, reread, refcoco; or "figures"
STAGES="${STAGES:-table,norms}"
SEED=20261015
CANVASES="256,384,512,768,1024,1536"   # coord stage, white canvas (disc diameter = canvas / 8)
NOISE_CANVASES="256,512,1024"          # coord stage, disc on gray pixel noise (subset of conditions)
T2_CANVASES="256,512,768,1024,1536"    # recognition control
SWAP_CANVASES="256,512,768,1024"       # swap stage (Qwen3.5 only)
BATCH_SIZE="${BATCH_SIZE:-16}"     # generate_batch / unpadded condition batches
VIT_BATCH=16                           # images per vision-only forward
MAX_NEW_TOKENS=64
N_SPOT=20
PROBE_TRAIN=320
PROBE_VAL=80
PROBE_TEST=100
LAYOUT_N=1600
LEACE_NATURAL=320
LEACE_SYNTHETIC=320
LEACE_DIAMETERS="32,64,128"           # disc diameters (pixels) of the erasure test, 36 images each
REREAD_N=48
NATURAL_RUN="$WORK_DIR/analysis/position/2026-10-14_probe_refinement"   # RefCOCO images of experiment 3.4
REFCOCO_CASES="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end/refcoco"       # RefCOCO val cases of experiment 4.2
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/vit_layout"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"   # host CPUs are shared; avoid BLAS oversubscription
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.position.vit_layout_analysis --run "$OUTPUT_DIR"
  uv run --frozen python -m analysis_tools.position.vit_layout_figures --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.vit_layout_run \
  --model "$MODEL" --tag "$TAG" --output "$OUTPUT_DIR" --natural "$NATURAL_RUN" --refcoco-cases "$REFCOCO_CASES" --stages "$STAGES" --seed "$SEED" \
  --canvases "$CANVASES" --noise-canvases "$NOISE_CANVASES" --t2-canvases "$T2_CANVASES" --swap-canvases "$SWAP_CANVASES" \
  --batch-size "$BATCH_SIZE" --vit-batch "$VIT_BATCH" --max-new-tokens "$MAX_NEW_TOKENS" --n-spot "$N_SPOT" \
  --probe-train "$PROBE_TRAIN" --probe-val "$PROBE_VAL" --probe-test "$PROBE_TEST" --layout-n "$LAYOUT_N" \
  --leace-natural "$LEACE_NATURAL" --leace-synthetic "$LEACE_SYNTHETIC" --leace-diameters "$LEACE_DIAMETERS" --reread-n "$REREAD_N" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run_${TAG}_${STAGES//,/-}.log"
