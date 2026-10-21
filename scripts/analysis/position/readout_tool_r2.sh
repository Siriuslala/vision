#!/usr/bin/env bash
# Experiment 3.9 round 2: revised coordinate readout tool (docs/analysis/position/design_readout_tool.md, round 2).
# Order: STAGES=prepare (CPU) -> pca (GPU) -> capture (GPU; SHARD=k/n per GPU) -> own,spotcheck,freecall (GPU)
#        -> fit (CPU/GPU) -> dissoc (GPU) -> evaluate, figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"               # check nvidia-smi and the GPU ledger first
STAGES="${STAGES:-prepare}"           # prepare | pca | capture | own,spotcheck,freecall | fit | dissoc | evaluate | figures
SHARD="${SHARD:-0/1}"                 # capture shard k/n (one process per GPU)
MAX_MEMORY="${MAX_MEMORY:-}"          # GiB per listed GPU to split the weights, e.g. "11,11"; empty = one GPU
RUN_NAME="2026-10-20_readout_tool_r2"
R1_RUN="$WORK_DIR/analysis/position/2026-10-16_readout_tool"      # round-1 cases, images, own answers
MAX_TOKENS=8000                       # padded visual-token budget per batch
MAX_BATCH=32
WORKERS=8
FIT_DEVICE="${FIT_DEVICE:-cuda}"      # reader training device (small: < 4 GB)
FIT_ONLY="${FIT_ONLY:-}"              # comma list of readers for this fitting process ("attnfree" = attention-free set); empty = all
FIT_TAG="${FIT_TAG:-main}"            # output tag of this fitting process (readers_<tag>.json, predictions_<tag>.npz)
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
SCRATCH="$DATA_DIR/scratch/analysis/position/$RUN_NAME"            # graded images and captured features
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/readout_tool_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
for d in images features features_ord features_digit; do
  mkdir -p "$SCRATCH/$d"
  [[ -e "$OUTPUT_DIR/$d" ]] || { mkdir -p "$OUTPUT_DIR"; ln -s "$SCRATCH/$d" "$OUTPUT_DIR/$d"; }
done

case "$STAGES" in
  fit|fit_digit|evaluate)
    uv run --frozen python -m analysis_tools.position.readout_tool_r2_analysis --run "$OUTPUT_DIR" --r1 "$R1_RUN" \
      --stage "$STAGES" --device "$FIT_DEVICE" --only "$FIT_ONLY" --tag "$FIT_TAG" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log" ;;
  figures)
    uv run --frozen python -m analysis_tools.position.readout_tool_r2_figures --run "$OUTPUT_DIR" --r1 "$R1_RUN" \
      --figures "$FIGURE_DIR" ;;
  *)
    uv run --frozen python -m analysis_tools.position.readout_tool_r2_run \
      --model "$MODEL" --output "$OUTPUT_DIR" --images "$SCRATCH/images" --features "$SCRATCH/features" \
      --r1 "$R1_RUN" --refcoco "$REFCOCO_DIR" --behavior-root "$WORK_DIR/analysis/behavior" --stages "$STAGES" \
      --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --max-batch "$MAX_BATCH" --shard "$SHARD" --max-memory "$MAX_MEMORY" \
      2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_s${SHARD%%/*}.log" ;;
esac
