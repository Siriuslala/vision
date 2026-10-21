#!/usr/bin/env bash
# Experiment 3.9: coordinate readout tool (docs/analysis/position/design_readout_tool.md).
# Order: STAGES=prepare -> verify,capture,own,spotcheck (GPU) -> fit (CPU) -> slot,loop (GPU) -> fit_slot, evaluate, figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"               # never GPU 1 (reserved); a comma list splits the weights
MAX_MEMORY="${MAX_MEMORY:-}"          # GiB per listed GPU for the split, e.g. "14,14"; empty = one GPU
RUN_NAME="2026-10-16_readout_tool"
STAGES="${STAGES:-prepare}"           # prepare | verify,capture,own,spotcheck | fit | slot,loop | fit_slot | evaluate | figures
MAX_TOKENS=8000                       # padded visual-token budget per batch (shared GPUs)
MAX_BATCH=32
MAX_BATCH_THINK=48
MAX_NEW_TOKENS=48
MAX_THINK_TOKENS=768
N_SPOT=20
N_FREE="refcoco:200,synth:100,disc:100,t6:96,layout:96,ring:96"
WORKERS=8
REFCOCO="$REFCOCO_DIR"
REFCOCO_SOURCE="$WORK_DIR/analysis/position/2026-10-14_probe_refinement"      # experiment-3.4 split
T6_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
SUPPLEMENT_SOURCE="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
SCRATCH="$DATA_DIR/scratch/analysis/position/$RUN_NAME"     # images (~3 GB) and features (~6 GB)
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/readout_tool"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR" "$SCRATCH/images" "$SCRATCH/features" "$SCRATCH/features_slot"
[[ -e "$OUTPUT_DIR/images" ]] || ln -s "$SCRATCH/images" "$OUTPUT_DIR/images"
[[ -e "$OUTPUT_DIR/features" ]] || ln -s "$SCRATCH/features" "$OUTPUT_DIR/features"
[[ -e "$OUTPUT_DIR/features_slot" ]] || ln -s "$SCRATCH/features_slot" "$OUTPUT_DIR/features_slot"

case "$STAGES" in
  fit|fit_slot|evaluate)
    uv run --frozen python -m analysis_tools.position.readout_tool_analysis --run "$OUTPUT_DIR" --stage "$STAGES" \
      2>&1 | tee -a "$OUTPUT_DIR/analysis.log" ;;
  figures)
    uv run --frozen python -m analysis_tools.position.readout_tool_figures --run "$OUTPUT_DIR" --figures "$FIGURE_DIR" ;;
  *)
    uv run --frozen python -m analysis_tools.position.readout_tool_run \
      --model "$MODEL" --output "$OUTPUT_DIR" --images "$SCRATCH/images" --features "$SCRATCH/features" \
      --refcoco "$REFCOCO" --refcoco-source "$REFCOCO_SOURCE" --t6-source "$T6_SOURCE" \
      --supplement-source "$SUPPLEMENT_SOURCE" --stages "$STAGES" --workers "$WORKERS" \
      --max-tokens "$MAX_TOKENS" --max-batch "$MAX_BATCH" --max-batch-think "$MAX_BATCH_THINK" \
      --max-new-tokens "$MAX_NEW_TOKENS" --max-think-tokens "$MAX_THINK_TOKENS" --n-spot "$N_SPOT" --n-free "$N_FREE" --max-memory "$MAX_MEMORY" \
      2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log" ;;
esac
