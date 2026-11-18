#!/usr/bin/env bash
# Experiment 3.16: prior localization of the coordinate-readout defect on the base model (no training)
# (docs/analysis/position/results_2026-11-05_readout_prior_localization.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=features,transplant,textsrc,firsttok,grad GPU_IDS=3 bash scripts/analysis/position/readout_prior.sh   # GPU
#   STAGES=analysis bash scripts/analysis/position/readout_prior.sh                                              # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                     # an idle GPU claimed through the claim helper; required for GPU stages
RUN_NAME="2026-11-05_readout_prior"
STAGES="${STAGES:-analysis}"
BATCH=24                                   # rows per batched prefill (features / transplant)
GRAD_BATCH=6                               # rows per gradient batch
GRAD_BATCHES=12                            # gradient batches (72 rows: disc x / y sweeps and RefCOCO points)
INJECT_SITE="${INJECT_SITE:-after_py}"      # text-given source site whose periodic code is injected into image tokens

OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"      # 3.6 disc sweeps and cases
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-11-03_targeted_finetune_r2" # 3.13 round 2 (test split, base capture)
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME"
CAPTURE_DIR="$DATA_DIR/scratch/analysis/position/2026-11-03_targeted_finetune_r2/features/capture"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/readout_prior"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  # Small dense solves: one BLAS thread is much faster than many on the shared, busy CPU.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  uv run --frozen python -m analysis_tools.position.readout_prior_analysis \
    --run "$OUTPUT_DIR" --features "$FEATURE_DIR" --capture "$CAPTURE_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
[[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.position.readout_prior_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --round1 "$ROUND1_DIR" --finetune-run "$FINETUNE_RUN" \
  --features "$FEATURE_DIR" --batch "$BATCH" --grad-batch "$GRAD_BATCH" --grad-batches "$GRAD_BATCHES" \
  --inject-site "$INJECT_SITE" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
