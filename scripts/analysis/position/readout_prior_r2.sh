#!/usr/bin/env bash
# Experiment 3.16 round 2: theory and precise location of the coordinate-readout defect (base model + cited 7.2
# adapters; docs/analysis/position/results_2026-11-11_readout_prior_r2.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=capture,mismatch,layers,steer,grad2 GPU_IDS=3 bash scripts/analysis/position/readout_prior_r2.sh   # GPU
#   STAGES=vlcapture MODEL=Qwen3-VL-8B-Instruct GPU_IDS=3 bash scripts/analysis/position/readout_prior_r2.sh
#   STAGES=analysis bash scripts/analysis/position/readout_prior_r2.sh                                        # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"
GPU_IDS="${GPU_IDS:-}"                      # an idle GPU claimed through the claim helper; required for GPU stages
RUN_NAME="${RUN_NAME:-2026-11-11_readout_prior_r2}"   # override only for smoke tests
STAGES="${STAGES:-analysis}"
CONDITIONS="${CONDITIONS:-base,mlp1_4_img,mlp1_4_first,mlp1_4_text}"   # capture: base model and cited 7.2 adapters
BATCH=16                                    # rows per batched prefill
JAC_BATCH=8                                 # rows per Jacobian batch (20 backward passes each)
GRAD_BATCH=6
GRAD_BATCHES=12
STEER_SITES="${STEER_SITES:-first,image,question}"
STEER_STEPS=60                              # Adam steps of the learned single vector per (site, layer)
STEER_BATCH=8
STEER_LR=0.001                              # Adam step per coordinate, in units of the median image-token residual norm (|v| grows ~0.064 x median per step)

OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-11-03_targeted_finetune_r2"
ADAPTER_RUN="${ADAPTER_RUN:-$WORK_DIR/analysis/application/2026-11-05_located_finetune}"     # 7.2 adapters
CAPTURE72="$DATA_DIR/scratch/analysis/application/2026-11-05_located_finetune/capture"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/readout_prior_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.position.readout_prior_r2_analysis \
    --run "$OUTPUT_DIR" --features "$FEATURE_DIR" --capture72 "$CAPTURE72" --figures "$FIGURE_DIR" --model-name "Qwen3.5-9B"
  exit 0
fi
[[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.position.readout_prior_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --round1 "$ROUND1_DIR" --finetune-run "$FINETUNE_RUN" \
  --adapter-run "$ADAPTER_RUN" --features "$FEATURE_DIR" --conditions "$CONDITIONS" --batch "$BATCH" \
  --jac-batch "$JAC_BATCH" --grad-batch "$GRAD_BATCH" --grad-batches "$GRAD_BATCHES" --steer-sites "$STEER_SITES" \
  --steer-steps "$STEER_STEPS" --steer-batch "$STEER_BATCH" --steer-lr "$STEER_LR" ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
