#!/usr/bin/env bash
# Experiment 7.2 round 2: mechanism check of the located coordinate fine-tuning
# (docs/analysis/application/results_2026-11-11_located_finetune_r2.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=link bash scripts/analysis/application/located_r2.sh                                          # CPU
#   STAGES=restore,qswap CONDITIONS=mlp1_4_text,mlp1_4_first,mlp1_4_img GPU_IDS=3 bash scripts/analysis/application/located_r2.sh
#   STAGES=prompts CONDITIONS=base,mlp1_4_img,mlp1_4_first,mlp1_4_text GPU_IDS=3 bash ...
#   STAGES=train,digits,coord,ordinal CONDITIONS=<round-2 condition> GPU_IDS=3 bash ...
#   STAGES=analysis bash scripts/analysis/application/located_r2.sh                                      # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"
RUN_NAME="${RUN_NAME:-2026-11-11_located_finetune_r2}"   # override only for smoke tests
STAGES="${STAGES:-analysis}"
CONDITIONS="${CONDITIONS:-mlp1_4_text}"
BATCH=12                                    # rows per batched prefill (restore / qswap)
TOKEN_BUDGET=16000                          # generation batch budget (visual + 200 tokens per item)
MAX_NEW_TOKENS=64

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
ROUND1_RUN="$WORK_DIR/analysis/application/2026-11-05_located_finetune"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-11-03_targeted_finetune_r2"
FEATURE_DIR="$DATA_DIR/scratch/analysis/application/$RUN_NAME"
RP2_RUN="$WORK_DIR/analysis/position/2026-11-11_readout_prior_r2"
RP2_FEATURES="$DATA_DIR/scratch/analysis/position/2026-11-11_readout_prior_r2"
CAPTURE72="$DATA_DIR/scratch/analysis/application/2026-11-05_located_finetune/capture"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/located_finetune_r2"
GENERAL_DATA="$DATA_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.located_r2_analysis \
    --run "$OUTPUT_DIR" --round1-run "$ROUND1_RUN" --rp2-run "$RP2_RUN" --rp2-features "$RP2_FEATURES" \
    --capture72 "$CAPTURE72" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" == "link" ]]; then
  uv run --frozen python -m analysis_tools.application.located_r2_run --output "$OUTPUT_DIR" --stages link \
    --round1-run "$ROUND1_RUN" --round1 "$ROUND1_DIR" --finetune-run "$FINETUNE_RUN" --features "$FEATURE_DIR"
  exit 0
fi
[[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.application.located_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --conditions "$CONDITIONS" --round1-run "$ROUND1_RUN" \
  --round1 "$ROUND1_DIR" --finetune-run "$FINETUNE_RUN" --features "$FEATURE_DIR" --batch "$BATCH" \
  --token-budget "$TOKEN_BUDGET" --max-new-tokens "$MAX_NEW_TOKENS" --data-dir "$GENERAL_DATA" ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
