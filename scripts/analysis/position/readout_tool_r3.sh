#!/usr/bin/env bash
# Experiment 3.9 round 3: reduced-input coordinate readout tool (docs/analysis/position/design_readout_tool.md, section 11).
# Reuses the round-2 capture.  Order: STAGES=annotate,zeroshot (CPU) -> fit (small GPU; FIT_ONLY splits configs
# over processes) -> capture_heads (GPU, head-output vectors) -> fit_vec (CPU) -> evaluate, figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"               # check nvidia-smi and the GPU ledger first
STAGES="${STAGES:-annotate}"          # annotate | zeroshot | edges | fit | capture_heads | fit_vec | capture_tf | compare | evaluate | figures
FIT_DEVICE="${FIT_DEVICE:-cuda}"      # reader training device (< 4 GB)
FIT_ONLY="${FIT_ONLY:-}"              # comma list of reader configs (readout_tool_r3_analysis.CONFIGS); empty = all
FIT_TAG="${FIT_TAG:-main}"            # output tag of this fitting process
SHARD="${SHARD:-0/1}"                 # capture shard k/n
MAX_MEMORY="${MAX_MEMORY:-}"          # GiB per listed GPU to split the weights, e.g. "11,11"; empty = one GPU
MAX_TOKENS="${MAX_TOKENS:-8000}"     # padded visual-token budget per capture batch
MAX_BATCH="${MAX_BATCH:-32}"
RUN_NAME="2026-10-21_readout_tool_r3"
R2_RUN="$WORK_DIR/analysis/position/2026-10-20_readout_tool_r2"   # capture, cases, own answers
R1_RUN="$WORK_DIR/analysis/position/2026-10-16_readout_tool"
TF_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"      # experiment 3.13 (test set and answers)
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
SCRATCH="$DATA_DIR/scratch/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/readout_tool_r3"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR" "$SCRATCH/heads"
[[ -e "$OUTPUT_DIR/heads" ]] || ln -s "$SCRATCH/heads" "$OUTPUT_DIR/heads"

IFS=',' read -ra STAGE_LIST <<< "$STAGES"
for stage in "${STAGE_LIST[@]}"; do
  case "$stage" in
    annotate|zeroshot|edges|fit|evaluate)
      uv run --frozen python -m analysis_tools.position.readout_tool_r3_analysis --run "$R2_RUN" --r1 "$R1_RUN" \
        --out "$OUTPUT_DIR" --stage "$stage" --refcoco "$REFCOCO_DIR" --device "$FIT_DEVICE" --only "$FIT_ONLY" \
        --tag "$FIT_TAG" 2>&1 | tee -a "$OUTPUT_DIR/analysis_${stage}_${FIT_TAG}.log" ;;
    capture_heads|fit_vec|capture_tf)
      uv run --frozen python -m analysis_tools.position.readout_tool_r3_run --model "$MODEL" --r2 "$R2_RUN" --r1 "$R1_RUN" \
        --out "$OUTPUT_DIR" --stage "$stage" --shard "$SHARD" --max-memory "$MAX_MEMORY" --max-tokens "$MAX_TOKENS" \
        --max-batch "$MAX_BATCH" --tf-test "$TF_RUN/test.jsonl" 2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${stage}_s${SHARD%%/*}.log" ;;
    compare)
      uv run --frozen python -m analysis_tools.position.readout_tool_r3_compare --r2 "$R2_RUN" --out "$OUTPUT_DIR" \
        --tf-run "$TF_RUN" 2>&1 | tee -a "$OUTPUT_DIR/compare.log" ;;
    figures)
      uv run --frozen python -m analysis_tools.position.readout_tool_r3_figures --run "$R2_RUN" --r1 "$R1_RUN" \
        --out "$OUTPUT_DIR" --figures "$FIGURE_DIR" ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
