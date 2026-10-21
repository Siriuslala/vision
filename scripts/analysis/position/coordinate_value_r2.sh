#!/usr/bin/env bash
# Experiment 3.6 round 2: digit-level output mechanism of coordinate values
# (docs/analysis/position/design_coordinate_value.md, section "第二轮").
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-4}"          # never use GPU 1 (reserved)
RUN_NAME="2026-10-16_coordinate_value_r2"
ROUND1_NAME="2026-10-15_coordinate_value"
STAGES="${STAGES:-formats,neurons,ablate,pathpatch}"   # GPU stages; "analysis" = CPU analysis + figures;
                                                       # "neuron_ablate" needs the analysis output first
BATCH_SIZE=16                    # rows per batched forward / greedy batch
MAX_ROWS=32                      # intervention rows per forward
MAX_NEW_TOKENS=48
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
ROUND1_DIR="$WORK_DIR/analysis/position/$ROUND1_NAME"
FEATURE_DIR="$DATA_DIR/scratch/analysis/position/$RUN_NAME/features"     # neuron activations (about 2 GB float16)
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"   # experiment-3.5 RefCOCO test cases
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_value_r2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"
[[ -e "$OUTPUT_DIR/features" ]] || ln -s "$FEATURE_DIR" "$OUTPUT_DIR/features"
if [[ "$STAGES" == "analysis" ]]; then
  export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
  uv run --frozen python -m analysis_tools.position.coordinate_value_r2_analysis \
    --run "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --features "$FEATURE_DIR"
  uv run --frozen python -m analysis_tools.position.coordinate_value_r2_figures \
    --run "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.coordinate_value_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --features "$FEATURE_DIR" --round1 "$ROUND1_DIR" \
  --refcoco-cases "$REFCOCO_CASES" --stages "$STAGES" --batch-size "$BATCH_SIZE" --max-rows "$MAX_ROWS" \
  --max-new-tokens "$MAX_NEW_TOKENS" 2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run.log"
