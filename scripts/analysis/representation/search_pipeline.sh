#!/usr/bin/env bash
# Experiment 2.7 round 4: the complete odd-entity pipeline from pixels to the (row, column) answer, see
# docs/analysis/representation/results_2026-11-11_search_pipeline.md.
# Configure this section, then run the script without terminal parameters.
# STAGE: prep (CPU: same-type images by replaying the round-3 build)
#        -> main (Qwen3.5: ViT exclusive detectors, captures, interventions) -> probe (Qwen3.5: reader outputs / values)
#        -> vl_behave -> vl_main (Qwen3-VL-8B: readers, mark decomposition, interventions)
#        -> analyze (CPU: tables, figures, summary.json).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
VL_MODEL="Qwen3-VL-8B-Instruct"
GPU_IDS="7"                                     # must be an idle GPU claimed through the group's claim helper
RUN_DATE="2026-11-11"
STAGE="analyze"
SEED=0
BATCH=16
LIMIT=0                                         # number of selected pairs to process (0 = all; smoke tests use 2)
VL_N_CMP=60                                     # Qwen3-VL: correctly answered comparison-class pairs (target cell >= 7)
VL_N_DET=40                                     # Qwen3-VL: correctly answered detector-class pairs
ROUND3_DIR="$WORK_DIR/analysis/representation/2026-11-05_search_circuit"
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_search_pipeline"
FIG_DIR="$ROOT_DIR/docs/analysis/representation/figures/search_pipeline"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
case "$STAGE" in
  prep|main|probe)
    uv run --frozen python -m analysis_tools.representation.search_pipeline_run --stage "$STAGE" --model "$MODEL" \
      --round3 "$ROUND3_DIR" --output "$OUTPUT_DIR" --batch "$BATCH" --seed "$SEED" --limit "$LIMIT" ;;
  vl_behave|vl_main)
    uv run --frozen python -m analysis_tools.representation.search_pipeline_vl --stage "${STAGE#vl_}" --model "$VL_MODEL" \
      --round3 "$ROUND3_DIR" --output "$OUTPUT_DIR" --batch "$BATCH" --seed "$SEED" --limit "$LIMIT" \
      --n-cmp "$VL_N_CMP" --n-det "$VL_N_DET" ;;
  analyze)
    uv run --frozen python -m analysis_tools.representation.search_pipeline_analysis --output "$OUTPUT_DIR" \
      --round3 "$ROUND3_DIR" --figures "$FIG_DIR" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
