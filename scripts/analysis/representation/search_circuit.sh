#!/usr/bin/env bash
# Experiment 2.7 round 3: the computation that locates the odd entity, see
# docs/analysis/representation/design_2026-11-05_search_circuit.md and results_2026-11-05_search_circuit.md.
# Configure this section, then run the script without terminal parameters.
# STAGE: build (CPU: odd / corrupted image pairs) -> selftest (instrumented forward = native forward)
#        -> behave (generation) -> spotcheck -> trace (DeltaNet prediction errors, reader-logit attribution)
#        -> knock (DeltaNet memory / full-attention knockouts, DeltaNet state priming) -> patch (noising at the target)
#        -> circuit (heads and neurons: per-head blocking/patching, neuron attribution, necessity and sufficiency)
#        -> analyze (CPU: tables, figures, summary.json).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="7"                                     # must be an idle GPU claimed through the group's claim helper
RUN_DATE="2026-11-05"
STAGE="build"
SEED=0
ATLAS_PAIRS="qo closure tilt curve sat size term plus dot circsq plusx"
ATLAS_SALIENT_PAIRS="qo closure"
ATLAS_ITEMS=24                                  # per pair, level and target variant (6 x 6)
PREV_CONDITIONS="qo:subtle:swap qo:salient:swap closure:subtle:swap closure:salient:swap dot:subtle:swap term:subtle:swap plus:subtle:swap qo:subtle:std closure:subtle:std dot:subtle:std term:subtle:std tilt:subtle:std"
PREV_ITEMS=12                                   # per condition and number of raster-earlier entities (0 2 5 10 21)
MAG="qo:3,6,10,14 closure:2,4,8,12"             # Q tail length / open-circle gap in pixels
MAG_ITEMS=36
SHIFT_PAIRS="qo closure"
SHIFT_BLANKS="0 6 12 18"                        # number of leading blank cells in homogeneous images
SHIFT_ITEMS=6
BATCH_DIRECT=32
BATCH_TRACE=16
BATCH_KNOCK=12
BATCH_PATCH=16
KNOCK_SETS="atlas prev shift"
KNOCK_LIMIT=0                                   # max items per condition cell (0 = all)
PATCH_LIMIT=12                                  # odd items per atlas condition
CIRCUIT_N_CMP=80                                # circuit stage: correctly answered comparison-class targets
CIRCUIT_N_DET=40                                # circuit stage: correctly answered detector-class targets
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_search_circuit"
FIG_DIR="$ROOT_DIR/docs/analysis/representation/figures/search_circuit"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
if [[ "$STAGE" == "circuit" ]]; then
  uv run --frozen python -m analysis_tools.representation.search_circuit_heads --model "$MODEL" --output "$OUTPUT_DIR" \
    --n-cmp "$CIRCUIT_N_CMP" --n-det "$CIRCUIT_N_DET" --batch "$BATCH_PATCH" --seed "$SEED"
  exit 0
fi
if [[ "$STAGE" == "analyze" ]]; then
  uv run --frozen python -m analysis_tools.representation.search_circuit_analysis --output "$OUTPUT_DIR" --figures "$FIG_DIR"
  exit 0
fi
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.search_circuit_run \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --seed "$SEED" \
  --atlas-pairs $ATLAS_PAIRS --atlas-salient-pairs $ATLAS_SALIENT_PAIRS --atlas-items "$ATLAS_ITEMS" \
  --prev-conditions $PREV_CONDITIONS --prev-items "$PREV_ITEMS" --mag $MAG --mag-items "$MAG_ITEMS" \
  --shift-pairs $SHIFT_PAIRS --shift-blanks $SHIFT_BLANKS --shift-items "$SHIFT_ITEMS" \
  --batch-direct "$BATCH_DIRECT" --batch-trace "$BATCH_TRACE" --batch-knock "$BATCH_KNOCK" --batch-patch "$BATCH_PATCH" \
  --knock-sets $KNOCK_SETS --knock-limit "$KNOCK_LIMIT" --patch-limit "$PATCH_LIMIT"
