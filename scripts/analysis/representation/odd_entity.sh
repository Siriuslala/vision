#!/usr/bin/env bash
# Experiment 2.6: comparison mechanism for locating the odd entity, see
# docs/analysis/representation/design_odd_entity.md.  Configure this section, then run the script
# without terminal parameters.
# STAGE: build (CPU: images and items) -> behave (generation) -> spotcheck -> capture (entity states)
#        -> patch (context replacement) -> analyze (CPU: metrics and figures).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="0"
RUN_DATE="2026-10-20"
STAGE="build"
SEED=0
CORE_KINDS="red color20 color10 gap16 gap2"   # 6x6 std / swap / homT / homX quadruples
AUX_KINDS="red color20 gap16 gap2"            # grid-size, heterogeneous and boundary groups
CORE_PAIRS=72                                 # two per target cell of the 6x6 grid
AUX_PAIRS=36
GRID_SIZES="3 10"
BOUNDARY_PER_ORIENT=12
NATURAL_PAIRS=48                              # 3x3 grids of RefCOCO object crops; 0 disables
THINK_PAIRS=16                                # think-then-answer subset per core kind and for natural crops
BATCH_DIRECT=32
BATCH_THINK=8
BATCH_CAPTURE=12
BATCH_PATCH=8
PATCH_KINDS="red color20 gap16 gap2"
PATCH_PAIRS=24                                # first 24 correctly answered images per kind and variant
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_odd_entity"
FIG_DIR="$ROOT_DIR/docs/analysis/representation/figures/odd_entity"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$STAGE" == "analyze" ]]; then
  uv run --frozen python -m analysis_tools.representation.odd_entity_analysis \
    --output "$OUTPUT_DIR" --figures "$FIG_DIR" --seed "$SEED"
  exit 0
fi
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.odd_entity_run \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --seed "$SEED" \
  --core-kinds $CORE_KINDS --aux-kinds $AUX_KINDS --core-pairs "$CORE_PAIRS" --aux-pairs "$AUX_PAIRS" \
  --grid-sizes $GRID_SIZES --boundary-per-orient "$BOUNDARY_PER_ORIENT" --natural-pairs "$NATURAL_PAIRS" \
  --refcoco-dir "${REFCOCO_DIR:-}" --think-pairs "$THINK_PAIRS" --batch-direct "$BATCH_DIRECT" \
  --batch-think "$BATCH_THINK" --batch-capture "$BATCH_CAPTURE" --batch-patch "$BATCH_PATCH" \
  --patch-kinds $PATCH_KINDS --patch-pairs "$PATCH_PAIRS"
