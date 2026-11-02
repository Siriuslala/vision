#!/usr/bin/env bash
# Experiment 2.7: mechanism of visual-search asymmetry, see
# docs/analysis/representation/design_search_asymmetry.md.  Configure this section, then run the script
# without terminal parameters.
# STAGE: build (CPU: images and items) -> behave (generation) -> spotcheck -> capture (ViT neurons and
#        entity states) -> select (CPU: neuron sets, writes neurons.json) -> edit (symmetric neuron edits)
#        -> mech (mixer patching, gate edits, attention edges) -> analyze (CPU: tables and figures).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="0"
MAX_MEMORY=""                                   # split over GPU_IDS when no single GPU has room, e.g. "0:12GiB,1:12GiB"
RUN_DATE="2026-10-21"
STAGE="build"
SEED=0
PAIRS="qo tilt length curve closure sat size"   # feature pairs (A = feature present / more, B = standard)
MAIN_GRID=6
MAIN_SUBTLE=72                                  # 6x6 subtle level: two items per target cell
MAIN_SALIENT=36                                 # 6x6 salient level: one item per target cell
SETSIZE_GRIDS="3 10"                            # set-size conditions for the subtle level
SETSIZE_ITEMS=36
HOM_ITEMS=24                                    # 6x6 subtle items that also get homA / homB donors
CONJ_GRIDS="3 6 10"                             # conjunction and feature-search controls
CONJ_ITEMS="36 72 36"
BATCH_DIRECT=32
BATCH_CAPTURE=12
BATCH_MECH=8
OE_KINDS="gap2 red"                             # experiment-2.6 quadruples added to capture / edit
NEURON_BLOCKS="3 6 9 12 15 18"
EDIT_BLOCKS="3 6 9 12"
TOP_K=32                                        # neurons per block and direction in the edit stage
EDIT_ITEMS=24
EDIT_TAG=""                                     # second edit round: TOP_K=256, EDIT_BLOCKS="3 6 9 12 15 18", EDIT_TAG="_k256"
MECH_KINDS="oe:gap2 oe:red oe:gap16 oe:color20 sa:qo sa:closure"
MECH_ITEMS=16                                   # correctly answered images per kind and variant
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_search_asymmetry"
ODD_ENTITY_DIR="$WORK_DIR/analysis/representation/2026-10-20_odd_entity"
FIG_DIR="$ROOT_DIR/docs/analysis/representation/figures/search_asymmetry"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$STAGE" == "select" || "$STAGE" == "analyze" ]]; then
  uv run --frozen python -m analysis_tools.representation.search_asymmetry_analysis \
    --stage "$STAGE" --output "$OUTPUT_DIR" --figures "$FIG_DIR" --odd-entity "$ODD_ENTITY_DIR" \
    --top-k "$TOP_K" --tag "$EDIT_TAG" --seed "$SEED"
  exit 0
fi
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.search_asymmetry_run \
  --stage "$STAGE" --model "$MODEL" ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} --output "$OUTPUT_DIR" --odd-entity "$ODD_ENTITY_DIR" --seed "$SEED" \
  --pairs $PAIRS --main-grid "$MAIN_GRID" --main-subtle "$MAIN_SUBTLE" --main-salient "$MAIN_SALIENT" \
  --setsize-grids $SETSIZE_GRIDS --setsize-items "$SETSIZE_ITEMS" --hom-items "$HOM_ITEMS" \
  --conj-grids $CONJ_GRIDS --conj-items $CONJ_ITEMS --batch-direct "$BATCH_DIRECT" \
  --batch-capture "$BATCH_CAPTURE" --batch-mech "$BATCH_MECH" --oe-kinds $OE_KINDS \
  --neuron-blocks $NEURON_BLOCKS --edit-blocks $EDIT_BLOCKS --edit-items "$EDIT_ITEMS" --edit-tag "$EDIT_TAG" \
  --mech-kinds $MECH_KINDS --mech-items "$MECH_ITEMS"
