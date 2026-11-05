#!/usr/bin/env bash
# Experiment 2.7 round 2: generality and mechanism of visual-search asymmetry, see
# docs/analysis/representation/results_2026-11-03_search_asymmetry_r2.md.  Configure this section, then run the
# script without terminal parameters.
# STAGE: build (CPU: images, items) -> behave (generation) -> spotcheck -> capture (neurons, states, attention)
#        -> states (entity states of the target-index and layout sets)
#        -> direction (CPU: oddness directions, writes directions.npz) -> steer (add / remove the direction)
#        -> steer2 (random-direction controls, reader-head ablation, projection ablation at all sites)
#        -> analyze (CPU: tables, figures, summary.json).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="2"                                     # must be an idle GPU claimed through the group's claim helper
RUN_DATE="2026-11-03"
STAGE="build"
SEED=0
NEW_PAIRS="term plus dot circsq plusx"          # three pixel-subset pairs, two mutually exclusive pairs
MAIN_SUBTLE=72
MAIN_SALIENT=36
HOM_ITEMS=24                                    # items with homA / homB donors (capture)
SALIENT_HOM_PAIRS="qo closure"
INDEX_PAIRS="qo closure tilt"                   # feature-absent targets on every 6x6 cell
INDEX_ITEMS=216
LAYOUT_PAIRS="qo closure tilt"
LAYOUTS="3x12 12x3"
LAYOUT_ITEMS=108
LAYOUT_PRESENT_ITEMS=36
R1_PAIRS="qo closure curve tilt"                # round-1 subtle quadruples re-captured with attention
BATCH_DIRECT=32
BATCH_CAPTURE=8
BATCH_MECH=8
NEURON_BLOCKS="6 9 12 18"
PATCH_BLOCK=9
PATCH_ITEMS=12
VIT_ATTN_BLOCKS="3 6 9 12 18 24"
STEER_PAIRS="qo closure curve tilt dot term"
STEER_ITEMS=12
STEER_SITES="dec0 dec8 dec16"
ALPHAS="1 2 4"
READ_LAYERS="12 16 20 24"
READER_HEADS="16:6 16:11 16:2 20:2 20:11 20:10"   # steer2: heads whose answer-query attention rises most under steering
PROJECT_SITES="dec0 dec4 dec8 dec12 dec16"      # steer2: sites of the simultaneous projection ablation (direction stage)
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_search_asymmetry_r2"
ROUND1_DIR="$WORK_DIR/analysis/representation/2026-10-21_search_asymmetry"
FIG_DIR="$ROOT_DIR/docs/analysis/representation/figures/search_asymmetry_r2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$STAGE" == "direction" || "$STAGE" == "analyze" ]]; then
  uv run --frozen python -m analysis_tools.representation.search_asymmetry_r2_analysis \
    --stage "$STAGE" --output "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --figures "$FIG_DIR" --steer-sites $PROJECT_SITES
  exit 0
fi
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.search_asymmetry_r2_run \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --round1 "$ROUND1_DIR" --seed "$SEED" \
  --new-pairs $NEW_PAIRS --main-subtle "$MAIN_SUBTLE" --main-salient "$MAIN_SALIENT" --hom-items "$HOM_ITEMS" \
  --salient-hom-pairs $SALIENT_HOM_PAIRS --index-pairs $INDEX_PAIRS --index-items "$INDEX_ITEMS" \
  --layout-pairs $LAYOUT_PAIRS --layouts $LAYOUTS --layout-items "$LAYOUT_ITEMS" \
  --layout-present-items "$LAYOUT_PRESENT_ITEMS" --r1-pairs $R1_PAIRS --batch-direct "$BATCH_DIRECT" \
  --batch-capture "$BATCH_CAPTURE" --batch-mech "$BATCH_MECH" --neuron-blocks $NEURON_BLOCKS \
  --patch-block "$PATCH_BLOCK" --patch-items "$PATCH_ITEMS" --vit-attn-blocks $VIT_ATTN_BLOCKS \
  --steer-pairs $STEER_PAIRS --steer-items "$STEER_ITEMS" --steer-sites $STEER_SITES --alphas $ALPHAS \
  --read-layers $READ_LAYERS --reader-heads $READER_HEADS --project-sites $PROJECT_SITES
