#!/usr/bin/env bash
# Experiment 3.11 round 2: coordinate readout on natural images
# (docs/analysis/position/results_2026-10-21_coordinate_flow_natural.md).
# Configure this section, then run this script without terminal parameters.
# STAGE: cases (CPU) | gpu (GPU_STAGES in one process) | analyze (CPU) | figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="7"                       # GPUs 0 and 1 are reserved by the PI
MAX_MEMORY=""                    # e.g. "0:11GiB,1:11GiB" to split the model over the GPUs in GPU_IDS
RUN_DATE="2026-10-21"
STAGE="gpu"
GPU_STAGES="gen spotcheck flip means ablate patch"  # every stage resumes from its records
SOURCE_DIR="$WORK_DIR/analysis/representation/2026-10-16_sees_right"     # 2.5e: RefCOCO cases, answers, attention
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-20_coordinate_flow"      # 3.11 round 1
PROBE_CASES="$WORK_DIR/analysis/position/2026-10-14_probe_refinement/cases.json"   # 3.4 RefCOCO source rows
N_MS=48                           # synthetic multi-shape images per size
GEN_GROUPS="rc ms_small ms_large disc t6g10"
PATCH_GROUPS="rc ms_small ms_large"
N_PATCH=160                       # cases per group in the patch stage
PATCH_ROWS=24                     # patched rows per forward
ABLATE_GROUPS="rc ms_small ms_large disc t6g10"
# Head sets for mean ablation (layer.head). edge_min / edge_max: heads whose RefCOCO attention centroid sits at the
# left-top / right-bottom box edges in the 2.5e readout (summary.json edges.classes); ctrl6: centred heads, same layers.
ABLATE_SETS="H11=20.11;edge_min=20.0,24.1,24.2,24.7,24.12;edge_max=20.10,24.5,24.6,24.8,24.9,28.0;edge_all=20.0,24.1,24.2,24.7,24.12,20.10,24.5,24.6,24.8,24.9,28.0;ctrl6=20.3,20.4,24.3,24.15,28.1,28.2"
ABLATE_FROM="answer"              # answer | text (ablate from the first post-image text token)
N_ABLATE=300
N_MEANS=64
BATCH_SIZE=16
MAX_NEW_TOKENS=48
SPOTCHECK=20
SEED=20261021
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_coordinate_flow_natural"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_flow_natural"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$STAGE" in
  analyze)
    uv run --frozen python -m analysis_tools.position.coordinate_flow_natural_analysis --run "$OUTPUT_DIR" --source "$SOURCE_DIR"
    exit 0 ;;
  figures)
    uv run --frozen python -m analysis_tools.position.coordinate_flow_natural_figures --run "$OUTPUT_DIR" --source "$SOURCE_DIR" --out "$FIGURE_DIR"
    exit 0 ;;
  cases) STAGES="cases" ;;
  gpu) STAGES="$GPU_STAGES" ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.position.coordinate_flow_natural \
  --stages $STAGES --model "$MODEL" --output "$OUTPUT_DIR" --source "$SOURCE_DIR" --round1 "$ROUND1_DIR" \
  --probe-cases "$PROBE_CASES" --refcoco-dir "$REFCOCO_DIR" --n-ms "$N_MS" --gen-groups $GEN_GROUPS \
  --patch-groups $PATCH_GROUPS --n-patch "$N_PATCH" --patch-rows "$PATCH_ROWS" --ablate-groups $ABLATE_GROUPS \
  --ablate-sets "$ABLATE_SETS" --ablate-from $ABLATE_FROM --n-ablate "$N_ABLATE" --n-means "$N_MEANS" \
  --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" --spotcheck "$SPOTCHECK" --seed "$SEED" \
  --max-memory "$MAX_MEMORY"
