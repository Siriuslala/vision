#!/usr/bin/env bash
# Experiment 3.11 round 3: roles of the edge-reading heads
# (docs/analysis/position/results_2026-11-03_edge_heads.md).
# Configure this section, then run this script without terminal parameters.
# STAGE: cases (CPU) | gpu (GPU_STAGES in one process) | analyze (CPU) | figures (CPU).
# The gpu stage claims an idle GPU through the shared claim helper and releases it on exit (also on failure).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS=""                        # empty: claim one idle GPU with GPU_CLAIM; otherwise a fixed index (never 0 or 1)
GPU_CLAIM="${GPU_CLAIM:-}"                    # path of the claim helper (gpu_claim.sh); required when GPU_IDS is empty
EXP_ID="3.11r3"
MAX_MEMORY=""                     # e.g. "0:40GiB,1:40GiB" to split the model over the GPUs in GPU_IDS
RUN_DATE="2026-11-03"
STAGE="${STAGE:-gpu}"
GPU_STAGES="gen spotcheck means ablate path"   # every stage resumes from its records
GROUPS_ALL="rc paco vocpart stuff"
ROUND2_DIR="$WORK_DIR/analysis/position/2026-10-21_coordinate_flow_natural"   # 3.11 r2: RefCOCO cases and masks
LISA_DIR="$DATA_DIR_PUBLIC/liweihan/LISA_dataset"                             # local copies of COCO, PACO, Pascal-Part
COCO_DIR="$LISA_DIR/coco"
PACO_DIR="$LISA_DIR/vlpart/paco/annotations"
VOC_DIR="$LISA_DIR/vlpart/pascal_part/VOCdevkit/VOC2010"
VOCPART_JSON="$LISA_DIR/vlpart/pascal_part/train.json"
STUFF_DIR="$LISA_DIR/cocostuff/train2017"
PER_CATEGORY_PACO=4
PER_CATEGORY_VOC=14
PER_CATEGORY_STUFF=6
# Head sets for mean ablation (layer.head); edge sets as in 3.11 round 2, ctrl6 = centred heads in the same layers.
ABLATE_SETS="H11=20.11;edge_min=20.0,24.1,24.2,24.7,24.12;edge_max=20.10,24.5,24.6,24.8,24.9,28.0;ctrl6=20.3,20.4,24.3,24.15,28.1,28.2"
N_MEANS=64
N_PATH=400
BATCH_SIZE=16
MAX_NEW_TOKENS=48
SPOTCHECK=20
SEED=20261103
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_edge_heads"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/edge_heads"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$STAGE" in
  cases)
    uv run --frozen python -m analysis_tools.position.edge_heads --stages cases --output "$OUTPUT_DIR" \
      --round2 "$ROUND2_DIR" --coco-dir "$COCO_DIR" --paco-dir "$PACO_DIR" --voc-dir "$VOC_DIR" \
      --vocpart-json "$VOCPART_JSON" --stuff-dir "$STUFF_DIR" --per-category-paco "$PER_CATEGORY_PACO" \
      --per-category-voc "$PER_CATEGORY_VOC" --per-category-stuff "$PER_CATEGORY_STUFF" --seed "$SEED"
    exit 0 ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.edge_heads_analysis --run "$OUTPUT_DIR" --round2 "$ROUND2_DIR"
    exit 0 ;;
  figures)
    uv run --frozen python -m analysis_tools.position.edge_heads_figures --run "$OUTPUT_DIR" --out "$FIGURE_DIR"
    exit 0 ;;
  gpu) ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
if [ -z "$GPU_IDS" ]; then
  [ -n "$GPU_CLAIM" ] || { echo "Set GPU_IDS or GPU_CLAIM" >&2; exit 1; }
  GPU_IDS=$(bash "$GPU_CLAIM" claim "$EXP_ID") || { echo "No idle GPU available" >&2; exit 1; }
  trap 'bash "$GPU_CLAIM" release "$EXP_ID" "$GPU_IDS"' EXIT
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Using GPU $GPU_IDS"
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.position.edge_heads \
  --stages $GPU_STAGES --model "$MODEL" --output "$OUTPUT_DIR" --round2 "$ROUND2_DIR" --groups $GROUPS_ALL \
  --ablate-sets "$ABLATE_SETS" --n-means "$N_MEANS" --n-path "$N_PATH" --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" --spotcheck "$SPOTCHECK" --seed "$SEED" --max-memory "$MAX_MEMORY"
