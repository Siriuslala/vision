#!/usr/bin/env bash
# Experiment 3.4: position probe refinement (docs/analysis/position/design_probe_refinement.md).
# Steps run in order and each step is resumable from its artifacts; remove a step from STEPS to skip it.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
: "${REFCOCO_DIR:?Set REFCOCO_DIR in the private .env file.}"

MODEL="Qwen3.5-9B"
GPU_IDS="2"
SPLIT="2400 300 300"                      # train / validation / test images
FEATURE_STAGES="patch_content vit_00_position vit_09 vit_18 vit_27 merger"
FEATURE_DECODER_LAYERS="3 4 5 7 8 9 11 12 13 14 15 16 17 20 24 28 32"
LC_DECODER_LAYERS="8 14 24 32"           # learning curves also include the merger
STEPS="prepare extract_trainval fit_ridge extract_test probes figures"
OUTPUT_DIR="$WORK_DIR/analysis/position/2026-10-14_probe_refinement"
# Statistics and features need about 90 GB; they live under DATA_DIR and OUTPUT_DIR links to them.
STORAGE_DIR="$DATA_DIR/scratch/analysis/position/2026-10-14_probe_refinement"
PREVIOUS_CASES="$WORK_DIR/analysis/position/2026-10-01_refcoco_layers/cases.json"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/probe_refinement"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vision-matplotlib"
MODULE="analysis_tools.position.position_probe_refinement"
if [[ ! -e "$OUTPUT_DIR" ]]; then
  mkdir -p "$STORAGE_DIR" "$(dirname "$OUTPUT_DIR")"
  ln -s "$STORAGE_DIR" "$OUTPUT_DIR"
fi
for step in $STEPS; do
  echo "== $step"
  case "$step" in
    prepare)
      [[ -f "$OUTPUT_DIR/cases.json" ]] || uv run --frozen python -m "$MODULE" prepare --output "$OUTPUT_DIR" \
        --refcoco "$REFCOCO_DIR" --previous-cases "$PREVIOUS_CASES" --split $SPLIT ;;
    extract_trainval)
      [[ -f "$OUTPUT_DIR/extract_trainval.json" ]] || uv run --frozen python -m "$MODULE" extract --phase trainval \
        --output "$OUTPUT_DIR" --model "$MODEL" --feature-stages $FEATURE_STAGES \
        --feature-decoder-layers $FEATURE_DECODER_LAYERS --lc-decoder-layers $LC_DECODER_LAYERS ;;
    fit_ridge)
      uv run --frozen python -m "$MODULE" fit-ridge --output "$OUTPUT_DIR" ;;
    extract_test)
      [[ -f "$OUTPUT_DIR/extract_test.json" ]] || uv run --frozen python -m "$MODULE" extract --phase test \
        --output "$OUTPUT_DIR" --model "$MODEL" --feature-stages $FEATURE_STAGES \
        --feature-decoder-layers $FEATURE_DECODER_LAYERS --lc-decoder-layers $LC_DECODER_LAYERS ;;
    probes)
      uv run --frozen python -m analysis_tools.position.position_probe_refinement_probes all --output "$OUTPUT_DIR" ;;
    figures)
      uv run --frozen python -m analysis_tools.position.position_probe_refinement_figures --output "$OUTPUT_DIR" \
        --figures "$FIGURE_DIR" ;;
    *) echo "Unknown step $step" >&2; exit 1 ;;
  esac
done
