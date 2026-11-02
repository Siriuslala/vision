#!/usr/bin/env bash
# Experiment 3.14: causal mask vs linear attention in row/column readout
# (docs/analysis/position/design_causal_vs_linear.md, results_2026-10-21_causal_vs_linear.md).
# Configure this section, then run this script without terminal parameters.
# JOB:
#   vl        Qwen3-VL-8B with DeepStack on and off: gen, spotcheck (ds_on), kv, ordinal
#   q35_gate  Qwen3.5-9B under prefill DeltaNet gate edits on the image span: gen, kv
#   q35_knock Qwen3.5-9B on 32/64-px slot grids under full-attention image-to-image knockouts: ordinal
#   toy       CPU toy models on synthetic raster grids
#   analyze   CPU statistics and figures
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
JOB="vl"
GPU_IDS="6"
MAX_MEMORY=""                                      # e.g. "0:10GiB,1:14GiB" splits the model over GPU_IDS
TORCH_DELTANET=""                                  # "--torch-deltanet" on GPUs whose shared memory is too small for FLA (A40)
RUN_DATE="2026-10-21"
VL_MODEL="Qwen3-VL-8B-Instruct"
Q35_MODEL="Qwen3.5-9B"
VL_MODES="ds_on ds_off"
GATE_MODES="nowrite_target freeze_l9_16 none freeze nodecay"   # prefill DeltaNet gate edits (all resumable); see causal_vs_linear.parse_mode
VL_STAGES="gen spotcheck ordinal kv"
GATE_STAGES="gen kv"
CONDITIONS="t6g10_rowcol t6g10_point disc_point rc_point t6g10_colrow"   # Qwen3-VL ds_on
OFF_CONDITIONS="t6g10_rowcol t6g10_point disc_point"                    # Qwen3-VL ds_off (main groups only)
GATE_CONDITIONS="t6g10_rowcol t6g10_point disc_point"
N_KV=16                                            # answers per full-set condition in the kv stage
N_KV_ORDER=12                                      # answers per order-variant condition
ORD_CONDITIONS="H32 H64 V32 V64"                   # 1.5 slot grids for Qwen3-VL
KNOCK_CONDITIONS="H32 H64"
KNOCK_VARIANTS="base insL delL"
KNOCKOUTS="none img:4 img:8 img:12 img:16 img:4,8,12,16 local:4,8,12,16"
BATCH_SIZE=16
KV_ROWS=16
MAX_NEW_TOKENS=48
SPOTCHECK=20
SEED=20261021
TOY_SEEDS="0 1"
TOY_STEPS=3000
TOY_ARCHS="causal_softmax bidir_softmax causal_delta bidir_delta hybrid"
TOY_LAYOUTS="6x8k1 4x5k2"                         # <cell rows>x<cell cols>k<tokens per cell side>
TOY_THREADS=4                                     # one CPU process per (arch, layout), run in parallel
FLOW_RUN="$WORK_DIR/analysis/position/2026-10-20_coordinate_flow"
ORDINAL_RUN="$WORK_DIR/analysis/behavior/2026-10-16_ordinal_mechanism"
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_causal_vs_linear"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/causal_vs_linear"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="uv run --frozen python -m analysis_tools.position.causal_vs_linear_run --output $OUTPUT_DIR --flow-run $FLOW_RUN --ordinal-run $ORDINAL_RUN --n-kv $N_KV --n-kv-order $N_KV_ORDER --batch-size $BATCH_SIZE --kv-rows $KV_ROWS --max-new-tokens $MAX_NEW_TOKENS --spotcheck $SPOTCHECK --seed $SEED --max-memory=$MAX_MEMORY $TORCH_DELTANET"
case "$JOB" in
  vl)
    for MODE in $VL_MODES; do
      if [ "$MODE" = "ds_on" ]; then
        # shellcheck disable=SC2086
        $RUN --model "$VL_MODEL" --mode "$MODE" --stages $VL_STAGES --conditions $CONDITIONS --ordinal-conditions $ORD_CONDITIONS
      else
        # shellcheck disable=SC2086
        $RUN --model "$VL_MODEL" --mode "$MODE" --stages ${VL_STAGES//spotcheck/} --conditions $OFF_CONDITIONS \
          --ordinal-conditions $ORD_CONDITIONS --reduced
      fi
    done ;;
  q35_gate)
    for MODE in $GATE_MODES; do
      # shellcheck disable=SC2086
      $RUN --model "$Q35_MODEL" --mode "$MODE" --stages $GATE_STAGES --conditions $GATE_CONDITIONS --reduced
    done ;;
  q35_knock)
    # shellcheck disable=SC2086
    $RUN --model "$Q35_MODEL" --mode none --stages ordinal --ordinal-conditions $KNOCK_CONDITIONS \
      --ordinal-variants $KNOCK_VARIANTS --knockouts $KNOCKOUTS ;;
  toy)
    mkdir -p "$OUTPUT_DIR/toy"
    for ARCH in $TOY_ARCHS; do
      for LAYOUT in $TOY_LAYOUTS; do
        # shellcheck disable=SC2086
        uv run --frozen python -m analysis_tools.position.causal_vs_linear_toy --output "$OUTPUT_DIR/toy/${ARCH}_${LAYOUT}" \
          --archs "$ARCH" --layouts "$LAYOUT" --seeds $TOY_SEEDS --steps "$TOY_STEPS" --threads "$TOY_THREADS" \
          > "$OUTPUT_DIR/toy/${ARCH}_${LAYOUT}.log" 2>&1 &
      done
    done
    wait ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.causal_vs_linear_analysis --run "$OUTPUT_DIR" --flow-run "$FLOW_RUN" \
      --ordinal-run "$ORDINAL_RUN" --figures "$FIGURE_DIR" ;;
  *) echo "Unknown JOB=$JOB" >&2; exit 1 ;;
esac
