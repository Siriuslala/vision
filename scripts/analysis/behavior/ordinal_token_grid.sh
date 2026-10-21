#!/usr/bin/env bash
# Experiment 1.4: ordinal offset and the visual token grid (Qwen3.5-9B, thinking off).
# Configure below, then run:
#   bash scripts/analysis/behavior/ordinal_token_grid.sh
# COMMAND: build | run | spotcheck | score | mech | convention | analyze
# "run" and "mech" resume from existing results; SHARD="i/k" processes every k-th pending case from i.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="build"
GPU_IDS="2"
SHARD="0/1"
LIMIT=""                                   # empty = all pending cases
MODEL="Qwen3.5-9B"
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-15_ordinal_token_grid"
FORMAL_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"              # read only
SUPPLEMENT_DIR="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"   # read only

# Conditions are c<cell>n<instances on the long axis>p<phase>[r<rows>]; grids have n columns and 6 rows unless r is given.
SEED=0
CELLS="32 48 64 80 96"
COUNTS="10 16 20 24"
PHASE_CONDITIONS="c64n16p8 c64n20p8 c64n24p8 c64n16p16 c64n20p16 c64n24p16 c32n20p16 c32n24p16"
ROW_CONDITIONS="c32n20p0r10 c32n20p0r20 c64n20p0r10 c64n20p0r20"   # same axis, more rows (r<rows> suffix)
T6_PER_CONDITION=192                       # rounded up to a multiple of n (targets stratified by column)
T0_PER_CONDITION=24
MECH_CONDITIONS="c32n20p0 c48n20p0 c64n20p0 c80n20p0 c96n20p0 c64n20p16 c32n20p16 c32n20p0r20 c64n20p0r20"

# Decoding.
MAX_NEW_TOKENS=256
BATCH_SIZE=8
SPOTCHECK_N=20

CONDITIONS=""
for c in $CELLS; do for n in $COUNTS; do CONDITIONS+="c${c}n${n}p0 "; done; done
CONDITIONS+="$PHASE_CONDITIONS $ROW_CONDITIONS"

ARGS=(--output "$OUTPUT_DIR" --seed "$SEED" --conditions $CONDITIONS --t6-per-condition "$T6_PER_CONDITION"
  --t0-per-condition "$T0_PER_CONDITION" --model "$MODEL" --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE" --shard "$SHARD" --spotcheck-n "$SPOTCHECK_N" --mech-conditions $MECH_CONDITIONS
  --formal "$FORMAL_DIR" --supplement "$SUPPLEMENT_DIR")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")

case "$COMMAND" in
  build|score|convention)
    exec uv run --frozen python -m analysis_tools.behavior.ordinal_token_grid "$COMMAND" "${ARGS[@]}" ;;
  run|spotcheck|mech)
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec uv run --frozen python -m analysis_tools.behavior.ordinal_token_grid \
      "$COMMAND" "${ARGS[@]}" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.behavior.ordinal_token_grid_analysis \
      --output "$OUTPUT_DIR" --formal "$FORMAL_DIR" --supplement "$SUPPLEMENT_DIR" ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
