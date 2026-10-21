#!/usr/bin/env bash
# Experiment 1.5: mechanism of row/column ordinal answers (Qwen3.5-9B, thinking off).
# Configure below, then run:
#   bash scripts/analysis/behavior/ordinal_mechanism.sh
# COMMAND: build | gen | spot | capture | directions | steer | analyze
# "gen", "capture" and "steer" resume from existing outputs; SHARD="i/k" runs every k-th pending case from i.
# GPU index 1 is reserved; choose GPU_IDS from nvidia-smi before launching.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="build"
GPU_IDS="4"
MODE="direct"                              # direct | think (gen, capture, directions, steer)
SHARD="0/1"
LIMIT=""                                   # empty = all pending cases
MODEL="Qwen3.5-9B"
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-16_ordinal_mechanism"
TOKEN_GRID_DIR="$WORK_DIR/analysis/behavior/2026-10-15_ordinal_token_grid"          # 1.4, read only
FORMAL_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"                   # 1.3, read only
SUPPLEMENT_DIR="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"    # 1.3 supplement, read only
COORD_VALUE_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"           # 3.6 directions, read only

# Stimuli: H<cell> = 6 x 21 slot grid (ordinal = column), V<cell> = 21 x 6 (ordinal = row).
SEED=0
CONDITIONS="H32 H64 H96 V32 V64"
THINK_CONDITIONS="H32 H64 V64"
BASES_PER_ORDINAL=4                        # bases per target ordinal 3..18; each base has 5 variants
THINK_PER_ORDINAL=2                        # of those bases, how many also get the think-then-answer prompt

BATCH_SIZE=16                              # gen; steering used 13 (one row per steering label)
SPOT_N=20
STEER_EVERY=3                              # steer every k-th base (direct: 3, think: 2 in the reported run)

case "$COMMAND" in
  build)
    exec uv run --frozen python -m analysis_tools.behavior.ordinal_mechanism build --output "$OUTPUT_DIR" \
      --seed "$SEED" --conditions $CONDITIONS --think-conditions $THINK_CONDITIONS \
      --bases-per-ordinal "$BASES_PER_ORDINAL" --think-per-ordinal "$THINK_PER_ORDINAL" ;;
  gen|spot|capture|steer)
    ARGS=(--output "$OUTPUT_DIR" --model "$MODEL" --mode "$MODE" --batch-size "$BATCH_SIZE" --shard "$SHARD"
      --spot-n "$SPOT_N" --every "$STEER_EVERY")
    [[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec uv run --frozen python -m analysis_tools.behavior.ordinal_mechanism_run \
      "$COMMAND" "${ARGS[@]}" ;;
  directions)
    exec uv run --frozen python -m analysis_tools.behavior.ordinal_mechanism_analysis directions \
      --output "$OUTPUT_DIR" --mode "$MODE" --coord-value "$COORD_VALUE_DIR" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.behavior.ordinal_mechanism_analysis analyze \
      --output "$OUTPUT_DIR" --token-grid "$TOKEN_GRID_DIR" --formal "$FORMAL_DIR" --supplement "$SUPPLEMENT_DIR" ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
