#!/usr/bin/env bash
# Experiment 3.11 round 5: how two edge reads become a centre (the factor 0.5) and why L20 H11 dominates on
# synthetic images; Qwen3.5-9B and Qwen3-VL-8B (docs/analysis/position/results_2026-11-11_edge_heads_r3.md).
# Configure this section, then run this script without terminal parameters.
# STAGE: cases (CPU) | gpu (GPU_STAGES in one process) | analyze (CPU) | figures (CPU).
# The gpu stage claims an idle GPU through the shared claim helper and releases it on exit (also on failure).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="${MODEL:-Qwen3.5-9B}"      # Qwen3.5-9B | Qwen3-VL-8B-Instruct
GPU_IDS="${GPU_IDS:-}"            # empty: claim one idle GPU with GPU_CLAIM; otherwise a fixed idle GPU index
GPU_CLAIM="${GPU_CLAIM:-}"        # path of the claim helper (gpu_claim.sh); required when GPU_IDS is empty
EXP_ID="3.11r5"
MAX_MEMORY=""
RUN_DATE="2026-11-11"
STAGE="${STAGE:-gpu}"
GPU_STAGES="${GPU_STAGES:-avg spot mix}"   # Qwen3.5: avg spot mix; Qwen3-VL: vlscan, then avg spot mix
LIMIT="${LIMIT:-0}"                         # >0: smoke test on a few items
ROUND3_DIR="$WORK_DIR/analysis/position/2026-11-03_edge_heads"      # 3.11 r3: natural sources, masks
ROUND4_DIR="$WORK_DIR/analysis/position/2026-11-05_edge_heads_r2"   # 3.11 r4: str / rect / nat item specs
N_MIX=16          # mix items per kind (8 kinds) and distractor count (0 / 2)
N_GRID=32         # 10x10 grids (target moved by one cell on both axes)
MIX_KINDS="${MIX_KINDS:-}"         # empty: all mix kinds; Qwen3-VL run: a subset (see results doc)
MIX_PER_KIND="${MIX_PER_KIND:-0}"  # >0: at most this many items per kind and distractor count
SCAN_TOP=128      # Qwen3-VL: heads with the largest image share that get a single-head patching row
# Qwen3.5 reader groups at point answers (3.11 r4 condition groups: left/top readers and right/bottom readers).
Q35_MINX="20.0,20.2,20.15,24.1,24.2,24.4,24.7,24.12,24.13"
Q35_MAXX="16.8,16.9,16.11,16.12,16.15,20.9,20.10,20.12,20.13,20.14,24.5,24.6,24.8,24.9,28.0,28.1"
Q35_MINY="20.0,24.2,24.7"
Q35_MAXY="16.0,16.1,16.2,16.3,16.6,16.8,16.9,16.10,16.11,16.12,16.15,20.8,20.10,20.14,24.0,24.5,24.6,24.8,24.9,24.14,28.0,28.3,28.4,28.8,28.9,28.10,28.11"
# Qwen3-VL reader groups, defined from the vlscan stage (causal heads reading the left/top vs right/bottom band
# at point answers; see the results doc, section on Qwen3-VL).
VL_MINX="${VL_MINX:-}"
VL_MAXX="${VL_MAXX:-}"
VL_MINY="${VL_MINY:-}"
VL_MAXY="${VL_MAXY:-}"
COND_BATCH=12
SCAN_BATCH=32
MAX_NEW_TOKENS=40
SEED=20261111
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_edge_heads_r3${OUT_SUFFIX:-}"   # OUT_SUFFIX: smoke-test directory
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/edge_heads_r3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16   # CPU stages: at most 16 threads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$MODEL" == Qwen3-VL* ]]; then G=(VL_MINX VL_MAXX VL_MINY VL_MAXY); else G=(Q35_MINX Q35_MAXX Q35_MINY Q35_MAXY); fi
COMMON=(--output "$OUTPUT_DIR" --round3 "$ROUND3_DIR" --round4 "$ROUND4_DIR" --seed "$SEED")
case "$STAGE" in
  cases)
    uv run --frozen python -m analysis_tools.position.edge_heads_r3 --stages cases "${COMMON[@]}" \
      --n-mix "$N_MIX" --n-grid "$N_GRID"
    exit 0 ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.edge_heads_r3_analysis --run "$OUTPUT_DIR" --round4 "$ROUND4_DIR"
    exit 0 ;;
  figures)
    uv run --frozen python -m analysis_tools.position.edge_heads_r3_figures --run "$OUTPUT_DIR" --round3 "$ROUND3_DIR" \
      --round4 "$ROUND4_DIR" --out "$FIGURE_DIR"
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
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
  | sed "s/^/[pre-launch $(date '+%F %T')] /"
echo "[pre-launch check] GPU $GPU_IDS: $(nvidia-smi -i "$GPU_IDS" --query-gpu=uuid --format=csv,noheader)"
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.position.edge_heads_r3 --stages $GPU_STAGES --model "$MODEL" "${COMMON[@]}" \
  --group-minx "${!G[0]}" --group-maxx "${!G[1]}" --group-miny "${!G[2]}" --group-maxy "${!G[3]}" \
  --scan-top "$SCAN_TOP" --cond-batch "$COND_BATCH" --scan-batch "$SCAN_BATCH" --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-memory "$MAX_MEMORY" --limit "$LIMIT" --mix-kinds "$MIX_KINDS" --mix-per-kind "$MIX_PER_KIND"
