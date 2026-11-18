#!/usr/bin/env bash
# Experiment 3.11 round 4: where the edge-reading heads read, how two edge reads combine, and the function
# of L20 H11 (docs/analysis/position/results_2026-11-05_edge_heads_r2.md).
# Configure this section, then run this script without terminal parameters.
# STAGE: cases (CPU) | gpu (GPU_STAGES in one process) | analyze (CPU) | figures (CPU).
# The gpu stage claims an idle GPU through the shared claim helper and releases it on exit (also on failure).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS=""                        # empty: claim one idle GPU with GPU_CLAIM; otherwise a fixed idle GPU index
GPU_CLAIM="${GPU_CLAIM:-}"        # path of the claim helper (gpu_claim.sh); required when GPU_IDS is empty
EXP_ID="3.11r4"
MAX_MEMORY=""
RUN_DATE="2026-11-05"
STAGE="${STAGE:-gpu}"
GPU_STAGES="${GPU_STAGES:-gen spotcheck means ablate behav probe mediate}"   # every stage resumes from its records
ROUND3_DIR="$WORK_DIR/analysis/position/2026-11-03_edge_heads"   # 3.11 r3: natural cases, images, masks, means
N_NAT=30          # nat items (translation pairs) per round-3 group
N_STR=15          # str items (one-edge moves on natural images) per round-3 group
N_RECT=24         # rect items (one solid rectangle)
N_COMP=16         # comp items per design cell (2 object types x 2 backgrounds x 4 sizes x 2 distractor counts)
N_MEANS=32
# Reader groups of round 3 at point answers (layers >= 16, edge-regression R^2 >= 0.5; L20 H11 kept separate).
GROUP_MINX="20.0,20.2,20.15,24.1,24.2,24.4,24.7,24.12,24.13"
GROUP_MAXX="16.8,16.9,16.11,16.12,16.15,20.9,20.10,20.12,20.13,20.14,24.5,24.6,24.8,24.9,28.0,28.1"
GROUP_CTRX="16.0,16.1,16.2,16.5,16.6,16.7,20.3,20.8,24.0,24.3,24.15,28.2,28.13"
GROUP_MINY="20.0,24.2,24.7"
GROUP_MAXY="16.0,16.1,16.2,16.3,16.6,16.8,16.9,16.10,16.11,16.12,16.15,20.8,20.10,20.14,24.0,24.5,24.6,24.8,24.9,24.14,28.0,28.3,28.4,28.8,28.9,28.10,28.11"
GROUP_CTRY="16.13,20.2,24.4,28.1,28.2"
ABLATE_SETS="H11=20.11;edge_min=20.0,24.1,24.2,24.7,24.12;edge_max=20.10,24.5,24.6,24.8,24.9,28.0;ctrl6=20.3,20.4,24.3,24.15,28.1,28.2;L16H1H2=16.1,16.2;H11_L16H1H2=20.11,16.1,16.2"
BATCH_SIZE=16
COND_BATCH=12
SCAN_BATCH=32
MAX_NEW_TOKENS=48
SPOTCHECK=20
SEED=20261105
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_edge_heads_r2"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/edge_heads_r2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16   # CPU stages: at most 16 threads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(--output "$OUTPUT_DIR" --round3 "$ROUND3_DIR" --seed "$SEED"
  --group-minx "$GROUP_MINX" --group-maxx "$GROUP_MAXX" --group-ctrx "$GROUP_CTRX"
  --group-miny "$GROUP_MINY" --group-maxy "$GROUP_MAXY" --group-ctry "$GROUP_CTRY")
case "$STAGE" in
  cases)
    uv run --frozen python -m analysis_tools.position.edge_heads_r2 --stages cases "${COMMON[@]}" \
      --n-nat "$N_NAT" --n-str "$N_STR" --n-rect "$N_RECT" --n-comp "$N_COMP"
    exit 0 ;;
  analyze)
    uv run --frozen python -m analysis_tools.position.edge_heads_r2_analysis --run "$OUTPUT_DIR" --round3 "$ROUND3_DIR"
    exit 0 ;;
  figures)
    uv run --frozen python -m analysis_tools.position.edge_heads_r2_figures --run "$OUTPUT_DIR" --round3 "$ROUND3_DIR" \
      --out "$FIGURE_DIR"
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
uv run --frozen python -m analysis_tools.position.edge_heads_r2 --stages $GPU_STAGES --model "$MODEL" "${COMMON[@]}" \
  --n-means "$N_MEANS" --ablate-sets "$ABLATE_SETS" --batch-size "$BATCH_SIZE" --cond-batch "$COND_BATCH" \
  --scan-batch "$SCAN_BATCH" --max-new-tokens "$MAX_NEW_TOKENS" --spotcheck "$SPOTCHECK" --max-memory "$MAX_MEMORY"
