#!/usr/bin/env bash
# Experiment 7.3 round 2: formation of ViT detectors for feature-absent targets
# (docs/analysis/application/results_2026-11-11_vit_detector_formation.md).
#
# Usage: edit the parameters below (or override them in the environment), then run, e.g.
#   STAGES=relocate GPU_IDS=1 bash scripts/analysis/application/detector_formation.sh             # pre-registration
#   NAME=lora_w10_s2 METHOD=lora SCOPE=vit_mlp:10-12 SEED=2 STAGES=train,probe GPU_IDS=1 bash ...
#   NAME=width_w10_s1 METHOD=width WIDTH=38 SCOPE=vit_mlp:10-12 STAGES=train,probe,detect GPU_IDS=1 bash ...
#   NAME=full_b11_s1 METHOD=full LR=1e-5 SCOPE=vit_mlp:11 STAGES=train,probe GPU_IDS=1 bash ...
#   MODEL=Qwen3-VL-8B NAME=base STAGES=vl GPU_IDS=1 bash ...                                      # mechanism comparison
#   STAGES=analysis bash ...                                                                       # CPU tables + figures
# GPU: claim with the helper first (gpu_claim.sh claim 7.3) and release when the job ends.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"
GPU_IDS="${GPU_IDS:-}"                    # required for GPU stages; must be a claimed, idle GPU
RUN_NAME="2026-11-11_vit_detector_formation"
SRC_RUN_NAME="2026-11-05_located_vit_finetune"   # 7.3 round 1: training data, probe, base results
STAGES="${STAGES:-relocate}"
NAME="${NAME:-base}"                      # adapter name; 'base' = original model
METHOD="${METHOD:-lora}"                  # lora | full | width
SCOPE="${SCOPE:-}"                        # vit_mlp:<blocks>
WIDTH="${WIDTH:-0}"                       # new neurons per block (METHOD=width)
SEED="${SEED:-1}"
N_ODD="${N_ODD:-1200}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-32}"
LR="${LR:-2e-4}"
STEPS=150                                 # identical to 7.3 (150 x 16 = 2400 sample visits)
STEP_SAMPLES=16
MICRO_TOKENS=3072
TOKEN_BUDGET=16000

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
SRC_RUN="$WORK_DIR/analysis/application/$SRC_RUN_NAME"
INTERP_RUN="$WORK_DIR/analysis/application/2026-11-03_interp_guided_optimization"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-10-21_targeted_finetune"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/vit_detector_formation"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$OUTPUT_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  uv run --frozen python -m analysis_tools.application.detector_formation_analysis \
    --run "$OUTPUT_DIR" --src-run "$SRC_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ -z "$GPU_IDS" ]]; then echo "Set GPU_IDS to a claimed idle GPU" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
uv run --frozen python -m analysis_tools.application.detector_formation_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --src-run "$SRC_RUN" --stages "$STAGES" --name "$NAME" --method "$METHOD" \
  --scope "$SCOPE" --width "$WIDTH" --seed "$SEED" --n-odd "$N_ODD" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --steps "$STEPS" --step-samples "$STEP_SAMPLES" --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" \
  --interp-run "$INTERP_RUN" --finetune-run "$FINETUNE_RUN" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" \
  --mmvp-csv "$MMVP_CSV" \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${NAME}_${STAGES//,/_}.log"
