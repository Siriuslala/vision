#!/usr/bin/env bash
# Experiment 5.4: cached re-reading pilots (docs/analysis/reread/results_2026-11-03_reread_pilots.md).
#   option 1: learned 2x cache from the single 1x ViT pass (train, fidelity, opt1_textvqa)
#   option 2: high-resolution documents, crop re-encode vs cache read (opt2_doc, opt2_timing)
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:?set GPU_IDS to a claimed idle GPU}"
RUN_NAME="2026-11-03_reread_pilots"
# comma list of: train, fidelity, opt1_textvqa, opt2_doc, opt2_timing; or "analysis" (CPU statistics and figures)
STAGES="${STAGES:-train,fidelity}"
SEED=20261020                          # same seed as experiments 5.2 / 5.3, hence the same TextVQA questions
MAX_PIXELS=$((512 * 512))              # natural-image input budget (5.2 / 5.3)
# option 1: learned 2x module
VARIANTS="l27 l27_0 l27_9"             # inputs: block 27; block 27 + block 0 (patch embedding); block 27 + block 9
N_TRAIN_TEXTVQA=12000                  # TextVQA train images
N_TRAIN_COCO=4000                      # COCO val2014 images
EPOCHS=1
BATCH=16
WORKERS=8
LR=1e-3
WIDTH=384
N_FIDELITY=300
N_TEXTVQA=200
OPT1_CONDS="${OPT1_CONDS:-base crop1 crop2 c27 u27 h27 l27 l27_0 l27_9}"
# option 2: documents (Visual-CoT DocVQA train questions; images extracted by experiment 4.4)
DOC_META="$DATA_DIR_PUBLIC/Visual-CoT/metadata/docvqa_cot_train.jsonl"
DOC_IMAGES="$WORK_DIR/analysis/deepstack/2026-10-21_natural/viscot_raw/cot_image_data/docvqa"
N_DOC=150
DOC_MAX_PIXELS="${DOC_MAX_PIXELS:-$((1024 * 1024))}"  # lo view budget; the hi image has twice lo's sides
DOC_TAG="${DOC_TAG:-}"                 # output suffix for a second budget (e.g. _512 for DOC_MAX_PIXELS=512*512)
OPT2_CONDS="${OPT2_CONDS:-base_lo base_hi lo_c27 lo_crop2 lo_crop_hi lo_cache_hi hi_crop1 hi_c27}"
TIMING_N=20
# decoding (closed-call lean protocol of 5.3)
MAX_CALL=40
MAX_NEW=32
CHUNK=8
TOKEN_BUDGET="${TOKEN_BUDGET:-24000}"
MAX_BATCH=16
OUTPUT_DIR="$WORK_DIR/analysis/reread/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/reread/figures/reread_pilots"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "analysis" ]]; then
  exec uv run --frozen python -m analysis_tools.reread.reread_pilots_analysis --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
fi
exec uv run --frozen python -m analysis_tools.reread.reread_pilots_run --output "$OUTPUT_DIR" --model "$MODEL" \
  --stages "$STAGES" --seed "$SEED" --data-dir "$DATA_DIR" --max-pixels "$MAX_PIXELS" --variants $VARIANTS \
  --n-train-textvqa "$N_TRAIN_TEXTVQA" --n-train-coco "$N_TRAIN_COCO" --epochs "$EPOCHS" --batch "$BATCH" \
  --workers "$WORKERS" --lr "$LR" --width "$WIDTH" --n-fidelity "$N_FIDELITY" --n-textvqa "$N_TEXTVQA" \
  --opt1-conds $OPT1_CONDS --doc-meta "$DOC_META" --doc-images "$DOC_IMAGES" --n-doc "$N_DOC" \
  --doc-max-pixels "$DOC_MAX_PIXELS" ${DOC_TAG:+--doc-tag "$DOC_TAG"} --opt2-conds $OPT2_CONDS --timing-n "$TIMING_N" --max-call "$MAX_CALL" \
  --max-new "$MAX_NEW" --chunk "$CHUNK" --token-budget "$TOKEN_BUDGET" --max-batch "$MAX_BATCH"
