#!/usr/bin/env bash
# Experiment 2.2: source, content and role of high-norm tokens
# (docs/analysis/representation/design_positional_component.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="${MODEL:-Qwen3.5-9B}"           # Qwen3.5-9B or Qwen3-VL-8B-Instruct
TAG="${TAG:-q35}"                      # model tag in artifact names: q35 or q3vl
GPU_IDS="${GPU_IDS:-4}"                # never GPU 1 (reserved)
RUN_NAME="2026-10-16_high_norm"
# comma list of: neurons, survey, source, attention, posvar, ablation, spot; or "figures"
STAGES="${STAGES:-neurons}"
SEED=20261016
PROBE_CANVASES="384,512,640"           # random-placement probe images, one group per canvas size
PROBE_N=160                            # probe images per canvas
GRID_N=24                              # experiment-1.3 grid images per grid size
NATURAL_N=300                          # RefCOCO 512-pixel images (experiment 3.4 test split)
MARKED_N=48                            # blue-box-marked T3 grids and their unmarked twins
DECODER_N=48                           # probe / grid images that also get decoder norms
SOURCE_N=16                            # synthetic and natural images per source intervention
ATTENTION_N=6                          # synthetic and natural images of the attention stage
POSVAR_N=160                           # images per group of the R^2_pos estimate
N_BINARY=240                           # T2 / T3-box / T3-rowcol cases (balanced over variant x grid)
N_LOCATE=200                           # T4 row/column and T4 point cases (balanced over grid)
WITH_EXPLICIT="${WITH_EXPLICIT:-1}"    # also run the 72 think-then-answer T4 row/column cases
CONDITIONS="${CONDITIONS:-base,hn_mean,hn_zero,rand_mean0,rand_mean1,neuron_zero}"
EXPLICIT_CONDITIONS="${EXPLICIT_CONDITIONS:-base,hn_mean,rand_mean0}"   # long reasoning cases are slow; subset of conditions
TASK_FILTER="${TASK_FILTER:-}"         # optional comma list of task tags
BATCH_SIZE="${BATCH_SIZE:-24}"
PIXEL_BUDGET=$((24 * 512 * 512))       # image pixels per generation batch
MAX_NEW_TOKENS=64
MAX_NEW_TOKENS_LONG=1024
N_SPOT=20
DECOMPOSITION_RUN="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
DECOMPOSITION_SUPP="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco"   # 3.5 cases on the 3.4 test split
OUTPUT_DIR="$WORK_DIR/analysis/representation/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/representation/figures/high_norm"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.representation.high_norm_analysis --run "$OUTPUT_DIR"
  uv run --frozen python -m analysis_tools.representation.high_norm_figures --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.representation.high_norm_run \
  --model "$MODEL" --tag "$TAG" --output "$OUTPUT_DIR" --stages "$STAGES" --seed "$SEED" \
  --decomposition "$DECOMPOSITION_RUN" --decomposition-supplement "$DECOMPOSITION_SUPP" --refcoco-cases "$REFCOCO_CASES" \
  --probe-canvases "$PROBE_CANVASES" --probe-n "$PROBE_N" --grid-n "$GRID_N" --natural-n "$NATURAL_N" \
  --marked-n "$MARKED_N" --decoder-n "$DECODER_N" --source-n "$SOURCE_N" --attention-n "$ATTENTION_N" \
  --posvar-n "$POSVAR_N" --n-binary "$N_BINARY" --n-locate "$N_LOCATE" --with-explicit "$WITH_EXPLICIT" \
  --conditions "$CONDITIONS" --explicit-conditions "$EXPLICIT_CONDITIONS" --task-filter "$TASK_FILTER" --batch-size "$BATCH_SIZE" --pixel-budget "$PIXEL_BUDGET" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-new-tokens-long "$MAX_NEW_TOKENS_LONG" --n-spot "$N_SPOT" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run_${TAG}_${STAGES//,/-}.log"
