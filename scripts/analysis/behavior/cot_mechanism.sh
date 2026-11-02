#!/usr/bin/env bash
# Experiment 1.7: behaviour and mechanism of chain-of-thought reasoning
# (docs/analysis/behavior/design_cot_mechanism.md, results_2026-10-21_cot_mechanism.md).
# COMMAND=run executes the GPU stages listed in STAGES (resumable); COMMAND=taxonomy codes all CoT corpora
# (CPU); COMMAND=analyze writes summary.json and the figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
COMMAND="${COMMAND:-run}"                  # run | taxonomy | analyze
GPU_IDS="${GPU_IDS:-3}"                    # check nvidia-smi and the GPU ledger first
MAX_MEMORY="${MAX_MEMORY:-}"               # optional split load over several GPUs, e.g. "0:12GiB,1:12GiB"
STAGES="${STAGES:-gen,faith,mech,mech_edit}"  # mech_edit: rowcol only (link to 3.10)
RUN_NAME="2026-10-21_cot_mechanism"
TASKS="${TASKS:-rowcol,odd,point,count}"   # fresh task sets (see cot_mechanism.TASK_SETS)
MECH_TASKS="rowcol,odd,point"              # tasks with matched image pairs
N_ITEMS=64                                 # images per task in the generation stage
N_THINKING_ON=8                            # thinking-on subset for rowcol and point
N_PAIRS=32                                 # matched pairs per task in the mechanism stage
BATCH_SIZE="${BATCH_SIZE:-16}"
MECH_MAX_NEW=768                           # decode budget of the mechanism stage (thinking off)
OUTPUT_DIR="$WORK_DIR/analysis/behavior/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/behavior/figures/cot_mechanism"

mkdir -p "$OUTPUT_DIR"
case "$COMMAND" in
  taxonomy)
    uv run --frozen python -m analysis_tools.behavior.cot_mechanism_taxonomy \
      --analysis-root "$WORK_DIR/analysis" --run-dir "$OUTPUT_DIR"
    ;;
  analyze)
    uv run --frozen python -m analysis_tools.behavior.cot_mechanism_analysis \
      --run-dir "$OUTPUT_DIR" --figures "$FIGURE_DIR"
    ;;
  run)
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    for STAGE in ${STAGES//,/ }; do
      if [ "$STAGE" = "mech" ]; then STAGE_TASKS="$MECH_TASKS"; else STAGE_TASKS="$TASKS"; fi
      uv run --frozen python -m analysis_tools.behavior.cot_mechanism_run \
        --model "$MODEL" --out "$OUTPUT_DIR" --stage "$STAGE" --tasks "$STAGE_TASKS" \
        --n "$N_ITEMS" --n-thinking "$N_THINKING_ON" --n-pairs "$N_PAIRS" --batch-size "$BATCH_SIZE" \
        --mech-max-new "$MECH_MAX_NEW" ${MAX_MEMORY:+--max-memory "$MAX_MEMORY"} 2>&1 | tee -a "$OUTPUT_DIR/run_$STAGE.log"
    done
    ;;
  *)
    echo "unknown COMMAND=$COMMAND" >&2; exit 1 ;;
esac
