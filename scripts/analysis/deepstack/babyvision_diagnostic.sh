#!/usr/bin/env bash
# Experiment 4.2 supplement: format-tolerant diagnostic rescoring of existing BabyVision runs.
# CPU only. Strict scores are verified and never overwritten. Run from the repository root:
#   bash scripts/analysis/deepstack/babyvision_diagnostic.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

RUNS_DIR="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end"   # babyvision/ holds the three new 4.2 runs
LEGACY_BABYVISION="$WORK_DIR/eval"                                  # existing Qwen3.5-9B thinking-off run
OUTPUT_DIR="$RUNS_DIR/diagnostic_scoring"

cd "$ROOT_DIR"
exec uv run --frozen python -m analysis_tools.deepstack.babyvision_diagnostic \
  --runs-dir "$RUNS_DIR" --legacy-babyvision "$LEGACY_BABYVISION" --data-root "$DATA_DIR" --output "$OUTPUT_DIR"
