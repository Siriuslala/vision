#!/usr/bin/env bash
# Format-tolerant diagnostic score for location answers of experiments 1.3 and 1.4 (CPU only).
#   bash scripts/analysis/behavior/tolerant_parse.sh
# COMMAND: score   -- write scores_*_tolerant.jsonl into the three run directories (existing score files unchanged)
#          analyze -- invalid-answer classification and diagnostic-vs-tolerant tables in OUTPUT_DIR
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="analyze"
FORMAL_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
SUPPLEMENT_DIR="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"
ORDINAL_DIR="$WORK_DIR/analysis/behavior/2026-10-15_ordinal_token_grid"
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-15_tolerant_parse"

exec uv run --frozen python -m analysis_tools.behavior.tolerant_parse "$COMMAND" \
  --formal "$FORMAL_DIR" --supplement "$SUPPLEMENT_DIR" --ordinal "$ORDINAL_DIR" --output "$OUTPUT_DIR"
