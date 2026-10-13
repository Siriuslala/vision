#!/usr/bin/env bash
# Configure below, then run: bash scripts/analysis/behavior_figures.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
SOURCE="$WORK_DIR/grounding/comparison_grounding_v1"
OUTPUT="$WORK_DIR/analysis/behavior/2026-09-30_selection_formats"
DOCUMENT_FIGURES="$ROOT_DIR/docs/analysis/behavior/figures"
uv run --frozen python -m analysis_tools.behavior_figures \
  --source "$SOURCE" --output "$OUTPUT" --document-figures "$DOCUMENT_FIGURES"
