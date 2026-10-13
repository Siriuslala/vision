#!/usr/bin/env bash
# Rebuild figures only from the finalized saved outputs; no model forward runs.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
INPUT_DIR="$WORK_DIR/analysis/representation/2026-10-01_layouts_random_targets"
DOCS_DIR="$ROOT_DIR/docs/analysis/representation/figures"
uv run --frozen python -m analysis_tools.representation_pattern_figures --input "$INPUT_DIR" --docs-dir "$DOCS_DIR"
