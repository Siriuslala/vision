#!/usr/bin/env bash
# Run from the repository root after configuring .env:
# bash scripts/env/install_linear_attention.sh
# Installs the locked FLA dependency; causal-conv1d needs a compatible wheel.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
uv sync --frozen --no-build
uv run --frozen --no-build python -c 'import fla; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; print("FLA", fla.__version__, "import check passed")'
