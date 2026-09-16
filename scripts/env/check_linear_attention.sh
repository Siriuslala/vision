#!/usr/bin/env bash
# Run from the repository root after configuring .env:
# bash scripts/env/check_linear_attention.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

# Configuration: choose a GPU with free memory for a small kernel test.
GPU_ID="0"
# Triton compiles specialized kernels at first use and caches them in WORK_DIR.
mkdir -p "$TRITON_CACHE_DIR"
exec uv run --frozen --no-build python -m utils.check_linear_attention --gpu-id "$GPU_ID"
