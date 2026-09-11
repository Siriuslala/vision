#!/usr/bin/env bash
# Run from the repository root after configuring .env:
# bash scripts/check_flash_attention.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"

# Configuration: choose a GPU with free memory for a small kernel test.
GPU_ID="3"

exec uv run --frozen --no-build python -m utils.check_flash_attention --gpu-id "$GPU_ID"
