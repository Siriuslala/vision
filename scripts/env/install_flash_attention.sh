#!/usr/bin/env bash
# Run from the repository root after configuring .env:
# bash scripts/env/install_flash_attention.sh
# The wheel URL and compatible Python/PyTorch versions are pinned in pyproject.toml.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

# Never fall back to compiling source distributions on the shared server.
uv sync --frozen --no-build
uv run --frozen --no-build python -m utils.check_flash_attention --import-only
