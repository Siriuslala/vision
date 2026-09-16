#!/usr/bin/env bash
# Run in your terminal:
# Run from the repository root after creating .env from .env_example:
# bash scripts/env/setup_env.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
exec uv sync --frozen
