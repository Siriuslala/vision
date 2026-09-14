#!/usr/bin/env bash
# Run in your terminal:
# Run from the repository root after creating .env from .env_example:
# bash scripts/check.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"
while IFS= read -r script; do bash -n "$script"; done < <(rg --files scripts -g "*.sh")
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m eva.simple_test --help >/dev/null
uv run --frozen python -m eva.evaluation --help >/dev/null
