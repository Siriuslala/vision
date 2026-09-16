#!/usr/bin/env bash
# Shared initialization file sourced by entry scripts; do not run directly.
# .env is trusted local configuration: KEY="absolute path"; spaces and trailing comments are supported.
_ENV_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$_ENV_PROJECT_DIR/.env" ]]; then
  echo 'Missing .env: copy .env_example to .env and fill in the paths.' >&2
  return 1
fi
# Prevent missing required settings from inheriting values from the parent shell.
unset ROOT_DIR DATA_DIR WORK_DIR
_ENV_HF_HUB_CACHE="${HF_HUB_CACHE:-}"
set -a
source "$_ENV_PROJECT_DIR/.env"
set +a
for _ENV_KEY in ROOT_DIR DATA_DIR WORK_DIR; do
  if [[ -z "${!_ENV_KEY:-}" || "${!_ENV_KEY}" != /* ]]; then
    echo "Set $_ENV_KEY to a nonempty absolute path in .env." >&2
    return 1
  fi
done
if [[ ! -f "$ROOT_DIR/pyproject.toml" || ! -d "$ROOT_DIR/scripts" ]]; then
  echo 'ROOT_DIR must point to this repository root.' >&2
  return 1
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-${_ENV_HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
# Keep runtime-generated Triton kernels with other experiment artifacts.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_DIR/cache/triton}"
PROJECT_ROOT="$ROOT_DIR"
cd "$ROOT_DIR"
unset _ENV_PROJECT_DIR _ENV_KEY _ENV_HF_HUB_CACHE
