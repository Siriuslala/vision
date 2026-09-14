#!/usr/bin/env bash
# Edit the configuration below, then run from the repository root:
# bash scripts/download.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"

# ==================== Configuration ====================
REPO_ID="allenai/pixmo-count"
LOCAL_DIR="$DATA_DIR_PUBLIC/pixmo-count"

REVISION="main"
MAX_WORKERS=8
DOWNLOAD_TIMEOUT=600
ETAG_TIMEOUT=60
ATTEMPTS=5
USE_XET=false

if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi
args=(
  --repo-id "$REPO_ID" --local-dir "$LOCAL_DIR" --revision "$REVISION"
  --max-workers "$MAX_WORKERS" --download-timeout "$DOWNLOAD_TIMEOUT"
  --etag-timeout "$ETAG_TIMEOUT" --attempts "$ATTEMPTS"
)
case "$USE_XET" in
  true) args+=(--use-xet) ;;
  false) ;;
  *) echo "USE_XET must be true or false." >&2; exit 2 ;;
esac
exec uv run --frozen python -m utils.download "${args[@]}"
