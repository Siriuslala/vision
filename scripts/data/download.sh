#!/usr/bin/env bash
# Edit the configuration below, then run from the repository root:
# bash scripts/data/download.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
: "${DATA_DIR_PUBLIC:?Set DATA_DIR_PUBLIC to the shared dataset root in .env}"

# export HF_ENDPOINT="https://huggingface.co"
# export http_proxy=http://proxy.812999.xyz:17890
# export https_proxy=http://proxy.812999.xyz:17890

unset http_proxy
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
# ==================== Configuration ====================
# lmms-lab/LLaVA-OneVision-Data
# Kwai-Keye/Thyme-SFT
# deepcs233/Visual-CoT
# allenai/pixmo-count
# allenai/CoSyn-400K

REPO_ID="Kwai-Keye/Thyme-SFT"
LOCAL_DIR="$DATA_DIR_PUBLIC/Thyme-SFT"

REVISION="main"
MAX_WORKERS=8
RANGE_MIB=64
DOWNLOAD_TIMEOUT=120
ETAG_TIMEOUT=60
ATTEMPTS=10
# Empty inherits HF_ENDPOINT from .env / the server environment.
# Set to https://huggingface.co to bypass a stale or rate-limited mirror.
ENDPOINT=""
# The previous downloader used HTTP. Stop it before migrating its partial files.
# Set false for directories with Xet partial files or unknown download provenance.
IMPORT_HTTP_PARTIALS=true
DRY_RUN=false

if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi
args=(
  --repo-id "$REPO_ID" --local-dir "$LOCAL_DIR" --revision "$REVISION"
  --max-workers "$MAX_WORKERS" --download-timeout "$DOWNLOAD_TIMEOUT"
  --etag-timeout "$ETAG_TIMEOUT" --attempts "$ATTEMPTS"
  --range-mib "$RANGE_MIB" --endpoint "$ENDPOINT"
)
for flag in IMPORT_HTTP_PARTIALS DRY_RUN; do
  case "${!flag}" in
    true|false) ;;
    *) echo "$flag must be true or false." >&2; exit 2 ;;
  esac
done
if [[ "$IMPORT_HTTP_PARTIALS" == true ]]; then args+=(--import-http-partials); fi
if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
exec uv run --frozen python -m utils.download "${args[@]}"
