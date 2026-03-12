#!/usr/bin/env bash
# ============================================================
# Oracle filter for SankalpKJ/swesmith-patched
# Run this on a machine with Python 3.12 and Daytona access.
#
# Prerequisites:
#   pip install "harbor[daytona] @ git+https://github.com/laude-institute/harbor.git@penfever/temp-override"
#   pip install -e ".[daytona]"   (from OT-Agent repo root)
#
# Required env vars:
#   HF_TOKEN         - HuggingFace write token
#   DAYTONA_API_KEY  - Daytona API key
#
# Usage:
#   # Test on 10 tasks first:
#   bash data/patchers/run_oracle_filter_swesmith.sh --limit 10
#
#   # Full run (31k tasks, ~18 hours):
#   bash data/patchers/run_oracle_filter_swesmith.sh
# ============================================================

set -euo pipefail

REPO_ID="SankalpKJ/swesmith-patched"
TARGET_REPO="SankalpKJ/swesmith-patched-oracle-filtered"
PROCESSES=32
LIMIT_ARG=""

# Parse optional --limit argument
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit)
      LIMIT_ARG="--limit $2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Running oracle filter on ${REPO_ID}"
echo "Target: ${TARGET_REPO}"
echo "Processes: ${PROCESSES}"
[[ -n "$LIMIT_ARG" ]] && echo "Limit: ${LIMIT_ARG}"

python scripts/daytona/validate_and_upload_from_hf.py \
  --repo_id "${REPO_ID}" \
  --oracle-check-only \
  --target_repo "${TARGET_REPO}" \
  --processes "${PROCESSES}" \
  --token "${HF_TOKEN}" \
  ${LIMIT_ARG}
