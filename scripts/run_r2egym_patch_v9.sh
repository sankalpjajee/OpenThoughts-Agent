#!/bin/bash
# =============================================================================
# R2E-Gym Patcher v9 — Full pipeline: patch → upload → validate
# =============================================================================
# Run this on the cluster (bmic-h100x2-jo) in the ot-agent conda env.
#
# Prerequisites:
#   conda activate ot-agent
#   cd /home/jajee/OpenThoughts-Agent
#   git pull origin sankalp/swesmith-patcher
#
# Usage:
#   bash scripts/run_r2egym_patch_v9.sh [--test]
#
# With --test: patches only 30 tasks (fast sanity check)
# Without --test: patches all 4,578 tasks and uploads to HF
# =============================================================================

set -euo pipefail

REPO_DIR="/home/jajee/OpenThoughts-Agent"
OUTPUT_BASE="/mnt/sda4T/home/jajee"
HF_TOKEN="${HF_TOKEN:-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}"
DAYTONA_API_KEY="${DAYTONA_API_KEY:-dtn_d23a3c4333715608ae3e7fb8c8fb47624fc8b7c2dd43e2cdf5cb8ccf326d3cf8}"

TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
    TEST_MODE=true
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Step 1: Patch the dataset
# ---------------------------------------------------------------------------
if $TEST_MODE; then
    echo "=== TEST MODE: Patching 30 tasks ==="
    OUTPUT_DIR="$OUTPUT_BASE/r2egym_patched_v9_test"
    HF_UPLOAD="SankalpKJ/r2egym-patched-v9-test"
    LIMIT_ARG="--limit 30"
else
    echo "=== FULL MODE: Patching all 4578 tasks ==="
    OUTPUT_DIR="$OUTPUT_BASE/r2egym_patched_v9_full"
    HF_UPLOAD="SankalpKJ/r2egym-patched-v9-full"
    LIMIT_ARG=""
fi

echo "Output dir: $OUTPUT_DIR"
echo "HF upload: $HF_UPLOAD"
echo ""

# Run the patcher
HF_TOKEN="$HF_TOKEN" python3 data/patchers/patch_r2egym_tasks.py \
    --output-dir "$OUTPUT_DIR" \
    --upload-to "$HF_UPLOAD" \
    $LIMIT_ARG

echo ""
echo "=== Patcher complete. Dataset at: $HF_UPLOAD ==="
echo ""

# ---------------------------------------------------------------------------
# Step 2: Validate with oracle
# ---------------------------------------------------------------------------
if $TEST_MODE; then
    VALIDATED_DIR="$OUTPUT_BASE/r2egym_validated_v9_test"
    FINAL_DIR="$OUTPUT_BASE/r2egym_validated_v9_test_final"
    TARGET_REPO="SankalpKJ/r2egym-patched-v9-test-validated"
else
    VALIDATED_DIR="$OUTPUT_BASE/r2egym_validated_v9_full"
    FINAL_DIR="$OUTPUT_BASE/r2egym_validated_v9_full_final"
    TARGET_REPO="SankalpKJ/r2egym-patched-v9-full-validated"
fi

echo "=== Running oracle validation ==="
echo "This will run test.sh + solve.sh for each task in Harbor/Daytona."
echo "Expected pass rate: ~52% (2400/4578 tasks have new_commit_res_code=0)"
echo ""

DAYTONA_API_KEY="$DAYTONA_API_KEY" HF_TOKEN="$HF_TOKEN" \
    python3 scripts/daytona/validate_and_upload_from_hf.py \
    --repo_id "$HF_UPLOAD" \
    --harbor_concurrency 8 \
    --timeout 300 \
    --cpu 2 \
    --memory_gb 4 \
    --disk_gb 10 \
    --extract_dir "$VALIDATED_DIR" \
    --final_output_dir "$FINAL_DIR" \
    --target_repo "$TARGET_REPO" \
    --oracle-check-only

echo ""
echo "=== Validation complete. Results at: $TARGET_REPO ==="
