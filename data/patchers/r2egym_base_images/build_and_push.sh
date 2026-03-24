#!/bin/bash
# Build and push custom R2E-Gym base images to ghcr.io
#
# Prerequisites:
#   - Docker installed and running
#   - Authenticated to ghcr.io:
#       echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin
#
# Usage:
#   bash data/patchers/r2egym_base_images/build_and_push.sh
#
# This builds 5 images for repos with compiled C extensions.
# Pure-Python repos (tornado, scrapy, pyramid, datalad, coveragepy) use
# python:3.11-bookworm directly and don't need custom images.

set -euo pipefail

REGISTRY="ghcr.io/open-thoughts"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPOS=(pandas numpy pillow aiohttp orange3)

for repo in "${REPOS[@]}"; do
    IMAGE="${REGISTRY}/r2egym-${repo}:latest"
    echo ""
    echo "=========================================="
    echo "Building: ${IMAGE}"
    echo "=========================================="
    docker build -t "${IMAGE}" "${SCRIPT_DIR}/${repo}/"
    echo ""
    echo "Pushing: ${IMAGE}"
    docker push "${IMAGE}"
    echo "Done: ${IMAGE}"
done

echo ""
echo "All images built and pushed."
echo "Images:"
for repo in "${REPOS[@]}"; do
    echo "  ghcr.io/open-thoughts/r2egym-${repo}:latest"
done
