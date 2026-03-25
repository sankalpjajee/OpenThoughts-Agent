#!/bin/bash
# Build and push custom R2E-Gym base images to Docker Hub (sankalpjajee)
# Also pushes to ghcr.io/sankalpjajee as a mirror.
#
# Prerequisites:
#   1. Login to Docker Hub:
#      docker login -u sankalpjajee
#   2. (Optional) Login to ghcr.io for the mirror:
#      echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u sankalpjajee --password-stdin
#
# Usage:
#   bash data/patchers/r2egym_base_images/build_and_push.sh
#
# This builds 5 images for repos with compiled C extensions.
# Pure-Python repos (tornado, scrapy, pyramid, datalad, coveragepy) use
# python:3.11-bookworm directly and don't need custom images.

set -euo pipefail

DOCKERHUB_REGISTRY="sankalpjajee"
GHCR_REGISTRY="ghcr.io/sankalpjajee"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPOS=(pandas numpy pillow aiohttp orange3)

for repo in "${REPOS[@]}"; do
    DH_IMAGE="${DOCKERHUB_REGISTRY}/r2egym-${repo}:v1"
    GHCR_IMAGE="${GHCR_REGISTRY}/r2egym-${repo}:v1"

    echo ""
    echo "=========================================="
    echo "Building: ${DH_IMAGE}"
    echo "=========================================="
    docker build -t "${DH_IMAGE}" "${SCRIPT_DIR}/${repo}/"

    echo ""
    echo "Pushing to Docker Hub: ${DH_IMAGE}"
    docker push "${DH_IMAGE}"

    # Mirror to ghcr.io if logged in
    if docker info 2>/dev/null | grep -q "ghcr.io"; then
        echo "Pushing to ghcr.io: ${GHCR_IMAGE}"
        docker tag "${DH_IMAGE}" "${GHCR_IMAGE}"
        docker push "${GHCR_IMAGE}" || echo "Warning: ghcr.io push failed, Docker Hub push succeeded"
    fi

    echo "Done: ${DH_IMAGE}"
done

echo ""
echo "All images built and pushed to Docker Hub."
echo "Pull with: docker pull sankalpjajee/r2egym-<repo>:v1"
