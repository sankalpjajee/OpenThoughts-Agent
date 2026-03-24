#!/bin/bash
# Build and push custom R2E-Gym base images to ghcr.io/sankalpjajee
#
# Prerequisites:
#   1. Create a GitHub PAT with write:packages scope at:
#      https://github.com/settings/tokens/new
#   2. Login to ghcr.io:
#      echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u sankalpjajee --password-stdin
#   3. After pushing, make packages public at:
#      https://github.com/sankalpjajee?tab=packages
#      (each package → Package settings → Change visibility → Public)
#
# Usage:
#   bash data/patchers/r2egym_base_images/build_and_push.sh
#
# This builds 5 images for repos with compiled C extensions.
# Pure-Python repos (tornado, scrapy, pyramid, datalad, coveragepy) use
# python:3.11-bookworm directly and don't need custom images.

set -euo pipefail

REGISTRY="ghcr.io/sankalpjajee"
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
echo ""
echo "IMPORTANT: Make each package public so Harbor can pull without auth:"
echo "  https://github.com/sankalpjajee?tab=packages"
echo "  (each r2egym-* package → Package settings → Change visibility → Public)"
echo ""
echo "Or use the GitHub API:"
for repo in "${REPOS[@]}"; do
    echo "  gh api --method PATCH /user/packages/container/r2egym-${repo} -f visibility=public"
done
