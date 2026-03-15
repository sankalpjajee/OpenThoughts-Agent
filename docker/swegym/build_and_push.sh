#!/bin/bash
# Build and push all custom swegym Docker images to ghcr.io
# Usage: bash docker/swegym/build_and_push.sh
# Requires: docker login to ghcr.io (done via: echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin)

set -e

REGISTRY="ghcr.io/sankalpjajee"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Login to ghcr.io using gh token
echo "Logging in to ghcr.io..."
echo "$(gh auth token)" | docker login ghcr.io -u sankalpjajee --password-stdin

IMAGES=(
    "pandas-1.5:Dockerfile.pandas-1.5"
    "pandas-2.0:Dockerfile.pandas-2.0"
    "pandas-2.1:Dockerfile.pandas-2.1"
    "pandas-2.2:Dockerfile.pandas-2.2"
    "pandas-3.0:Dockerfile.pandas-3.0"
    "monai:Dockerfile.monai"
    "modin:Dockerfile.modin"
)

for entry in "${IMAGES[@]}"; do
    name="${entry%%:*}"
    dockerfile="${entry##*:}"
    image="${REGISTRY}/swegym-${name}:latest"

    echo ""
    echo "=== Building ${image} ==="
    docker build \
        -f "${SCRIPT_DIR}/${dockerfile}" \
        -t "${image}" \
        "${SCRIPT_DIR}"

    echo "=== Pushing ${image} ==="
    docker push "${image}"
    echo "=== Done: ${image} ==="
done

echo ""
echo "All images built and pushed successfully!"
echo ""
echo "Images:"
for entry in "${IMAGES[@]}"; do
    name="${entry%%:*}"
    echo "  ${REGISTRY}/swegym-${name}:latest"
done
