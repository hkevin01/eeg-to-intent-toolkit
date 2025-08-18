#!/usr/bin/env bash
set -euo pipefail
IMAGE_TAG=${1:-eegintent:latest}
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  $IMAGE_TAG bash
