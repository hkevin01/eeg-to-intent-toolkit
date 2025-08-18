#!/usr/bin/env bash
set -euo pipefail
IMAGE_TAG=${1:-eegintent:latest}
docker build -t "$IMAGE_TAG" .
