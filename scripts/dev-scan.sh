#!/usr/bin/env bash
set -euo pipefail

# Lightweight local scans: Hadolint (Dockerfile) and Trivy filesystem scan
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Hadolint (requires hadolint installed locally)
if command -v hadolint &> /dev/null; then
  echo "Running hadolint..."
  hadolint Dockerfile || true
else
  echo "hadolint not found; skipping"
fi

# Trivy filesystem scan (requires trivy installed locally)
if command -v trivy &> /dev/null; then
  echo "Running trivy filesystem scan..."
  trivy fs --exit-code 0 --severity HIGH,CRITICAL . || true
else
  echo "trivy not found; skipping"
fi
