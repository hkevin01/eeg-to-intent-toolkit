#!/usr/bin/env bash
set -euo pipefail

# Exec into the app container (default) or target container
SERVICE=${1:-app}
shift || true

docker compose exec "$SERVICE" bash "$@"
