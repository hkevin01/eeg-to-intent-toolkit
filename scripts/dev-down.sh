#!/usr/bin/env bash
set -euo pipefail

# Tear down the dev stack, removing volumes only if --volumes is passed through
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

docker compose down "$@"
