#!/usr/bin/env bash
set -euo pipefail

# Follow logs for specified service or all
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

docker compose logs -f "$@"
