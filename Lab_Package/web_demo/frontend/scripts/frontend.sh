#!/usr/bin/env bash
# Run with: bash scripts/frontend.sh install|dev|build|typecheck
# npm bootstraps Linux Node/pnpm in its cache; system Node is unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This launcher is for Linux/WSL. On Windows use pnpm with Node 24.' >&2
  exit 1
fi
if ! command -v npm >/dev/null; then
  echo 'npm is required to bootstrap the local runtime.' >&2
  exit 1
fi
# Bootstrap Node first, then run npm itself under that Node. This avoids the
# old system npm checking pnpm's engine requirement against system Node 18.
exec npm exec --yes --package=node@24 -- node "$(command -v npm)" exec --yes --package=pnpm@11.25.0 -- pnpm "$@"
