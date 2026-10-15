#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
python=${TORCH_DOJO_TUI_PYTHON:-}
if [[ -z $python ]]; then
  if [[ -x target/insula/tui-venv/bin/python ]]; then
    python=target/insula/tui-venv/bin/python
  else
    python=python3
  fi
fi
exec "$python" -m insula.scenarios.tui_host "$@"
