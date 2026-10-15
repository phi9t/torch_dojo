#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$repo/target/insula/session"
exec timeout 150 "$repo/insula/capsule/enter.sh" run --role emacs -- \
  /usr/bin/emacs --batch -l /workspace/torch_dojo/insula/emacs/session_test.el
