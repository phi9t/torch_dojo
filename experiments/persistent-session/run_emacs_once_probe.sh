#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$repo/target/insula/research/emacs-once"
timeout 180 "$repo/insula/capsule/enter.sh" run --role emacs -- \
  /usr/bin/emacs --batch -l /workspace/torch_dojo/experiments/persistent-session/emacs_once_probe.el \
  > "$repo/target/insula/research/emacs-once/run.log" 2>&1
cat "$repo/target/insula/research/emacs-once/run.log"
