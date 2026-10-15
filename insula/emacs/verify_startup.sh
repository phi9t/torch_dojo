#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$repo/target/insula/session"
# A real terminal startup is needed: --batch skips Doom initialization.
printf -v startup_command '%q ' bash "$repo/insula/emacs/run.sh" --eval \
  '(progn (unless (featurep (quote torch-dojo-session)) (error "Dojo transport not loaded")) (kill-emacs 0))'
TERM=xterm timeout 30 script --quiet --return --command "$startup_command" \
  "$repo/target/insula/session/startup.log"
