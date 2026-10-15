#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$repo/insula/capsule/enter.sh" run --role emacs -- /usr/bin/emacs --init-directory /home/dojo/.emacs.d -nw \
  -l /workspace/torch_dojo/insula/emacs/torch-dojo-session.el "$@"
