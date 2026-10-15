#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$repo/insula/capsule/enter.sh" run --role emacs -- /usr/bin/emacs --batch -l /workspace/torch_dojo/insula/emacs/verify_attach.el
