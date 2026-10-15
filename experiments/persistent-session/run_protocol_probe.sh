#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec timeout 180 "$repo/insula/capsule/enter.sh" run --role jupyter -- \
  python /workspace/torch_dojo/experiments/persistent-session/protocol_probe.py
