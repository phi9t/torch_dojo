#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"$repo/insula/capsule/enter.sh" run --role jupyter -- python /workspace/torch_dojo/insula/jupyter_rootfs/verify.py
[[ $(cat "$repo/target/insula/shared/kernel-write.txt") == 'visible from host' ]]
printf 'Host sees kernel-written file\n'
