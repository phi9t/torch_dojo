#!/usr/bin/env bash
# Run the integrated CUDA checks (toolchain, compilers, torch interop) inside the GPU insula.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"$repo/insula/capsule/enter.sh" run --role jupyter -- python /workspace/torch_dojo/insula/jupyter_rootfs/verify_cuda.py "$@"
