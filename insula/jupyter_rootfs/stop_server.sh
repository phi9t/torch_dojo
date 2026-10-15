#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
state="$repo/target/insula/shared"
if [[ -f $state/token ]]; then
  python3 - "$state/token" <<'PY'
import json,sys,urllib.request
try:
    token=open(sys.argv[1]).read().strip()
    req=urllib.request.Request('http://127.0.0.1:8899/api/kernels',headers={'Authorization':'token '+token})
    with urllib.request.urlopen(req,timeout=4) as r: kernels=json.load(r)
    for kernel in kernels:
        req=urllib.request.Request('http://127.0.0.1:8899/api/kernels/'+kernel['id'],method='DELETE',headers={'Authorization':'token '+token})
        urllib.request.urlopen(req,timeout=4).close()
except Exception as exc:
    print('kernel cleanup:',exc,file=sys.stderr)
PY
fi
if tmux has-session -t torch-dojo-jupyter 2>/dev/null; then tmux kill-session -t torch-dojo-jupyter; fi
rm -f "$state/token"
