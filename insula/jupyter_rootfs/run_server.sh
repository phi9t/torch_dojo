#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python3 "$repo/insula/rootfs/select.py" >/dev/null
command -v tmux >/dev/null
state="$repo/target/insula/shared"
mkdir -p "$state"
chmod 700 "$state"
if tmux has-session -t torch-dojo-jupyter 2>/dev/null; then echo 'Jupyter session already running' >&2; exit 1; fi
if python3 - <<'PY'
import socket
s=socket.socket(); result=s.connect_ex(('127.0.0.1',8899)); s.close(); raise SystemExit(result != 0)
PY
then echo '127.0.0.1:8899 is occupied' >&2; exit 1; fi
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$state/token"
chmod 600 "$state/token"
printf -v command '%q ' "$repo/insula/capsule/enter.sh" run --role jupyter -- /bin/bash /workspace/torch_dojo/insula/jupyter_rootfs/server_entry.sh
tmux new-session -d -s torch-dojo-jupyter -c "$repo" "exec $command >>'$state/server.log' 2>&1"
printf 'Jupyter: http://127.0.0.1:8899/?token=%s\n' "$(cat "$state/token")"
printf 'Token: %s\nLog: %s\n' "$state/token" "$state/server.log"
