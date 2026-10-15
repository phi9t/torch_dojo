#!/usr/bin/env bash
set -euo pipefail
cd /workspace/torch_dojo
mkdir -p "$HOME/.local/share/jupyter/kernels/torch-dojo-gpu" "$JUPYTER_RUNTIME_DIR"
python - <<'PY'
import json, pathlib, sys
p=pathlib.Path.home()/'.local/share/jupyter/kernels/torch-dojo-gpu/kernel.json'
p.write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Torch Dojo GPU','language':'python'}))
import torch
assert torch.version.cuda, 'CUDA PyTorch build required'
assert torch.cuda.is_available(), 'CUDA device unavailable'
torch.randn(16,device='cuda').sum().item()
torch.cuda.synchronize()
PY
token=$(cat /workspace/torch_dojo/target/insula/shared/token)
exec python -m jupyter_server --ServerApp.ip=127.0.0.1 --ServerApp.port=8899 --ServerApp.port_retries=0 --ServerApp.token="$token" --ServerApp.root_dir=/workspace/torch_dojo --ServerApp.open_browser=False --ServerApp.allow_remote_access=False
