"""Host-side lifecycle for the terminal Emacs Jupyter scenario."""

import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


URL = 'http://127.0.0.1:8899'
CONTAINER_ROOT = '/workspace/torch_dojo'


def token(root):
    path = root / 'target/insula/shared/token'
    if not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError('Jupyter token file missing or insecure')
    value = path.read_text().strip()
    if not value:
        raise RuntimeError('Jupyter token file empty')
    return value


def api(root, path):
    req = urllib.request.Request(URL + path,
                                 headers={'Authorization': 'token ' + token(root)})
    with urllib.request.urlopen(req, timeout=3) as response:
        return json.load(response)


def server_responds(root):
    try:
        specs = api(root, '/api/kernelspecs')
        return 'torch-dojo-gpu' in specs.get('kernelspecs', {})
    except (OSError, ValueError, urllib.error.URLError, RuntimeError):
        return False


def port_occupied():
    with socket.socket() as sock:
        server = urllib.parse.urlsplit(URL)
        return sock.connect_ex((server.hostname, server.port)) == 0


def start_server(root):
    # run_server.sh prints an authenticated URL; keep it out of scenario output.
    result = subprocess.run([str(root / 'insula/jupyter_rootfs/run_server.sh')],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Jupyter server start failed; see target/insula/shared/server.log')


def stop_server(root):
    subprocess.run([str(root / 'insula/jupyter_rootfs/stop_server.sh')], check=True)


def ensure_server(root):
    if server_responds(root):
        return False
    if port_occupied():
        address = urllib.parse.urlsplit(URL).netloc
        try:
            token(root)
        except RuntimeError as exc:
            raise RuntimeError(f'{address} is occupied; {str(exc).lower()}') from exc
        raise RuntimeError(f'{address} is occupied by another service or rejects the token')
    start_server(root)
    for _ in range(30):
        if server_responds(root):
            return True
        time.sleep(1)
    raise RuntimeError('Jupyter server did not become ready; see target/insula/shared/server.log')


def kernel_count(root):
    return len(api(root, '/api/kernels'))


def finish_server(root, owned):
    if not owned:
        return
    try:
        count = kernel_count(root)
    except (OSError, ValueError, urllib.error.URLError, RuntimeError) as exc:
        print(f'Leaving owned Jupyter server running: kernel count unavailable ({type(exc).__name__})',
              file=sys.stderr)
        return
    if count:
        print(f'Leaving owned Jupyter server running: {count} kernel(s) remain', file=sys.stderr)
    else:
        stop_server(root)


def resolve_scenario_source(root: Path, argument: str | None = None) -> Path:
    """Resolve a readable Org file within this checkout before starting a server."""
    checkout = root.resolve()
    source = (checkout / (argument or 'insula/scenarios/default.org')).resolve()
    if not source.is_relative_to(checkout) or source.suffix.lower() != '.org':
        raise RuntimeError('Scenario must be an Org file inside the checkout')
    if not source.is_file() or not os.access(source, os.R_OK):
        raise RuntimeError(f'Scenario Org file is missing or unreadable: {argument or source}')
    return source


def create_run(root, source=None):
    base = root / 'target/insula/scenario'
    base.mkdir(parents=True, exist_ok=True)
    path = base / (time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(4))
    path.mkdir(mode=0o700)
    default_source = root / 'insula/scenarios/default.org'
    chosen = source or resolve_scenario_source(root)
    template = chosen.read_bytes()
    container_path = f'{CONTAINER_ROOT}/target/insula/scenario/{path.name}'
    if chosen.resolve() == default_source.resolve():
        template = template.replace(b'__RUN_DIR__', container_path.encode())
    (path / 'run.org').write_bytes(template)
    return path


def check_workspace_evidence(run):
    """Check the bundled notebook's host-visible Jupyter workspace file."""
    evidence = run / 'workspace.txt'
    if not evidence.exists() or evidence.read_text() != 'visible from jupyter\n':
        raise RuntimeError('Host workspace evidence missing or mismatched')
    return True


def main():
    root = Path(__file__).resolve().parents[2]
    # Validate before starting or reusing a server.
    subprocess.run(['python3', str(root / 'insula/rootfs/select.py')],
                   check=True, stdout=subprocess.DEVNULL)
    owned = False
    status = 1
    try:
        owned = ensure_server(root)
        run = create_run(root)
        print(f'Scenario: {run / "run.org"}', flush=True)
        print(f'Jupyter log: {root / "target/insula/shared/server.log"}', flush=True)
        env = os.environ.copy()
        env.pop('DOJO_VERIFY_TOKEN', None)
        env.pop('DOJO_VERIFY_PORT', None)
        env['DOJO_SCENARIO_DIR'] = f'{CONTAINER_ROOT}/target/insula/scenario/{run.name}'
        status = subprocess.run([
            str(root / 'insula/emacs/run.sh'), '--load',
            f'{CONTAINER_ROOT}/insula/emacs/scenario.el',
            '--eval', '(torch-dojo-scenario-open)'], env=env).returncode
        if (run / 'cleanup-error').exists():
            print(f'Scenario cleanup error; inspect {run / "run.org"}', file=sys.stderr)
            status = 1
        try:
            if check_workspace_evidence(run):
                print(f'Host verified workspace file: {run / "workspace.txt"}')
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            status = 1
    finally:
        finish_server(root, owned)
    return status


def verify_main():
    """Drive the bundled Org cells in batch and check host evidence."""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['python3', str(root / 'insula/rootfs/select.py')],
                   check=True, stdout=subprocess.DEVNULL)
    owned = False
    try:
        owned = ensure_server(root)
        before = {item['id'] for item in api(root, '/api/kernels')}
        run = create_run(root)
        env = os.environ.copy()
        env['DOJO_SCENARIO_DIR'] = f'{CONTAINER_ROOT}/target/insula/scenario/{run.name}'
        result = subprocess.run([
            str(root / 'insula/capsule/enter.sh'), 'run', '--role', 'emacs', '--',
            '/usr/bin/emacs', '--batch', '-Q', '-l',
            f'{CONTAINER_ROOT}/insula/emacs/scenario.el', '-l',
            f'{CONTAINER_ROOT}/insula/emacs/verify_scenario.el'], env=env)
        if (run / 'cleanup-error').exists():
            print(f'Scenario cleanup error; inspect {run / "run.org"}', file=sys.stderr)
            return 1
        after = {item['id'] for item in api(root, '/api/kernels')}
        if after - before:
            raise RuntimeError(f'Scenario left {len(after - before)} kernel(s) running')
        if result.returncode:
            return result.returncode
        if not check_workspace_evidence(run):
            raise RuntimeError('Host workspace evidence missing')
        print(f'Host verified workspace file: {run / "workspace.txt"}')
        return 0
    finally:
        finish_server(root, owned)


if __name__ == '__main__':
    try:
        raise SystemExit(verify_main() if sys.argv[1:] == ['--verify'] else main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Scenario: {exc}', file=sys.stderr)
        raise SystemExit(1)
