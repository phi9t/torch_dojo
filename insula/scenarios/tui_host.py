"""Own the host side of a TUI scenario run."""

import os
from pathlib import Path
import subprocess
import sys
import time

from insula.emacs import scenario_host
from insula.scenarios.tui_client import OutcomeUnknown, ScenarioClient, ScenarioError


ROOT = Path(__file__).resolve().parents[2]


def validate_rootfs() -> None:
    subprocess.run(['python3', str(ROOT / 'insula/rootfs/select.py')],
                   check=True, stdout=subprocess.DEVNULL)


def start_worker(run: Path, *, verification: bool = False):
    log = (run / 'worker.log').open('w')
    env = os.environ.copy()
    if not verification:
        env.pop('DOJO_VERIFY_TOKEN', None)
        env.pop('DOJO_VERIFY_PORT', None)
    env['DOJO_SCENARIO_DIR'] = f'{scenario_host.CONTAINER_ROOT}/target/insula/scenario/{run.name}'
    try:
        process = subprocess.Popen([
            str(ROOT / 'insula/capsule/enter.sh'), 'run', '--role', 'emacs', '--',
            '/usr/bin/emacs', '--batch', '-Q',
            '-l', f'{scenario_host.CONTAINER_ROOT}/insula/emacs/scenario.el',
            '-l', f'{scenario_host.CONTAINER_ROOT}/insula/emacs/scenario_bridge.el',
            '--eval', '(torch-dojo-scenario-bridge-serve)'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    except BaseException:
        log.close()
        raise
    return process, log


def wait_ready(process, client: ScenarioClient) -> None:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError('Emacs worker died during startup')
        try:
            client.request('snapshot')
            return
        except ScenarioError:
            time.sleep(.1)
    raise RuntimeError('Emacs worker did not become ready')


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    verification = argv == ['--verify']
    if len(argv) > 1 or (argv and argv[0].startswith('--') and not verification):
        print('usage: run_tui.sh [path/to/file.org | --verify]', file=sys.stderr)
        return 2
    owned = False
    run = None
    process = None
    log = None
    client = None
    clean_shutdown = False
    unknown_outcome = False
    status = 1
    try:
        source = scenario_host.resolve_scenario_source(ROOT, None if verification else (argv[0] if argv else None))
        validate_rootfs()
        server_log = ROOT / 'target/insula/shared/server.log'
        server_log_offset = server_log.stat().st_size if server_log.exists() else 0
        owned = scenario_host.ensure_server(ROOT)
        run = scenario_host.create_run(ROOT, source)
        print(f'Scenario: {run / "run.org"}', flush=True)
        process, log = start_worker(run, verification=verification)
        client = ScenarioClient(run / 'bridge.sock')
        wait_ready(process, client)
        if verification:
            from insula.scenarios.verify_tui import verify
            verify(client, run)
            status = 0
        else:
            from insula.scenarios.tui_app import ScenarioApp
            app = ScenarioApp(
                client, worker_log=run / 'worker.log', server_log=server_log,
                token_path=ROOT / 'target/insula/shared/token',
                server_log_offset=server_log_offset)
            app.run()
            if app.failure:
                print(f'TUI: {app.failure}', file=sys.stderr)
                status = 1
            else:
                status = 0
            clean_shutdown = app.quit_acknowledged
        if not clean_shutdown and process.poll() is None:
            client.request('quit')
            clean_shutdown = True
        if clean_shutdown:
            exit_status = process.wait(timeout=10)
            if exit_status:
                status = 1
            if (run / 'cleanup-error').exists():
                print(f'Kernel cleanup error; inspect {run / "run.org"}', file=sys.stderr)
                status = 1
                clean_shutdown = False
            if verification and scenario_host.check_workspace_evidence(run):
                print(f'Host verified workspace file: {run / "workspace.txt"}')
    except OutcomeUnknown as exc:
        print(f'Scenario TUI: {exc}', file=sys.stderr)
        unknown_outcome = True
        status = 1
    except (RuntimeError, ScenarioError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(f'Scenario TUI: {exc}', file=sys.stderr)
        status = 1
    finally:
        if (process and process.poll() is None and client and
                client.last_snapshot and not clean_shutdown and not unknown_outcome):
            try:
                client.request('quit')
                clean_shutdown = True
                if process.wait(timeout=10):
                    status = 1
            except (ScenarioError, subprocess.TimeoutExpired):
                pass
        if run and (run / 'cleanup-error').exists():
            clean_shutdown = False
            status = 1
        if log:
            log.close()
        if process and process.poll() is None and not clean_shutdown:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        if run and not clean_shutdown and process:
            kernel_id = ((client.last_snapshot or {}).get('kernel_id') if client else None)
            print(f'Possible orphan kernel {kernel_id or "unknown"}; inspect {run}',
                  file=sys.stderr)
        if owned and (clean_shutdown or process is None or
                      (client is not None and client.last_snapshot is None
                       and not unknown_outcome)):
            scenario_host.finish_server(ROOT, True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
