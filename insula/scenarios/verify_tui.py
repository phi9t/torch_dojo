"""Verify the bundled GPU notebook through the Emacs bridge."""

from pathlib import Path

from insula.emacs import scenario_host


def verify(client, run) -> None:
    steps = client.request('snapshot')['steps']
    kinds = [step['kind'] for step in steps]
    cell_count = kinds.count('cell')
    if kinds != ['setup', 'connect', *(['cell'] * cell_count), 'final', 'cleanup']:
        raise RuntimeError('Unexpected scenario lifecycle')
    if cell_count < 3:
        raise RuntimeError('Bundled GPU notebook is missing a cell')

    kernel_id = None
    for index, kind in enumerate(kinds):
        state = client.request('next')
        if state.get('failed') or state.get('pending_index') != index + 1:
            raise RuntimeError(f'Checkpoint {index + 1} failed through Emacs worker')
        result = state['steps'][index].get('result') or ''
        if kind == 'setup' and ('rootfs=' not in result or
                                'repo=/workspace/torch_dojo' not in result):
            raise RuntimeError('Setup evidence missing')
        if kind == 'connect':
            kernel_id = state.get('kernel_id')
            if not kernel_id or f'kernel-id={kernel_id}' not in result:
                raise RuntimeError('Connection evidence missing')
        if kind in ('cell', 'final', 'cleanup') and state.get('kernel_id') != kernel_id:
            raise RuntimeError('Kernel ID changed during scenario')
        if index == 2 and ('executable=/usr/bin/python' not in result or
                           'cwd=/workspace/torch_dojo' not in result or
                           'home=/home/dojo' not in result):
            raise RuntimeError('Kernel context evidence missing')
        if index == 3 and 'device=cuda:0 sum=10.0' not in result:
            raise RuntimeError('CUDA evidence missing')
        if index == 4 and '/workspace/torch_dojo/target/insula/scenario/' not in result:
            raise RuntimeError('Workspace write evidence missing')
        if kind == 'final' and f'completed-cells={cell_count}' not in result:
            raise RuntimeError('Final summary missing')
        if kind == 'cleanup' and f'kernel-id={kernel_id} absent' not in result:
            raise RuntimeError('Kernel cleanup evidence missing')
        print(f'{index + 1}/{len(steps)} {result}', flush=True)
    root = Path(__file__).resolve().parents[2]
    scenario_host.check_workspace_evidence(run)
    if kernel_id in {item['id'] for item in scenario_host.api(root, '/api/kernels')}:
        raise RuntimeError('Scenario-owned kernel remains on server')
