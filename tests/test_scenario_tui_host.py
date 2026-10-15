import tempfile
import unittest
from pathlib import Path
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import sys
from types import ModuleType

from insula.scenarios import tui_host
from insula.scenarios.tui_client import OutcomeUnknown
from insula.scenarios.verify_tui import verify


class ScenarioTuiHostTest(unittest.TestCase):
    def test_verifier_uses_discovered_step_count(self):
        class Worker:
            def __init__(self):
                self.index = 0
                self.commands = []

            def request(self, command):
                self.commands.append(command)
                if command == 'next':
                    self.index += 1
                results = [
                    ('setup', 'rootfs=demo repo=/workspace/torch_dojo'),
                    ('connect', 'kernel-id=k-1'),
                    ('cell', 'executable=/usr/bin/python cwd=/workspace/torch_dojo home=/home/dojo'),
                    ('cell', 'device=cuda:0 sum=10.0'),
                    ('cell', 'workspace=/workspace/torch_dojo/target/insula/scenario/demo/workspace.txt'),
                    ('cell', 'extra cell'),
                    ('final', 'completed-cells=4'),
                    ('cleanup', 'kernel-id=k-1 absent from server'),
                ]
                return {'pending_index': self.index, 'failed': False,
                        'kernel_id': 'k-1' if self.index >= 2 else None,
                        'steps': [{'kind': kind, 'result': result if n < self.index else None}
                                  for n, (kind, result) in enumerate(results)]}

        worker = Worker()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'workspace.txt').write_text('visible from jupyter\n')
            with mock.patch.object(tui_host.scenario_host, 'api', return_value=[]):
                verify(worker, run)
        self.assertEqual(worker.commands.count('next'), 8)

    def test_rootfs_validation_precedes_server_start(self):
        with mock.patch.object(tui_host, 'validate_rootfs', side_effect=RuntimeError('bad rootfs')), \
             mock.patch.object(tui_host.scenario_host, 'ensure_server') as server:
            self.assertEqual(tui_host.main([]), 1)
            server.assert_not_called()

    def test_worker_start_failure_finishes_owned_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with mock.patch.object(tui_host, 'validate_rootfs'), \
                 mock.patch.object(tui_host.scenario_host, 'ensure_server', return_value=True), \
                 mock.patch.object(tui_host.scenario_host, 'create_run', return_value=run), \
                 mock.patch.object(tui_host, 'start_worker', side_effect=RuntimeError('worker died')), \
                 mock.patch.object(tui_host.scenario_host, 'finish_server') as finish:
                self.assertEqual(tui_host.main([]), 1)
                finish.assert_called_once_with(tui_host.ROOT, True)

    def test_worker_readiness_failure_finishes_owned_server_when_no_kernel_was_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            process = mock.Mock()
            process.poll.return_value = 1
            client = mock.Mock(last_snapshot=None)
            with mock.patch.object(tui_host, 'validate_rootfs'), \
                 mock.patch.object(tui_host.scenario_host, 'ensure_server', return_value=True), \
                 mock.patch.object(tui_host.scenario_host, 'create_run', return_value=run), \
                 mock.patch.object(tui_host, 'start_worker', return_value=(process, mock.Mock())), \
                 mock.patch.object(tui_host, 'ScenarioClient', return_value=client), \
                 mock.patch.object(tui_host, 'wait_ready', side_effect=RuntimeError('startup failed')), \
                 mock.patch.object(tui_host.scenario_host, 'finish_server') as finish:
                self.assertEqual(tui_host.main([]), 1)
                finish.assert_called_once_with(tui_host.ROOT, True)

    def test_lost_step_leaves_owned_server_and_reports_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            process = mock.Mock()
            process.poll.return_value = 1
            client = mock.Mock()
            client.last_snapshot = {'kernel_id': 'owned-123'}
            app = mock.Mock()
            app.run.side_effect = OutcomeUnknown('outcome unknown')
            frontend = ModuleType('insula.scenarios.tui_app')
            frontend.ScenarioApp = mock.Mock(return_value=app)
            output = StringIO()
            with mock.patch.object(tui_host, 'validate_rootfs'), \
                 mock.patch.object(tui_host.scenario_host, 'ensure_server', return_value=True), \
                 mock.patch.object(tui_host.scenario_host, 'create_run', return_value=run), \
                 mock.patch.object(tui_host, 'start_worker', return_value=(process, mock.Mock())), \
                 mock.patch.object(tui_host, 'wait_ready'), \
                 mock.patch.object(tui_host, 'ScenarioClient', return_value=client), \
                 mock.patch.dict(sys.modules, {'insula.scenarios.tui_app': frontend}), \
                 mock.patch.object(tui_host.scenario_host, 'finish_server') as finish, \
                 redirect_stderr(output), redirect_stdout(output):
                self.assertEqual(tui_host.main([]), 1)
            self.assertIn('owned-123', output.getvalue())
            self.assertIn(str(run), output.getvalue())
            finish.assert_not_called()

    def test_verifier_advances_bundled_steps(self):
        class Worker:
            def __init__(self):
                self.index = 0
                self.commands = []

            def request(self, command):
                self.commands.append(command)
                if command == 'next':
                    self.index += 1
                results = [
                    'rootfs=demo repo=/workspace/torch_dojo',
                    'kernel-id=k-1',
                    'executable=/usr/bin/python cwd=/workspace/torch_dojo home=/home/dojo',
                    'device=cuda:0 sum=10.0',
                    'workspace=/workspace/torch_dojo/target/insula/scenario/demo/workspace.txt',
                    'completed-cells=3',
                    'kernel-id=k-1 absent from server',
                ]
                kinds = ['setup', 'connect', 'cell', 'cell', 'cell', 'final', 'cleanup']
                return {'pending_index': self.index, 'failed': False,
                        'kernel_id': 'k-1' if self.index >= 2 else None,
                        'steps': [{'kind': kinds[n], 'result': result if n < self.index else None}
                                  for n, result in enumerate(results)]}

        worker = Worker()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'workspace.txt').write_text('visible from jupyter\n')
            with mock.patch.object(tui_host.scenario_host, 'api',
                                   return_value=[]):
                verify(worker, run)
        self.assertEqual(worker.commands.count('next'), 7)

    def test_verifier_rejects_python_outside_selected_rootfs(self):
        results = [
            'rootfs=demo repo=/workspace/torch_dojo', 'kernel-id=k-1',
            'executable=/usr/local/bin/python cwd=/workspace/torch_dojo home=/home/dojo',
            'device=cuda:0 sum=10.0',
            'workspace=/workspace/torch_dojo/target/insula/scenario/demo/workspace.txt',
            'completed-cells=3', 'kernel-id=k-1 absent from server',
        ]
        client = mock.Mock()
        kinds = ['setup', 'connect', 'cell', 'cell', 'cell', 'final', 'cleanup']
        client.request.side_effect = [
            {'steps': [{'kind': kind} for kind in kinds]},
            *[
            {'pending_index': index + 1, 'failed': False,
             'kernel_id': 'k-1' if index >= 1 else None,
             'steps': [{'kind': kinds[position], 'result': result if position <= index else None}
                       for position, result in enumerate(results)]}
            for index in range(7)]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'Kernel context evidence missing'):
                verify(client, run)
        self.assertEqual(client.request.call_count, 4)


if __name__ == '__main__':
    unittest.main()
