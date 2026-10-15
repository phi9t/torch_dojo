import pathlib
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

from insula.emacs import scenario_host


class ScenarioLauncherTest(unittest.TestCase):
    def test_custom_org_is_copied_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / 'lesson.org'
            source.write_bytes(b'* Lesson\n#+BEGIN_SRC python\nprint(7)\n#+END_SRC\n')
            chosen = scenario_host.resolve_scenario_source(root, 'lesson.org')
            run = scenario_host.create_run(root, chosen)
            self.assertEqual((run / 'run.org').read_bytes(), source.read_bytes())
            self.assertEqual(source.read_bytes(), b'* Lesson\n#+BEGIN_SRC python\nprint(7)\n#+END_SRC\n')

    def test_bad_selection_fails_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / 'folder').mkdir()
            (root / 'wrong.txt').write_text('hi')
            (root / 'outside.org').symlink_to(pathlib.Path(temporary).parent / 'outside.org')
            for argument in ('missing.org', 'folder', 'wrong.txt', 'outside.org', '../outside.org'):
                with self.subTest(argument=argument):
                    with self.assertRaises(RuntimeError):
                        scenario_host.resolve_scenario_source(root, argument)

    def test_create_run_copies_template_to_unique_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / 'insula/scenarios').mkdir(parents=True)
            (root / 'insula/scenarios/default.org').write_text('* Demo\n')
            first = scenario_host.create_run(root)
            second = scenario_host.create_run(root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'run.org').read_text(), '* Demo\n')
            self.assertEqual((second / 'run.org').read_text(), '* Demo\n')

    def test_custom_org_preserves_literal_run_dir_placeholder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / 'lesson.org'
            source.write_bytes(b'* Lesson\n#+BEGIN_SRC python\nprint("__RUN_DIR__")\n#+END_SRC\n')
            run = scenario_host.create_run(root, source)
            self.assertEqual((run / 'run.org').read_bytes(), source.read_bytes())

    def test_bundled_template_expands_run_dir_placeholder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            template = root / 'insula/scenarios/default.org'
            template.parent.mkdir(parents=True)
            template.write_text('workspace = "__RUN_DIR__/workspace.txt"\n')
            run = scenario_host.create_run(root)
            self.assertEqual((run / 'run.org').read_text(),
                             f'workspace = "/workspace/torch_dojo/target/insula/scenario/{run.name}/workspace.txt"\n')

    def test_foreign_service_is_not_started_or_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host, 'server_responds', return_value=False), \
                 mock.patch.object(scenario_host, 'port_occupied', return_value=True), \
                 mock.patch.object(scenario_host, 'start_server') as start:
                with self.assertRaisesRegex(RuntimeError, 'occupied'):
                    scenario_host.ensure_server(root)
                start.assert_not_called()

    def test_occupied_port_with_missing_token_names_token_problem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host, 'server_responds', return_value=False), \
                 mock.patch.object(scenario_host, 'port_occupied', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'token file missing or insecure'):
                    scenario_host.ensure_server(root)

    def test_wrong_token_names_the_server_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            token = root / 'target/insula/shared/token'
            token.parent.mkdir(parents=True)
            token.write_text('wrong-token\n')
            token.chmod(0o600)
            with mock.patch.object(scenario_host, 'URL', 'http://127.0.0.1:8898'), \
                 mock.patch.object(scenario_host, 'server_responds', return_value=False), \
                 mock.patch.object(scenario_host, 'port_occupied', return_value=True):
                with self.assertRaisesRegex(RuntimeError, '127.0.0.1:8898 is occupied.*rejects the token'):
                    scenario_host.ensure_server(root)

    def test_reused_server_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host, 'server_responds', return_value=True), \
                 mock.patch.object(scenario_host, 'stop_server') as stop:
                owned = scenario_host.ensure_server(root)
                scenario_host.finish_server(root, owned)
                self.assertFalse(owned)
                stop.assert_not_called()

    def test_owned_server_with_other_kernel_is_left_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host, 'kernel_count', return_value=1), \
                 mock.patch.object(scenario_host, 'stop_server') as stop:
                scenario_host.finish_server(root, True)
                stop.assert_not_called()

    def test_owned_server_without_kernels_is_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host, 'kernel_count', return_value=0), \
                 mock.patch.object(scenario_host, 'stop_server') as stop:
                scenario_host.finish_server(root, True)
                stop.assert_called_once_with(root)

    def test_rootfs_validation_precedes_server_start(self):
        with mock.patch.object(scenario_host.subprocess, 'run',
                               side_effect=subprocess.CalledProcessError(1, 'select.py')), \
             mock.patch.object(scenario_host, 'ensure_server') as ensure:
            with self.assertRaises(subprocess.CalledProcessError):
                scenario_host.main()
            ensure.assert_not_called()

    def test_server_start_output_does_not_expose_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with mock.patch.object(scenario_host.subprocess, 'run',
                                   return_value=subprocess.CompletedProcess([], 0,
                                                                            stdout='secret-token',
                                                                            stderr='')):
                output = StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    scenario_host.start_server(root)
                self.assertNotIn('secret-token', output.getvalue())

    def test_host_workspace_evidence_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = pathlib.Path(temporary)
            (run / 'workspace.txt').write_text('wrong\n')
            with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                scenario_host.check_workspace_evidence(run)


if __name__ == '__main__':
    unittest.main()
