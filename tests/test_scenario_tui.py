import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from insula.scenarios.tui_client import OutcomeUnknown, ScenarioClient
try:
    from insula.scenarios.tui_app import ScenarioApp
except ModuleNotFoundError as exc:
    if exc.name != 'textual':
        raise
    ScenarioApp = None


class ScenarioClientTest(unittest.TestCase):
    def test_snapshot_connects_to_long_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / ('long-' * 24)
            run.mkdir()
            path = run / 'bridge.sock'
            self.assertGreater(len(str(path).encode()), 107)
            directory_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
            server = socket.socket(socket.AF_UNIX)
            try:
                server.bind(f'/proc/self/fd/{directory_fd}/bridge.sock')
                server.listen()

                def answer():
                    conn, _ = server.accept()
                    with conn:
                        conn.recv(1024)
                        conn.sendall(b'{"id":1,"ok":true,"snapshot":{"pending_index":0}}\n')

                thread = threading.Thread(target=answer, daemon=True)
                thread.start()
                try:
                    self.assertEqual(ScenarioClient(path).request('snapshot')['pending_index'], 0)
                finally:
                    thread.join(timeout=2)
            finally:
                server.close()
                os.close(directory_fd)

    def test_snapshot_request_reads_worker_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bridge.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen()
            seen = []

            def answer():
                conn, _ = server.accept()
                with conn:
                    seen.append(json.loads(conn.recv(1024)))
                    conn.sendall(b'{"id":1,"ok":true,"snapshot":{"pending_index":0}}\n')

            thread = threading.Thread(target=answer)
            thread.start()
            try:
                state = ScenarioClient(path).request('snapshot')
            finally:
                thread.join(timeout=2)
                server.close()
            self.assertEqual(state['pending_index'], 0)
            self.assertEqual(seen[0]['command'], 'snapshot')

    def test_lost_next_response_has_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bridge.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen()
            seen = []

            def drop():
                conn, _ = server.accept()
                with conn:
                    seen.append(json.loads(conn.recv(1024)))

            thread = threading.Thread(target=drop)
            thread.start()
            try:
                with self.assertRaises(OutcomeUnknown):
                    ScenarioClient(path).request('next')
            finally:
                thread.join(timeout=2)
                server.close()
            self.assertEqual([item['command'] for item in seen], ['next'])

    def test_lost_source_edit_response_has_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bridge.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen()
            seen = []

            def drop():
                conn, _ = server.accept()
                with conn:
                    seen.append(json.loads(conn.recv(1024)))

            thread = threading.Thread(target=drop)
            thread.start()
            try:
                with self.assertRaises(OutcomeUnknown):
                    ScenarioClient(path).request('set_source',
                                                 {'cell_id': 'cell-2', 'source': 'print("é")'})
            finally:
                thread.join(timeout=2)
                server.close()
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]['args']['cell_id'], 'cell-2')

    def test_malformed_next_response_has_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bridge.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen()

            def corrupt():
                conn, _ = server.accept()
                with conn:
                    conn.recv(1024)
                    conn.sendall(b'{bad response}\n')

            thread = threading.Thread(target=corrupt)
            thread.start()
            try:
                with self.assertRaises(OutcomeUnknown):
                    ScenarioClient(path).request('next')
            finally:
                thread.join(timeout=2)
                server.close()

    def test_mismatched_edit_response_has_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bridge.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen()

            def wrong_id():
                conn, _ = server.accept()
                with conn:
                    conn.recv(1024)
                    conn.sendall(b'{"id":99,"ok":true,"snapshot":{}}\n')

            thread = threading.Thread(target=wrong_id)
            thread.start()
            try:
                with self.assertRaises(OutcomeUnknown):
                    ScenarioClient(path).request('set_source',
                                                 {'cell_id': 'cell-1', 'source': 'print(2)'})
            finally:
                thread.join(timeout=2)
                server.close()


@unittest.skipIf(ScenarioApp is None, 'optional Textual dependency not installed')
class ScenarioAppTest(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_preserves_editor_caret_selection_undo_and_scroll(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Lesson', 'cell_id': 'cell-1',
                                   'state': 'pending', 'source': '\n'.join(
                                       f'print({i})' for i in range(40)),
                                   'result': None}], 'pending_index': 0,
                        'viewed_index': 0, 'busy': False, 'failed': False,
                        'status': 'ready', 'emacs_pid': 7, 'kernel_id': 'k'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            editor = app.query_one('#editor')
            editor.focus()
            editor.cursor_location = (20, 3)
            editor.selection = ((20, 3), (20, 6))
            editor.scroll_y = 18
            old_selection = editor.selection
            old_scroll = editor.scroll_y
            app._receive(worker.request('snapshot'), 'snapshot', None)
            app._refresh_logs()
            await pilot.pause()
            self.assertEqual(editor.selection, old_selection)
            self.assertEqual(editor.scroll_y, old_scroll)
            self.assertIs(app.focused, editor)
            editor.selection = ((20, 3), (20, 3))
            await pilot.press('x')
            await pilot.pause()
            edited = editor.text
            app._render()
            await pilot.press('ctrl+z')
            await pilot.pause()
            self.assertNotEqual(editor.text, edited)

    async def test_structured_text_output_and_redaction_at_both_sizes(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Cell secret-token', 'cell_id': 'cell-1',
                                   'heading': 'Lesson', 'text': 'secret-token context',
                                   'source': 'print("secret-token")', 'state': 'complete',
                                   'result': 'secret-token',
                                   'output': {'stdout': 'α' * 150 + ' secret-token',
                                              'text': 'value=2', 'error': 'Traceback\n' + 'boom ' * 80,
                                              'mime': ['image/png']}}],
                        'pending_index': 1, 'viewed_index': 0, 'busy': False,
                        'failed': False, 'status': 'secret-token status',
                        'emacs_pid': 7, 'kernel_id': 'k'}

        with tempfile.TemporaryDirectory() as temporary:
            token = Path(temporary) / 'token'
            token.write_text('secret-token')
            for size in ((80, 24), (120, 36)):
                app = ScenarioApp(Worker(), token_path=token)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    result = '\n'.join(line.text for line in app.query_one('#result').lines)
                    self.assertIn('STDOUT', result)
                    self.assertIn('VALUE', result)
                    self.assertIn('ERROR', result)
                    self.assertIn('MIME', result)
                    self.assertIn('image/png', result)
                    displayed = result + app.query_one('#description').content
                    displayed += app.query_one('#editor').text
                    displayed += app.query_one('#status').content
                    displayed += app.query_one('#steps').children[0].query_one('Label').content
                    self.assertNotIn('secret-token', displayed)
                    self.assertGreater(app.query_one('#result').size.height, 1)
                    self.assertGreater(app.query_one('#emacs-log').size.height, 0)
                    self.assertGreater(app.query_one('#server-log').size.height, 0)

    async def test_resize_rewraps_result_and_process_logs(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Cell', 'cell_id': 'cell-1',
                                   'source': 'print(1)', 'state': 'complete',
                                   'result': None,
                                   'output': {'stdout': 'x' * 100, 'text': '',
                                              'error': '', 'mime': []}}],
                        'pending_index': 1, 'viewed_index': 0, 'busy': False,
                        'failed': False, 'status': 'ready'}

        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'worker.log'
            log.write_text('y' * 100 + '\n')
            app = ScenarioApp(Worker(), worker_log=log)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                narrow_result = len(app.query_one('#result').lines)
                narrow_log = len(app.query_one('#emacs-log').lines)
                await pilot.resize_terminal(120, 36)
                await pilot.pause()
                self.assertLess(len(app.query_one('#result').lines), narrow_result)
                self.assertLess(len(app.query_one('#emacs-log').lines), narrow_log)

    async def test_tab_reaches_notebook_panes_and_controls(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Cell', 'cell_id': 'cell-1',
                                   'source': 'print(1)', 'state': 'pending',
                                   'result': None}], 'pending_index': 0,
                        'viewed_index': 0, 'busy': False, 'failed': False,
                        'status': 'ready'}

        app = ScenarioApp(Worker())
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.query_one('#steps').focus()
            seen = []
            for _ in range(7):
                seen.append(app.focused.id)
                await pilot.press('tab')
            self.assertEqual(seen[:5], ['steps', 'editor', 'result',
                                        'emacs-log', 'server-log'])
            self.assertEqual(seen[5:7], ['next', 'previous'])

    async def test_pending_cell_shows_preceding_structured_output_and_empty_output(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [
                    {'name': 'First', 'cell_id': 'cell-1', 'source': '1',
                     'state': 'complete', 'result': '1',
                     'output': {'stdout': '', 'text': '1', 'error': None, 'mime': []}},
                    {'name': 'Second', 'cell_id': 'cell-2', 'source': 'pass',
                     'state': 'pending', 'result': None, 'output': None}],
                    'pending_index': 1, 'viewed_index': 1, 'busy': False,
                    'failed': False, 'status': 'ready'}

        app = ScenarioApp(Worker())
        async with app.run_test() as pilot:
            await pilot.pause()
            shown = '\n'.join(line.text for line in app.query_one('#result').lines)
            self.assertIn('VALUE', shown)
            app.snapshot['steps'][0]['output'] = {
                'stdout': '', 'text': '', 'error': None, 'mime': []}
            app._render()
            self.assertIn('EMPTY OUTPUT', '\n'.join(
                line.text for line in app.query_one('#result').lines))

    async def test_edit_around_redacted_token_preserves_original_source(self):
        class Worker:
            def __init__(self):
                self.saved = None

            def request(self, command, args=None):
                if command == 'set_source':
                    self.saved = args['source']
                return {'steps': [{'name': 'Cell', 'cell_id': 'cell-1',
                                   'source': 'secret-token\nprint(1)',
                                   'state': 'pending', 'result': None}],
                        'pending_index': 0, 'viewed_index': 0,
                        'busy': False, 'failed': False, 'status': 'ready'}

        with tempfile.TemporaryDirectory() as temporary:
            token = Path(temporary) / 'token'
            token.write_text('secret-token')
            worker = Worker()
            app = ScenarioApp(worker, token_path=token)
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertIn('[redacted]', app.query_one('#editor').text)
                app.query_one('#editor').text = '[redacted]\nprint(2)'
                await pilot.pause()
                await pilot.press('f10')
                await pilot.pause()
                self.assertEqual(worker.saved, 'secret-token\nprint(2)')

    async def test_saved_result_freshness_comes_from_worker_snapshot(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [
                    {'name': 'Edited cell', 'cell_id': 'cell-1', 'state': 'complete',
                     'source': 'print(2)', 'result': '1', 'stale': True},
                    {'name': 'Final', 'state': 'pending', 'source': None, 'result': None}],
                    'pending_index': 1, 'viewed_index': 0, 'busy': False,
                    'failed': False, 'status': 'ready', 'kernel_id': None}
        app = ScenarioApp(Worker())
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn('STALE', app.query_one('#result-label').content)
            self.assertNotIn('UNSAVED', app.query_one('#source-label').content)
            self.assertIn('F10 run next', app.query_one('#status').content)
            self.assertEqual(app.query_one('#editor').text, 'print(2)')

    async def test_completed_cell_edit_is_dirty_and_stale_until_restart(self):
        class Worker:
            def __init__(self):
                self.calls = []
                self.source = 'print(1)\n'
                self.restarted = False

            def request(self, command, args=None):
                self.calls.append((command, args))
                if command == 'set_source':
                    self.source = args['source']
                if command == 'restart':
                    self.restarted = True
                return {'steps': [
                    {'name': 'Setup', 'state': 'pending' if self.restarted else 'complete',
                     'source': None, 'result': None if self.restarted else 'ready'},
                    {'name': 'Connect', 'state': 'locked' if self.restarted else 'complete',
                     'source': None, 'result': None if self.restarted else 'kernel'},
                    {'name': 'Unnamed', 'cell_id': 'cell-1', 'state': 'locked' if self.restarted else 'complete',
                     'source': self.source, 'result': None if self.restarted else 'old result'},
                    {'name': 'Final', 'state': 'locked' if self.restarted else 'pending',
                     'source': None, 'result': None}],
                    'pending_index': 0 if self.restarted else 3,
                    'viewed_index': 0 if self.restarted else 2,
                    'busy': False, 'failed': False, 'status': 'ready',
                    'kernel_id': None, 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#editor').text = 'print("café")\nprint(2)'
            await pilot.pause()
            self.assertIn('UNSAVED', app.query_one('#source-label').content)
            self.assertIn('STALE', app.query_one('#result-label').content)
            app._render()
            self.assertEqual(app.query_one('#editor').text, 'print("café")\nprint(2)')
            await pilot.press('ctrl+r')
            await pilot.pause()
            self.assertEqual([call[0] for call in worker.calls[-2:]], ['set_source', 'restart'])
            self.assertEqual(worker.source, 'print("café")\nprint(2)')
            self.assertNotIn('old result', ''.join(line.text for line in app.query_one('#result').lines))

    async def test_unknown_source_save_blocks_retries_but_allows_cleanup(self):
        class Worker:
            def __init__(self):
                self.commands = []

            def request(self, command, args=None):
                self.commands.append(command)
                if command == 'set_source':
                    raise OutcomeUnknown('source save outcome unknown')
                return {'steps': [
                    {'name': 'Cell', 'cell_id': 'cell-1', 'state': 'pending',
                     'source': 'print(1)\n', 'result': None}],
                    'pending_index': 0, 'viewed_index': 0, 'busy': False,
                    'failed': False, 'status': 'ready', 'kernel_id': 'owned-kernel',
                    'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#editor').text = 'print(2)'
            await pilot.pause()
            await pilot.press('f10')
            await pilot.pause()
            await pilot.press('f10', 'ctrl+r')
            await pilot.pause()
            self.assertEqual(worker.commands.count('set_source'), 1)
            self.assertNotIn('next', worker.commands)
            self.assertNotIn('restart', worker.commands)
            self.assertIn('owned-kernel', app.query_one('#status').content)
            await pilot.press('ctrl+q')
            await pilot.pause()
            self.assertEqual(worker.commands.count('set_source'), 1)
            self.assertIn('quit', worker.commands)

    async def test_edit_flushes_before_navigation_and_quit(self):
        class Worker:
            def __init__(self):
                self.calls = []
                self.source = 'print(1)\n'

            def request(self, command, args=None):
                self.calls.append((command, args))
                if command == 'set_source':
                    self.source = args['source']
                return {'steps': [
                    {'name': 'Setup', 'state': 'complete', 'source': None, 'result': 'ready'},
                    {'name': 'Connect', 'state': 'complete', 'source': None, 'result': 'kernel'},
                    {'name': 'Cell', 'cell_id': 'cell-1', 'state': 'complete',
                     'source': self.source, 'result': 'old', 'stale': self.source != 'print(1)\n'}],
                    'pending_index': 3,
                    'viewed_index': args['index'] if command == 'previous' else 2,
                    'busy': False, 'failed': False, 'status': 'ready',
                    'kernel_id': 'owned', 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#editor').text = 'print(2)'
            await pilot.pause()
            await pilot.press('f9')
            await pilot.pause()
            self.assertEqual([name for name, _ in worker.calls[-2:]],
                             ['set_source', 'previous'])
            self.assertEqual(worker.calls[-2][1]['cell_id'], 'cell-1')
            app.query_one('#steps').index = 2
            await pilot.pause()
            self.assertIn('STALE', app.query_one('#result-label').content)
            app.query_one('#editor').text = 'print(3)'
            await pilot.pause()
            await pilot.press('ctrl+q')
            await pilot.pause()
            self.assertEqual([name for name, _ in worker.calls[-2:]],
                             ['set_source', 'quit'])
            self.assertEqual(worker.source, 'print(3)')

    async def test_one_cell_snapshot_shows_five_steps_and_skipped_count(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [
                    {'name': name, 'state': 'pending' if i == 0 else 'locked',
                     'source': 'print(7)\n' if i == 2 else None, 'result': None}
                    for i, name in enumerate(('Setup', 'Connect', 'Cell', 'Final', 'Cleanup'))],
                    'pending_index': 0, 'viewed_index': 0, 'busy': False,
                    'failed': False, 'status': 'ready', 'skipped_count': 1,
                    'kernel_id': None, 'run_file': '/tmp/run.org'}

        app = ScenarioApp(Worker())
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn('05', app.query_one('#rail-title').content)
            self.assertIn('1 non-Python', app.query_one('#status').content)

    async def test_f10_runs_one_step_and_displays_result(self):
        class Worker:
            def __init__(self):
                self.commands = []
                self.index = 0

            def request(self, command, args=None):
                self.commands.append(command)
                if command == 'next':
                    self.index += 1
                return {
                    'steps': [{'name': 'Setup',
                               'state': 'complete' if self.index else 'pending',
                               'source': None,
                               'result': 'rootfs=demo' if self.index else None},
                              {'name': 'Connect', 'state': 'pending' if self.index else 'locked',
                               'source': None, 'result': None}],
                    'pending_index': self.index,
                    'viewed_index': 0 if command == 'previous' else self.index,
                    'busy': False, 'failed': False, 'status': 'ready',
                    'kernel_id': None, 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('f10')
            await pilot.pause()
            self.assertEqual(worker.commands.count('next'), 1)
            self.assertIn('rootfs=demo', app.query_one('#result').lines[1].text)
            await pilot.press('f9')
            await pilot.pause()
            self.assertIn('rootfs=demo', app.query_one('#result').lines[0].text)

    async def test_edit_is_saved_before_cell_execution(self):
        class Worker:
            def __init__(self):
                self.commands = []
                self.source = 'print(1)\n'

            def request(self, command, args=None):
                self.commands.append(command)
                if command == 'set_source':
                    self.source = args['source']
                executed = command == 'next'
                return {'steps': [
                    {'name': 'Setup', 'state': 'complete', 'source': None, 'result': 'ready'},
                    {'name': 'Connect', 'state': 'complete', 'source': None, 'result': 'kernel-id=k'},
                    {'name': 'Context cell', 'state': 'complete' if executed else 'pending',
                     'cell_id': 'cell-1', 'block_name': 'context', 'source': self.source,
                     'result': self.source if executed else None}],
                    'pending_index': 3 if executed else 2,
                    'viewed_index': 2, 'busy': False, 'failed': False,
                    'status': 'ready', 'kernel_id': 'k', 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#editor').text = 'print(42)'
            await pilot.pause()
            await pilot.press('f10')
            await pilot.pause()
            self.assertEqual(worker.commands[-2:], ['set_source', 'next'])
            self.assertEqual(worker.source, 'print(42)')

    async def test_unknown_next_outcome_blocks_a_second_f10(self):
        class Worker:
            def __init__(self):
                self.commands = []

            def request(self, command, args=None):
                self.commands.append(command)
                if command == 'next':
                    raise OutcomeUnknown('outcome unknown')
                return {'steps': [{'name': 'Setup', 'state': 'pending',
                                   'source': None, 'result': None}],
                        'pending_index': 0, 'viewed_index': 0,
                        'busy': False, 'failed': False, 'status': 'ready',
                        'kernel_id': None, 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('f10')
            await pilot.pause()
            await pilot.press('f10')
            await pilot.pause()
            self.assertEqual(worker.commands.count('next'), 1)
            self.assertIn('outcome unknown', app.query_one('#status').content)

    async def test_two_f10_presses_while_busy_send_one_next(self):
        gate = threading.Event()

        class Worker:
            def __init__(self):
                self.commands = []

            def request(self, command, args=None):
                self.commands.append(command)
                if command == 'next':
                    gate.wait(2)
                return {'steps': [{'name': 'Setup',
                                   'state': 'complete' if command == 'next' else 'pending',
                                   'source': None, 'result': 'done' if command == 'next' else None}],
                        'pending_index': 1 if command == 'next' else 0,
                        'viewed_index': 0, 'busy': False, 'failed': False,
                        'status': 'ready', 'kernel_id': None, 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('f10', 'f10')
            self.assertEqual(worker.commands.count('next'), 1)
            gate.set()
            await pilot.pause()

    async def test_step_text_is_visible(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Setup', 'text': 'Inspect the server',
                                   'state': 'pending', 'source': None, 'result': None}],
                        'pending_index': 0, 'viewed_index': 0,
                        'busy': False, 'failed': False, 'status': 'ready',
                        'kernel_id': None, 'run_file': '/tmp/run.org'}

        app = ScenarioApp(Worker())
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn('Inspect the server', app.query_one('#description').content)

    async def test_named_cell_shows_heading_and_prose_context(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [
                    {'name': 'Setup', 'state': 'complete', 'source': None, 'result': 'ok'},
                    {'name': 'Connect', 'state': 'complete', 'source': None, 'result': 'k'},
                    {'name': 'named', 'heading': 'Lesson / Deep', 'text': 'Why this runs.',
                     'cell_id': 'cell-1', 'source': 'print(1)\n', 'state': 'pending', 'result': None},
                    {'name': 'Final', 'state': 'locked', 'source': None, 'result': None},
                    {'name': 'Cleanup', 'state': 'locked', 'source': None, 'result': None}],
                    'pending_index': 2, 'viewed_index': 2, 'busy': False,
                    'failed': False, 'status': 'ready', 'kernel_id': 'k',
                    'run_file': '/tmp/run.org'}

        app = ScenarioApp(Worker())
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn('Lesson / Deep', app.query_one('#title').content)
            self.assertIn('Why this runs.', app.query_one('#description').content)
            self.assertIn('05', app.query_one('#rail-title').content)

    async def test_f9_moves_back_from_selected_completed_step(self):
        class Worker:
            def __init__(self):
                self.previous_indices = []

            def request(self, command, args=None):
                if command == 'previous':
                    self.previous_indices.append(args['index'])
                return {'steps': [{'name': f'Step {i}', 'state': 'complete',
                                   'source': None, 'result': str(i)} for i in range(5)] +
                                 [{'name': 'Pending', 'state': 'pending',
                                   'source': None, 'result': None}],
                        'pending_index': 5,
                        'viewed_index': args['index'] if command == 'previous' else 5,
                        'busy': False, 'failed': False, 'status': 'ready',
                        'kernel_id': None, 'run_file': '/tmp/run.org'}

        worker = Worker()
        app = ScenarioApp(worker)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#steps').index = 1
            await pilot.pause()
            await pilot.press('f9')
            await pilot.pause()
            self.assertEqual(worker.previous_indices, [1, 0])

    async def test_provenance_and_both_process_logs_are_visible(self):
        class Worker:
            def request(self, command, args=None):
                return {'steps': [{'name': 'Connect', 'state': 'complete',
                                   'source': None, 'result': 'kernel-id=kernel-abc'}],
                        'pending_index': 1, 'viewed_index': 0,
                        'busy': False, 'failed': False, 'status': 'connected',
                        'emacs_pid': 7, 'kernel_id': 'kernel-abc',
                        'run_file': '/tmp/run.org'}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'worker.log').write_text('Emacs launched kernel\n')
            (root / 'server.log').write_text(
                'Kernel started: kernel-abc\nOpen /?token=private-token\n')
            (root / 'token').write_text('private-token')
            app = ScenarioApp(Worker(), worker_log=root / 'worker.log',
                              server_log=root / 'server.log',
                              token_path=root / 'token')
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                app._refresh_logs()
                await pilot.pause()
                self.assertIn('Emacs insula PID 7', app.query_one('#provenance').content)
                self.assertIn('kernel-abc', app.query_one('#provenance').content)
                self.assertIn('Emacs launched kernel',
                              ''.join(line.text for line in app.query_one('#emacs-log').lines))
                server_lines = ''.join(line.text for line in app.query_one('#server-log').lines)
                self.assertIn('Kernel started: kernel-abc', server_lines)
                self.assertNotIn('private-token', server_lines)


if __name__ == '__main__':
    unittest.main()
