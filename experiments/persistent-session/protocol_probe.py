#!/usr/bin/env python3
"""Disposable Jupyter lifecycle/debugger probe; never resubmits execution.

Run inside the jupyter insula. Owns and deletes only its newly created kernel.
"""
import datetime
import json
import pathlib
import socket
import time
import urllib.request
import uuid

import websocket

ROOT = pathlib.Path('/workspace/torch_dojo')
OUT = ROOT / 'target/insula/research/protocol'
URL = 'http://127.0.0.1:8899'


class Probe:
    def __init__(self):
        self.token = (ROOT / 'target/insula/shared/token').read_text().strip()
        self.client_session = uuid.uuid4().hex
        self.events = []
        self.results = []
        self.seq = 0
        self.kernel = self.api('POST', '/api/kernels', {'name': 'torch-dojo-gpu'})['id']
        self.ws = None
        self.connect()

    def api(self, method, path, body=None):
        request = urllib.request.Request(URL + path, method=method,
            headers={'Authorization': 'token ' + self.token, 'Content-Type': 'application/json'},
            data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
            return json.loads(data) if data else None

    def connect(self):
        self.ws = websocket.create_connection(
            URL.replace('http:', 'ws:') + '/api/kernels/' + self.kernel + '/channels?session_id=' + self.client_session,
            header={'Authorization': 'token ' + self.token}, suppress_origin=True, timeout=30)

    def send(self, kind, content, channel='shell'):
        ident = uuid.uuid4().hex
        self.ws.send(json.dumps({'header': {'msg_id': ident, 'username': 'research',
            'session': self.client_session, 'msg_type': kind, 'version': '5.3',
            'date': datetime.datetime.now(datetime.timezone.utc).isoformat()},
            'parent_header': {}, 'metadata': {}, 'content': content, 'channel': channel, 'buffers': []}))
        return ident

    def drop_transport(self):
        """Lose TCP without a WebSocket close frame or debugger detach request."""
        self.ws.sock.shutdown(socket.SHUT_RDWR)
        self.ws.sock.close()

    def receive(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ws.settimeout(max(.01, deadline - time.monotonic()))
            msg = json.loads(self.ws.recv())
            self.events.append(msg)
            if predicate(msg):
                return msg
        raise TimeoutError('expected message not observed')

    def reply(self, ident, kind):
        return self.receive(lambda m: m['header']['msg_type'] == kind and m.get('parent_header', {}).get('msg_id') == ident)

    def start(self, code):
        return self.send('execute_request', {'code': code, 'silent': False, 'store_history': True,
            'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True})

    def finish(self, ident):
        # Collect both reply and idle, without assuming cross-channel ordering.
        def done(_):
            related = [m for m in self.events if m.get('parent_header', {}).get('msg_id') == ident]
            return (any(m['header']['msg_type'] == 'execute_reply' for m in related)
                    and any(m['header']['msg_type'] == 'status' and m['content']['execution_state'] == 'idle' for m in related))
        if not done(None):
            self.receive(done, timeout=90)
        messages = [m for m in self.events if m.get('parent_header', {}).get('msg_id') == ident]
        reply = next(m for m in messages if m['header']['msg_type'] == 'execute_reply')
        assert reply['content']['status'] == 'ok', reply['content']
        return ''.join(m['content'].get('text', '') + m['content'].get('data', {}).get('text/plain', '') for m in messages)

    def execute(self, code):
        return self.finish(self.start(code))

    def debug(self, command, arguments=None):
        self.seq += 1
        ident = self.send('debug_request', {'seq': self.seq, 'type': 'request',
            'command': command, 'arguments': arguments or {}}, 'control')
        result = self.reply(ident, 'debug_reply')['content']
        assert result.get('success'), result
        return result.get('body', {})

    def record(self, name, evidence):
        self.results.append({'name': name, 'evidence': evidence})
        print(name, json.dumps(evidence), flush=True)

    def stopped(self):
        return self.receive(lambda m: m['header']['msg_type'] == 'debug_event' and m['content'].get('event') == 'stopped')['content']['body']

    def run(self):
        info = self.reply(self.send('kernel_info_request', {}), 'kernel_info_reply')
        generation = info['header']['session']
        self.record('kernel_info', {'kernel_id': self.kernel, 'generation': generation,
            'advertised_debugger': info['content'].get('debugger'), 'debug_info': self.debug('debugInfo')})
        setup = self.execute("import torch, time, json, sys, ipykernel, jupyter_server, debugpy\nmodel = torch.nn.Linear(8, 8).cuda()\nx = torch.randn(8, 8, device='cuda')\nidentity = (id(model), id(x), x.data_ptr())\nprint(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, ipykernel=ipykernel.__version__, server=jupyter_server.__version__, debugpy=debugpy.__version__, gpu=torch.cuda.get_device_name(0), identity=identity)))")
        self.ws.close()
        self.connect()
        retained = self.execute("assert identity == (id(model), id(x), x.data_ptr())\nprint(model(x).shape, x.device)")
        self.record('idle_reconnect', {'setup': setup, 'retained': retained})

        ident = self.start("execution_count_probe = globals().get('execution_count_probe', 0) + 1\nprint('before-disconnect', flush=True)\ntime.sleep(3)\nprint('offline-output', execution_count_probe, flush=True)")
        self.receive(lambda m: m.get('parent_header', {}).get('msg_id') == ident and 'before-disconnect' in m['content'].get('text', ''))
        self.drop_transport()
        time.sleep(4)
        self.connect()
        text = self.finish(ident)
        assert 'offline-output 1' in text, text
        self.record('busy_reconnect_same_session', {'output': text, 'verification': self.execute('assert execution_count_probe == 1\nassert identity == (id(model), id(x), x.data_ptr())')})

        # Deliberately use another client session: old buffered output is not an execution journal.
        lost = self.start("gap_probe = globals().get('gap_probe', 0) + 1\nprint('gap-start', flush=True)\ntime.sleep(2)\nprint('gap-offline-output', flush=True)")
        self.receive(lambda m: m.get('parent_header', {}).get('msg_id') == lost and 'gap-start' in m['content'].get('text', ''))
        self.ws.close()
        time.sleep(3)
        self.client_session = uuid.uuid4().hex
        self.connect()
        check = self.execute('assert gap_probe == 1\nprint("state-survived-new-client")')
        recovered = any(m.get('parent_header', {}).get('msg_id') == lost and 'gap-offline-output' in m['content'].get('text', '') for m in self.events)
        assert not recovered
        self.record('new_client_output_gap', {'state': check, 'offline_output_recovered': recovered, 'original_request_outcome': 'unknown until explicit state inspection'})

        self.debug('initialize', {'clientID': 'torch-dojo-research', 'adapterID': 'python', 'pathFormat': 'path', 'linesStartAt1': True, 'columnsStartAt1': True, 'supportsVariableType': True})
        self.debug('attach', {'justMyCode': False})
        code = 'def research_step(x):\n    local_tensor = x + 2\n    result = local_tensor.sum()\n    return result\nresearch_result = research_step(x)'
        path = self.debug('dumpCell', {'code': code})['sourcePath']
        breaks = self.debug('setBreakpoints', {'source': {'path': path}, 'breakpoints': [{'line': 3}]})
        ident = self.start(code)
        stop = self.stopped()
        thread = stop['threadId']
        frames = self.debug('stackTrace', {'threadId': thread})['stackFrames']
        frame = frames[0]
        assert frame['line'] == 3, frame
        inspected = self.debug('evaluate', {'expression': '(tuple(local_tensor.shape), str(local_tensor.device), bool(torch.equal(local_tensor, x + 2)))', 'frameId': frame['id'], 'context': 'repl'})
        assert inspected['result'] == "((8, 8), 'cuda:0', True)", inspected
        scopes = self.debug('scopes', {'frameId': frame['id']})['scopes']
        local_scope = next(s for s in scopes if s['name'] == 'Locals')
        variables = self.debug('variables', {'variablesReference': local_scope['variablesReference']})['variables']
        assert any(v['name'] == 'local_tensor' for v in variables), variables
        self.record('breakpoint_frame_evaluation', {'source': path, 'breakpoints': breaks,
            'frame': frame, 'evaluation': inspected, 'local_names': [v['name'] for v in variables]})
        self.drop_transport()
        self.connect()
        debug_info = self.debug('debugInfo')
        assert thread in debug_info['stoppedThreads'], debug_info
        new_frame = self.debug('stackTrace', {'threadId': thread})['stackFrames'][0]
        value = self.debug('evaluate', {'expression': 'local_tensor.shape', 'frameId': new_frame['id'], 'context': 'repl'})
        self.record('paused_reconnect', {'debug_info': debug_info, 'evaluation': value})
        self.debug('next', {'threadId': thread})
        stepped = self.stopped()
        stepped_frame = self.debug('stackTrace', {'threadId': stepped['threadId']})['stackFrames'][0]
        assert stepped_frame['line'] == 4, stepped_frame
        self.debug('continue', {'threadId': thread})
        self.finish(ident)
        self.record('step_continue', {'stepped_line': stepped_frame['line'], 'retained': self.execute('assert identity == (id(model), id(x), x.data_ptr())\nprint(research_result.item())')})

        edited = code.replace('x + 2', 'x + 3').replace('    result =', '    # edited source shifts the breakpoint\n    result =')
        new_path = self.debug('dumpCell', {'code': edited})['sourcePath']
        assert new_path != path
        self.debug('setBreakpoints', {'source': {'path': new_path}, 'breakpoints': [{'line': 4}]})
        ident = self.start(edited)
        stop = self.stopped()
        frame = self.debug('stackTrace', {'threadId': stop['threadId']})['stackFrames'][0]
        assert frame['source']['path'] == new_path
        assert frame['line'] == 4, frame
        inspected = self.debug('evaluate', {'expression': 'bool(torch.equal(local_tensor, x + 3))', 'frameId': frame['id'], 'context': 'repl'})
        assert inspected['result'] == 'True', inspected
        self.debug('continue', {'threadId': stop['threadId']})
        self.finish(ident)
        self.record('edited_source_mapping', {'old_path': path, 'new_path': new_path, 'new_line': frame['line'], 'evaluation': inspected})

        self.api('POST', '/api/kernels/' + self.kernel + '/restart', {})
        self.ws.close()
        self.connect()
        restarted = self.reply(self.send('kernel_info_request', {}), 'kernel_info_reply')
        assert restarted['header']['session'] != generation
        absent = self.execute("assert 'model' not in globals() and 'x' not in globals()\nprint('namespace-cleared')")
        self.record('restart_detection', {'same_kernel_id': self.kernel, 'old_generation': generation, 'new_generation': restarted['header']['session'], 'namespace': absent})

    def close(self):
        if self.ws:
            self.ws.close()
        self.api('DELETE', '/api/kernels/' + self.kernel)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    probe = Probe()
    try:
        probe.run()
    finally:
        (OUT / 'results.json').write_text(json.dumps(probe.results, indent=2))
        (OUT / 'messages.json').write_text(json.dumps(probe.events, indent=2))
        probe.close()
