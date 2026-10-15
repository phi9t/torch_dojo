#!/usr/bin/env python3
"""Test linear workout reevaluation against an owned GPU kernel."""
import json
from pathlib import Path
from protocol_probe import Probe, ROOT

OUT = ROOT / 'target/insula/research/once'
p = Probe()
OUT.mkdir(parents=True, exist_ok=True)
try:
    versions = p.execute("import sys, torch, json\nprint(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))))")
    p.record('versions', versions)
    p.execute('''import sys, socket, socketserver, threading
sys.path.insert(0, '/workspace/torch_dojo/experiments/persistent-session')
from once_lab import once, invalidate
counts = dict(model=0, data=0, service=0, closed=0, attempts=0)
class Echo(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.sendall(self.request.recv(64))
def start_service():
    counts['service'] += 1
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
def stop_service(pair):
    pair[0].shutdown()
    pair[0].server_close()
    pair[1].join(timeout=2)
    assert not pair[1].is_alive()
    counts['closed'] += 1
def model_setup():
    counts['model'] += 1
    return torch.nn.Linear(16, 8).cuda()
def data_setup():
    counts['data'] += 1
    return torch.randn(4, 16, device='cuda')
''')
    # Exactly this text is submitted twice, then edited and explicitly submitted.
    workout = '''from once_lab import once, invalidate
model = once('model', model_setup)
x = once('data', data_setup)
service = once('service', start_service, healthy=lambda p: p[1].is_alive(), close=stop_service)
with socket.create_connection(service[0].server_address, timeout=2) as conn:
    conn.sendall(b'workout')
    assert conn.recv(64) == b'workout'
y = model(x) + 1
assert y.shape == (4, 8) and y.is_cuda
'''
    (OUT / 'workout.py').write_text(workout)
    p.execute(workout)
    p.execute("identity = (id(model), id(x), x.data_ptr(), service[0].server_address)\nfirst_y = y.detach().clone()")
    p.execute(workout)
    p.record('repeat_buffer', p.execute("assert counts == dict(model=1, data=1, service=1, closed=0, attempts=0)\nassert identity == (id(model), id(x), x.data_ptr(), service[0].server_address)\nassert torch.equal(y, first_y)\nprint(counts, y.device)"))
    p.execute(workout.replace('model(x) + 1', 'model(x) + 2'))
    p.record('edited_experiment_reuses_fixture', p.execute("assert torch.allclose(y, first_y + 1)\nassert counts['model'] == counts['data'] == counts['service'] == 1\nprint('changed result; same CUDA allocations and live service')"))
    p.drop_transport()
    p.connect()
    p.record('reconnect_cache', p.execute("assert identity == (id(model), id(x), x.data_ptr(), service[0].server_address)\nassert once('model', lambda: 1) is model\nprint('same objects after transport loss')"))
    p.record('none_and_failed_setup', p.execute('''assert once('none', lambda: None) is None
assert once('none', lambda: 42) is None
def flaky():
    counts['attempts'] += 1
    if counts['attempts'] == 1:
        raise ValueError('setup failed')
    return 42
try:
    once('flaky', flaky)
except ValueError:
    pass
else:
    raise AssertionError('failure was hidden')
assert once('flaky', flaky) == once('flaky', lambda: 99) == 42
assert counts['attempts'] == 2
print('None cached; failure visible; explicit retry succeeds')'''))
    p.record('changed_factory_is_not_invalidation', p.execute("assert once('model', lambda: 'new model') is model\nprint('stable key returns old object even with a changed factory')"))
    p.record('failed_side_effect_is_not_exactly_once', p.execute('''side_effects = 0
def partial_setup():
    global side_effects
    side_effects += 1
    raise ValueError('failed after side effect')
for _ in range(2):
    try:
        once('partial', partial_setup)
    except ValueError:
        pass
assert side_effects == 2
print('explicit retries repeated a side effect; cache is not transactional')'''))
    p.record('health_and_explicit_invalidation', p.execute('''service[0].shutdown()
service[1].join(timeout=2)
try:
    once('service', start_service, healthy=lambda p: p[1].is_alive())
except RuntimeError:
    pass
else:
    raise AssertionError('dead service was silently reused')
assert counts['service'] == 1
invalidate('service')
service = once('service', start_service, close=stop_service)
assert counts['service'] == 2 and counts['closed'] == 1
invalidate('service')
assert counts['closed'] == 2
invalidate('model')
assert once('model', model_setup) is not model
assert counts['model'] == 2
print(counts)'''))
    p.record('cleanup_failure_remains_reachable', p.execute('''sentinel = object()
def bad_close(value):
    raise RuntimeError('cleanup failed')
assert once('bad-close', lambda: sentinel, close=bad_close) is sentinel
try:
    invalidate('bad-close')
except RuntimeError:
    pass
else:
    raise AssertionError('cleanup failure hidden')
assert once('bad-close', lambda: object()) is sentinel
print('retained handle for explicit recovery')'''))
    old = p.reply(p.send('kernel_info_request', {}), 'kernel_info_reply')['header']['session']
    p.api('POST', '/api/kernels/' + p.kernel + '/restart', {})
    p.ws.close()
    p.connect()
    new = p.reply(p.send('kernel_info_request', {}), 'kernel_info_reply')['header']['session']
    assert new != old
    p.record('restart_loses_cache', p.execute("import sys\nassert 'once_lab' not in sys.modules and 'model' not in globals()\nprint('same kernel ID, new generation, no session cache')"))
finally:
    (OUT / 'results.json').write_text(json.dumps(p.results, indent=2))
    (OUT / 'messages.json').write_text(json.dumps(p.events, indent=2))
    p.close()
