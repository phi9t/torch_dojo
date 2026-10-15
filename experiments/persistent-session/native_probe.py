#!/usr/bin/env python3
"""Owned Jupyter kernel: retained C++/Rust handles beside a retained CUDA model.

C ABI/ctypes only; deliberately does not claim pybind11, PyO3 or native CUDA.
"""
import json
import pathlib
from protocol_probe import Probe

OUT = pathlib.Path('/workspace/torch_dojo/target/insula/research/native')

SETUP = '''
import ctypes, json, platform, torch
from pathlib import Path
class Native:
    def __init__(self, language, version):
        self.lib = ctypes.CDLL(str(Path('/workspace/torch_dojo/target/insula/research/native') / f'libdojo_{language}_v{version}.so'))
        self.lib.fixture_new.argtypes = []
        self.lib.fixture_new.restype = ctypes.c_void_p
        self.lib.fixture_delete.argtypes = [ctypes.c_void_p]
        self.lib.fixture_delete.restype = None
        self.lib.fixture_calls.argtypes = [ctypes.c_void_p]
        self.lib.fixture_calls.restype = ctypes.c_uint64
        self.lib.fixture_version.argtypes = []
        self.lib.fixture_version.restype = ctypes.c_int
        self.lib.fixture_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.lib.fixture_apply.restype = None
        self.handle = self.lib.fixture_new()
        assert self.handle
    def apply(self, tensor):
        assert self.handle, 'fixture was closed'
        assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and tensor.is_contiguous()
        self.lib.fixture_apply(self.handle, ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)), tensor.numel())
    def calls(self):
        assert self.handle
        return self.lib.fixture_calls(self.handle)
    def close(self):
        if self.handle:
            self.lib.fixture_delete(self.handle)
            self.handle = None
cpp, rust = Native('cpp', 1), Native('rust', 1)
model = torch.nn.Linear(4, 4).cuda()
gpu_x = torch.ones(4, 4, device='cuda')
cpu_x = torch.zeros(4, 4)
identity = (id(model), gpu_x.data_ptr(), cpu_x.data_ptr(), cpp.handle, rust.handle)
print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), identity=identity)))
'''

WORKOUT = '''
cpp.apply(cpu_x)
rust.apply(cpu_x)
gpu_result = model(cpu_x.cuda() + gpu_x)
assert identity == (id(model), gpu_x.data_ptr(), cpu_x.data_ptr(), cpp.handle, rust.handle)
assert gpu_result.device.type == 'cuda'
print(json.dumps(dict(cpp_calls=cpp.calls(), rust_calls=rust.calls(), first=cpu_x[0,0].item(), result_shape=list(gpu_result.shape))))
'''

def main():
    probe = Probe()
    try:
        probe.record('gpu_and_native_setup', {'output': probe.execute(SETUP)})
        first = probe.execute(WORKOUT)
        second = probe.execute(WORKOUT)
        probe.execute('assert cpp.calls() == rust.calls() == 2\nassert torch.equal(cpu_x, torch.full_like(cpu_x, 4))')
        probe.record('repeated_linear_workout', {'first': first, 'second': second})
        probe.drop_transport()
        probe.connect()
        third = probe.execute(WORKOUT)
        probe.execute('assert cpp.calls() == rust.calls() == 3\nassert torch.equal(cpu_x, torch.full_like(cpu_x, 6))')
        probe.record('native_handles_survive_frontend_reconnect', {'output': third})
        versioning = probe.execute('''
cpp2, rust2 = Native('cpp', 2), Native('rust', 2)
assert cpp.lib.fixture_version() == rust.lib.fixture_version() == 1
assert cpp2.lib.fixture_version() == rust2.lib.fixture_version() == 2
cpp2.apply(cpu_x)
rust2.apply(cpu_x)
assert torch.equal(cpu_x, torch.full_like(cpu_x, 10))
assert cpp.calls() == rust.calls() == 3
assert cpp2.calls() == rust2.calls() == 1
print(json.dumps(dict(old_versions=[cpp.lib.fixture_version(), rust.lib.fixture_version()], new_versions=[cpp2.lib.fixture_version(), rust2.lib.fixture_version()], first=cpu_x[0,0].item())))
''')
        probe.record('distinct_artifact_versions_coexist', {'output': versioning})
        teardown = probe.execute('''
for fixture in [cpp, rust, cpp2, rust2]:
    fixture.close()
    fixture.close()
    assert fixture.handle is None
    try:
        fixture.apply(cpu_x)
    except AssertionError as error:
        assert str(error) == 'fixture was closed'
    else:
        raise AssertionError('closed handle accepted')
assert identity[:3] == (id(model), gpu_x.data_ptr(), cpu_x.data_ptr())
print('all four handles closed; reuse rejected; GPU fixture retained')
''')
        probe.record('explicit_native_teardown', {'output': teardown})
    finally:
        if probe.ws is not None:
            probe.ws.close()
        probe.api('DELETE', '/api/kernels/' + probe.kernel)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'results.json').write_text(json.dumps(probe.results, indent=2) + '\n')
        (OUT / 'messages.json').write_text(json.dumps(probe.events, indent=2) + '\n')
    print('PASS: 5 native orchestration scenarios; owned kernel deleted')

if __name__ == '__main__':
    main()
