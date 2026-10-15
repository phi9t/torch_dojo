"""Host-side verification of the insula's CUDA toolchain wiring and integrated CUDA checks."""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTER = ROOT / 'insula/capsule/enter.sh'
VERIFY = ROOT / 'insula/jupyter_rootfs/verify_cuda.py'


def selected_rootfs():
    result = subprocess.run([sys.executable, str(ROOT / 'insula/rootfs/select.py')], capture_output=True, text=True)
    return result.returncode == 0


def inside_gpu_insula():
    return pathlib.Path('/usr/local/cuda/bin/nvcc').exists() and os.environ.get('CUDA_HOME') == '/usr/local/cuda'


class CapsulePlanTest(unittest.TestCase):
    """The launcher decides what nvcc can see; pin that without needing a GPU."""

    @classmethod
    def setUpClass(cls):
        content = ROOT / 'target/insula/rootfs/content/test-plan-cuda'
        for name in ('bin/bash', 'usr/bin/emacs-nox', 'opt/dojo/doom/bin/doom', 'opt/dojo/doom-config/init.el'):
            (content / name).parent.mkdir(parents=True, exist_ok=True)
            (content / name).touch()
        marker = content / 'etc/torch-dojo-rootfs.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'kind': 'torch-dojo-gpu', 'image_id': 'sha256:test-plan-cuda'}))
        cls.content = content

    def plan(self, role):
        return json.loads(subprocess.check_output(
            [str(ENTER), 'plan', '--role', role, '--rootfs', str(self.content), '--', '/bin/true'], text=True))['argv']

    @staticmethod
    def setenv(argv, name):
        return next((argv[i + 2] for i, a in enumerate(argv) if a == '--setenv' and argv[i + 1] == name), None)

    def test_jupyter_role_gets_cuda_home_and_cuda_bin_first_after_doom(self):
        argv = self.plan('jupyter')
        self.assertEqual(self.setenv(argv, 'CUDA_HOME'), '/usr/local/cuda')
        path = self.setenv(argv, 'PATH').split(':')
        self.assertLess(path.index('/usr/local/cuda/bin'), path.index('/usr/local/bin'))

    def test_emacs_role_has_no_cuda_home_or_devices(self):
        argv = self.plan('emacs')
        self.assertIsNone(self.setenv(argv, 'CUDA_HOME'))
        self.assertNotIn('/run/nvidia-driver', argv)

    def test_environment_is_cleared_so_host_cuda_cannot_leak_in(self):
        self.assertIn('--clearenv', self.plan('jupyter'))

    def test_containerfile_pins_nvcc_companions_to_one_release(self):
        text = (ROOT / 'insula/rootfs/Containerfile').read_text()
        pins = {name: text.split(f"'{name}==", 1)[1].split("'", 1)[0]
                for name in ('nvidia-cuda-nvcc', 'nvidia-nvvm', 'nvidia-cuda-crt')}
        self.assertEqual(len(set(pins.values())), 1, pins)
        self.assertIn('g++', text)


@unittest.skipUnless(inside_gpu_insula() or (shutil.which('bwrap') and selected_rootfs()),
                     'needs the GPU insula (build it with insula/rootfs/build.sh)')
class IntegratedCudaChecksTest(unittest.TestCase):
    """Run insula/jupyter_rootfs/verify_cuda.py where nvcc and the GPU live."""

    def test_toolchain_compile_launch_and_interop_checks_pass(self):
        if inside_gpu_insula():
            command = [sys.executable, str(VERIFY)]
        else:
            command = [str(ENTER), 'run', '--role', 'jupyter', '--', 'python',
                       '/workspace/torch_dojo/insula/jupyter_rootfs/verify_cuda.py']
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertRegex(result.stderr, r'Ran \d+ tests')
        self.assertIn('OK', result.stderr)

    def test_verifier_goes_red_when_the_toolchain_is_broken(self):
        """Prove the checks have teeth: hide CUDA_HOME's include dir by pointing nvcc at nothing."""
        if not inside_gpu_insula():
            self.skipTest('mutation run happens inside the insula only')
        env = {**os.environ, 'PATH': '/usr/bin:/bin', 'CUDA_HOME': '/nonexistent'}
        result = subprocess.run([sys.executable, str(VERIFY)], capture_output=True, text=True, env=env, timeout=600)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAILED', result.stderr)


if __name__ == '__main__':
    unittest.main()
