import json
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class InsulaPlanTest(unittest.TestCase):
    def test_two_roles_share_rootfs_and_repo_but_not_state(self):
        content = ROOT / 'target/insula/rootfs/content/test-plan'
        for name in ('bin/bash', 'usr/bin/emacs-nox', 'opt/dojo/doom/bin/doom', 'opt/dojo/doom-config/init.el'):
            path = content / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        marker = content / 'etc/torch-dojo-rootfs.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'kind': 'torch-dojo-gpu', 'image_id': 'sha256:test-plan'}))

        def plan(role):
            return json.loads(subprocess.check_output(
                [str(ROOT / 'insula/capsule/enter.sh'), 'plan', '--role', role,
                 '--rootfs', str(content), '--', '/bin/true'], text=True))

        server, emacs = plan('jupyter'), plan('emacs')
        self.assertEqual(server['rootfs'], emacs['rootfs'])
        self.assertEqual(server['repo'], emacs['repo'])
        self.assertNotEqual(server['state'], emacs['state'])
        self.assertEqual(server['network'], emacs['network'])
        self.assertEqual(server['network'], 'shared-loopback')
        for entry in (server, emacs):
            argv = entry['argv']
            self.assertEqual(argv[argv.index('--bind'):argv.index('--bind') + 3],
                             ['--bind', str(ROOT), '/workspace/torch_dojo'])
            self.assertIn('--share-net', argv)
        self.assertIn('/run/nvidia-driver', server['argv'])
        self.assertNotIn('/run/nvidia-driver', emacs['argv'])
        # The rootfs is read-only and PYTHONNOUSERSITE=1, so `pip install -e .` cannot make
        # torch_judge importable; both roles must find it from any working directory (dojo/).
        for entry in (server, emacs):
            argv = entry['argv']
            values = [argv[i + 2] for i, a in enumerate(argv) if a == '--setenv' and argv[i + 1] == 'PYTHONPATH']
            self.assertEqual(values, ['/workspace/torch_dojo'])


if __name__ == '__main__':
    unittest.main()
