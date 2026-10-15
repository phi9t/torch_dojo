#!/usr/bin/env python3
"""Return the validated Torch Dojo rootfs selected for both insulas."""
import json
import pathlib
import sys

repo = pathlib.Path(__file__).resolve().parents[2]
store = repo / 'target/insula/rootfs'

def validate(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to((store / 'content').resolve()):
        raise ValueError('rootfs is outside Torch Dojo content store')
    for name in ('bin/bash', 'usr/bin/emacs-nox', 'opt/dojo/doom/bin/doom', 'opt/dojo/doom-config/init.el'):
        if not (path / name).exists():
            raise ValueError(f'rootfs lacks {name}')
    marker = json.loads((path / 'etc/torch-dojo-rootfs.json').read_text())
    if marker.get('kind') != 'torch-dojo-gpu' or marker.get('image_id') != 'sha256:' + path.name:
        raise ValueError('rootfs marker invalid')
    return path

try:
    if len(sys.argv) == 3 and sys.argv[1] == '--validate':
        print(validate(pathlib.Path(sys.argv[2])))
    elif len(sys.argv) == 1:
        selection = json.loads((store / 'selected.json').read_text())
        if selection.get('kind') != 'torch-dojo-gpu':
            raise ValueError('selection invalid')
        print(validate(store / 'content' / selection['id']))
    else:
        raise ValueError('usage: select.py [--validate PATH]')
except (OSError, ValueError, KeyError) as exc:
    print(f'rootfs selection: {exc}; run insula/rootfs/build.sh', file=sys.stderr)
    sys.exit(1)
