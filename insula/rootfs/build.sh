#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
store="$repo/target/insula/rootfs"
base='ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime@sha256:7492928e093d67276716440161f694a0e4ea27796d599d64b23cb76cdd665e71'
tag=torch-dojo-gpu:local
mkdir -p "$store/content"
docker build --network host -f "$repo/insula/rootfs/Containerfile" --build-arg "BASE_IMAGE=$base" -t "$tag" "$repo"
image_id=$(docker image inspect "$tag" --format '{{.Id}}')
id=${image_id#sha256:}
stage=$(mktemp -d "$store/.stage.XXXXXX")
cid=
trap '[[ -z ${cid:-} ]] || docker rm -f "$cid" >/dev/null 2>&1; [[ ! -d ${stage:-} ]] || rm -rf "$stage"' EXIT
cid=$(docker create "$tag")
docker export "$cid" | tar -C "$stage" -xf -
docker rm "$cid" >/dev/null
cid=
python3 - "$stage" "$image_id" "$base" <<'PY'
import json,pathlib,sys
root,image,base=sys.argv[1:]
(pathlib.Path(root)/'etc/torch-dojo-rootfs.json').write_text(json.dumps({'kind':'torch-dojo-gpu','image_id':image,'base_image':base})+'\n')
PY
# Validation happens before selection, while the stage still lives in the content store.
target="$store/content/$id"
if [[ -d $target ]]; then rm -rf "$stage"; else mv "$stage" "$target"; fi
stage=
python3 "$repo/insula/rootfs/select.py" --validate "$target" >/dev/null
python3 - "$store/selected.json" "$id" <<'PY'
import json,os,pathlib,sys,tempfile
p=pathlib.Path(sys.argv[1]); content=json.dumps({'kind':'torch-dojo-gpu','id':sys.argv[2]})+'\n'
fd,name=tempfile.mkstemp(dir=p.parent,prefix='.selection.')
with os.fdopen(fd,'w') as out: out.write(content)
os.replace(name,p)
PY
printf 'selected rootfs: %s\n' "$target"
