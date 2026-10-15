#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
sources="$repo/experiments/persistent-session"
out="$repo/target/insula/research/native"
mkdir -p "$out"
gcc -shared -fPIC -O2 "$sources/native_probe_dependency.c" -o "$out/libdojo_dependency.so"
for version in 1 2; do
  g++ -std=c++17 -shared -fPIC -O2 -DPROBE_VERSION="$version" \
    "$sources/native_probe.cpp" -L "$out" -ldojo_dependency \
    '-Wl,-rpath,$ORIGIN' -o "$out/libdojo_cpp_v$version.so"
  options=()
  [[ $version == 1 ]] || options+=(--cfg version2)
  rustc --edition=2024 --crate-type cdylib -O "${options[@]}" \
    "$sources/native_probe.rs" -L "$out" -l dylib=dojo_dependency \
    '-C' 'link-arg=-Wl,-rpath,$ORIGIN' -o "$out/libdojo_rust_v$version.so"
done
python3 - "$out/build.json" <<'PY'
import json, subprocess, sys
versions = {name: subprocess.check_output([name, '--version'], text=True).splitlines()[0]
            for name in ['gcc', 'g++', 'rustc']}
open(sys.argv[1], 'w').write(json.dumps(versions, indent=2) + '\n')
PY
exec timeout 120 "$repo/insula/capsule/enter.sh" run --role jupyter -- \
  python /workspace/torch_dojo/experiments/persistent-session/native_probe.py
