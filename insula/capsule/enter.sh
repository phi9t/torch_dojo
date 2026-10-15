#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
usage() { echo 'usage: enter.sh {plan|run} --role {jupyter|emacs} [--rootfs PATH] -- COMMAND' >&2; exit 2; }
[[ $# -gt 0 ]] || usage
mode=$1; shift
[[ $mode == plan || $mode == run ]] || usage
role=""
rootfs=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) role=${2:?}; shift 2;;
    --rootfs) rootfs=${2:?}; shift 2;;
    --) shift; break;;
    *) usage;;
  esac
done
[[ $role == jupyter || $role == emacs ]] || usage
[[ $# -gt 0 ]] || usage
if [[ -z $rootfs ]]; then
  rootfs=$(python3 "$repo/insula/rootfs/select.py")
fi
rootfs=$(readlink -f -- "$rootfs")
case "$rootfs" in "$repo"/target/insula/rootfs/content/*) ;; *) echo 'rootfs must be selected under target/insula/rootfs/content' >&2; exit 1;; esac
python3 "$repo/insula/rootfs/select.py" --validate "$rootfs" >/dev/null
if [[ $mode == run ]]; then
  selected=$(python3 "$repo/insula/rootfs/select.py")
  [[ $rootfs == "$selected" ]] || { echo "run requires selected rootfs: $selected" >&2; exit 1; }
fi
state="$repo/target/insula/state/$role"
mkdir -p "$state/home" "$state/run" "$state/tmp" "$state/etc"
if [[ $mode == run ]]; then
  if [[ $(cat "$state/etc/.torch-dojo-rootfs" 2>/dev/null || true) != "$rootfs" ]]; then
    rm -rf "${state:?}/etc"
    mkdir -p "$state/etc"
    cp -a --no-preserve=mode,ownership "$rootfs/etc/." "$state/etc/"
    printf "%s\n" "$rootfs" > "$state/etc/.torch-dojo-rootfs"
  fi
  cp -L /etc/resolv.conf "$state/etc/resolv.conf"
  cp -L /etc/hosts "$state/etc/hosts"
  if ! grep -q "^dojo:" "$state/etc/passwd"; then
    printf "dojo:x:%s:%s:Torch Dojo:/home/dojo:/bin/bash\n" "$(id -u)" "$(id -g)" >> "$state/etc/passwd"
    printf "dojo:x:%s:\n" "$(id -g)" >> "$state/etc/group"
  fi
  if [[ $role == emacs && $(cat "$state/home/.emacs.d/.torch-dojo-ready" 2>/dev/null || true) != "$rootfs" ]]; then
    rm -rf "$state/home/.emacs.d.new"
    mkdir -p "$state/home/.emacs.d.new"
    cp -a --no-preserve=mode,ownership "$rootfs/opt/dojo/doom/." "$state/home/.emacs.d.new/"
    printf "%s\n" "$rootfs" > "$state/home/.emacs.d.new/.torch-dojo-ready"
    rm -rf "$state/home/.emacs.d"
    mv "$state/home/.emacs.d.new" "$state/home/.emacs.d"
  fi
  # Doom's image-generated startup files abbreviate /opt/dojo/doom as
  # ~/doom because the build HOME was /opt/dojo. Preserve that source path
  # when runtime HOME moves to /home/dojo; local state stays in .emacs.d.
  if [[ $role == emacs && ! -e "$state/home/doom" && ! -L "$state/home/doom" ]]; then
    ln -s /opt/dojo/doom "$state/home/doom"
  fi
fi
args=(bwrap --die-with-parent --unshare-user --uid "$(id -u)" --gid "$(id -g)" --unshare-ipc --unshare-pid --unshare-uts --unshare-cgroup-try --share-net --clearenv --ro-bind "$rootfs" / --proc /proc --dev /dev --ro-bind-try /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /workspace --tmpfs /home --bind "$repo" /workspace/torch_dojo --bind "$state/home" /home/dojo --bind "$state/run" /run/user --bind "$state/tmp" /tmp --ro-bind "$state/etc" /etc --setenv HOME /home/dojo --setenv USER dojo --setenv PATH /opt/dojo/doom/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin --setenv XDG_RUNTIME_DIR /run/user --setenv XDG_CACHE_HOME /home/dojo/.cache --setenv JUPYTER_RUNTIME_DIR /run/user/jupyter --setenv DOOMDIR /opt/dojo/doom-config --setenv DOOMLOCALDIR /home/dojo/.emacs.d/.local --setenv PYTHONNOUSERSITE 1 --setenv PYTHONPATH /workspace/torch_dojo --setenv TERM "${TERM:-xterm-256color}" --setenv LANG "${LANG:-C.UTF-8}")
for name in DOJO_VERIFY_TOKEN DOJO_VERIFY_PORT; do
  [[ -z ${!name+x} ]] || args+=(--setenv "$name" "${!name}")
done
if [[ $role == emacs && -n ${DOJO_SCENARIO_DIR:-} ]]; then
  args+=(--setenv DOJO_SCENARIO_DIR "$DOJO_SCENARIO_DIR")
fi
if [[ $role == jupyter ]]; then
  args+=(--dir /run/nvidia-driver --setenv LD_LIBRARY_PATH /run/nvidia-driver --setenv CUDA_HOME /usr/local/cuda)
  shopt -s nullglob
  for dev in /dev/nvidia* /dev/nvidia-caps; do args+=(--dev-bind "$dev" "$dev"); done
  for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia-*.so*; do
    [[ -e $lib ]] && args+=(--ro-bind "$lib" "/run/nvidia-driver/$(basename "$lib")")
  done
  shopt -u nullglob
  [[ ! -x /usr/bin/nvidia-smi ]] || args+=(--ro-bind /usr/bin/nvidia-smi /run/nvidia-driver/nvidia-smi)
  [[ -z ${CUDA_VISIBLE_DEVICES+x} ]] || args+=(--setenv CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES")
fi
args+=(--chdir /workspace/torch_dojo -- "$@")
if [[ $mode == plan ]]; then
  python3 - "$repo" "$rootfs" "$state" "$role" "${args[@]}" <<'PY'
import json,sys
repo,rootfs,state,role,*argv=sys.argv[1:]
print(json.dumps(dict(schema_version=1,repo=repo,rootfs=rootfs,state=state,role=role,network='shared-loopback',argv=argv)))
PY
else
  exec "${args[@]}"
fi
