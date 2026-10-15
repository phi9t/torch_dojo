"""Build the release-qualified Browser Gym JupyterLite site.

The normal JupyterLite build can load Pyodide and wheels from public CDNs. This
entrypoint creates the stricter release artifact used by ticket 16: all runtime
assets needed by the accepted exercises live under ``dist`` and the generated
configuration disables PyPI fallback.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PYODIDE_VERSION = "314.0.6"
PYODIDE_TARBALL_URL = f"https://registry.npmjs.org/pyodide/-/pyodide-{PYODIDE_VERSION}.tgz"
PYODIDE_TARBALL_INTEGRITY = (
    "sha512-BKDTJyIqFxC4BExLqeRS3f5xvXZIjOt8C3zGLN/Cc7tFxSwvKVhVkchQQ2AGLOtR4YrVOIFxbV8poyDOOmWwxQ=="
)

PYODIDE_BOOTSTRAP_PACKAGES = (
    "numpy",
    "micropip",
    "ipython",
    "jedi",
    "sortedcontainers",
)

PYODIDE_PYPI_WHEELS = (
    "hypothesis==6.168.5",
)

UNIVERSAL_PYPI_WHEELS = (
    "sortedcontainers==2.4.0",
    "comm==0.2.3",
    "ipywidgets==8.1.9",
    "jupyterlab_widgets==3.0.17",
    "widgetsnbextension==4.0.16",
)
HYPOTHESIS_WHEEL = "hypothesis-6.168.5-cp314-cp314-pyemscripten_2026_0_wasm32.whl"

CHECKER_WORKER_ONLINE_PYODIDE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide.mjs"
PYODIDE_CDN_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"
GYM_EXTENSION = "extensions/@torch-dojo/browser-gym"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def venv_bin(self) -> Path:
        return self.root / ".venv" / "bin"

    @property
    def python(self) -> Path:
        return self.venv_bin / "python"

    @property
    def jupyter(self) -> Path:
        return self.venv_bin / "jupyter"

    @property
    def dist(self) -> Path:
        return self.root / "dist"

    @property
    def cache(self) -> Path:
        return self.root / ".offline-cache"

    @property
    def pyodide_dir(self) -> Path:
        return self.root / "static" / "pyodide"

    @property
    def wheel_dir(self) -> Path:
        return self.root / "pypi"


def run(cmd: list[str], *, cwd: Path, env_path: Path | None = None) -> None:
    env = None
    if env_path is not None:
        import os

        env = os.environ.copy()
        env["PATH"] = f"{env_path}:{env['PATH']}"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_integrity(path: Path, integrity: str) -> None:
    algo, encoded = integrity.split("-", 1)
    if algo != "sha512":
        raise ValueError(f"unsupported integrity algorithm: {algo}")
    expected = base64.b64decode(encoded)
    actual = hashlib.sha512(path.read_bytes()).digest()
    if actual != expected:
        raise RuntimeError(f"{path.name} did not match pinned npm integrity")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_pyodide_packages(lock: dict) -> list[str]:
    packages = lock["packages"]
    seen: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        key = _canonical_package_name(name)
        if key in seen:
            return
        if key not in packages:
            raise RuntimeError(f"pyodide-lock.json does not contain package {name!r}")
        seen.add(key)
        ordered.append(key)
        for dep in packages[key].get("depends", []):
            visit(dep)

    for package in PYODIDE_BOOTSTRAP_PACKAGES:
        visit(package)
    return ordered


def _download_pyodide_packages(paths: Paths, lock: dict) -> None:
    for package in _required_pyodide_packages(lock):
        info = lock["packages"][package]
        filename = info.get("file_name")
        if not filename:
            continue
        target = paths.pyodide_dir / filename
        _download(f"{PYODIDE_CDN_BASE_URL}/{filename}", target)
        actual = sha256(target)
        expected = info["sha256"]
        if actual != expected:
            raise RuntimeError(f"{filename} sha256 mismatch: expected {expected}, got {actual}")


def _extract_pyodide_runtime(paths: Paths) -> dict:
    tarball = paths.cache / f"pyodide-{PYODIDE_VERSION}.tgz"
    _download(PYODIDE_TARBALL_URL, tarball)
    _verify_integrity(tarball, PYODIDE_TARBALL_INTEGRITY)

    if paths.pyodide_dir.exists():
        shutil.rmtree(paths.pyodide_dir)
    paths.pyodide_dir.mkdir(parents=True)

    with tarfile.open(tarball) as archive:
        lock = None
        required = {
            "pyodide.mjs",
            "pyodide.asm.mjs",
            "pyodide.asm.wasm",
            "python_stdlib.zip",
            "pyodide-lock.json",
        }
        for member_name in archive.getnames():
            if not member_name.startswith("package/"):
                continue
            rel = member_name.removeprefix("package/")
            if rel in required or rel.endswith(".whl"):
                member = archive.getmember(member_name)
                if member.isdir():
                    continue
                target = paths.pyodide_dir / rel
                with archive.extractfile(member) as source:
                    if source is None:
                        continue
                    target.write_bytes(source.read())
            if rel == "pyodide-lock.json":
                with archive.extractfile(member_name) as source:
                    if source is not None:
                        lock = json.loads(source.read().decode("utf-8"))
        if lock is None:
            raise RuntimeError("pyodide-lock.json missing from pyodide tarball")

    _download_pyodide_packages(paths, lock)
    return lock


def _download_pypi_wheels(paths: Paths) -> None:
    if paths.wheel_dir.exists():
        shutil.rmtree(paths.wheel_dir)
    paths.wheel_dir.mkdir(parents=True)
    run(
        [
            str(paths.python),
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--platform",
            "pyemscripten_2026_0_wasm32",
            "--python-version",
            "3.14",
            "--implementation",
            "cp",
            "--abi",
            "cp314",
            "--dest",
            str(paths.wheel_dir),
            *PYODIDE_PYPI_WHEELS,
        ],
        cwd=paths.root,
    )
    run(
        [
            str(paths.python),
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--platform",
            "any",
            "--python-version",
            "3.14",
            "--implementation",
            "py",
            "--abi",
            "none",
            "--dest",
            str(paths.wheel_dir),
            *UNIVERSAL_PYPI_WHEELS,
        ],
        cwd=paths.root,
    )


def _build_site(paths: Paths) -> None:
    run([str(paths.python), "build_catalog.py"], cwd=paths.root)
    run(["npm", "--prefix", "extension", "run", "build"], cwd=paths.root, env_path=paths.venv_bin)
    labextension_dir = paths.root / ".venv" / "share" / "jupyter" / "labextensions" / "@torch-dojo"
    target = labextension_dir / "browser-gym"
    labextension_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(paths.root / "extension" / "labextension", target)
    # doit skips the jupyter-lite.json merge when only dist/ changed, which drops disabledExtensions.
    (paths.root / ".jupyterlite.doit.db").unlink(missing_ok=True)
    run(
        [
            str(paths.jupyter),
            "lite",
            "build",
            "--force",
            "--lite-dir",
            ".",
            "--contents",
            "build",
            "--output-dir",
            "dist",
        ],
        cwd=paths.root,
        env_path=paths.venv_bin,
    )


def _copy_gym_assets(paths: Paths) -> None:
    gym_assets = paths.dist / "gym-assets"
    checker_dir = gym_assets / "checkers"
    checker_dir.mkdir(parents=True, exist_ok=True)
    for checker in sorted((paths.root / "build" / "checkers").glob("*.json")):
        shutil.copy2(checker, checker_dir / checker.name)
    lesson_dir = gym_assets / "lessons"
    if lesson_dir.exists():
        shutil.rmtree(lesson_dir)
    shutil.copytree(paths.root / "build" / "lessons", lesson_dir)
    shutil.copy2(paths.root / "build" / "catalog.json", gym_assets / "catalog.json")
    shutil.copy2(paths.root / "build" / "motifs.json", gym_assets / "motifs.json")
    shutil.copy2(paths.root / "check_runtime.py", gym_assets / "check_runtime.py")
    checker_pypi = gym_assets / "pypi"
    if checker_pypi.exists():
        shutil.rmtree(checker_pypi)
    shutil.copytree(paths.dist / "pypi", checker_pypi)

    worker = (paths.root / "checker-worker.js").read_text()
    offline_worker = (
        worker.replace(CHECKER_WORKER_ONLINE_PYODIDE, "../static/pyodide/pyodide.mjs")
        .replace("__DOJO_CHECKER_OFFLINE__", "true")
    )
    (gym_assets / "checker-worker.js").write_text(offline_worker)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _patch_jupyterlite_config(paths: Paths) -> None:
    config_path = paths.dist / "jupyter-lite.json"
    config = _read_json(config_path)
    jupyter_config = config.setdefault("jupyter-config-data", {})
    settings = jupyter_config.setdefault("litePluginSettings", {})
    kernel = settings.setdefault("@jupyterlite/pyodide-kernel-extension:kernel", {})
    urls = kernel.get("pipliteUrls", [])
    local_url = "./pypi/all.json"
    urls = [local_url, *[url for url in urls if url.split("?")[0].split("#")[0] != local_url]]
    kernel.update(
        {
            "pyodideUrl": "./static/pyodide/pyodide.mjs",
            "disablePyPIFallback": True,
            "pipliteUrls": urls,
            "loadPyodideOptions": {"packages": sorted(PYODIDE_BOOTSTRAP_PACKAGES)},
        }
    )
    jupyter_config["enableServiceWorkerCache"] = True
    _write_json(config_path, config)


def _patch_service_worker(paths: Paths) -> None:
    # JupyterLite serves bundled notebooks through static /api/contents/*.json
    # indexes. Its generated worker bypasses every /api/ URL, so a previously
    # opened notebook disappears on offline reload unless these indexes cache.
    path = paths.dist / "service-worker.js"
    source = path.read_text()
    needle = 'a.pathname.includes("/api/")'
    if source.count(needle) != 1:
        raise RuntimeError("JupyterLite service worker API filter changed")
    path.write_text(source.replace(
        needle,
        '(a.pathname.includes("/api/")&&!a.pathname.includes("/api/contents/"))',
    ))


def _asset_entry(dist: Path, path: Path) -> dict:
    return {"path": path.relative_to(dist).as_posix(), "sha256": sha256(path), "size": path.stat().st_size}


def _manifest(paths: Paths, lock: dict) -> dict:
    pypi_index = _read_json(paths.dist / "pypi" / "all.json")
    assets = []
    for rel in [
        "jupyter-lite.json",
        "service-worker.js",
        "static/pyodide/pyodide.mjs",
        "static/pyodide/pyodide.asm.mjs",
        "static/pyodide/pyodide.asm.wasm",
        "static/pyodide/python_stdlib.zip",
        "static/pyodide/pyodide-lock.json",
        "pypi/all.json",
        "gym-assets/check_runtime.py",
        "gym-assets/checker-worker.js",
        "gym-assets/catalog.json",
        "gym-assets/motifs.json",
        "gym-assets/pypi/all.json",
    ]:
        assets.append(_asset_entry(paths.dist, paths.dist / rel))
    for wheel in sorted((paths.dist / "pypi").glob("*.whl")):
        assets.append(_asset_entry(paths.dist, wheel))
    for wheel in sorted((paths.dist / "gym-assets" / "pypi").glob("*.whl")):
        assets.append(_asset_entry(paths.dist, wheel))
    for lesson in sorted((paths.dist / "gym-assets" / "lessons").glob("*.py")):
        assets.append(_asset_entry(paths.dist, lesson))
    # MathJax's fonts, which typeset the briefs and drills.
    for font in sorted((paths.dist / "build").glob("MathJax_*.woff")):
        assets.append(_asset_entry(paths.dist, font))
    # The gym extension itself: its scripts, styles and bundled fonts.
    for asset in sorted(path for path in (paths.dist / GYM_EXTENSION).rglob("*") if path.is_file()):
        assets.append(_asset_entry(paths.dist, asset))

    def pypi_version(name: str) -> str:
        releases = pypi_index[name]["releases"]
        return sorted(releases)[-1]

    packages = {
        "numpy": {"version": lock["packages"]["numpy"]["version"], "source": "pyodide-lock"},
        "micropip": {"version": lock["packages"]["micropip"]["version"], "source": "pyodide-lock"},
        "sortedcontainers": {
            "version": lock["packages"]["sortedcontainers"]["version"],
            "source": "pyodide-lock",
        },
        "hypothesis": {"version": pypi_version("hypothesis"), "source": "local-piplite-wheel"},
        "comm": {"version": pypi_version("comm"), "source": "local-piplite-wheel"},
        "ipywidgets": {"version": pypi_version("ipywidgets"), "source": "local-piplite-wheel"},
        "jupyterlab_widgets": {
            "version": pypi_version("jupyterlab-widgets"),
            "source": "local-piplite-wheel",
        },
        "widgetsnbextension": {
            "version": pypi_version("widgetsnbextension"),
            "source": "local-piplite-wheel",
        },
    }
    return {
        "offline_scope": "previously-loaded-site",
        "pyodide": {
            "version": PYODIDE_VERSION,
            "python": lock["info"]["python"],
            "abi": lock["info"]["abi_version"],
            "platform": lock["info"]["platform"],
            "npm_tarball": PYODIDE_TARBALL_URL,
            "npm_integrity": PYODIDE_TARBALL_INTEGRITY,
        },
        "packages": packages,
        "assets": assets,
    }


def manifest_problems(dist: Path) -> list[str]:
    """Every manifest asset that is missing from ``dist`` or no longer matches its checksum."""
    manifest = _read_json(dist / "gym-assets" / "offline-manifest.json")
    problems = []
    for item in manifest["assets"]:
        path = dist / item["path"]
        if not path.is_file():
            problems.append(f"missing: {item['path']}")
        elif sha256(path) != item["sha256"]:
            problems.append(f"altered: {item['path']}")
    return problems


def build(root: Path) -> None:
    paths = Paths(root=root)
    lock = _extract_pyodide_runtime(paths)
    _download_pypi_wheels(paths)
    _build_site(paths)
    _copy_gym_assets(paths)
    _patch_jupyterlite_config(paths)
    _patch_service_worker(paths)
    manifest_path = paths.dist / "gym-assets" / "offline-manifest.json"
    _write_json(manifest_path, _manifest(paths, lock))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check dist against its offline manifest instead of building",
    )
    args = parser.parse_args()
    if args.verify:
        problems = manifest_problems(Paths(root=args.root.resolve()).dist)
        for problem in problems:
            print(problem)
        raise SystemExit(1 if problems else 0)
    build(args.root.resolve())


if __name__ == "__main__":
    main()
