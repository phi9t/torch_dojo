import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from browser_gym.offline_build import GYM_EXTENSION, manifest_problems


ROOT = Path(__file__).resolve().parents[2]
BROWSER_GYM = ROOT / "browser_gym"
DIST = BROWSER_GYM / "dist"


def _load_json(path: Path):
    return json.loads(path.read_text())


def test_offline_build_bundles_runtime_packages_and_manifest():
    completed = subprocess.run(
        ["make", "offline-build"],
        cwd=BROWSER_GYM,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout

    pyodide_dir = DIST / "static" / "pyodide"
    pypi_dir = DIST / "pypi"
    manifest_path = DIST / "gym-assets" / "offline-manifest.json"
    assert (pyodide_dir / "pyodide.mjs").is_file()
    assert (pyodide_dir / "pyodide.asm.wasm").is_file()
    assert (pyodide_dir / "python_stdlib.zip").is_file()
    assert (pyodide_dir / "pyodide-lock.json").is_file()
    assert (pypi_dir / "all.json").is_file()
    assert manifest_path.is_file()
    assert (DIST / "gym-assets" / "lessons" / "0059-0.py").is_file()

    config = _load_json(DIST / "jupyter-lite.json")
    plugin_settings = config["jupyter-config-data"]["litePluginSettings"][
        "@jupyterlite/pyodide-kernel-extension:kernel"
    ]
    assert plugin_settings["pyodideUrl"] == "./static/pyodide/pyodide.mjs"
    assert plugin_settings["disablePyPIFallback"] is True
    assert any(value.startswith("./pypi/all.json") for value in plugin_settings["pipliteUrls"])
    assert not any("://" in value for value in json.dumps(plugin_settings).split('"'))

    worker = (DIST / "gym-assets" / "checker-worker.js").read_text()
    assert "cdn.jsdelivr.net" not in worker
    assert "pypi.org" not in worker
    assert "static/pyodide/pyodide.mjs" in worker
    assert "gym-assets/pypi/" in worker
    service_worker = (DIST / "service-worker.js").read_text()
    assert 'a.pathname.includes("/api/contents/")' in service_worker

    wheel_index = _load_json(pypi_dir / "all.json")
    indexed_names = set(wheel_index)
    assert {"hypothesis", "sortedcontainers", "ipywidgets"}.issubset(indexed_names)

    manifest = _load_json(manifest_path)
    assert manifest["offline_scope"] == "previously-loaded-site"
    assert manifest["pyodide"]["version"] == "314.0.6"
    assert manifest["packages"]["numpy"]["version"] == "2.4.6"
    assert manifest["packages"]["hypothesis"]["version"] == "6.168.5"
    assert manifest["packages"]["ipywidgets"]["version"].startswith("8.")
    for item in manifest["assets"]:
        asset_path = DIST / item["path"]
        assert asset_path.is_file(), item
        actual = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        assert item["sha256"] == actual
    assert any(item["path"] == "gym-assets/lessons/0059-0.py" for item in manifest["assets"])
    assert any(item["path"] == "gym-assets/motifs.json" for item in manifest["assets"])
    assert _load_json(DIST / "gym-assets/motifs.json") == _load_json(ROOT / "torch_judge/motifs.json")


def _write_manifest(dist: Path, paths: list[str]) -> None:
    assets = [
        {"path": rel, "sha256": hashlib.sha256((dist / rel).read_bytes()).hexdigest(), "size": (dist / rel).stat().st_size}
        for rel in paths
    ]
    (dist / "gym-assets").mkdir(parents=True, exist_ok=True)
    (dist / "gym-assets" / "offline-manifest.json").write_text(json.dumps({"assets": assets}))


def test_manifest_verification_names_missing_and_altered_assets(tmp_path):
    font = tmp_path / GYM_EXTENSION / "static/newsreader.woff2"
    lesson = tmp_path / "gym-assets/lessons/0059-0.py"
    for path, data in ((font, b"font"), (lesson, b"print(1)\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _write_manifest(tmp_path, [f"{GYM_EXTENSION}/static/newsreader.woff2", "gym-assets/lessons/0059-0.py"])
    assert manifest_problems(tmp_path) == []

    font.write_bytes(b"tampered")
    lesson.unlink()
    assert manifest_problems(tmp_path) == [
        f"altered: {GYM_EXTENSION}/static/newsreader.woff2",
        "missing: gym-assets/lessons/0059-0.py",
    ]


def test_built_manifest_covers_the_gym_fonts():
    manifest_path = DIST / "gym-assets" / "offline-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("run make offline-build first")
    paths = {item["path"] for item in _load_json(manifest_path)["assets"]}
    fonts = {path.relative_to(DIST).as_posix() for path in (DIST / GYM_EXTENSION).rglob("*.woff*")}
    assert fonts and fonts <= paths
