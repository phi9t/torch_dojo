const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.mjs';
const DOJO_CHECKER_OFFLINE = typeof __DOJO_CHECKER_OFFLINE__ !== 'undefined' && __DOJO_CHECKER_OFFLINE__;

self.onmessage = async event => {
  const { requestId, source, bundleUrl, stage } = event.data;
  try {
    const { loadPyodide } = await import(PYODIDE_URL);
    const pyodide = await loadPyodide();
    await pyodide.loadPackage(['numpy', 'micropip', 'sortedcontainers']);
    if (DOJO_CHECKER_OFFLINE) {
      const hypothesisWheel = new URL('../gym-assets/pypi/hypothesis-6.168.5-cp314-cp314-pyemscripten_2026_0_wasm32.whl', self.location.href).toString();
      await pyodide.runPythonAsync(
        `import micropip\nawait micropip.install("${hypothesisWheel}", deps=False)`
      );
    } else {
      await pyodide.runPythonAsync('import micropip\nawait micropip.install(["hypothesis==6.168.5"])');
    }
    const bundle = await fetch(bundleUrl).then(response => {
      if (!response.ok) throw new Error(`Checker bundle HTTP ${response.status}`);
      return response.json();
    });
    const runtimeUrl = new URL('../check_runtime.py', bundleUrl);
    const runtime = await fetch(runtimeUrl).then(response => {
      if (!response.ok) throw new Error(`Checker runtime HTTP ${response.status}`);
      return response.text();
    });
    pyodide.FS.writeFile('/home/pyodide/check_runtime.py', runtime);
    pyodide.globals.set('submitted_source', source);
    pyodide.globals.set('checker_bundle', JSON.stringify(bundle));
    pyodide.globals.set('check_stage', stage);
    const resultJson = await pyodide.runPythonAsync(
      'import json, check_runtime\njson.dumps(check_runtime.run_check_bundle(submitted_source, json.loads(checker_bundle), check_stage))'
    );
    self.postMessage({ requestId, result: JSON.parse(resultJson) });
  } catch (error) {
    self.postMessage({ requestId, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  }
};
