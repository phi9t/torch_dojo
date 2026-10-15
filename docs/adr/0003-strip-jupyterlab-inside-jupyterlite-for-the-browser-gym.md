# Strip JupyterLab inside JupyterLite for the browser gym

The browser gym should feel like a calm, two-column workout page, not an IDE. We keep running it as a JupyterLite site: `disabledExtensions` in `jupyter-lite.json` turns off the menus, document tabs, status bar, file browser, and launcher, and the gym extension lays out its own shell from Lumino widgets. We do not build a separate page around the JupyterLite kernel with our own editor and markdown renderer.

## Considered Options

- **Standalone page over the kernel**: total visual control, but we would rewrite the notebook editor, output and markdown/HTML renderers, widget support, and the offline-build and e2e qualification that JupyterLite already gives us.
- **Restyle full JupyterLab**: cheapest, but the IDE chrome stays and the gym never stops looking like a notebook app.

## Consequences

The learner keeps exactly two JupyterLab pieces, the Python cell editor and rendered markdown/HTML outputs, and both are themed by the gym. Upgrading JupyterLite may rename the plugin IDs listed in `disabledExtensions`, so the e2e suite must assert the chrome stays hidden.

Two pieces of chrome are not plugins and cannot be disabled by ID. Document tabs go away because the lab app starts in single-document mode (`lab/jupyter-lite.json`), and the shell's own title and menu strips are hidden by the gym extension through the shell's API. The same e2e check guards them.
