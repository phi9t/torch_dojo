"""Small display helpers used by generated browser-gym lesson cells."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import inspect
import textwrap
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HtmlFragment:
    data: str

    def _repr_html_(self) -> str:
        return self.data


def _html(data: str):
    try:
        from IPython.display import HTML
    except Exception:
        return HtmlFragment(data)
    return HTML(data)


def _preview(value: np.ndarray, max_items: int) -> str:
    flat = value.reshape(-1) if value.ndim else value.reshape(1)
    shown = flat[:max_items]
    preview = np.array2string(shown, precision=4, separator=", ", threshold=max_items + 1)
    if flat.size > max_items:
        preview = preview.removesuffix("]") + ", ...]"
    return preview


def tensor_card_html(
    name: str,
    value: np.ndarray,
    axes: Iterable[str] = (),
    *,
    max_items: int = 12,
):
    """Return compact, keyboard-focusable tensor summary markup."""

    array = np.asarray(value)
    axis_labels = tuple(axes)
    axis_text = " x ".join(axis_labels) if axis_labels else "unnamed"
    preview = escape(_preview(array, max_items))
    label = escape(name)
    shape = escape(str(tuple(array.shape)))
    dtype = escape(str(array.dtype))
    axis_text = escape(axis_text)
    return f"""
<style>
  .dojo-tensor-card {{
    border: 1px solid var(--jp-border-color1, #d1d5db);
    border-radius: 8px;
    padding: 10px 12px;
    max-width: 520px;
    background: var(--jp-layout-color0, #ffffff);
    color: var(--jp-content-font-color1, #111827);
  }}
  .dojo-tensor-card:focus {{
    outline: 2px solid var(--jp-brand-color1, #2563eb);
    outline-offset: 2px;
  }}
  .dojo-tensor-card figcaption {{
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .dojo-tensor-card dl {{
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px 10px;
    margin: 0 0 8px;
  }}
  .dojo-tensor-card dt {{
    font-weight: 600;
  }}
  .dojo-tensor-card dd {{
    margin: 0;
  }}
  .dojo-tensor-card pre {{
    margin: 0;
    max-width: 100%;
    overflow: auto;
    white-space: pre-wrap;
  }}
</style>
<figure class="dojo-tensor-card" role="group" tabindex="0" aria-label="Tensor card for {label}">
  <figcaption>Tensor: {label}</figcaption>
  <dl>
    <dt>shape</dt><dd>{shape}</dd>
    <dt>dtype</dt><dd>{dtype}</dd>
    <dt>axes</dt><dd>{axis_text}</dd>
  </dl>
  <pre aria-label="bounded value preview">{preview}</pre>
</figure>
""".strip()


def tensor_card(
    name: str,
    value: np.ndarray,
    axes: Iterable[str] = (),
    *,
    max_items: int = 12,
):
    """Render a compact, keyboard-focusable tensor summary for supplied values."""

    return _html(tensor_card_html(name, value, axes, max_items=max_items))


def notebook_tensor_card_source() -> str:
    """Return the helper source embedded into standalone browser notebooks."""

    return "\n\n".join(
        (
            "from html import escape\nfrom typing import Iterable\n\nimport numpy as np",
            textwrap.dedent(inspect.getsource(_preview)).strip(),
            textwrap.dedent(inspect.getsource(tensor_card_html)).strip(),
            textwrap.dedent(
                """
                def tensor_card(name: str, value: np.ndarray, axes: Iterable[str] = (), *, max_items: int = 12):
                    return HTML(tensor_card_html(name, value, axes, max_items=max_items))
                """
            ).strip(),
        )
    )
