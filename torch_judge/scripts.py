"""Interpret tagged question/eval scripts without executing their teaching cells.

Both the native judge and browser authoring use this contract. Execution and
packaging consume the same selected program, including its original order.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tokenize

_CELL_MARKER = re.compile(r"#\s*%%(?:\s*\[([^\]]+)\])?")


def script_cells(source: str) -> list[tuple[str | None, int, int]]:
    markers: list[tuple[str | None, int]] = [(None, 0)]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and token.start[1] == 0:
            match = _CELL_MARKER.fullmatch(token.string.strip())
            if match:
                markers.append((match.group(1), token.start[0]))
    final_line = len(source.splitlines()) + 1
    return [(tag, line, markers[i + 1][1] if i + 1 < len(markers) else final_line)
            for i, (tag, line) in enumerate(markers)]


@dataclass(frozen=True)
class ScriptProgram:
    """Selected Python, executable with original locations or portable as source."""
    filename: str
    nodes: list[ast.stmt]
    text: str
    candidate: str | None = None
    verifier_names: tuple[str, ...] = ()

    def compile(self):
        return compile(ast.Module(body=self.nodes, type_ignores=[]), self.filename, "exec")

    def source(self) -> str:
        return self.text


class TaggedScript:
    """Own cell validation and the selection rules for one canonical script."""
    def __init__(self, source: str, filename: str | Path = "script.py"):
        self.source = source
        self.path = Path(filename)
        self._tree = ast.parse(source, filename=str(self.path))
        self._cells = script_cells(source)

    def _one_cell(self, tag: str) -> tuple[str | None, int, int]:
        cells = [cell for cell in self._cells if cell[0] == tag]
        if not cells:
            raise ValueError(f"{self.path.name}:1: missing # %% [{tag}] section")
        if len(cells) > 1:
            raise ValueError(f"{self.path.name}:{cells[1][1]}: duplicate # %% [{tag}] section")
        return cells[0]

    def _nodes(self, cell) -> list[ast.stmt]:
        _, start, end = cell
        nodes = []
        for node in self._tree.body:
            first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
            last = node.end_lineno or node.lineno
            if first < end and last > start:
                if first <= start or last >= end:
                    raise ValueError(f"{self.path.name}:{node.lineno}: statement crosses a cell marker")
                nodes.append(node)
        return nodes

    def cell_source(self, tag: str) -> str:
        cell = self._one_cell(tag)
        self._nodes(cell)
        return "\n".join(self.source.splitlines()[cell[1]:cell[2] - 1]).strip() + "\n"

    def _program(self, nodes, cells, **metadata) -> ScriptProgram:
        ordered = sorted(dict.fromkeys(nodes), key=lambda node: (node.lineno, node.col_offset))
        # Keep author comments and formatting in the learner's answer cell.
        # Unselected imports retain only their statement, never a teaching probe
        # sharing the same source line.
        lines = self.source.splitlines()
        parts = [(cell[1], "\n".join(lines[cell[1]:cell[2] - 1]).strip()) for cell in cells]
        parts.extend((node.lineno, ast.get_source_segment(self.source, node))
                     for node in ordered if not any(cell[1] < node.lineno < cell[2] for cell in cells))
        text = "\n\n".join(part for _, part in sorted(parts, key=lambda item: item[0]) if part) + "\n"
        return ScriptProgram(str(self.path), ordered, text, **metadata)

    def _imports_before(self, line: int) -> list[ast.stmt]:
        return [node for node in self._tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < line]

    def answer(self) -> ScriptProgram:
        answer = self._one_cell("answer")
        # Drill imports belong to their teaching probes, not the saved answer.
        drills = [cell for cell in self._cells if cell[0] == "drill"]
        nodes = [node for node in self._imports_before(answer[1])
                 if not any(start < node.lineno < end for _, start, end in drills)]
        cells = [cell for cell in self._cells if cell[0] == "setup"] + [answer]
        for cell in cells:
            nodes.extend(self._nodes(cell))
        return self._program(nodes, cells)

    def verifiers(self) -> ScriptProgram:
        cell = self._one_cell("verifiers")
        nodes = self._nodes(cell)
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("torch_judge"):
                raise ValueError(f"{self.path.name}:{node.lineno}: close the verifier section before the runner")
        verifiers = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name.startswith("verify_")]
        if not verifiers:
            raise ValueError(f"{self.path.name}:{cell[1]}: verifier section needs a verify_* function")
        if any(len(node.args.args) != 1 or node.args.posonlyargs or node.args.kwonlyargs
               or node.args.vararg or node.args.kwarg or node.args.defaults or node.args.kw_defaults
               for node in verifiers):
            raise ValueError(f"{self.path.name}:{cell[1]}: verifiers need the same candidate parameter")
        names = {node.args.args[0].arg for node in verifiers}
        if len(names) != 1:
            raise ValueError(f"{self.path.name}:{cell[1]}: verifiers need the same candidate parameter")
        return self._program(self._imports_before(cell[1]) + nodes, [cell],
                             candidate=names.pop(), verifier_names=tuple(node.name for node in verifiers))
