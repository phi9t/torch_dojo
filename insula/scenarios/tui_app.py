"""Full-screen learner view of the Emacs-owned Org scenario."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ListItem, ListView, RichLog, Static, TextArea
from textual import work
from rich.text import Text

from insula.scenarios.tui_client import OutcomeUnknown, ScenarioError
from insula.scenarios.tui_logs import LogTail, redact_text


class ReflowLog(RichLog):
    """RichLog that rewraps retained source text when its width changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entries = []
        self._last_width = None

    def write(self, content, *args, **kwargs):
        self._entries.append(content)
        del self._entries[:-200]
        return super().write(content, *args, **kwargs)

    def clear(self):
        self._entries.clear()
        return super().clear()

    def on_resize(self, event):
        if not self._size_known:
            # RichLog replays deferred writes through our write override.
            self._entries.clear()
        super().on_resize(event)
        if self._last_width is not None and event.size.width != self._last_width:
            entries = self._entries[:]
            position = self.scroll_y
            super().clear()
            for entry in entries:
                super().write(entry, scroll_end=False)
            self.scroll_y = min(position, self.max_scroll_y)
        self._last_width = event.size.width


class ScenarioApp(App):
    CSS = """
    Screen { layout: vertical; background: #f5f8fc; color: #183956; }
    #masthead { height: 2; background: #ffffff; color: #173b5c; padding: 0 1; }
    #brand { width: 1fr; text-style: bold; content-align: left middle; }
    #progress { width: 17; color: #5c7790; text-align: right; content-align: right middle; }
    #provenance { height: 2; padding: 0 1; background: #dceaf5; color: #235475; }
    #body { height: 1fr; }
    #rail { width: 26; background: #eaf2f8; border-right: solid #c7d9e8; }
    #rail-title { height: 2; padding: 0 1; color: #57748c; text-style: bold; content-align: left middle; }
    #steps { height: 1fr; background: #eaf2f8; }
    #steps ListItem { height: 2; padding: 0 1; color: #788da0; }
    #steps ListItem.complete { color: #437264; }
    #steps ListItem.pending { color: #1a527c; text-style: bold; }
    #steps ListItem.failed { color: #9c5555; text-style: bold; }
    #steps ListItem.-highlight { background: #cbdfee; color: #153b5c; text-style: bold; }
    #detail { width: 1fr; padding: 0 1; background: #f9fbfd; }
    #title { height: 2; color: #123758; text-style: bold; content-align: left middle; }
    #description { height: 2; color: #587188; }
    #source-label, #result-label { height: 1; color: #50718e; text-style: bold; }
    #editor { height: 30%; min-height: 4; border: round #adc9df; background: #ffffff; color: #153b5b; }
    #editor:focus { border: round #4f83ab; }
    #result { height: 1fr; min-height: 2; background: #ffffff; border-left: thick #9dc1dc; padding: 0 1; color: #213f58; }
    #monitor-row { height: 25%; min-height: 4; max-height: 11; }
    .monitor { width: 1fr; background: #eef4f9; border-left: thick #a6c5db; }
    #jupyter-panel { margin-left: 1; }
    .monitor-title { height: 1; padding: 0 1; background: #dceaf5; color: #315e7e; text-style: bold; }
    .monitor-log { height: 1fr; padding: 0 1; color: #36536b; }
    #controls { height: 2; background: #ffffff; padding: 0 1; }
    Button { height: 1; min-width: 14; margin: 0 1 0 0; border: none; background: #e7eff6; color: #254f70; }
    Button:hover { background: #d1e3f1; }
    #next { background: #235b84; color: #ffffff; text-style: bold; }
    #next:hover { background: #174568; }
    Button:disabled { background: #edf1f4; color: #92a3b1; }
    #status { height: 1; padding: 0 1; background: #eaf2f8; color: #587188; }
    """
    BINDINGS = [
        ("f10", "next_step", "Next step"),
        ("f9", "previous_step", "Previous"),
        ("ctrl+r", "restart", "Restart"),
        ("ctrl+q", "quit_scenario", "Quit"),
    ]

    def __init__(self, client, *, worker_log: Path | None = None,
                 server_log: Path | None = None, token_path: Path | None = None,
                 server_log_offset: int = 0):
        super().__init__()
        self.client = client
        self.snapshot = None
        self.selected_index = 0
        self._busy = False
        self._dirty = False
        self._rendering = False
        self.quit_acknowledged = False
        self.failure = None
        self.unknown_outcome = False
        self.token_path = token_path
        self._result_key = None
        self._log_tails = {
            'emacs-log': LogTail(worker_log, token_path=token_path)
            if worker_log else None,
            'server-log': LogTail(server_log, token_path=token_path,
                                  start_offset=server_log_offset)
            if server_log else None,
        }

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static("TORCH DOJO  /  GPU SCENARIO", id="brand")
            yield Static("— / —", id="progress")
        yield Static("TUI  →  EMACS  →  emacs-jupyter  →  JUPYTER\nAwaiting worker and kernel", id="provenance")
        with Horizontal(id="body"):
            with Vertical(id="rail"):
                yield Static("CHECKPOINTS", id="rail-title")
                yield ListView(id="steps")
            with Vertical(id="detail"):
                yield Label("Loading scenario…", id="title")
                yield Static("", id="description")
                yield Static("PYTHON CELL  /  EDITABLE", id="source-label")
                yield TextArea("", language="python", theme="github_light",
                               show_line_numbers=True, id="editor")
                yield Static("RESULT  /  CURRENT CHECKPOINT", id="result-label")
                yield ReflowLog(id="result", min_width=1, wrap=True, markup=False)
                with Horizontal(id="monitor-row"):
                    with Vertical(classes="monitor", id="emacs-panel"):
                        yield Static("EMACS  /  WORKER", classes="monitor-title",
                                     id="emacs-title")
                        yield ReflowLog(id="emacs-log", classes="monitor-log",
                                      min_width=1, wrap=True, max_lines=200)
                    with Vertical(classes="monitor", id="jupyter-panel"):
                        yield Static("JUPYTER  /  SERVER", classes="monitor-title")
                        yield ReflowLog(id="server-log", classes="monitor-log",
                                      min_width=1, wrap=True, max_lines=200)
        with Horizontal(id="controls"):
            yield Button("Next  F10", id="next")
            yield Button("Previous  F9", id="previous")
            yield Button("Restart  Ctrl+R", id="restart")
            yield Button("Quit  Ctrl+Q", id="quit")
        yield Static("Connecting to Emacs worker…", id="status")

    def on_mount(self) -> None:
        self.theme = "textual-light"
        self._refresh_logs()
        self.set_interval(.5, self._refresh_logs)
        self._request("snapshot")

    def _refresh_logs(self) -> None:
        for widget_id, tail in self._log_tails.items():
            if tail is None:
                continue
            pane = self.query_one(f'#{widget_id}', RichLog)
            for line in tail.poll():
                pane.write(line)

    def _safe(self, value) -> str:
        return redact_text(value or '', self.token_path)

    def _source_draft(self, displayed: str, original: str) -> str:
        # The editable view hides the live token. Keep an untouched placeholder
        # backed by its original value when the learner edits another line.
        if self.token_path:
            try:
                token = self.token_path.read_text().strip()
                if token and token in original:
                    return displayed.replace('[redacted]', token)
            except OSError:
                pass
        return displayed

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self.query_one("#next", Button).disabled = not self._can_advance()
        self.query_one("#restart", Button).disabled = value
        self.query_one("#quit", Button).disabled = value

    def _can_advance(self) -> bool:
        return not (self._busy or self.unknown_outcome or
                    (self.snapshot and (self.snapshot.get('failed') or
                                        self.snapshot.get('busy') or
                                        self.snapshot['pending_index'] >= len(self.snapshot['steps']))))

    def _request(self, command: str, target_index: int | None = None) -> None:
        if self._busy:
            return
        if self.unknown_outcome and command != 'quit':
            return
        if command == 'next' and not self._can_advance():
            return
        self._set_busy(True)
        draft = None
        if self._dirty and self.snapshot is not None and not self.unknown_outcome:
            step = self.snapshot['steps'][self.selected_index]
            if step.get('source') is not None:
                draft = (step['cell_id'], self._source_draft(
                    self.query_one('#editor', TextArea).text, step['source']))
        self._perform(command, draft, target_index)

    @work(thread=True)
    def _perform(self, command: str, draft, target_index: int | None) -> None:
        try:
            if draft is not None:
                self.client.request('set_source', {'cell_id': draft[0], 'source': draft[1]})
            args = {'index': target_index} if command == 'previous' and target_index is not None else None
            state = self.client.request(command, args)
            self.call_from_thread(self._receive, state, command, target_index, draft)
        except OutcomeUnknown as exc:
            self.call_from_thread(self._show_error, str(exc), True)
        except ScenarioError as exc:
            self.call_from_thread(self._show_error, str(exc), False)

    def _receive(self, state: dict, command: str, target_index: int | None,
                 draft=None) -> None:
        self._dirty = False
        self.snapshot = state
        if target_index is not None:
            self.selected_index = target_index
        elif command in {'next', 'restart', 'snapshot', 'previous'}:
            view = state.get('viewed_index')
            self.selected_index = view if isinstance(view, int) else min(
                state['pending_index'], len(state['steps']) - 1)
        self._render()
        self._set_busy(False)
        if command == 'quit':
            self.quit_acknowledged = True
            self.exit()

    def _show_error(self, detail: str, unknown: bool) -> None:
        self.unknown_outcome = unknown
        context = ''
        if unknown and self.snapshot:
            context = (f" Inspect {self.snapshot.get('run_file')}"
                       f"; possible owned kernel {self.snapshot.get('kernel_id') or 'unknown'}.")
        self.failure = self._safe(detail + context)
        self.query_one('#status', Static).update(self.failure)
        self._set_busy(False)

    def _render(self) -> None:
        if self.snapshot is None:
            return
        self._rendering = True
        steps = self.snapshot['steps']
        self.query_one('#rail-title', Static).update(f"CHECKPOINTS  /  {len(steps):02d}")
        listing = self.query_one('#steps', ListView)
        listing.clear()
        for number, step in enumerate(steps, 1):
            state = step['state']
            marker = {'complete': '✓', 'pending': '●', 'failed': '×',
                      'locked': '·'}.get(state, '·')
            if step.get('stale', False) or (
                    self._dirty and number - 1 == self.selected_index):
                marker = '✎'
            listing.append(ListItem(Label(self._safe(f"{number:02d}  {marker}  {step['name']}")),
                                    classes=state))
        listing.index = self.selected_index
        step = steps[self.selected_index]
        current = min(self.snapshot['pending_index'] + 1, len(steps))
        self.query_one('#progress', Static).update(f"{current:02d} / {len(steps):02d}")
        emacs_pid = self.snapshot.get('emacs_pid') or '?'
        kernel_id = self.snapshot.get('kernel_id')
        kernel_label = kernel_id[:12] if kernel_id else 'awaiting kernel'
        self.query_one('#provenance', Static).update(self._safe(
            "TUI  →  EMACS  →  emacs-jupyter  →  JUPYTER\n"
            f"Emacs insula PID {emacs_pid}   ·   Jupyter kernel {kernel_label}"))
        self.query_one('#emacs-title', Static).update(
            f"EMACS  /  PID {emacs_pid}")
        heading = step.get('heading')
        title = step['name']
        if heading and heading not in title:
            title = f"{heading} · {title}"
        self.query_one('#title', Label).update(self._safe(title))
        self.query_one('#description', Static).update(self._safe(step.get('text')))
        editor = self.query_one('#editor', TextArea)
        source = self._safe(step.get('source'))
        if editor.text != source and not self._dirty:
            editor.text = source
        editor.display = step.get('source') is not None
        self.query_one('#source-label', Static).display = editor.display
        self.query_one('#source-label', Static).update(
            'PYTHON CELL  /  UNSAVED' if self._dirty else 'PYTHON CELL  /  EDITABLE')
        result = self.query_one('#result', RichLog)
        lines = []
        label = "RESULT  /  AWAITING EXECUTION"
        stale = step.get('stale', False) or self._dirty
        preceding = (not step.get('result') and not isinstance(step.get('output'), dict)
                     and self.selected_index == self.snapshot['pending_index']
                     and self.selected_index > 0)
        shown_step = steps[self.selected_index - 1] if preceding else step
        output = shown_step.get('output')
        if isinstance(output, dict) or shown_step.get('result') or preceding:
            label = (f"RESULT  /  {shown_step['name'].upper()}" if preceding else
                     "RESULT  /  STALE UNTIL RESTART" if stale else
                     "RESULT  /  SELECTED CHECKPOINT")
        if isinstance(output, dict):
            for heading, key, color in (('STDOUT', 'stdout', '#356886'),
                                        ('VALUE', 'text', '#315d50'),
                                        ('ERROR', 'error', '#a0424b')):
                value = output.get(key)
                if value:
                    lines.append(Text(heading, style=f'bold {color}'))
                    lines.append(Text(self._safe(value)))
            mime = output.get('mime') or []
            if mime:
                lines.append(Text('MIME  /  unsupported', style='bold #8c6b3c'))
                lines.append(Text(self._safe(', '.join(mime))))
            if not lines:
                lines.append(Text('EMPTY OUTPUT', style='italic #607d94'))
        if shown_step.get('result'):
            if not isinstance(output, dict):
                plain = (f"{shown_step['name']} result:\n{shown_step['result']}"
                         if preceding else shown_step['result'])
                lines.append(Text(self._safe(plain)))
        elif not lines and preceding:
            lines.append(Text(self._safe(f"{shown_step['name']} result:\nNo result yet.")))
        elif not lines:
            lines.append(Text('No result yet.'))
        if stale and not preceding and (step.get('result') or isinstance(output, dict)):
            lines.insert(0, Text('This result is from before your edit. Ctrl+R saves your edits and restarts.',
                                 style='italic #607d94'))
        self.query_one('#result-label', Static).update(self._safe(label))
        result_key = (self.selected_index, tuple((line.plain, str(line.style)) for line in lines))
        if result_key != self._result_key:
            result.clear()
            for line in lines:
                result.write(line)
            self._result_key = result_key
        skipped = self.snapshot.get('skipped_count', 0)
        notice = f"  ·  {skipped} non-Python block(s) skipped" if skipped else ''
        if self.snapshot.get('failed'):
            hint = 'Ctrl+R restart · Ctrl+Q finish'
        elif self.snapshot['pending_index'] >= len(steps):
            hint = 'Ctrl+Q finish · Ctrl+R start again'
        else:
            hint = 'F10 run next · F9 review'
        self.query_one('#status', Static).update(self._safe(
            self.snapshot.get('status', '') + '  ·  ' + hint + notice))
        self.query_one('#next', Button).disabled = not self._can_advance()
        self._rendering = False

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not self._rendering and self.snapshot is not None:
            source = self.snapshot['steps'][self.selected_index].get('source')
            self._dirty = source is not None and event.text_area.text != self._safe(source)
            self._render()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if self._rendering or self._busy or self.snapshot is None:
            return
        listing = self.query_one('#steps', ListView)
        index = listing.index
        if index is None or index == self.selected_index:
            return
        if index > self.snapshot['pending_index']:
            self._render()
            return
        if self._dirty:
            if index < self.snapshot['pending_index']:
                self._request('previous', target_index=index)
            else:
                self._request('snapshot', target_index=index)
            return
        if index < self.snapshot['pending_index']:
            self._request('previous', target_index=index)
        else:
            self.selected_index = index
            self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {'next': self.action_next_step, 'previous': self.action_previous_step,
         'restart': self.action_restart, 'quit': self.action_quit_scenario}[event.button.id]()

    def action_next_step(self) -> None:
        self._request('next')

    def action_previous_step(self) -> None:
        if self.snapshot and self.selected_index > 0:
            self._request('previous', target_index=self.selected_index - 1)

    def action_restart(self) -> None:
        self._request('restart')

    def action_quit_scenario(self) -> None:
        self._request('quit')
