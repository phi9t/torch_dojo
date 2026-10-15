import { acceptCompletion, autocompletion, startCompletion, type Completion, type CompletionContext } from '@codemirror/autocomplete';
import {
  cursorCharLeft,
  cursorCharRight,
  cursorGroupBackward,
  cursorGroupForward,
  cursorLineDown,
  cursorLineEnd,
  cursorLineStart,
  cursorLineUp,
  deleteCharBackward,
  deleteCharForward,
  deleteGroupBackward,
  deleteGroupForward
} from '@codemirror/commands';
import { syntaxTree } from '@codemirror/language';
import { Prec, type Extension } from '@codemirror/state';
import { keymap, type Command, type EditorView } from '@codemirror/view';

/** The kernel request this module needs; a Jupyter kernel connection satisfies it. */
export interface CompletingKernel {
  requestComplete(content: { code: string; cursor_pos: number }): Promise<{
    content: {
      status: string;
      matches?: string[];
      cursor_start?: number;
      cursor_end?: number;
      metadata?: { _jupyter_types_experimental?: Array<{ text: string; type?: string; signature?: string }> };
    };
  }>;
}

const EMACS_KEY = 'torch-dojo-emacs-keys';

export function emacsKeysEnabled(): boolean {
  try {
    return window.localStorage.getItem(EMACS_KEY) === 'true';
  } catch {
    return false;
  }
}

export function setEmacsKeys(enabled: boolean): void {
  try {
    window.localStorage.setItem(EMACS_KEY, String(enabled));
  } catch {
    // The choice still applies until the page reloads.
  }
  emacsOverride = enabled;
}

let emacsOverride: boolean | undefined;

function emacsOn(): boolean {
  return emacsOverride ?? emacsKeysEnabled();
}

const COMPLETION_TYPES: Record<string, string> = {
  function: 'function',
  method: 'method',
  class: 'class',
  module: 'namespace',
  instance: 'variable',
  statement: 'variable',
  keyword: 'keyword',
  param: 'variable',
  property: 'property',
  path: 'text'
};

/**
 * Completions from the notebook kernel, as you type code: after a dot, or
 * once a name has two characters (not in comments or strings), or on Ctrl+Space. The kernel knows the names
 * defined so far, including NumPy's API once `np` is imported.
 */
function kernelCompletions(kernel: () => CompletingKernel | null | undefined) {
  return async (context: CompletionContext) => {
    const token = context.matchBefore(/\w*/);
    if (!token) return null;
    const afterDot = context.state.sliceDoc(token.from - 1, token.from) === '.';
    if (!context.explicit && !afterDot && token.text.length < 2) return null;
    // Typing prose in a comment or a string isn't asking for names.
    if (!context.explicit && /Comment|String/.test(syntaxTree(context.state).resolveInner(context.pos, -1).name)) return null;
    const connection = kernel();
    if (!connection) return null;
    let reply;
    try {
      reply = (await connection.requestComplete({ code: context.state.doc.toString(), cursor_pos: context.pos })).content;
    } catch {
      return null;
    }
    if (context.aborted || reply.status !== 'ok' || !reply.matches?.length) return null;
    const types = reply.metadata?._jupyter_types_experimental ?? [];
    const options: Completion[] = reply.matches.map(label => {
      const info = types.find(item => item.text === label);
      return {
        label,
        type: COMPLETION_TYPES[info?.type ?? ''] ?? 'variable',
        ...(info?.signature ? { detail: info.signature } : {})
      };
    });
    return { from: reply.cursor_start ?? token.from, to: reply.cursor_end ?? context.pos, options, validFor: /^\w*$/ };
  };
}

/** Text removed by the last Ctrl+K, for Ctrl+Y. */
let killed = '';

const killLine: Command = view => {
  const { head } = view.state.selection.main;
  const line = view.state.doc.lineAt(head);
  // At the end of a line, Ctrl+K joins the next one, as in Emacs.
  const to = head === line.to ? Math.min(line.to + 1, view.state.doc.length) : line.to;
  if (to === head) return true;
  killed = view.state.sliceDoc(head, to);
  view.dispatch({ changes: { from: head, to }, scrollIntoView: true });
  return true;
};

const yank: Command = view => {
  if (!killed) return true;
  view.dispatch(view.state.replaceSelection(killed));
  return true;
};

/** An Emacs binding that only acts while Emacs keys are on. */
function emacs(key: string, run: Command) {
  return { key, run: (view: EditorView) => emacsOn() && run(view), preventDefault: false };
}

const EMACS_BINDINGS = [
  emacs('Ctrl-a', cursorLineStart),
  emacs('Ctrl-e', cursorLineEnd),
  emacs('Ctrl-f', cursorCharRight),
  emacs('Ctrl-b', cursorCharLeft),
  emacs('Ctrl-n', cursorLineDown),
  emacs('Ctrl-p', cursorLineUp),
  emacs('Ctrl-d', deleteCharForward),
  emacs('Ctrl-h', deleteCharBackward),
  emacs('Ctrl-k', killLine),
  emacs('Ctrl-y', yank),
  emacs('Alt-f', cursorGroupForward),
  emacs('Alt-b', cursorGroupBackward),
  emacs('Alt-d', deleteGroupForward),
  emacs('Alt-Backspace', deleteGroupBackward)
];

/** Everything the gym adds to Python editors: kernel completion and optional Emacs keys. */
export function gymEditorExtension(kernel: () => CompletingKernel | null | undefined): Extension {
  return [
    // Tab accepts a visible suggestion at once, without CodeMirror's default grace period.
    autocompletion({ override: [kernelCompletions(kernel)], activateOnTyping: true, interactionDelay: 0, maxRenderedOptions: 60 }),
    Prec.highest(keymap.of([
      { key: 'Tab', run: acceptCompletion },
      { key: 'Ctrl-Space', run: startCompletion },
      ...EMACS_BINDINGS
    ]))
  ];
}

/** Whether a key event should be left to the editor's Emacs keys rather than the page's shortcuts. */
export function emacsOwnsKey(event: KeyboardEvent): boolean {
  if (!emacsOn() || !event.ctrlKey || event.metaKey) return false;
  return Boolean((event.target as Element | null)?.closest?.('.cm-editor'));
}
