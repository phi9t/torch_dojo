/** The Bench's keyboard shortcuts, named the way this platform writes them. */
const MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);

/** `Mod` is ⌘ on a Mac and Ctrl elsewhere. */
export function keys(combo: string): string {
  return combo.replace(/Mod/g, MAC ? '⌘' : 'Ctrl');
}

export const SHORTCUTS: Array<{ combo: string; action: string }> = [
  { combo: 'Shift+Enter', action: 'Run the cell you are in' },
  { combo: 'Mod+Shift+Enter', action: 'Run the answer, then each scratch cell' },
  { combo: 'Mod+Enter', action: 'Check the answer at the selected stage' },
  { combo: 'Mod+.', action: 'Stop a running check' },
  { combo: 'Mod+Shift+E', action: 'Expand the answer, or show the problem again' },
  { combo: 'Mod+K', action: 'Switch to another workout' },
  { combo: 'Tab', action: 'Accept a completion (they appear as you type; Ctrl+Space asks for them)' }
];

/** A tooltip that names the action and its shortcut. */
export function withShortcut(action: string, combo: string): string {
  return `${action} (${keys(combo)})`;
}
