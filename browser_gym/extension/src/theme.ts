export type GymTheme = 'light' | 'dark';

const STORAGE_KEY = 'torch-dojo-theme';

function opposite(theme: GymTheme): GymTheme {
  return theme === 'dark' ? 'light' : 'dark';
}

function storedTheme(): GymTheme | undefined {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === 'light' || value === 'dark' ? value : undefined;
  } catch {
    return undefined;
  }
}

function storeTheme(theme: GymTheme): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // The toggle still works for this page view without storage.
  }
}

/**
 * Follows the system colour scheme until the learner picks a theme, then
 * remembers that choice. The resolved theme is written to
 * `<html data-dojo-theme>`, which the gym's design tokens key off.
 */
export class ThemeToggle {
  readonly node = document.createElement('button');
  private readonly media = window.matchMedia('(prefers-color-scheme: dark)');
  private explicit = storedTheme();

  constructor() {
    this.node.type = 'button';
    this.node.className = 'dojo-theme-toggle';
    this.node.onclick = () => {
      this.explicit = opposite(this.current);
      storeTheme(this.explicit);
      this.apply();
    };
    this.media.addEventListener('change', () => this.apply());
    this.apply();
  }

  get current(): GymTheme {
    return this.explicit ?? (this.media.matches ? 'dark' : 'light');
  }

  private apply(): void {
    const theme = this.current;
    document.documentElement.dataset.dojoTheme = theme;
    const next = opposite(theme);
    this.node.setAttribute('aria-label', `Switch to ${next} theme`);
    this.node.title = `Switch to ${next} theme`;
    this.node.textContent = theme === 'dark' ? '☾' : '☀';
  }
}
