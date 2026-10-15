import { emacsOwnsKey } from './editor';
import { matchesWorkout, sinceLastRep } from './home';

export interface SwitcherWorkout {
  id: string;
  title: string;
  topics: string[];
  lastRepAt?: string;
}

/**
 * ⌘/Ctrl+K: a dialog listing every workout, filtered as the learner types and
 * operated from the keyboard (arrows move, Enter opens, Escape closes). It
 * listens on the window in the capture phase, ahead of JupyterLab's own
 * shortcuts, so it also opens while typing a drill prediction.
 */
export class WorkoutSwitcher {
  readonly node = document.createElement('dialog');
  private readonly input = document.createElement('input');
  private readonly list = document.createElement('ul');
  private shown: SwitcherWorkout[] = [];
  private selected = 0;

  constructor(
    private readonly workouts: SwitcherWorkout[],
    private readonly open: (id: string) => void
  ) {
    this.node.className = 'dojo-switcher';
    this.node.setAttribute('aria-label', 'Switch workout');
    this.input.type = 'text';
    this.input.setAttribute('role', 'combobox');
    this.input.setAttribute('aria-label', 'Find a workout');
    this.input.setAttribute('aria-controls', 'torch-dojo-switcher-list');
    this.input.setAttribute('aria-expanded', 'true');
    this.input.setAttribute('aria-autocomplete', 'list');
    this.input.setAttribute('data-lm-suppress-shortcuts', '');
    this.input.placeholder = 'Workout id, title or topic';
    this.list.id = 'torch-dojo-switcher-list';
    this.list.setAttribute('role', 'listbox');
    this.list.setAttribute('aria-label', 'Workouts');
    this.node.append(this.input, this.list);
    this.input.addEventListener('input', () => this.render(0));
    this.input.addEventListener('keydown', event => this.onKey(event));
    this.node.addEventListener('click', event => {
      if (event.target === this.node) this.close();
    });
    document.body.append(this.node);
    window.addEventListener('keydown', event => {
      if (event.key.toLowerCase() !== 'k' || !(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return;
      // With Emacs keys on, Ctrl+K in an editor deletes to the end of the line.
      if (emacsOwnsKey(event)) return;
      event.preventDefault();
      event.stopPropagation();
      if (this.node.open) this.close();
      else this.show();
    }, true);
  }

  show(): void {
    this.input.value = '';
    this.render(0);
    this.node.showModal();
    this.input.focus();
  }

  close(): void {
    this.node.close();
  }

  private onKey(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      if (this.shown.length) this.render((this.selected + step + this.shown.length) % this.shown.length);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const workout = this.shown[this.selected];
      if (workout) this.choose(workout.id);
    }
  }

  private choose(id: string): void {
    this.close();
    this.open(id);
  }

  private render(selected: number): void {
    this.shown = this.workouts.filter(workout => matchesWorkout(workout, this.input.value));
    this.selected = Math.min(selected, Math.max(0, this.shown.length - 1));
    const now = new Date();
    this.list.replaceChildren(...this.shown.map((workout, index) => {
      const option = document.createElement('li');
      option.id = `torch-dojo-switcher-${workout.id}`;
      option.setAttribute('role', 'option');
      option.setAttribute('aria-selected', String(index === this.selected));
      const name = document.createElement('span');
      name.textContent = `${workout.id} · ${workout.title}`;
      const since = document.createElement('span');
      since.className = 'dojo-switcher-since';
      since.textContent = workout.lastRepAt ? `last rep ${sinceLastRep(workout.lastRepAt, now)}` : 'never repped';
      option.append(name, ' ', since);
      option.onclick = () => this.choose(workout.id);
      return option;
    }));
    if (!this.shown.length) {
      const empty = document.createElement('li');
      empty.className = 'dojo-switcher-empty';
      empty.textContent = 'No workout matches.';
      this.list.append(empty);
    }
    const active = this.shown[this.selected];
    if (active) {
      this.input.setAttribute('aria-activedescendant', `torch-dojo-switcher-${active.id}`);
      document.getElementById(`torch-dojo-switcher-${active.id}`)?.scrollIntoView({ block: 'nearest' });
    } else {
      this.input.removeAttribute('aria-activedescendant');
    }
  }
}
