import { formatDuration } from './splits';

const IDLE_MS = 2 * 60 * 1000;
/** Longest gap one tick may add, so a sleeping laptop never counts as practice. */
const MAX_TICK_MS = 2000;
const HIDDEN_KEY = 'torch-dojo-timer-hidden';

const INPUT_EVENTS = ['keydown', 'pointerdown', 'pointermove', 'wheel', 'input'];

function spoken(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes} minute${minutes === 1 ? '' : 's'} ${secs} second${secs === 1 ? '' : 's'}`;
}

/**
 * The rep timer. It counts only active time: the page is visible, the learner
 * gave input within the last two minutes, and the rep's final stage has not
 * passed yet. It shows quietly in the rail and can be hidden.
 */
export class ActiveTimer {
  readonly node = document.createElement('div');
  private readonly display = document.createElement('span');
  private readonly toggle = document.createElement('button');
  private seconds = 0;
  private stopped = false;
  private paused = false;
  private lastTick = Date.now();
  private lastInput = Date.now();

  constructor() {
    this.node.className = 'dojo-timer';
    this.display.className = 'dojo-timer-value';
    this.display.setAttribute('role', 'timer');
    this.toggle.type = 'button';
    this.toggle.className = 'dojo-timer-toggle';
    this.toggle.onclick = () => this.setHidden(!this.display.hidden);
    this.node.append(this.display, this.toggle);
    let hidden = false;
    try {
      hidden = window.localStorage.getItem(HIDDEN_KEY) === 'true';
    } catch {
      // The timer shows when its preference can't be read.
    }
    this.setHidden(hidden, false);
    const onInput = () => { this.lastInput = Date.now(); };
    for (const type of INPUT_EVENTS) document.addEventListener(type, onInput, { capture: true, passive: true });
    document.addEventListener('visibilitychange', () => { this.lastTick = Date.now(); });
    window.setInterval(() => this.tick(), 1000);
    this.render();
  }

  get value(): number {
    return this.seconds;
  }

  get isStopped(): boolean {
    return this.stopped;
  }

  /** Starts showing a rep: its saved active time, and whether it is already finished. */
  reset(seconds: number, finished: boolean): void {
    this.seconds = seconds;
    this.stopped = finished;
    this.lastTick = Date.now();
    this.lastInput = Date.now();
    this.render();
  }

  /** Loading or recovering a rep is waiting time, not practice time. */
  setPaused(paused: boolean): void {
    if (this.paused === paused) return;
    this.paused = paused;
    this.lastTick = Date.now();
  }

  private tick(): void {
    const now = Date.now();
    const elapsed = Math.min(now - this.lastTick, MAX_TICK_MS);
    this.lastTick = now;
    const active = !this.paused && !this.stopped && document.visibilityState === 'visible' && now - this.lastInput < IDLE_MS;
    if (active && elapsed > 0) {
      this.seconds += elapsed / 1000;
      this.render();
    } else if (this.node.dataset.paused !== String(!active)) {
      this.render();
    }
    this.node.dataset.paused = String(!active);
  }

  private render(): void {
    this.display.textContent = formatDuration(this.seconds);
    this.display.setAttribute('aria-label', `Active time ${spoken(this.seconds)}${this.stopped ? ', final' : ''}`);
    this.node.dataset.final = String(this.stopped);
  }

  private setHidden(hidden: boolean, remember = true): void {
    this.display.hidden = hidden;
    this.toggle.textContent = hidden ? 'Show timer' : 'Hide';
    this.toggle.setAttribute('aria-label', hidden ? 'Show timer' : 'Hide timer');
    if (!remember) return;
    try {
      window.localStorage.setItem(HIDDEN_KEY, String(hidden));
    } catch {
      // Hiding still works for this view.
    }
  }
}
