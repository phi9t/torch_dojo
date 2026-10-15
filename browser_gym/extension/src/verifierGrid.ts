import { element, replayCode, sampleCard } from './cases';
import { failurePlaces, type FailureSources } from './failure';
import type { CheckResult } from './checker';
import { stageName, type StageOrder } from './seals';
import { buildSquares, firstFailing, type VerifierSquare } from './squares';

export type LoadCases = (source: string, caseIndex: number) => Promise<void>;

export interface ShownResult {
  /** The SHA-256 of the checked source. */
  sourceHash?: string;
  /** Whether the answer has changed since this check. */
  stale?: boolean;
  /** For independent checkpoints, the latest results of the other checkpoints. */
  others?: CheckResult[];
  /** The checked answer's source, so a failure inside it can point at its line. */
  submitted?: string;
}

function verifierLabel(name: string): string {
  return name.replace(/^verify_/, '').replace(/_/g, ' ');
}

const STATE_WORDS: Record<VerifierSquare['state'], string> = {
  pass: 'passed',
  fail: 'failed',
  'not-run': 'not run',
  locked: 'not reached'
};

/**
 * Check results as one square per verifier, grouped by stage, with a detail
 * panel for the selected square: its error and the captured case behind it.
 */
export class VerifierGrid {
  readonly node = element('section', 'dojo-verifier-grid');
  private verifiers: Record<string, string[]> = {};
  private squares: VerifierSquare[] = [];
  private selected?: VerifierSquare;
  private cases: unknown[] = [];
  private caseOffsets = new Map<VerifierSquare, number>();
  private sourceHash = '';
  private key = '';
  private resultKey = '';
  private shown?: { result: CheckResult } & ShownResult;
  private sources: Omit<FailureSources, 'submitted'> = { verifiers: {} };
  private submitted?: string;
  private readonly squaresNode = element('div', 'dojo-square-groups');
  private readonly detail = element('section', 'dojo-verifier-detail');

  constructor(
    private readonly stages: string[],
    private readonly order: StageOrder,
    private readonly load: LoadCases
  ) {
    this.node.setAttribute('aria-label', 'Check results');
    this.detail.setAttribute('aria-label', 'Verifier detail');
    this.squaresNode.addEventListener('keydown', event => this.onKey(event));
    this.showIdle();
  }

  /** The verifier names of each stage, from the checker bundle. */
  setVerifiers(verifiers: Record<string, string[]>): void {
    this.verifiers = verifiers;
    this.key = '';
    this.resultKey = '';
    this.restore();
  }

  /** The verifiers' and their helpers' source, so a failure can point at its line. */
  setSources(sources: Omit<FailureSources, 'submitted'>): void {
    this.sources = sources;
    this.key = '';
    this.restore();
  }

  /** Shows the last result again, e.g. after a check that never reached the worker. */
  restore(): void {
    if (this.shown) this.showResult(this.shown.result, this.shown);
    else this.showIdle();
  }

  showIdle(): void {
    this.key = 'idle';
    this.node.replaceChildren(element('p', 'dojo-grid-quiet', 'Check your answer to see each verifier here.'));
  }

  showRunning(stage: string): void {
    this.key = `running:${stage}`;
    const line = element('p', 'dojo-grid-quiet dojo-grid-running');
    line.setAttribute('aria-live', 'polite');
    line.append(element('span', 'dojo-spinner'), document.createTextNode(`Checking ${stageName(stage).toLowerCase()} in a fresh worker… Stop with ⌘/Ctrl+.`));
    this.node.replaceChildren(line);
  }

  /** Shows a check result and how it relates to the code it checked. */
  showResult(result: CheckResult | undefined, shown: ShownResult = {}): void {
    if (!result) {
      this.showIdle();
      return;
    }
    const { sourceHash = '', stale = false, others = [], submitted } = shown;
    const key = JSON.stringify([result, sourceHash, stale, this.verifiers, others, submitted]);
    if (key === this.key) return;
    this.key = key;
    this.shown = { result, sourceHash, stale, others, submitted };
    this.submitted = submitted;
    // Editing after a check only marks it stale; keep what the learner selected.
    const resultKey = JSON.stringify([result, sourceHash, this.verifiers]);
    const keepSelection = resultKey === this.resultKey ? this.selected?.name : undefined;
    this.resultKey = resultKey;
    this.sourceHash = sourceHash;
    this.squares = buildSquares(this.stages, this.verifiers, result, this.order, others);
    this.cases = [];
    this.caseOffsets.clear();
    for (const square of this.squares) {
      const samples = square.check?.samples ?? [];
      if (!samples.length) continue;
      this.caseOffsets.set(square, this.cases.length);
      this.cases.push(...samples.map(sample => ({ ...sample, check: square.name, check_status: square.check?.status })));
    }

    const count = (state: VerifierSquare['state']) => this.squares.filter(square => square.state === state).length;
    const totals = element('div', 'dojo-case-totals');
    totals.append(element('span', 'dojo-badge dojo-pass', `${count('pass')} passed`), element('span', 'dojo-badge dojo-fail', `${count('fail')} failed`));
    if (count('locked')) totals.append(element('span', 'dojo-badge', `${count('locked')} not reached`));
    if (stale) totals.append(element('span', 'dojo-badge dojo-stale', 'Code changed since this check'));
    const caption = element('p', 'dojo-case-caption', stale ? 'Your answer has changed. Check it again when you’re ready; these results are from the earlier version.'
      : count('fail') ? 'Start with one failing check. Inspect its example below, try it in scratch, then revise your answer.'
      : result.status === 'passed' ? (count('locked')
        ? 'This stage passed. Explore a passing example, or choose the next stage when you’re ready.'
        : 'All checks passed. Explore a passing example, or start a fresh rep when you’re ready.')
      : 'Your answer is saved. Inspect the message below, then check again when you’re ready.');
    this.node.replaceChildren(totals, caption);
    if (result.error) this.node.append(element('pre', 'dojo-case-error', result.error));
    this.node.append(this.squaresNode, this.detail);
    const kept = keepSelection === undefined ? undefined : this.squares.find(square => square.name === keepSelection);
    this.select(kept ?? (keepSelection === undefined ? firstFailing(this.squares) : undefined));
  }

  private renderSquares(focusTarget = this.selected ?? this.squares.find(square => square.state !== 'locked')): void {
    this.squaresNode.replaceChildren();
    for (const stage of this.stages) {
      const own = this.squares.filter(square => square.stage === stage);
      if (!own.length) continue;
      const group = element('div', 'dojo-square-group');
      group.setAttribute('role', 'group');
      group.setAttribute('aria-label', stageName(stage));
      const passed = own.filter(square => square.state === 'pass').length;
      const heading = element('div', 'dojo-square-heading');
      heading.append(element('span', '', stageName(stage)), element('span', 'dojo-square-count', `${passed}/${own.length}`));
      const row = element('div', 'dojo-square-row');
      for (const square of own) {
        const label = `${verifierLabel(square.name)}: ${STATE_WORDS[square.state]}`;
        if (square.state === 'locked') {
          const locked = element('span', 'dojo-square');
          locked.dataset.state = square.state;
          locked.setAttribute('role', 'img');
          locked.setAttribute('aria-label', label);
          locked.title = label;
          row.append(locked);
          continue;
        }
        const button = element('button', 'dojo-square');
        button.type = 'button';
        button.dataset.state = square.state;
        button.setAttribute('aria-label', label);
        button.title = label;
        button.setAttribute('aria-pressed', String(square === this.selected));
        button.tabIndex = square === focusTarget ? 0 : -1;
        button.dataset.name = square.name;
        button.onclick = () => this.select(square === this.selected ? undefined : square, square);
        row.append(button);
      }
      group.append(heading, row);
      this.squaresNode.append(group);
    }
  }

  /** Selects a square (or none). `focus` is the square that keeps keyboard focus, if any. */
  private select(square: VerifierSquare | undefined, focus?: VerifierSquare): void {
    this.selected = square;
    this.renderSquares(focus);
    this.renderDetail();
    if (focus) this.squaresNode.querySelector<HTMLButtonElement>('button[tabindex="0"]')?.focus();
  }

  private onKey(event: KeyboardEvent): void {
    const buttons = Array.from(this.squaresNode.querySelectorAll<HTMLButtonElement>('button.dojo-square'));
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    const target = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : step === undefined ? undefined : Math.min(buttons.length - 1, Math.max(0, index + step));
    if (target === undefined) return;
    event.preventDefault();
    buttons[index].tabIndex = -1;
    buttons[target].tabIndex = 0;
    buttons[target].focus();
  }

  /** The check line that failed, and the answer line too when the answer raised. */
  private renderWhereItFailed(traceback: string): void {
    const places = failurePlaces(traceback, { ...this.sources, submitted: this.submitted });
    if (!places.length) return;
    const section = element('section', 'dojo-failure-places');
    section.setAttribute('aria-label', 'Where it failed');
    section.append(element('h5', 'dojo-detail-heading', 'Where it failed'));
    for (const place of places) {
      const block = element('figure', 'dojo-failure-place');
      block.append(element('figcaption', '', place.label));
      const code = element('pre', 'dojo-failure-code');
      for (const line of place.lines) {
        const row = element('span', line.failing ? 'dojo-failure-line dojo-failing-line' : 'dojo-failure-line',
          `${String(line.number).padStart(3)}  ${line.text}\n`);
        if (line.failing) row.setAttribute('aria-label', `Failing line ${line.number}: ${line.text.trim()}`);
        code.append(row);
      }
      block.append(code);
      section.append(block);
    }
    this.detail.append(section);
  }

  /** The check's own variables at the failing line: the example it built and the values it compared. */
  private renderCheckVariables(check: VerifierSquare['check']): void {
    const locals = Object.entries(check?.locals ?? {});
    if (!locals.length) return;
    const section = element('section', 'dojo-check-variables');
    section.setAttribute('aria-label', 'The check’s variables when it failed');
    section.append(element('h5', 'dojo-detail-heading', 'The check’s variables when it failed'));
    const list = element('dl', '');
    for (const [name, value] of locals) list.append(element('dt', '', name), element('dd', '', value));
    section.append(list);
    this.detail.append(section);
  }

  private renderDetail(): void {
    const square = this.selected;
    this.detail.replaceChildren();
    this.detail.hidden = !square;
    if (!square) return;
    const check = square.check;
    const title = element('h4', 'dojo-detail-title', verifierLabel(square.name));
    const meta = element('p', 'dojo-detail-meta', `${stageName(square.stage)} · ${STATE_WORDS[square.state]}` +
      (square.elapsedMs !== undefined ? ` · ${square.elapsedMs} ms` : '') + (square.earlier ? ' · from an earlier check' : ''));
    this.detail.append(title, meta);
    for (const substitution of square.substitutions ?? []) {
      // A pass that leaned on a reference component must never read as a full pass.
      this.detail.append(element('p', 'dojo-detail-substitution',
        `Reference substitution: this check used the reference ${substitution.function} ` +
        `(${stageName(substitution.checkpoint)} checkpoint)` +
        (substitution.independent_credit ? '.' : ', so its pass is not independent credit.')));
    }
    if (check?.error) {
      // A bare assert says nothing on its own; the line below shows which one.
      const message = /^AssertionError:\s*$/.test(check.error) ? 'AssertionError: the highlighted assertion was false' : check.error;
      this.detail.append(element('pre', 'dojo-case-error', message));
    }
    if (check?.traceback) this.renderWhereItFailed(check.traceback);
    const samples = check?.samples ?? [];
    const offset = this.caseOffsets.get(square) ?? 0;
    if (samples.length) {
      const load = element('button', 'dojo-load-cases', 'Load case into scratch');
      load.type = 'button';
      const notice = element('span', 'dojo-case-caption');
      notice.setAttribute('aria-live', 'polite');
      // The last call is the one the check was making, or had just made, when it failed.
      const failing = square.state === 'fail' ? samples.length - 1 : -1;
      load.onclick = async () => {
        load.disabled = true;
        notice.textContent = 'Loading examples…';
        try {
          await this.load(replayCode(this.cases, this.sourceHash), offset + Math.max(failing, 0));
          notice.textContent = `Ready: gym_case(${offset + Math.max(failing, 0)}) · gym_try(${offset + Math.max(failing, 0)}) · gym_cases`;
        } catch (error) {
          notice.textContent = String(error);
        } finally {
          load.disabled = false;
        }
      };
      const actions = element('div', 'dojo-case-toolbar');
      actions.append(load, notice);
      if (failing >= 0) {
        const section = element('section', 'dojo-failing-example');
        section.setAttribute('aria-label', 'The failing example');
        section.append(element('h5', 'dojo-detail-heading', 'The failing example: the last call before the check failed'),
          sampleCard(samples[failing], offset + failing));
        this.detail.append(section);
      }
      this.renderCheckVariables(check);
      this.detail.append(actions);
      const earlier = samples.map((sample, index) => ({ sample, index })).filter(({ index }) => index !== failing);
      if (earlier.length) {
        const holder = failing >= 0 ? element('details', 'dojo-earlier-calls') : this.detail;
        if (holder instanceof HTMLDetailsElement) {
          holder.append(element('summary', '', `Earlier calls in this check (${earlier.length})`));
          this.detail.append(holder);
        }
        for (const { sample, index } of earlier) holder.append(sampleCard(sample, offset + index));
      }
      if ((check?.sample_calls ?? samples.length) > samples.length) {
        this.detail.append(element('p', 'dojo-case-caption', `Showing ${samples.length} of ${check?.sample_calls} calls: the first three and latest three per function. Large or unsupported inputs are preview only.`));
      }
    } else if (check) {
      this.renderCheckVariables(check);
      this.detail.append(element('p', 'dojo-case-caption', 'No replayable call samples were captured for this check.'));
    }
    if (check?.evidence) this.detail.append(element('pre', 'dojo-case-evidence', JSON.stringify(check.evidence, null, 2)));
  }
}
