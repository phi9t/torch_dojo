import { CodeCell, type Cell } from '@jupyterlab/cells';
import type { NotebookPanel } from '@jupyterlab/notebook';
import { MessageLoop } from '@lumino/messaging';
import { Panel, Widget } from '@lumino/widgets';
import { paneForCell, type Pane } from './cellPanes';
import type { StageOrder } from './seals';
import { keys, SHORTCUTS, withShortcut } from './shortcuts';
import { emacsKeysEnabled, setEmacsKeys } from './editor';

export type BenchTab = 'brief' | 'drills' | 'explore' | 'reps';

const TABS: Array<{ id: BenchTab; label: string }> = [
  { id: 'brief', label: 'Brief' },
  { id: 'drills', label: 'Drills' },
  { id: 'explore', label: 'Explore' },
  { id: 'reps', label: 'Reps' }
];

/** The cells each tab shows from the problem view (Reps shows none). */
const TAB_PANES: Record<BenchTab, Pane | undefined> = { brief: 'brief', drills: 'drill', explore: 'explore', reps: undefined };

/** A drill's card text from the catalog. Its probe is the notebook's drill cell. */
export interface DrillCard {
  title: string;
  prompt_html: string;
  reasoning_html: string;
}

export interface BenchOptions {
  exerciseId: string;
  title: string;
  topics: string[];
  stages: string[];
  /** Cumulative stages unlock in order; a lab's checkpoints pass in any order. */
  stageOrder: StageOrder;
  drills: DrillCard[];
  /** Typesets `\(…\)` math in card text the way the notebook typesets markdown. */
  typeset: (node: HTMLElement) => void;
  /** Called as the learner types a prediction. */
  onPrediction: (drill: number, text: string) => void;
}

function cellPane(cell: Cell): Pane {
  const dojo = cell.model.metadata.dojo as { role?: string } | undefined;
  return paneForCell({ cellType: cell.model.type, source: cell.model.sharedModel.getSource(), role: dojo?.role });
}

function readStoredTab(key: string): BenchTab | undefined {
  try {
    const value = window.localStorage.getItem(key);
    return TABS.some(tab => tab.id === value) ? value as BenchTab : undefined;
  } catch {
    return undefined;
  }
}

const EXPANDED_KEY = 'torch-dojo-answer-expanded';

function readExpanded(): boolean {
  try {
    return window.localStorage.getItem(EXPANDED_KEY) === 'true';
  } catch {
    return false;
  }
}

function storeExpanded(expanded: boolean): void {
  try {
    window.localStorage.setItem(EXPANDED_KEY, String(expanded));
  } catch {
    // Expanding still works for this page view without storage.
  }
}

function storeTab(key: string, tab: BenchTab): void {
  try {
    window.localStorage.setItem(key, tab);
  } catch {
    // The tab still switches; it just isn't remembered.
  }
}

/** Hosts one notebook view and keeps its Lumino layout sized to the CSS box it sits in. */
class ViewHost extends Panel {
  private readonly observer = new ResizeObserver(() => {
    for (const widget of this.widgets) MessageLoop.sendMessage(widget, Widget.ResizeMessage.UnknownSize);
  });

  constructor(className: string) {
    super();
    this.addClass(className);
    this.observer.observe(this.node);
  }

  dispose(): void {
    this.observer.disconnect();
    super.dispose();
  }
}

/**
 * The workout page: a side rail, a left pane with one tab visible at a time,
 * and a right pane holding only the answer cell. Both panes are views of the
 * same notebook model, each showing only the cells that belong to it.
 */
export class Bench extends Panel {
  /** Workout identity, stage seals and quiet controls. Filled by the caller. */
  readonly rail = document.createElement('aside');
  /** Check and Run controls above the answer cell. Filled by the caller. */
  readonly answerActions = document.createElement('div');
  /** The Reps tab's content. Filled by the caller. */
  readonly repsPanel = document.createElement('div');
  /** Opens and closes the scratch cells under the answer. Placed by the caller. */
  readonly scratchToggle = document.createElement('button');
  /** Runs the answer and then the scratch cells. Placed by the caller, beside the scratch toggle. */
  readonly scratchRun = document.createElement('button');
  /** Gives the answer the whole page beside the rail. */
  readonly expandToggle = document.createElement('button');
  /** Says how scratch runs, shown while the scratch cells are open. Placed by the caller. */
  readonly scratchNote = document.createElement('p');
  /** Opens the list of keyboard shortcuts. Placed by the caller, beside Run scratch. */
  readonly shortcutsToggle = document.createElement('button');
  /** The keyboard shortcuts. Placed by the caller, under the scratch controls. */
  readonly shortcutsPanel = document.createElement('dl');
  /** Links from the brief to help. Filled by the caller; the Bench keeps it under the brief. */
  readonly briefFooter = document.createElement('nav');

  private readonly tabKey: string;
  private readonly tabButtons = new Map<BenchTab, HTMLButtonElement>();
  private readonly tabCounts = new Map<BenchTab, HTMLSpanElement>();
  private readonly problemHost = new ViewHost('dojo-bench-viewhost');
  private readonly answerHost = new ViewHost('dojo-bench-viewhost');
  private readonly repsWidget: Widget;
  private readonly observers: MutationObserver[] = [];
  private readonly controlledCells = new WeakSet<Cell>();
  private readonly drills: DrillCard[];
  private readonly typeset: (node: HTMLElement) => void;
  private readonly onPrediction: (drill: number, text: string) => void;
  private readonly predictionFields = new Map<number, HTMLTextAreaElement>();
  private predictions: Record<string, string> = {};
  private readonly mutedCells = new WeakSet<Cell>();
  private problemView?: NotebookPanel;
  private answerView?: NotebookPanel;
  private tab: BenchTab;
  private scratchOpen = false;
  private expanded = readExpanded();
  private problemPane!: Panel;
  private readonly stages: string[];
  private readonly stageOrder: StageOrder;
  private briefStages = { passed: new Set<string>(), current: '' };
  private refreshScheduled = false;

  constructor(options: BenchOptions) {
    super();
    this.id = 'torch-dojo-bench';
    this.addClass('dojo-bench');
    this.addClass('dojo-gym-ui');
    this.title.label = `${options.exerciseId} · ${options.title}`;
    this.title.closable = false;
    this.tabKey = `torch-dojo-tab:${options.exerciseId}`;
    this.stages = options.stages;
    this.stageOrder = options.stageOrder;
    this.drills = options.drills;
    this.typeset = options.typeset;
    this.onPrediction = options.onPrediction;
    this.briefStages = { passed: new Set(), current: options.stages[0] ?? '' };
    this.briefFooter.className = 'dojo-brief-footer';
    this.briefFooter.setAttribute('aria-label', 'Help with this workout');
    this.tab = readStoredTab(this.tabKey) ?? 'brief';
    if (!this.hasTab(this.tab)) this.tab = 'brief';

    this.rail.className = 'dojo-bench-rail';
    this.rail.setAttribute('aria-label', 'Workout');
    this.addWidget(new Widget({ node: this.rail }));

    const problem = new Panel();
    problem.addClass('dojo-bench-pane');
    problem.addClass('dojo-bench-problem');
    problem.node.setAttribute('role', 'region');
    problem.node.setAttribute('aria-label', 'Problem');
    problem.addWidget(new Widget({ node: this.problemHeader(options) }));
    this.problemHost.node.setAttribute('role', 'tabpanel');
    problem.addWidget(this.problemHost);
    this.repsPanel.className = 'dojo-bench-reps';
    this.repsPanel.setAttribute('role', 'tabpanel');
    this.repsPanel.id = 'torch-dojo-tab-panel-reps';
    this.repsPanel.setAttribute('aria-labelledby', 'torch-dojo-tab-reps');
    this.repsWidget = new Widget({ node: this.repsPanel });
    problem.addWidget(this.repsWidget);
    this.addWidget(problem);
    this.problemPane = problem;

    const answer = new Panel();
    answer.addClass('dojo-bench-pane');
    answer.addClass('dojo-bench-answer');
    answer.node.setAttribute('role', 'region');
    answer.node.setAttribute('aria-label', 'Answer');
    const answerHeader = document.createElement('header');
    answerHeader.className = 'dojo-answer-header';
    const answerLabel = document.createElement('div');
    answerLabel.className = 'dojo-answer-label';
    const answerTitle = document.createElement('h2');
    answerTitle.textContent = 'Answer cell';
    const answerNote = document.createElement('p');
    answerNote.textContent = 'The only cell that is checked.';
    answerLabel.append(answerTitle, answerNote);
    this.answerActions.className = 'dojo-answer-actions';
    this.expandToggle.type = 'button';
    this.expandToggle.className = 'dojo-expand-toggle';
    this.expandToggle.onclick = () => this.setExpanded(!this.expanded);
    answerHeader.append(answerLabel, this.expandToggle, this.answerActions);
    answer.addWidget(new Widget({ node: answerHeader }));
    answer.addWidget(this.answerHost);
    this.addWidget(answer);

    this.scratchToggle.type = 'button';
    this.scratchToggle.className = 'dojo-scratch-toggle';
    this.scratchToggle.setAttribute('aria-expanded', 'false');
    this.scratchToggle.onclick = () => this.setScratchOpen(!this.scratchOpen);
    this.scratchRun.type = 'button';
    this.scratchRun.className = 'dojo-scratch-run';
    this.scratchRun.textContent = 'Run scratch';
    this.scratchRun.title = withShortcut('Run the answer, then each scratch cell', 'Mod+Shift+Enter');
    this.scratchToggle.title = 'Show or hide scratch cells for trying your answer; they are never checked';
    this.scratchNote.className = 'dojo-scratch-note';
    this.scratchNote.textContent = `Run scratch (${keys('Mod+Shift+Enter')}) reruns your answer first, then each scratch cell, ` +
      `so scratch always sees your latest code. Shift+Enter runs only the cell you are in. Scratch cells are never checked.`;
    this.shortcutsToggle.type = 'button';
    this.shortcutsToggle.className = 'dojo-shortcuts-toggle';
    this.shortcutsToggle.textContent = 'Shortcuts';
    this.shortcutsToggle.title = 'Show the keyboard shortcuts';
    this.shortcutsPanel.className = 'dojo-shortcuts';
    this.shortcutsPanel.id = 'torch-dojo-shortcuts';
    this.shortcutsPanel.setAttribute('aria-label', 'Keyboard shortcuts');
    this.shortcutsPanel.hidden = true;
    for (const shortcut of SHORTCUTS) {
      const term = document.createElement('dt');
      term.append(...keys(shortcut.combo).split('+').flatMap((key, index) => {
        const kbd = document.createElement('kbd');
        kbd.textContent = key;
        return index ? ['+', kbd] : [kbd];
      }));
      const detail = document.createElement('dd');
      detail.textContent = shortcut.action;
      this.shortcutsPanel.append(term, detail);
    }
    // Emacs keys are opt-in: they take over Ctrl+A (select all) and Ctrl+K in editors.
    const emacsLabel = document.createElement('label');
    emacsLabel.className = 'dojo-emacs-toggle';
    const emacsBox = document.createElement('input');
    emacsBox.type = 'checkbox';
    emacsBox.id = 'torch-dojo-emacs-keys';
    emacsBox.checked = emacsKeysEnabled();
    emacsBox.onchange = () => setEmacsKeys(emacsBox.checked);
    emacsLabel.append(emacsBox, ' Emacs keys in the editor: Ctrl+A/E/F/B/N/P/D/K/Y, Alt+F/B/D. ' +
      'Ctrl+K then deletes to the end of the line inside an editor; the workout switcher still opens from outside it.');
    this.shortcutsPanel.append(emacsLabel);
    this.shortcutsToggle.setAttribute('aria-controls', this.shortcutsPanel.id);
    this.shortcutsToggle.setAttribute('aria-expanded', 'false');
    this.shortcutsToggle.onclick = () => {
      this.shortcutsPanel.hidden = !this.shortcutsPanel.hidden;
      this.shortcutsToggle.setAttribute('aria-expanded', String(!this.shortcutsPanel.hidden));
    };
    this.scratchRun.onclick = () => void this.runScratch();
    this.renderScratchToggle();
    this.setExpanded(this.expanded);
    this.setTabCount('drills', options.drills.length);
    this.showTab(this.tab);
  }

  get currentTab(): BenchTab {
    return this.tab;
  }

  /** The two views must share one notebook model (one a clone of the other). */
  setViews(problemView: NotebookPanel, answerView: NotebookPanel): void {
    for (const observer of this.observers.splice(0)) observer.disconnect();
    this.problemView = problemView;
    this.answerView = answerView;
    this.problemHost.addWidget(problemView);
    this.answerHost.addWidget(answerView);
    for (const view of [problemView, answerView]) {
      view.toolbar.hide();
      const observer = new MutationObserver(() => this.scheduleRefresh());
      observer.observe(view.content.node, { childList: true, subtree: true });
      this.observers.push(observer);
    }
    this.refresh();
  }

  /** Whether a notebook view is one of the Bench's two panes. */
  holds(widget: Widget | null | undefined): boolean {
    return Boolean(widget) && (widget === this.problemView || widget === this.answerView);
  }

  selectTab(tab: BenchTab): void {
    if (tab === this.tab) return;
    this.tab = tab;
    storeTab(this.tabKey, tab);
    this.showTab(tab);
  }

  setRepCount(count: number): void {
    this.setTabCount('reps', count);
  }

  /** Shows in the brief which stages have passed and which one is being checked. */
  setBriefStages(passed: Set<string>, current: string): void {
    this.briefStages = { passed: new Set(passed), current };
    this.markBriefStages();
  }

  /** Shows the active rep's saved predictions, keyed by drill index. */
  setPredictions(predictions: Record<string, string>): void {
    this.predictions = { ...predictions };
    for (const [index, field] of this.predictionFields) {
      const text = this.predictions[index] ?? '';
      if (field.value !== text && document.activeElement !== field) field.value = text;
    }
  }

  setScratchOpen(open: boolean): void {
    this.scratchOpen = open;
    this.renderScratchToggle();
    this.refresh();
  }

  /** Runs a drill or exploration cell once the kernel is ready; an early click would otherwise do nothing. */
  private async runProblemCell(cell: CodeCell): Promise<void> {
    const view = this.problemView;
    if (!view) return;
    await view.sessionContext.ready;
    await CodeCell.execute(cell, view.sessionContext);
  }

  /** Workouts without drills have no Drills tab. */
  private hasTab(tab: BenchTab): boolean {
    return tab !== 'drills' || this.drills.length > 0;
  }

  /** The problem pane's code cells of one pane kind. */
  private problemCodeCells(pane: Pane): CodeCell[] {
    return (this.problemView?.content.widgets ?? [])
      .filter((cell): cell is CodeCell => cell instanceof CodeCell && cellPane(cell) === pane);
  }

  /** Re-applies which cells each pane shows. Safe to call often. */
  refresh(): void {
    const shown = TAB_PANES[this.tab];
    this.filter(this.problemView, pane => pane === shown, pane => pane !== 'answer' && pane !== 'scratch');
    this.filter(
      this.answerView,
      pane => pane === 'answer' || (pane === 'scratch' && this.scratchOpen),
      pane => pane === 'answer' || pane === 'scratch'
    );
    const explore = this.problemCodeCells('explore');
    this.setTabCount('explore', explore.length);
    for (const cell of explore) this.addExploreControls(cell);
    for (const cell of this.problemCodeCells('drill')) this.addDrillCard(cell);
    const brief = this.problemView?.content.widgets.find(cell => cell.model.type === 'markdown' && cellPane(cell) === 'brief');
    if (brief && this.briefFooter.parentElement !== brief.node) brief.node.append(this.briefFooter);
    this.markBriefStages();
  }

  dispose(): void {
    for (const observer of this.observers.splice(0)) observer.disconnect();
    super.dispose();
  }

  /**
   * Emphasises the stage being checked, marks passed stages, and quiets later
   * ones, on the brief's stage sections (`dojo-stage-<stage>`). Idempotent, since
   * re-rendered markdown brings fresh sections.
   */
  private markBriefStages(): void {
    const { passed, current } = this.briefStages;
    const currentIndex = this.stages.indexOf(current);
    for (const stage of this.stages) {
      for (const section of this.problemView?.content.node.querySelectorAll<HTMLElement>(`.dojo-brief-stage.dojo-stage-${stage}`) ?? []) {
        this.markBriefStage(section, stage, passed, current, currentIndex);
      }
    }
  }

  private markBriefStage(section: HTMLElement, stage: string, passed: Set<string>, current: string, currentIndex: number): void {
    const state = passed.has(stage) ? 'passed' : stage === current ? 'current'
      : this.stageOrder === 'cumulative' && this.stages.indexOf(stage) > currentIndex ? 'later' : 'open';
    if (section.dataset.state !== state) section.dataset.state = state;
    const heading = section.querySelector('h3');
    if (!heading) return;
    let badge = heading.querySelector<HTMLElement>('.dojo-stage-badge');
    const text = state === 'passed' ? 'passed' : state === 'current' ? 'checking now' : '';
    if (!text) {
      badge?.remove();
      return;
    }
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'dojo-stage-badge';
      heading.append(badge);
    }
    if (badge.textContent !== text) badge.textContent = text;
  }

  private problemHeader(options: BenchOptions): HTMLElement {
    const header = document.createElement('header');
    header.className = 'dojo-problem-header';
    const identity = document.createElement('div');
    identity.className = 'dojo-problem-identity';
    const id = document.createElement('span');
    id.className = 'dojo-workout-id';
    id.textContent = options.exerciseId;
    const title = document.createElement('h1');
    title.textContent = options.title;
    identity.append(id, title);
    const topics = document.createElement('div');
    topics.className = 'dojo-concepts';
    for (const topic of options.topics) {
      const chip = document.createElement('span');
      chip.textContent = topic;
      topics.append(chip);
    }

    const tablist = document.createElement('div');
    tablist.className = 'dojo-tablist';
    tablist.setAttribute('role', 'tablist');
    tablist.setAttribute('aria-label', 'Workout views');
    for (const { id: tab, label } of TABS) {
      const button = document.createElement('button');
      button.type = 'button';
      button.id = `torch-dojo-tab-${tab}`;
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-controls', tab === 'reps' ? 'torch-dojo-tab-panel-reps' : 'torch-dojo-tab-panel-notebook');
      button.textContent = label;
      const count = document.createElement('span');
      count.className = 'dojo-tab-count';
      count.hidden = true;
      button.append(count);
      button.onclick = () => this.selectTab(tab);
      button.hidden = !this.hasTab(tab);
      this.tabButtons.set(tab, button);
      this.tabCounts.set(tab, count);
      tablist.append(button);
    }
    tablist.addEventListener('keydown', event => this.onTabKey(event));
    this.problemHost.id = 'torch-dojo-tab-panel-notebook';
    header.append(identity, topics, tablist);
    return header;
  }

  private onTabKey(event: KeyboardEvent): void {
    const order = TABS.map(tab => tab.id).filter(tab => !this.tabButtons.get(tab)?.hidden);
    const index = order.indexOf(this.tab);
    const next = {
      ArrowRight: order[(index + 1) % order.length],
      ArrowLeft: order[(index - 1 + order.length) % order.length],
      Home: order[0],
      End: order[order.length - 1]
    }[event.key];
    if (!next) return;
    event.preventDefault();
    this.selectTab(next);
    this.tabButtons.get(next)?.focus();
  }

  private showTab(tab: BenchTab): void {
    for (const [id, button] of this.tabButtons) {
      const selected = id === tab;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    this.problemHost.node.setAttribute('aria-labelledby', `torch-dojo-tab-${tab === 'reps' ? 'brief' : tab}`);
    this.problemHost.setHidden(tab === 'reps');
    this.repsWidget.setHidden(tab !== 'reps');
    this.refresh();
  }

  private setTabCount(tab: BenchTab, count: number): void {
    const node = this.tabCounts.get(tab);
    if (!node) return;
    node.hidden = count === 0;
    node.textContent = count ? String(count) : '';
  }

  private renderScratchToggle(): void {
    this.scratchToggle.setAttribute('aria-expanded', String(this.scratchOpen));
    this.scratchToggle.textContent = this.scratchOpen ? 'Hide scratch cell' : 'Scratch cell';
    this.scratchRun.hidden = !this.scratchOpen;
    this.scratchNote.hidden = !this.scratchOpen;
  }

  /** Shows the answer across the whole page beside the rail, or beside the problem again. */
  setExpanded(expanded: boolean): void {
    this.expanded = expanded;
    this.toggleClass('dojo-answer-expanded', expanded);
    this.problemPane.setHidden(expanded);
    this.expandToggle.setAttribute('aria-pressed', String(expanded));
    this.expandToggle.textContent = expanded ? 'Show problem' : 'Expand';
    this.expandToggle.title = withShortcut(expanded ? 'Show the problem beside the answer' : 'Give the answer the whole page', 'Mod+Shift+E');
    storeExpanded(expanded);
  }

  toggleExpanded(): void {
    this.setExpanded(!this.expanded);
  }

  /**
   * Runs the answer cell and then each scratch cell, in order, so scratch code
   * always sees the latest answer. Stops at the first cell that raises.
   */
  async runScratch(): Promise<void> {
    const view = this.answerView;
    if (!view) return;
    this.setScratchOpen(true);
    await view.sessionContext.ready;
    const cells = view.content.widgets.filter((cell): cell is CodeCell =>
      cell instanceof CodeCell && (cellPane(cell) === 'answer' || cellPane(cell) === 'scratch'));
    for (const cell of cells) {
      const reply = await CodeCell.execute(cell, view.sessionContext);
      if (reply && reply.content.status !== 'ok') break;
    }
  }

  private scheduleRefresh(): void {
    if (this.refreshScheduled) return;
    this.refreshScheduled = true;
    queueMicrotask(() => {
      this.refreshScheduled = false;
      if (!this.isDisposed) this.refresh();
    });
  }

  /**
   * Shows a view's cells for the current state, and stops the view rendering
   * outputs of cells it never shows. Otherwise both views would render every
   * output, duplicating element ids and running exploration scripts twice.
   */
  private filter(view: NotebookPanel | undefined, visible: (pane: Pane) => boolean, owns: (pane: Pane) => boolean): void {
    for (const cell of view?.content.widgets ?? []) {
      const pane = cellPane(cell);
      const hidden = !visible(pane);
      if (cell.isHidden !== hidden) cell.setHidden(hidden);
      // The answer editor gets room to work in; see .dojo-answer-cell.
      if (view === this.answerView && pane === 'answer' && cell instanceof CodeCell) cell.addClass('dojo-answer-cell');
      if (cell instanceof CodeCell && !owns(pane) && !this.mutedCells.has(cell)) {
        this.mutedCells.add(cell);
        cell.outputArea.dispose();
      }
    }
  }

  /**
   * A drill is a card around its notebook cell: the prompt, a field for the
   * learner's prediction, Run (the probe, in the notebook kernel), and the
   * author's reasoning on request. Drills are never checked.
   */
  private addDrillCard(cell: CodeCell): void {
    if (this.controlledCells.has(cell)) return;
    // The build numbers drill cells from 0; authors write `# Drill 1:` for index 0.
    const index = (cell.model.metadata.dojo as { drill?: number } | undefined)?.drill ?? -1;
    const drill = this.drills[index];
    if (!drill) return;
    this.controlledCells.add(cell);
    cell.addClass('dojo-drill-cell');
    const card = document.createElement('div');
    card.className = 'dojo-drill-card';
    const title = document.createElement('h3');
    title.textContent = drill.title;
    // Named by its title; JupyterLab rewrites the cell's own aria-label as outputs change.
    title.id = `torch-dojo-drill-${index}`;
    cell.node.setAttribute('role', 'group');
    cell.node.setAttribute('aria-labelledby', title.id);
    const prompt = document.createElement('div');
    prompt.className = 'dojo-drill-prompt';
    prompt.innerHTML = drill.prompt_html;
    const label = document.createElement('label');
    label.textContent = 'Your prediction';
    const field = document.createElement('textarea');
    field.rows = 2;
    // Typing a prediction must not trigger Run, Check or Stop.
    field.setAttribute('data-lm-suppress-shortcuts', '');
    field.value = this.predictions[index] ?? '';
    field.addEventListener('input', () => {
      this.predictions[index] = field.value;
      this.onPrediction(index, field.value);
    });
    label.append(field);
    this.predictionFields.set(index, field);
    const actions = document.createElement('div');
    actions.className = 'dojo-drill-actions';
    const run = document.createElement('button');
    run.type = 'button';
    run.textContent = 'Run';
    run.onclick = () => void this.runProblemCell(cell);
    const reveal = document.createElement('button');
    reveal.type = 'button';
    reveal.textContent = 'Reveal reasoning';
    reveal.setAttribute('aria-expanded', 'false');
    const reasoning = document.createElement('div');
    reasoning.className = 'dojo-drill-reasoning';
    reasoning.hidden = true;
    reasoning.innerHTML = drill.reasoning_html;
    reveal.onclick = () => {
      reasoning.hidden = !reasoning.hidden;
      reveal.textContent = reasoning.hidden ? 'Reveal reasoning' : 'Hide reasoning';
      reveal.setAttribute('aria-expanded', String(!reasoning.hidden));
    };
    actions.append(run, reveal);
    card.append(title, prompt, label, actions);
    cell.node.prepend(card);
    cell.node.append(reasoning);
    this.typeset(prompt);
    this.typeset(reasoning);
  }

  /** Exploration panels show their output; the loader code sits behind a disclosure. */
  private addExploreControls(cell: CodeCell): void {
    if (this.controlledCells.has(cell)) return;
    this.controlledCells.add(cell);
    cell.addClass('dojo-explore-cell');
    cell.inputHidden = true;
    const controls = document.createElement('div');
    controls.className = 'dojo-explore-controls';
    const run = document.createElement('button');
    run.type = 'button';
    run.textContent = 'Run';
    run.onclick = () => void this.runProblemCell(cell);
    const code = document.createElement('button');
    code.type = 'button';
    const renderCode = () => {
      code.textContent = cell.inputHidden ? 'Show code' : 'Hide code';
      code.setAttribute('aria-expanded', String(!cell.inputHidden));
    };
    code.onclick = () => {
      cell.inputHidden = !cell.inputHidden;
      renderCode();
    };
    renderCode();
    controls.append(run, code);
    cell.node.prepend(controls);
  }
}
