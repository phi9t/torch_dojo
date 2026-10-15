import type { JupyterFrontEndPlugin } from '@jupyterlab/application';
import { JupyterFrontEnd, type LabShell } from '@jupyterlab/application';
import { IDocumentManager } from '@jupyterlab/docmanager';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { EditorExtensionRegistry, IEditorExtensionRegistry } from '@jupyterlab/codemirror';
import { INotebookTracker, NotebookActions, type NotebookPanel } from '@jupyterlab/notebook';
import { type PanelLayout, Widget } from '@lumino/widgets';
import { captureSolution, type NotebookSnapshot } from './source';
import {
  runCheck,
  CheckStoppedError,
  CheckTimeoutError,
  type CheckResult,
  type CheckRun
} from './checker';
import {
  AttemptStore,
  type AssistanceEvent,
  type AttemptRecord,
  type CheckReceipt
} from './attempts';
import { RepLifecycle, type RepChange, type RepPhase } from './repLifecycle';
import { ActiveTimer } from './activeTimer';
import { Bench, type DrillCard } from './bench';
import { helpUsed, repTable } from './repTable';
import { bestSplits, finishTime, formatDuration, repSplits } from './splits';
import { passedStages, stageName, StageSeals, type StageOrder } from './seals';
import { otherCheckpointResults, stageVerifiers } from './squares';
import { VerifierGrid } from './verifierGrid';
import { ThemeToggle } from './theme';
import { nextUp, workoutStats } from './home';
import { HomePage } from './homePage';
import { MotifPanel } from './motifPanel';
import type { Motif } from './review';
import { WorkoutSwitcher } from './switcher';
import { withShortcut } from './shortcuts';
import { gymEditorExtension } from './editor';

interface CatalogEntry {
  id: string;
  title: string;
  notebook: string;
  checker: string;
  checker_version: string;
  hints?: string[];
  solution?: string;
  stages?: string[];
  stage_order?: StageOrder;
  drills?: DrillCard[];
  concepts?: string[];
  previous?: { id: string; title: string };
  next?: { id: string; title: string };
  learning_path?: Array<{ id: string; title: string; concepts?: string[] }>;
}

type NotebookModelWithJSON = {
  toJSON: () => unknown;
  fromJSON: (value: unknown) => void;
};

type AttemptTransitionKind = RepChange['kind'];
type AttemptTransitionPhase = RepPhase;
type CheckGatePhase = 'before-receipt' | 'after-receipt' | 'after-render';
type RenderAttemptGatePhase = 'before-ui-write';
type RenderHistoryGatePhase = 'after-list';

type BrowserGymTestWindow = Window & {
  __torchDojoAttemptTransitionGate?: (event: {
    kind: AttemptTransitionKind;
    attemptId: string;
    phase: AttemptTransitionPhase;
  }) => void | Promise<void>;
  __torchDojoCheckGate?: (event: {
    attemptId: string;
    phase: CheckGatePhase;
  }) => void | Promise<void>;
  __torchDojoRenderAttemptGate?: (event: {
    attemptId: string;
    phase: RenderAttemptGatePhase;
  }) => void | Promise<void>;
  __torchDojoRenderHistoryGate?: (event: {
    attemptId?: string;
    phase: RenderHistoryGatePhase;
  }) => void | Promise<void>;
  __dojoRunAutosaveTick?: () => Promise<string | undefined>;
  __dojoOpenNotebookPath?: (path: string) => Promise<void>;
  __dojoActivateNotebook?: (name: string) => Promise<void>;
};

function stagesOf(entry: CatalogEntry): string[] {
  return entry.stages ?? ['baseline', 'followup', 'production'];
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function repLogHeading(count: number): string {
  return `Rep log: ${count} rep${count === 1 ? '' : 's'}`;
}

function repLabel(repNumber: number, current = false): string {
  return `Rep ${repNumber}${current ? ' (current)' : ''}`;
}

function assistanceSummary(events: AssistanceEvent[]): string {
  if (!events.length) return '';
  const labels = events.map(event => event.kind === 'hint' ? `hint ${event.level}` : 'solution reveal');
  return `Assistance recorded: ${labels.join(', ')}`;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * jupyter-lite.json disables the chrome plugins (ADR 0003). The document
 * title strip belongs to the shell itself, so it is turned off here.
 */
function keepShellChromeHidden(app: JupyterFrontEnd): void {
  const shell = app.shell as LabShell;
  const hide = () => {
    if (shell.mode !== 'single-document') return;
    // The title strip carries only the document name; the menu strip is empty
    // because the main menu is disabled. Layout restores can show them again.
    for (const area of (shell.layout as PanelLayout).widgets) {
      if ((area.id === 'jp-top-panel' || area.id === 'jp-menu-panel') && !area.isHidden) area.hide();
    }
  };
  shell.modeChanged.connect(hide);
  shell.layoutModified.connect(hide);
  void app.restored.then(hide);
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@torch-dojo/browser-gym:plugin',
  autoStart: true,
  requires: [INotebookTracker, IDocumentManager, IRenderMimeRegistry, IEditorExtensionRegistry],
  activate: async (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    docManager: IDocumentManager,
    rendermime: IRenderMimeRegistry,
    editorExtensions: IEditorExtensionRegistry
  ) => {
    const assets = new URL('../gym-assets/', window.location.href);
    const catalog = await fetch(new URL('catalog.json', assets)).then(response => {
      if (!response.ok) throw new Error(`Gym catalog HTTP ${response.status}`);
      return response.json() as Promise<CatalogEntry[]>;
    });
    keepShellChromeHidden(app);
    const attempts = new AttemptStore();
    // Pages under a repository prefix keep it: links are built from the current path.
    const homeHref = window.location.pathname;
    const exerciseHref = (id: string) => `${homeHref}?exercise=${encodeURIComponent(id)}`;
    const learningPath = catalog.find(item => item.learning_path?.length)?.learning_path?.map(item => item.id) ?? [];
    const repsByWorkout = new Map<string, AttemptRecord[]>();
    for (const rep of await attempts.listAttempts()) {
      repsByWorkout.set(rep.exerciseId, [...(repsByWorkout.get(rep.exerciseId) ?? []), rep]);
    }
    const statsFor = (item: CatalogEntry) => workoutStats(repsByWorkout.get(item.id) ?? [], stagesOf(item), item.stage_order ?? 'cumulative');
    const summaries = catalog.map(item => ({ id: item.id, title: item.title, topics: item.concepts ?? [], lastRepAt: statsFor(item).lastRepAt }));
    new WorkoutSwitcher(summaries, id => { window.location.href = exerciseHref(id); });

    const requested = new URLSearchParams(window.location.search).get('exercise');
    if (!requested) {
      // The site root is the home page; `?exercise=<id>` opens a workout directly.
      document.body.classList.add('dojo-browser-gym');
      const home = new HomePage(
        catalog.map(item => ({ id: item.id, title: item.title, topics: item.concepts ?? [], stats: statsFor(item), href: exerciseHref(item.id) })),
        learningPath,
        [new ThemeToggle().node]
      );
      // Reviews are optional: a failed review asset must not block workouts.
      try {
        const response = await fetch(new URL('motifs.json', assets));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const motifs: Motif[] = await response.json();
        const panel = new MotifPanel(motifs, new Map(catalog.map(item => [item.id, { title: item.title, href: exerciseHref(item.id) }])));
        home.node.querySelector('.dojo-home-header')!.after(panel.node);
      } catch (error) {
        const notice = document.createElement('p');
        notice.textContent = 'Motif reviews are unavailable. Reload to try again; workouts are ready below.';
        home.node.querySelector('.dojo-home-header')!.after(notice);
        console.warn('Motif reviews unavailable', error);
      }
      app.shell.add(home, 'main');
      app.shell.activateById(home.id);
      return;
    }
    const entry = catalog.find(item => item.id === requested);
    if (!entry) throw new Error(`Exercise ${requested} is missing from the gym catalog.`);
    const exercise = entry;
    // Code editors (not markdown) complete from the notebook kernel and can use Emacs keys.
    editorExtensions.addExtension({
      name: '@torch-dojo/browser-gym:editor',
      factory: options => !/markdown/.test(options.model.mimeType)
        ? EditorExtensionRegistry.createImmutableExtension(gymEditorExtension(() => answerView?.sessionContext.session?.kernel))
        : null
    });
    const hints = exercise.hints ?? [];
    const referenceSolution = exercise.solution ?? '';
    const stages = stagesOf(exercise);
    const stageOrder = exercise.stage_order ?? 'cumulative';
    const panel = document.createElement('section');
    const contents = app.serviceManager.contents;
    const rep = new RepLifecycle({
      exerciseId: exercise.id, store: attempts, contents,
      notebook: {
        read: currentExerciseNotebook, open: openExerciseNotebook,
        replace: loadNotebookIntoCurrentExerciseModel,
        starter: async () => starterNotebook ?? await loadStarterNotebook()
      },
      onBusy: busy => {
        setAttemptTransitionControlsDisabled(busy);
        if (busy) timer.setPaused(true);
      },
      beforeChange: saveActiveTime,
      onPhase: waitForAttemptTransitionGate
    });
    let starterNotebook: NotebookSnapshot | undefined;
    let lastRenderedHash: string | undefined;
    let hintsVisible = false;
    let shownHintCount = 0;
    let solutionVisible = false;
    let initialized = false;
    let notesRevision = 0;
    /** The in-flight check, if any. Checked by the Stop button and the check click guard. */
    let currentCheck: CheckRun | undefined;
    /** Covers the full check transaction, including append-only receipt commit and UI render. */
    let checkInProgress = false;
    panel.id = 'torch-dojo-gym';
    panel.className = 'dojo-gym-ui dojo-inline-eval';
    document.body.classList.add('dojo-browser-gym');

    /** One pending save per drill, so typing in one drill never cancels another's save. */
    const predictionSaves = new Map<number, number>();
    const bench = new Bench({
      exerciseId: exercise.id,
      title: exercise.title,
      topics: exercise.concepts ?? [],
      stages,
      stageOrder,
      drills: exercise.drills ?? [],
      typeset: node => { void rendermime.latexTypesetter?.typeset(node); },
      onPrediction: (drill, text) => {
        // Predictions belong to the rep; save shortly after typing pauses.
        const attemptId = rep.id;
        if (!attemptId || isAttemptTransitioning()) return;
        window.clearTimeout(predictionSaves.get(drill));
        predictionSaves.set(drill, window.setTimeout(() => {
          predictionSaves.delete(drill);
          if (attemptId === rep.id) void attempts.savePrediction(attemptId, drill, text).catch(() => undefined);
        }, 300));
      }
    });
    const seals = new StageSeals(stages, stageOrder);
    const timer = new ActiveTimer();
    /** The rep whose active time the timer is counting, and the time last saved for it. */
    let timedRepId: string | undefined;
    let savedSeconds = 0;
    const repSummary = document.createElement('p');
    repSummary.className = 'dojo-rep-summary';
    repSummary.setAttribute('aria-live', 'polite');
    repSummary.hidden = true;
    const repTableHost = document.createElement('div');
    repTableHost.className = 'dojo-rep-table-host';

    const pathNav = document.createElement('nav');
    pathNav.setAttribute('aria-label', 'Learning path');
    pathNav.setAttribute('data-testid', 'dojo-learning-path');
    if (!exercise.learning_path?.length) pathNav.style.display = 'none';
    const pathTitle = document.createElement('strong');
    pathTitle.textContent = 'Learning path';
    const pathList = document.createElement('div');
    pathList.setAttribute('role', 'list');
    for (const item of exercise.learning_path ?? []) {
      const row = document.createElement('div');
      row.setAttribute('role', 'listitem');
      const link = document.createElement('a');
      link.href = exerciseHref(item.id);
      link.textContent = `${item.id} · ${item.title}`;
      link.setAttribute('data-testid', `dojo-path-link-${item.id}`);
      if (item.id === exercise.id) link.setAttribute('aria-current', 'page');
      row.append(link);
      if (item.concepts?.length) {
        const concepts = document.createElement('span');
        concepts.textContent = ` · ${item.concepts.join(', ')}`;
        row.append(concepts);
      }
      pathList.append(row);
    }
    pathNav.append(pathTitle, pathList);

    const nextExercise = document.createElement('a');
    nextExercise.setAttribute('data-testid', 'dojo-next-exercise');
    if (exercise.next) {
      nextExercise.href = exerciseHref(exercise.next.id);
      nextExercise.textContent = `Next: ${exercise.next.id} · ${exercise.next.title}`;
    } else {
      nextExercise.textContent = exercise.learning_path?.length ? 'End of path' : '';
      if (!nextExercise.textContent) nextExercise.style.display = 'none';
    }

    const open = document.createElement('button');
    open.textContent = 'Open exercise';
    open.onclick = async () => {
      if (rep.id) await changeRep({ kind: 'resume', id: rep.id }, 'Your current rep is open.');
    };

    const run = document.createElement('button');
    run.textContent = 'Run cell';
    run.title = withShortcut('Run the selected cell in the notebook kernel', 'Shift+Enter');
    run.onclick = () => { void runActiveCell(exerciseNotebookWidget()); };
    const interrupt = document.createElement('button');
    interrupt.textContent = 'Interrupt';
    interrupt.title = 'Request a cooperative interrupt of the current notebook execution. ' +
      'Requires SharedArrayBuffer (cross-origin isolation). If interrupt is unavailable, use Restart kernel below.';
    interrupt.onclick = async () => {
      // Pyodide kernels honour interrupts only with an interrupt buffer
      // (SharedArrayBuffer under cross-origin isolation).
      try {
        const kernel = exerciseNotebookWidget()?.sessionContext.session?.kernel;
        if (!kernel) throw new Error('No notebook kernel.');
        await kernel.interrupt();
        void renderAttempt('Notebook kernel interrupt requested.');
      } catch {
        void renderAttempt(
          'Interrupt is unavailable because SharedArrayBuffer / cross-origin isolation is not active. ' +
          'Use Restart kernel to recover from a stuck cell.'
        );
      }
    };
    const restart = document.createElement('button');
    restart.textContent = 'Restart kernel';
    restart.title = 'Discard the live notebook kernel state and continue with a fresh kernel. ' +
      'Saved reps, notebooks, notes, and check receipts remain available.';
    restart.onclick = async () => {
      if (!confirm(
        'Restart the notebook kernel?\n\n' +
        'Live kernel state (variables, imports, open file handles, half-loaded packages in memory) is DISCARDED.\n\n' +
        'Saved state (rep log, editable notebooks, your notes, and every append-only check receipt including ' +
        'stopped/timeout receipts bound to the source you submitted) remains available. You can re-run cells to rebuild the live state.'
      )) return;
      try {
        const widget = exerciseNotebookWidget();
        if (!widget) throw new Error('Open the exercise notebook first.');
        await widget.sessionContext.restartKernel();
        NotebookActions.clearOutputs(widget.content);
        void renderAttempt('Notebook kernel restarted. Live state was discarded; saved reps and receipts remain.');
      } catch (error) {
        void renderAttempt('Kernel restart failed: ' + (error instanceof Error ? error.message : String(error)));
      }
    };
    const stop = document.createElement('button');
    stop.textContent = 'Stop';
    stop.title = withShortcut('Stop the running check; it still records a receipt', 'Mod+.');
    stop.disabled = true;
    stop.onclick = () => {
      if (!currentCheck) return;
      currentCheck.abort();
    };
    const check = document.createElement('button');

    const hint = document.createElement('button');
    hint.textContent = 'Hint';
    hint.disabled = hints.length === 0;

    const collapseHints = document.createElement('button');
    collapseHints.textContent = 'Collapse hints';
    collapseHints.disabled = true;

    const revealSolution = document.createElement('button');
    revealSolution.textContent = 'Reveal solution';
    revealSolution.disabled = referenceSolution.length === 0;

    const newAttempt = document.createElement('button');
    newAttempt.textContent = 'New rep';

    const exportAttempt = document.createElement('button');
    exportAttempt.textContent = 'Export rep';

    const exportAll = document.createElement('button');
    exportAll.textContent = 'Export all';

    const importInput = document.createElement('input');
    importInput.type = 'file';
    importInput.accept = 'application/json,.json';
    importInput.setAttribute('aria-label', 'Import reps');

    const stageLabel = document.createElement('label');
    stageLabel.textContent = 'Check stage ';
    stageLabel.htmlFor = 'torch-dojo-stage';
    const stageSelect = document.createElement('select');
    stageSelect.id = 'torch-dojo-stage';
    for (const stage of stages) {
      const option = document.createElement('option');
      option.value = stage;
      option.textContent = stage;
      stageSelect.append(option);
    }
    stageLabel.append(stageSelect);
    stageSelect.addEventListener('change', () => {
      renderCheckLabel();
      void renderAttempt();
    });
    renderCheckLabel();

    const notesLabel = document.createElement('label');
    notesLabel.textContent = 'Rep notes';
    notesLabel.htmlFor = 'torch-dojo-notes';
    const notes = document.createElement('textarea');
    notes.id = 'torch-dojo-notes';
    notes.rows = 4;
    notes.addEventListener('input', () => {
      notesRevision += 1;
      if (isAttemptTransitioning()) return;
      if (rep.id) void attempts.saveNotes(rep.id, notes.value);
    });

    const hintsRegion = document.createElement('div');

    const solutionRegion = document.createElement('pre');
    solutionRegion.style.display = 'none';

    const historyRegion = document.createElement('div');
    historyRegion.setAttribute('role', 'region');
    historyRegion.setAttribute('aria-label', 'Rep log');
    historyRegion.setAttribute('data-testid', 'dojo-attempt-history');

    const status = document.createElement('div');
    status.className = 'dojo-technical-log';
    status.textContent = 'Browser gym is starting…';
    const grid = new VerifierGrid(stages, stageOrder, async (source, caseIndex) => {
      try {
        await rep.editNotebook(async () => {
          const widget = exerciseNotebookWidget();
          if (!widget?.content.model) throw new Error('Open the exercise notebook first.');
          await widget.sessionContext.ready;
          const kernel = widget.sessionContext.session?.kernel;
          if (!kernel) throw new Error('Start the notebook kernel first.');
          const reply = await kernel.requestExecute({ code: source, store_history: false }).done;
          if (reply.content.status !== 'ok') throw new Error('Could not load examples into the notebook kernel.');
          const model = widget.content.model.sharedModel;
          const index = model.cells.length;
          model.insertCell(index, {
            cell_type: 'code',
            source: '# Captured check examples — fresh copies, ready to inspect.\n' +
              `case = gym_case(${caseIndex})\ncase\n` +
              '# case["args"], case.get("expected"), case.get("actual")\n' +
              `# Run your answer cell, then try: gym_try(${caseIndex})\n# More examples: gym_passed, gym_failed\n`,
            metadata: { dojo: { role: 'scratch', exercise_id: exercise.id } }
          });
          bench.setScratchOpen(true);
          widget.content.activeCellIndex = index;
          await widget.content.scrollToItem(index);
        });
      } finally {
        await renderAttempt();
      }
    });
    // The grid shows stages a check has not reached, so it needs every verifier's name,
    // and a failure's detail shows the line that failed, so it needs their source.
    void fetch(new URL(exercise.checker, assets))
      .then(response => response.ok ? response.json() as Promise<{ verifiers?: Array<{ name: string; source: string }>; helpers?: string; program?: string }> : undefined)
      .then(bundle => {
        if (!bundle?.verifiers) return;
        grid.setVerifiers(stageVerifiers(stages, bundle.verifiers.map(item => item.name)));
        // A failure points at the line of the check (or helper) that failed.
        grid.setSources({ verifiers: Object.fromEntries(bundle.verifiers.map(item => [item.name, item.source])), helpers: bundle.helpers, program: bundle.program });
      })
      .catch(() => { /* The grid still shows every verifier a check reports. */ });

    const readout = document.createElement('p');
    readout.className = 'dojo-readout';
    function renderReadout(result: CheckResult | undefined): void {
      const identity = result?.package_identity ?? {};
      readout.textContent = result
        ? [identity.runtime ?? result.runtime, identity.python && `Python ${identity.python}`,
          identity.numpy && `NumPy ${identity.numpy}`, `checker ${result.checker_version}`].filter(Boolean).join(' · ')
        : '';
    }

    function renderCheckLabel(): void {
      check.textContent = `Check ${stageName(stageSelect.value).toLowerCase()}`;
      check.title = withShortcut(`Check the answer cell at ${stageName(stageSelect.value).toLowerCase()} in a fresh worker`, 'Mod+Enter');
    }


    function receiptId(): string {
      if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
      }
      return `receipt-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function resultText(result: CheckResult, sourceHash: string): string {
      return `${result.status.toUpperCase()} · source SHA-256 ${sourceHash}\n` +
        (result.error ? `${result.error}\n` : '') +
        (result.traceback ? `${result.traceback}\n` : '') +
        result.checks.map(item => `${item.status === 'passed' ? '✓' : '✗'} ${item.name} (${item.elapsed_ms ?? 0} ms)` +
          `${item.error ? ': ' + item.error : ''}` +
          `${item.evidence ? '\n' + JSON.stringify(item.evidence) : ''}` +
          `${item.traceback ? '\n' + item.traceback : ''}`).join('\n');
    }

    function attemptHistoryText(
      all: AttemptRecord[],
      renderedActiveAttemptId: string | undefined,
      currentHash?: string
    ): string {
      const lines = [repLogHeading(all.length)];
      for (const [attemptIndex, attempt] of all.entries()) {
        lines.push(repLabel(attemptIndex + 1, attempt.id === renderedActiveAttemptId));
        if (attempt.assistance.length) lines.push(`  ${assistanceSummary(attempt.assistance)}`);
        for (const [receiptIndex, receipt] of attempt.receipts.entries()) {
          const hashForReceipt = attempt.id === renderedActiveAttemptId ? (currentHash ?? receipt.sha256) : receipt.sha256;
          const bound = Array.isArray(receipt.sourceBound) && receipt.sourceBound.length
            ? ` · source cells [${receipt.sourceBound.join(', ')}]`
            : '';
          const description = attempts.describeReceipt(attempt.id, receipt.id, hashForReceipt);
          lines.push(`  Receipt ${receiptIndex + 1}: ${description.label}${bound} · Receipt source SHA-256 ${receipt.sha256} · ${receipt.checkedAt}`);
        }
      }
      return lines.join('\n');
    }

    function isCurrentRenderContext(attemptId: string | undefined, transitionVersion: number): boolean {
      return attemptId === rep.id &&
        transitionVersion === rep.version &&
        !isAttemptTransitioning();
    }

    async function renderAttemptHistory(
      renderedActiveAttemptId: string | undefined,
      transitionVersion: number,
      currentHash?: string
    ): Promise<string | undefined> {
      const all = await attempts.listAttempts(exercise.id);
      await waitForRenderHistoryGate(renderedActiveAttemptId, 'after-list');
      if (!isCurrentRenderContext(renderedActiveAttemptId, transitionVersion)) return undefined;
      historyRegion.replaceChildren();
      bench.setRepCount(all.length);
      const active = all.find(attempt => attempt.id === renderedActiveAttemptId);
      const passed = passedStages(stages, active?.receipts ?? [], stageOrder);
      const splits = active ? repSplits(active, stages, stageOrder) : {};
      const best = bestSplits(all.filter(attempt => attempt.id !== renderedActiveAttemptId), stages, stageOrder);
      seals.render(passed, stageSelect.value, { splits, best }, renderedActiveAttemptId);
      bench.setBriefStages(passed, stageSelect.value);
      bench.setPredictions(active?.predictions ?? {});
      const finish = active ? finishTime(active, stages, stageOrder) : undefined;
      if (active) syncTimer(active, finish);
      renderRepSummary(active, finish, all.filter(attempt => attempt.id !== renderedActiveAttemptId));
      repTableHost.replaceChildren(repTable(all, stages, stageOrder, renderedActiveAttemptId));
      const heading = document.createElement('div');
      heading.textContent = repLogHeading(all.length);
      historyRegion.append(heading);

      for (const [attemptIndex, attempt] of all.entries()) {
        const attemptNumber = attemptIndex + 1;
        const attemptBlock = document.createElement('div');
        attemptBlock.style.marginTop = '8px';
        const attemptLine = document.createElement('div');
        const summary = document.createElement('span');
        summary.textContent = repLabel(attemptNumber, attempt.id === renderedActiveAttemptId);
        attemptLine.append(summary);
        if (attempt.id !== renderedActiveAttemptId) {
          const resume = document.createElement('button');
          resume.type = 'button';
          resume.textContent = `Resume rep ${attemptNumber}`;
          resume.disabled = isAttemptTransitioning();
          resume.style.marginLeft = '8px';
          resume.onclick = () => {
            void resumeAttempt(attempt.id, attemptNumber);
          };
          attemptLine.append(resume);
        }
        attemptBlock.append(attemptLine);
        if (attempt.assistance.length) {
          const assistance = document.createElement('div');
          assistance.textContent = assistanceSummary(attempt.assistance);
          assistance.style.marginLeft = '16px';
          attemptBlock.append(assistance);
        }
        for (const [receiptIndex, receipt] of attempt.receipts.entries()) {
          const hashForReceipt = attempt.id === renderedActiveAttemptId ? (currentHash ?? receipt.sha256) : receipt.sha256;
          const bound = Array.isArray(receipt.sourceBound) && receipt.sourceBound.length
            ? ` · source cells [${receipt.sourceBound.join(', ')}]`
            : '';
          const description = attempts.describeReceipt(attempt.id, receipt.id, hashForReceipt);
          const receiptLine = document.createElement('div');
          receiptLine.textContent =
            `Receipt ${receiptIndex + 1}: ${description.label}${bound} · Receipt source SHA-256 ${receipt.sha256} · ${receipt.checkedAt}`;
          receiptLine.style.marginLeft = '16px';
          attemptBlock.append(receiptLine);
        }
        historyRegion.append(attemptBlock);
      }

      return attemptHistoryText(all, renderedActiveAttemptId, currentHash);
    }

    function isLoadedExerciseNotebook(notebook: NotebookSnapshot | undefined): notebook is NotebookSnapshot {
      return notebook?.metadata?.dojo?.exercise_id === exercise.id &&
        notebook.cells.some(cell => cell.cell_type === 'code' &&
          cell.metadata?.dojo?.role === 'solution' &&
          cell.metadata?.dojo?.exercise_id === exercise.id);
    }

    function notebookFromWidget(widget: NotebookPanel | null | undefined): NotebookSnapshot | undefined {
      const model = widget?.content.model as unknown as NotebookModelWithJSON | null | undefined;
      const notebook = model?.toJSON();
      return notebook && typeof notebook === 'object' ? notebook as NotebookSnapshot : undefined;
    }

    function isExerciseNotebookWidget(widget: NotebookPanel | null | undefined): widget is NotebookPanel {
      if (!widget) return false;
      return widget.context.path === exercise.notebook;
    }

    /** The answer pane's view. The problem pane is a second view of the same model. */
    let answerView: NotebookPanel | undefined;
    let problemView: NotebookPanel | undefined;

    function exerciseNotebookWidget(): NotebookPanel | undefined {
      if (answerView && !answerView.isDisposed) return answerView;
      return tracker.find(widget => isExerciseNotebookWidget(widget));
    }

    /** Runs the pane's selected cell, or its first visible code cell if the selection is hidden there. */
    /**
     * The Bench pane the learner is typing in. Both panes are views of one
     * notebook, so the tracker's current widget can be the other one.
     */
    function focusedBenchView(): NotebookPanel | undefined {
      const focused = document.activeElement;
      const views = [answerView, problemView].filter((view): view is NotebookPanel => Boolean(view && !view.isDisposed));
      return views.find(view => focused && view.node.contains(focused)) ?? views[0];
    }

    async function runActiveCell(view: NotebookPanel | null | undefined): Promise<void> {
      if (!view) return;
      // A run pressed while the kernel is still starting waits instead of being
      // dropped. Once it is ready, run straight away so the editor keeps focus.
      if (!view.sessionContext.isReady) await view.sessionContext.ready;
      const content = view.content;
      if (!content.activeCell || content.activeCell.isHidden) {
        const visible = content.widgets.findIndex(cell => !cell.isHidden && cell.model.type === 'code');
        if (visible < 0) return;
        content.activeCellIndex = visible;
      }
      const wasEditing = content.mode === 'edit';
      const done = NotebookActions.run(content, view.sessionContext);
      // Running switches the notebook to command mode; on the Bench the learner
      // keeps typing in the cell they ran.
      if (wasEditing) {
        content.mode = 'edit';
        content.activeCell?.editor?.focus();
      }
      await done;
    }

    let observedNotebook: NotebookPanel | undefined;
    let inlineObserver: MutationObserver | undefined;
    let mountScheduled = false;

    function mountInlineEval(): void {
      // Generated teaching cells stay rendered even in restored older attempts.
      for (const view of [problemView, answerView]) {
        for (const cell of view?.content.widgets ?? []) {
          if (cell.model.type !== 'markdown') continue;
          const marker = cell.model.metadata.dojo as { role?: string } | undefined;
          const source = cell.model.sharedModel.getSource();
          const legacyInstruction = /^(# (?:\d{4} ·|MLP Modeling Lab)|### Explore|Edit the solution below\.|Scratch space:)/.test(source);
          if (marker?.role !== 'instruction' && !legacyInstruction) continue;
          if (cell.model.sharedModel.getMetadata('editable') !== false) {
            cell.model.sharedModel.setMetadata('editable', false);
          }
          cell.readOnly = true;
          if ('showEditorForReadOnlyMarkdown' in cell) cell.showEditorForReadOnlyMarkdown = false;
          if ('rendered' in cell && cell.rendered === false) cell.rendered = true;
        }
      }
      const solution = answerView?.content.widgets.find(cell => {
        const dojo = cell.model.metadata.dojo as { role?: string; exercise_id?: string } | undefined;
        return dojo?.role === 'solution' && dojo.exercise_id === exercise.id;
      });
      if (solution?.node.isConnected && panel.parentElement !== solution.node) {
        solution.node.append(panel);
      }
    }

    /**
     * Puts the exercise notebook on the Bench: the opened view becomes the answer
     * pane and a clone of it (same model, same kernel) becomes the problem pane.
     */
    function mountBench(): void {
      const views: NotebookPanel[] = [];
      tracker.forEach(widget => { if (isExerciseNotebookWidget(widget)) views.push(widget); });
      const previous = [problemView, answerView];
      if (problemView?.isDisposed) problemView = undefined;
      if (!answerView || answerView.isDisposed) answerView = views.find(view => view !== problemView);
      if (!answerView) return;
      // A restored workspace can bring back extra views of the same notebook.
      for (const view of views) if (view !== answerView && view !== problemView) view.dispose();
      problemView ??= docManager.cloneWidget(answerView) as NotebookPanel | undefined;
      if (!problemView) throw new Error('Could not open a second view of the exercise notebook.');
      if (previous[0] !== problemView || previous[1] !== answerView) bench.setViews(problemView, answerView);
      if (!bench.isAttached) app.shell.add(bench, 'main');
      app.shell.activateById(bench.id);

      const widget = answerView;
      if (observedNotebook !== widget) {
        inlineObserver?.disconnect();
        observedNotebook = widget;
        const showKernelStatus = () => {
          if (observedNotebook !== widget) return;
          const kernel = widget.sessionContext.kernelDisplayStatus;
          kernelStatus.textContent = `Python · ${kernel === 'unknown' || kernel === 'initializing' ? 'starting' : kernel}`;
        };
        widget.sessionContext.statusChanged.connect(showKernelStatus);
        widget.sessionContext.connectionStatusChanged.connect(showKernelStatus);
        showKernelStatus();
        // Completion asks the kernel, so have NumPy ready under its usual name, and
        // ask once up front: the kernel's first completion is slow while it warms up.
        void widget.sessionContext.ready.then(async () => {
          const kernel = widget.sessionContext.session?.kernel;
          if (!kernel) return;
          await kernel.requestExecute({ code: 'import numpy as np', silent: true, store_history: false }).done;
          await kernel.requestComplete({ code: 'np.ar', cursor_pos: 5 });
        }).catch(() => undefined);
        inlineObserver = new MutationObserver(() => {
          if (mountScheduled) return;
          mountScheduled = true;
          queueMicrotask(() => {
            mountScheduled = false;
            mountInlineEval();
          });
        });
        inlineObserver.observe(widget.content.node, { childList: true, subtree: true });
      }
      mountInlineEval();
    }

    async function currentExerciseNotebook(): Promise<NotebookSnapshot | undefined> {
      const notebook = notebookFromWidget(exerciseNotebookWidget());
      return isLoadedExerciseNotebook(notebook) ? notebook : undefined;
    }

    async function currentExerciseModel(): Promise<NotebookModelWithJSON | undefined> {
      const widget = exerciseNotebookWidget();
      return isLoadedExerciseNotebook(notebookFromWidget(widget))
        ? widget?.content.model as unknown as NotebookModelWithJSON
        : undefined;
    }

    async function loadNotebookIntoCurrentExerciseModel(notebook: NotebookSnapshot): Promise<void> {
      const model = await currentExerciseModel();
      if (!model) throw new Error('Open the exercise notebook first.');
      const visibleNotebook = clone(notebook);
      const brief = visibleNotebook.cells[0];
      const currentBrief = starterNotebook?.cells[0];
      const briefSource = brief && (Array.isArray(brief.source) ? brief.source.join('') : brief.source);
      if (brief?.cell_type === 'markdown' && currentBrief?.cell_type === 'markdown' &&
          briefSource?.includes('dojo-brief-stage')) {
        // Refresh generated instructions in old attempts without changing learner code.
        brief.source = clone(currentBrief.source);
        brief.metadata = { ...brief.metadata, ...clone(currentBrief.metadata) };
      }
      for (const cell of visibleNotebook.cells) {
        if (cell.cell_type !== 'code' || cell.metadata?.dojo?.role !== 'exploration') continue;
        const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
        if (source.split('\n').length <= 20) continue;
        cell.metadata.jupyter = { ...cell.metadata.jupyter, source_hidden: true };
      }
      model.fromJSON(visibleNotebook);
      queueMicrotask(() => {
        try {
          mountBench();
        } catch (error) {
          status.textContent = errorMessage(error);
        }
      });
    }

    async function openExerciseNotebook(): Promise<void> {
      await app.commands.execute('docmanager:open', { path: exercise.notebook });
      mountBench();
      await waitForExerciseNotebook();
      mountBench();
    }

    async function waitForAttemptTransitionGate(
      kind: AttemptTransitionKind,
      attemptId: string,
      phase: AttemptTransitionPhase = 'before-swap'
    ): Promise<void> {
      await (window as BrowserGymTestWindow).__torchDojoAttemptTransitionGate?.({ kind, attemptId, phase });
    }

    async function waitForCheckGate(attemptId: string, phase: CheckGatePhase): Promise<void> {
      await (window as BrowserGymTestWindow).__torchDojoCheckGate?.({ attemptId, phase });
    }

    async function waitForRenderAttemptGate(attemptId: string, phase: RenderAttemptGatePhase): Promise<void> {
      await (window as BrowserGymTestWindow).__torchDojoRenderAttemptGate?.({ attemptId, phase });
    }

    async function waitForRenderHistoryGate(
      attemptId: string | undefined,
      phase: RenderHistoryGatePhase
    ): Promise<void> {
      await (window as BrowserGymTestWindow).__torchDojoRenderHistoryGate?.({ attemptId, phase });
    }

    function isAttemptTransitioning(): boolean {
      return rep.locked;
    }

    function setAttemptTransitionControlsDisabled(disabled: boolean): void {
      open.disabled = disabled;
      run.disabled = disabled;
      interrupt.disabled = disabled;
      restart.disabled = disabled;
      check.disabled = disabled || checkInProgress;
      hint.disabled = disabled || hints.length === 0;
      collapseHints.disabled = disabled || !hintsVisible || shownHintCount === 0;
      revealSolution.disabled = disabled || referenceSolution.length === 0;
      briefHint.disabled = hint.disabled;
      briefSolution.disabled = revealSolution.disabled;
      newAttempt.disabled = disabled;
      importInput.disabled = disabled || checkInProgress;
      notes.disabled = disabled;
      stageSelect.disabled = disabled;
      stop.disabled = disabled || !currentCheck;
      historyRegion.querySelectorAll<HTMLButtonElement>('button').forEach(button => {
        if (button.textContent?.startsWith('Resume rep')) button.disabled = disabled;
      });
    }

    function setCheckInProgress(inProgress: boolean): void {
      checkInProgress = inProgress;
      if (!isAttemptTransitioning()) {
        check.disabled = inProgress;
        importInput.disabled = inProgress;
        stop.disabled = !currentCheck;
      }
    }

    function errorMessage(error: unknown): string {
      return error instanceof Error ? error.message : String(error);
    }

    async function waitForExerciseNotebook(timeoutMs = 15_000): Promise<NotebookSnapshot> {
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        const notebook = await currentExerciseNotebook();
        if (isLoadedExerciseNotebook(notebook)) return notebook;
        await new Promise(resolve => window.setTimeout(resolve, 50));
      }
      throw new Error('The exercise notebook did not finish loading.');
    }

    async function loadStarterNotebook(): Promise<NotebookSnapshot> {
      const response = await fetch(new URL(`../files/${exercise.notebook}`, window.location.href));
      if (!response.ok) throw new Error(`Starter notebook HTTP ${response.status}`);
      return response.json() as Promise<NotebookSnapshot>;
    }

    async function currentHashAndSave(): Promise<string | undefined> {
      return (await rep.save()).hash;
    }

    async function renderAttempt(prefix = ''): Promise<void> {
      const attemptId = rep.id;
      const transitionVersion = rep.version;
      const notesRevisionAtStart = notesRevision;
      if (!attemptId) return;
      const attempt = await attempts.getAttempt(attemptId);
      if (!attempt) return;
      const currentHash = await currentHashAndSave();
      await waitForRenderAttemptGate(attemptId, 'before-ui-write');
      if (!isCurrentRenderContext(attemptId, transitionVersion)) return;
      const latestAttempt = await attempts.getAttempt(attemptId);
      if (!latestAttempt) return;
      const assistance = assistanceSummary(latestAttempt.assistance);
      const history = await renderAttemptHistory(attemptId, transitionVersion, currentHash);
      if (history === undefined || !isCurrentRenderContext(attemptId, transitionVersion)) return;
      lastRenderedHash = currentHash;
      if (notesRevision === notesRevisionAtStart && document.activeElement !== notes) {
        notes.value = latestAttempt.notes;
      }
      status.textContent = [prefix, assistance, history].filter(Boolean).join('\n\n');
      const latestReceipt = latestAttempt.receipts.at(-1);
      if (!checkInProgress) {
        grid.showResult(latestReceipt?.result, {
          sourceHash: latestReceipt?.sha256,
          stale: Boolean(latestReceipt && currentHash !== latestReceipt.sha256),
          others: otherCheckpointResults(stages, stageOrder, latestAttempt.receipts, latestReceipt?.result),
          submitted: latestReceipt?.source
        });
        renderReadout(latestReceipt?.result);
      }
    }

    function failedResult(error: unknown, checkerVersion: string): CheckResult {
      return {
        status: 'failed',
        checks: [],
        error: error instanceof Error ? error.message : String(error),
        traceback: error instanceof Error ? error.stack : undefined,
        checker_version: checkerVersion,
        runtime: 'pyodide',
        package_identity: { runtime: 'pyodide' },
        fixture_id: `canonical-${exercise.id}`
      };
    }

    /**
     * Turn a controller-level exception (timeout / stopped / worker error) into
     * the right receipt status, the right synthetic CheckResult, and a short
     * human-facing prefix that explains the saved vs. discarded state.
     */
    function classifyControllerError(
      error: unknown,
      checkerVersion: string,
      snapshot: { source: string; sha256: string }
    ): { receiptStatus: 'timeout' | 'stopped' | 'failed'; result: CheckResult; prefix: string } {
      if (error instanceof CheckTimeoutError) {
        return {
          receiptStatus: 'timeout',
          result: {
            status: 'failed',
            checks: [],
            error: 'Check deadline reached before the worker returned results. ' +
              'This often means an infinite loop, very slow check, or missing return. The submitted source was: ' +
              snapshot.sha256,
            checker_version: checkerVersion,
            runtime: 'pyodide',
            package_identity: { runtime: 'pyodide' },
            fixture_id: `canonical-${exercise.id}`
          },
          prefix:
            'TIMEOUT · source SHA-256 ' + snapshot.sha256 + '\n' +
            'The check was hard-terminated at the controller deadline. No live checker worker state remains.\n' +
            'Saved: this source-bound timeout receipt, your notebook snapshot, and all prior receipts.\n' +
            'Discarded: the halted worker\'s in-memory state only.'
        };
      }
      if (error instanceof CheckStoppedError) {
        return {
          receiptStatus: 'stopped',
          result: {
            status: 'failed',
            checks: [],
            error: 'Check stopped by the learner before the worker returned results. Source SHA-256: ' +
              snapshot.sha256,
            checker_version: checkerVersion,
            runtime: 'pyodide',
            package_identity: { runtime: 'pyodide' },
            fixture_id: `canonical-${exercise.id}`
          },
          prefix:
            'STOPPED · source SHA-256 ' + snapshot.sha256 + '\n' +
            'You stopped the check. No live checker worker state remains.\n' +
            'Saved: this source-bound stopped receipt, your notebook snapshot, and all prior receipts.\n' +
            'Discarded: the halted worker\'s in-memory state only.'
        };
      }
      return {
        receiptStatus: 'failed',
        result: failedResult(error, checkerVersion),
        prefix: ''
      };
    }

    /**
     * Counts the active rep's time; switching reps saves the time of the one left
     * behind. `finalSplit` is set once the final stage has passed (null if that
     * split predates timing): the rep's time is then final.
     */
    function syncTimer(active: AttemptRecord, finalSplit: number | null | undefined): void {
      timer.setPaused(false);
      const finished = finalSplit !== undefined;
      if (timedRepId !== active.id) {
        if (timedRepId && !timer.isStopped) void attempts.saveActiveSeconds(timedRepId, timer.value).catch(() => undefined);
        timedRepId = active.id;
        savedSeconds = active.activeSeconds;
        timer.reset(finalSplit ?? active.activeSeconds, finished);
      } else if (finished && !timer.isStopped) {
        // The time is final at the finishing check, not when its result arrived.
        const final = finalSplit ?? timer.value;
        timer.reset(final, true);
        savedSeconds = final;
        void attempts.saveActiveSeconds(active.id, final, { final: true }).catch(() => undefined);
      }
    }

    async function saveActiveTime(): Promise<void> {
      if (!timedRepId || timedRepId !== rep.id || timer.isStopped) return;
      const seconds = timer.value;
      if (seconds <= savedSeconds) return;
      savedSeconds = seconds;
      await attempts.saveActiveSeconds(timedRepId, seconds);
    }

    /** One line once the final stage passes: time, difference from best, help used, and what's next. */
    function renderRepSummary(active: AttemptRecord | undefined, finish: number | null | undefined, others: AttemptRecord[]): void {
      repSummary.hidden = !active || finish === undefined;
      if (!active || finish === undefined) return;
      const parts = [`Rep done in ${formatDuration(finish)}`];
      const best = Math.min(...others.map(rep => finishTime(rep, stages, stageOrder)).filter((time): time is number => typeof time === 'number'));
      if (typeof finish === 'number' && Number.isFinite(best)) {
        const difference = best - finish;
        parts.push(Math.abs(difference) < 1 ? 'level with your best'
          : `${formatDuration(Math.abs(difference))} ${difference > 0 ? 'faster than' : 'slower than'} your best`);
      } else if (typeof finish === 'number') {
        parts.push('your first timed finish');
      }
      parts.push(helpUsed(active) || 'no hints');
      repSummary.replaceChildren(document.createTextNode(`${parts.join(' · ')}.`));
      // Next up follows the home page's rule, counting this rep as the latest.
      const suggested = nextUp(summaries, learningPath, exercise.id);
      if (suggested) {
        const next = document.createElement('a');
        next.href = exerciseHref(suggested.id);
        next.textContent = `Next up: ${suggested.id} · ${suggested.title}`;
        repSummary.append(' ', next);
      }
    }

    async function appendReceipt(
      attemptId: string,
      snapshot: { source: string; sha256: string; sourceBound: number[] },
      result: CheckResult,
      overrideStatus?: CheckReceipt['status'],
      activeSecondsAtCheck?: number
    ): Promise<AttemptRecord> {
      const receipt: CheckReceipt = {
        id: receiptId(),
        exerciseId: exercise.id,
        checkedAt: new Date().toISOString(),
        source: snapshot.source,
        sha256: snapshot.sha256,
        sourceBound: snapshot.sourceBound,
        status: overrideStatus ?? (result.status === 'passed' ? 'passed' : 'failed'),
        checkerVersion: result.checker_version,
        runtime: result.runtime,
        packageIdentity: result.package_identity ?? { runtime: result.runtime },
        fixtureId: result.fixture_id,
        result,
        assistance: [],
        ...(activeSecondsAtCheck === undefined ? {} : { activeSecondsAtCheck })
      };
      return attempts.appendReceipt(attemptId, receipt);
    }

    async function renderResultWithAttempt(result: CheckResult, sourceHash: string, attempt: AttemptRecord, prefix = ''): Promise<void> {
      const attemptId = attempt.id;
      const transitionVersion = rep.version;
      const notesRevisionAtStart = notesRevision;
      if (attemptId !== rep.id) {
        await renderAttempt(prefix || 'Check completed for a background rep. Receipt saved.');
        return;
      }
      if (isAttemptTransitioning()) return;
      const body = resultText(result, sourceHash);
      const currentHash = await currentHashAndSave();
      if (!isCurrentRenderContext(attemptId, transitionVersion)) {
        await renderAttempt(prefix || 'Check completed for a background rep. Receipt saved.');
        return;
      }
      const latestAttempt = await attempts.getAttempt(attemptId) ?? attempt;
      const displayHash = currentHash ?? sourceHash;
      const assistance = assistanceSummary(latestAttempt.assistance);
      const history = await renderAttemptHistory(attemptId, transitionVersion, displayHash);
      if (history === undefined || !isCurrentRenderContext(attemptId, transitionVersion)) return;
      lastRenderedHash = displayHash;
      if (notesRevision === notesRevisionAtStart && document.activeElement !== notes) {
        notes.value = latestAttempt.notes;
      }
      status.textContent = [prefix, body, assistance, history].filter(Boolean).join('\n\n');
      grid.showResult(result, {
        sourceHash,
        stale: displayHash !== sourceHash,
        others: otherCheckpointResults(stages, stageOrder, latestAttempt.receipts, result),
        submitted: [...latestAttempt.receipts].reverse().find(receipt => receipt.sha256 === sourceHash)?.source
      });
      renderReadout(result);
    }

    async function changeRep(change: RepChange, prefix?: string): Promise<void> {
      const outcome = await rep.change(change);
      if (outcome.status === 'ignored') return;
      if (outcome.context.version !== rep.version || outcome.context.id !== rep.id) return;
      try {
        if (outcome.status === 'failed') {
          status.textContent = outcome.message;
          if (outcome.recovered) await renderAttempt(outcome.message);
          return;
        }
        shownHintCount = 0;
        hintsVisible = false;
        solutionVisible = false;
        renderHints();
        solutionRegion.textContent = '';
        solutionRegion.style.display = 'none';
        revealSolution.textContent = 'Reveal solution';
        const label = change.kind === 'import' ? `Imported reps: ${outcome.imported}`
          : prefix ?? 'New rep — take it one stage at a time.';
        await renderAttempt(label);
      } catch (error) {
        if (rep.isCurrent(outcome.context)) status.textContent = errorMessage(error);
      }
    }

    async function resumeAttempt(attemptId: string, attemptNumber?: number): Promise<void> {
      await changeRep({ kind: 'resume', id: attemptId }, attemptNumber ? `Resumed rep ${attemptNumber}` : 'Resumed rep');
    }

    function renderHints(): void {
      hintsRegion.replaceChildren();
      if (!hintsVisible || shownHintCount === 0) {
        collapseHints.disabled = true;
        return;
      }
      const list = document.createElement('ol');
      for (const text of hints.slice(0, shownHintCount)) {
        const item = document.createElement('li');
        item.textContent = text;
        list.append(item);
      }
      hintsRegion.append(list);
      collapseHints.disabled = false;
    }

    async function recordAssistance(kind: AssistanceEvent['kind'], level: number): Promise<void> {
      const { attempt } = await rep.capture();
      await attempts.recordAssistance(attempt.id, kind, level);
      await renderAttempt();
    }

    hint.onclick = async () => {
      if (isAttemptTransitioning()) return;
      try {
        shownHintCount = Math.min(shownHintCount + 1, hints.length);
        hintsVisible = true;
        renderHints();
        await recordAssistance('hint', shownHintCount);
      } catch (error) {
        status.textContent = error instanceof Error ? error.message : String(error);
      }
    };

    collapseHints.onclick = async () => {
      if (isAttemptTransitioning()) return;
      hintsVisible = false;
      renderHints();
      await renderAttempt();
    };

    revealSolution.onclick = async () => {
      if (isAttemptTransitioning()) return;
      try {
        solutionVisible = !solutionVisible;
        solutionRegion.textContent = solutionVisible ? referenceSolution : '';
        solutionRegion.style.display = solutionVisible ? 'block' : 'none';
        revealSolution.textContent = solutionVisible ? 'Hide solution' : 'Reveal solution';
        if (solutionVisible) await recordAssistance('solution', 0);
        else await renderAttempt();
      } catch (error) {
        status.textContent = error instanceof Error ? error.message : String(error);
      }
    };

    newAttempt.onclick = () => changeRep({ kind: 'new' });

    exportAttempt.onclick = async () => {
      if (!rep.id) return;
      await currentHashAndSave();
      const blob = await attempts.exportAttempt(rep.id);
      downloadBlob(blob, `${exercise.id}-rep-${rep.id}.json`);
    };

    exportAll.onclick = async () => {
      await currentHashAndSave();
      const blob = await attempts.exportAll();
      downloadBlob(blob, `${exercise.id}-reps.json`);
    };

    importInput.onchange = async () => {
      const file = importInput.files?.[0];
      if (!file) return;
      if (checkInProgress) {
        status.textContent = 'Wait for the current check to finish before importing reps.';
        importInput.value = '';
        return;
      }
      try {
        await changeRep({ kind: 'import', archive: file });
      } finally {
        importInput.value = '';
      }
    };

    check.onclick = async () => {
      if (checkInProgress || currentCheck || isAttemptTransitioning()) {
        // Guard: there's already a check in flight. Let the Stop button handle it.
        return;
      }
      setCheckInProgress(true);
      const checkContext = rep.context();
      const checkedStage = stageSelect.value;
      // A split is the rep's active time when the passing check was submitted.
      let activeAtCheck = timedRepId === rep.id && !timer.isStopped ? timer.value : undefined;
      void saveActiveTime().catch(() => undefined);
      status.textContent = 'Checking your answer…';
      grid.showRunning(checkedStage);
      let snapshot: { source: string; sha256: string; sourceBound: number[] } | undefined;
      let attemptId: string | undefined;
      try {
        const { attempt: checkedAttempt, notebook } = await rep.capture();
        attemptId = checkedAttempt.id;
        // Right after a rep change the timer may not have caught up; use the rep's saved time.
        if (timedRepId !== attemptId && finishTime(checkedAttempt, stages, stageOrder) === undefined) {
          activeAtCheck = checkedAttempt.activeSeconds;
        }
        snapshot = await captureSolution(notebook, exercise.id);
        const bundleUrl = new URL(exercise.checker, assets);
        const run = runCheck(snapshot.source, snapshot.sha256, bundleUrl, checkedStage);
        currentCheck = run;
        stop.disabled = isAttemptTransitioning();
        const result = await run.result;
        currentCheck = undefined;
        stop.disabled = true;
        await waitForCheckGate(attemptId, 'before-receipt');
        const updatedAttempt = await appendReceipt(attemptId, snapshot, result, undefined, activeAtCheck);
        await waitForCheckGate(attemptId, 'after-receipt');
        await renderResultWithAttempt(result, snapshot.sha256, updatedAttempt);
        await waitForCheckGate(attemptId, 'after-render');
      } catch (error) {
        currentCheck = undefined;
        stop.disabled = true;
        if (snapshot && attemptId) {
          const classified = classifyControllerError(error, exercise.checker_version, snapshot);
          const { receiptStatus, prefix } = classified;
          // The worker never reported, so record which stage was being checked.
          const result: CheckResult = { ...classified.result, stage: checkedStage };
          await waitForCheckGate(attemptId, 'before-receipt');
          const updatedAttempt = await appendReceipt(attemptId, snapshot, result, receiptStatus, activeAtCheck);
          await waitForCheckGate(attemptId, 'after-receipt');
          await renderResultWithAttempt(result, snapshot.sha256, updatedAttempt, prefix);
          await waitForCheckGate(attemptId, 'after-render');
        } else if (rep.isCurrent(checkContext)) {
          grid.restore();
          status.textContent = `Check failed${snapshot ? ` · source SHA-256 ${snapshot.sha256}` : ''}: ` +
            (error instanceof Error ? error.message : String(error));
        }
      } finally {
        setCheckInProgress(false);
        stop.disabled = true;
      }
    };

    const importLabel = document.createElement('label');
    importLabel.textContent = 'Import archive ';
    importLabel.append(importInput);

    const pathSection = document.createElement('section');
    pathSection.id = 'torch-dojo-path';
    pathSection.className = 'dojo-path-card';
    pathSection.hidden = true;
    pathSection.append(pathNav, nextExercise);

    function cardHeading(label: string): HTMLHeadingElement {
      const heading = document.createElement('h3');
      heading.className = 'dojo-card-heading';
      heading.textContent = label;
      return heading;
    }

    const kernelStatus = document.createElement('span');
    kernelStatus.className = 'dojo-kernel-status';
    kernelStatus.setAttribute('data-testid', 'dojo-kernel-status');
    kernelStatus.textContent = 'Python · starting';
    const kernelActions = document.createElement('div');
    kernelActions.className = 'dojo-action-row dojo-kernel-actions';
    kernelActions.append(kernelStatus, interrupt, restart);

    // Side rail: identity, stage seals, then quiet controls.
    const brand = document.createElement('a');
    brand.className = 'dojo-eyebrow dojo-home-link';
    brand.href = homeHref;
    brand.textContent = '← All workouts';
    const navigation = document.createElement('button');
    navigation.textContent = 'Learning path';
    navigation.setAttribute('aria-controls', pathSection.id);
    navigation.setAttribute('aria-expanded', 'false');
    navigation.disabled = !exercise.learning_path?.length && !nextExercise.textContent;
    navigation.onclick = () => {
      pathSection.hidden = !pathSection.hidden;
      navigation.setAttribute('aria-expanded', String(!pathSection.hidden));
    };
    const theme = new ThemeToggle();
    const railControls = document.createElement('div');
    railControls.id = 'torch-dojo-notebook-controls';
    railControls.className = 'dojo-rail-controls';
    railControls.append(navigation, open, kernelActions, theme.node);
    bench.rail.append(brand, timer.node, seals.node, repSummary, railControls, pathSection);

    // Answer pane header: Run, the stage to check, Check and Stop.
    stageLabel.classList.add('dojo-stage-label');
    bench.answerActions.append(run, stageLabel, check, stop);

    const helpSection = document.createElement('section');
    helpSection.className = 'dojo-card dojo-help-card';
    const helpActions = document.createElement('div');
    helpActions.className = 'dojo-action-row';
    helpActions.append(hint, collapseHints, revealSolution);
    helpSection.append(cardHeading('Need a nudge?'), helpActions, hintsRegion, solutionRegion);

    // Help is reachable from the brief: these open the hint and reference controls under the answer.
    const briefHint = document.createElement('button');
    briefHint.type = 'button';
    briefHint.textContent = 'Show a hint';
    briefHint.disabled = hints.length === 0;
    briefHint.onclick = () => {
      hint.click();
      helpSection.scrollIntoView({ block: 'nearest' });
    };
    const briefSolution = document.createElement('button');
    briefSolution.type = 'button';
    briefSolution.textContent = 'See the reference solution';
    briefSolution.disabled = referenceSolution.length === 0;
    briefSolution.onclick = () => {
      if (!solutionVisible) revealSolution.click();
      helpSection.scrollIntoView({ block: 'nearest' });
    };
    bench.briefFooter.append(document.createTextNode('Stuck? '), briefHint, briefSolution);

    const attemptSection = document.createElement('section');
    attemptSection.className = 'dojo-attempt-card';
    const attemptActions = document.createElement('div');
    attemptActions.className = 'dojo-action-row';
    attemptActions.append(newAttempt, exportAttempt, exportAll);
    attemptSection.append(attemptActions, repTableHost, notesLabel, notes, importLabel, historyRegion);
    bench.repsPanel.append(attemptSection);

    const feedbackSection = document.createElement('section');
    feedbackSection.className = 'dojo-card dojo-feedback-card';
    const reportRegion = document.createElement('div');
    reportRegion.setAttribute('role', 'status');
    const report = document.createElement('details');
    const reportSummary = document.createElement('summary');
    reportSummary.textContent = 'Check log & receipts';
    const reportHeadline = document.createElement('p');
    reportHeadline.className = 'dojo-report-headline';
    const updateReportHeadline = () => {
      reportHeadline.textContent = (status.textContent ?? '').split('\n')[0].split(' · source SHA-256')[0].slice(0, 140);
    };
    new MutationObserver(updateReportHeadline).observe(status, { childList: true, characterData: true, subtree: true });
    updateReportHeadline();
    report.append(reportSummary, status);
    reportRegion.append(reportHeadline, report);
    feedbackSection.append(cardHeading('Check receipt'), reportRegion);

    check.classList.add('dojo-primary-button');
    stop.classList.add('dojo-stop-button');
    const scratchControls = document.createElement('div');
    scratchControls.className = 'dojo-action-row dojo-scratch-controls';
    scratchControls.append(bench.scratchToggle, bench.scratchRun, bench.shortcutsToggle);
    panel.append(grid.node, readout, feedbackSection, helpSection, scratchControls, bench.scratchNote, bench.shortcutsPanel);
    setAttemptTransitionControlsDisabled(true);

    // Bench shortcuts. The selectors are more specific than the notebook's own
    // bindings, so Shift+Enter runs without moving on and Accel+Enter checks.
    app.commands.addCommand('torch-dojo:run-cell', {
      label: 'Run cell',
      execute: () => runActiveCell(focusedBenchView() ?? tracker.currentWidget)
    });
    app.commands.addCommand('torch-dojo:check', {
      label: 'Check',
      execute: () => { if (!check.disabled) check.click(); }
    });
    app.commands.addCommand('torch-dojo:run-scratch', {
      label: 'Run answer and scratch',
      execute: () => bench.runScratch()
    });
    app.commands.addCommand('torch-dojo:toggle-expand', {
      label: 'Expand the answer',
      execute: () => bench.toggleExpanded()
    });
    app.commands.addCommand('torch-dojo:stop-check', {
      label: 'Stop check',
      execute: () => { if (!stop.disabled) stop.click(); }
    });
    for (const selector of ['.dojo-bench .jp-Notebook.jp-mod-editMode', '.dojo-bench .jp-Notebook:focus']) {
      app.commands.addKeyBinding({ command: 'torch-dojo:run-cell', keys: ['Shift Enter'], selector });
      app.commands.addKeyBinding({ command: 'torch-dojo:check', keys: ['Accel Enter'], selector });
      app.commands.addKeyBinding({ command: 'torch-dojo:run-scratch', keys: ['Accel Shift Enter'], selector });
    }
    app.commands.addKeyBinding({ command: 'torch-dojo:run-scratch', keys: ['Accel Shift Enter'], selector: '.dojo-bench' });
    app.commands.addKeyBinding({ command: 'torch-dojo:toggle-expand', keys: ['Accel Shift E'], selector: '.dojo-bench' });
    app.commands.addKeyBinding({ command: 'torch-dojo:check', keys: ['Accel Enter'], selector: '.dojo-bench' });
    app.commands.addKeyBinding({ command: 'torch-dojo:stop-check', keys: ['Accel .'], selector: '.dojo-bench' });

    void app.restored.then(async () => {
      try {
        await openExerciseNotebook();
        starterNotebook = await loadStarterNotebook();
        const notebook = await waitForExerciseNotebook();
        if (notebook) {
          await rep.initialize(notebook);
          await renderAttempt();
          initialized = true;
        }
      } catch (error) {
        status.textContent = `Browser gym startup failed: ${errorMessage(error)}`;
      } finally {
        setAttemptTransitionControlsDisabled(!initialized);
      }
    }).catch(error => {
      status.textContent = `Browser gym startup failed: ${errorMessage(error)}`;
      setAttemptTransitionControlsDisabled(true);
    });

    async function autosaveTick(): Promise<string | undefined> {
      if (!rep.id) return;
      if (timer.value - savedSeconds >= 5) await saveActiveTime();
      const result = await rep.save();
      if (result.saved && !isAttemptTransitioning() && result.hash !== lastRenderedHash) {
        await renderAttempt();
      }
      return result.hash;
    }

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') void saveActiveTime().catch(() => undefined);
    });
    window.addEventListener('pagehide', () => { void saveActiveTime().catch(() => undefined); });

    const testWindow = window as BrowserGymTestWindow;
    testWindow.__dojoRunAutosaveTick = autosaveTick;
    testWindow.__dojoOpenNotebookPath = async (path: string) => {
      await app.commands.execute('docmanager:open', { path });
    };
    // Single-document mode has no document tabs; tests switch notebooks here.
    testWindow.__dojoActivateNotebook = async (name: string) => {
      const widget = tracker.find(item => item.title.label === name);
      if (!widget) throw new Error(`No open notebook named ${name}`);
      app.shell.activateById(bench.holds(widget) ? bench.id : widget.id);
    };
    window.setInterval(() => {
      return autosaveTick();
    }, 500);
  }
};

export default plugin;
