import { AttemptStore, restoreAttemptNotebook, saveAttemptNotebook, type AttemptRecord, type ContentsLike } from './attempts';
import { captureSolution, type NotebookSnapshot } from './source';

export type RepChange = { kind: 'new' } | { kind: 'resume'; id: string } | { kind: 'import'; archive: Blob };
export type RepPhase = 'before-swap' | 'after-load' | 'after-activation';
export interface RepContext { id?: string; version: number }
export type RepChangeResult =
  | { status: 'ignored' }
  | { status: 'changed'; attempt: AttemptRecord; imported: number; context: RepContext }
  | { status: 'failed'; message: string; recovered: boolean; context: RepContext };

/** The visible gym notebook, even when an unrelated notebook has focus. */
export interface RepNotebook {
  read(): Promise<NotebookSnapshot | undefined>;
  open(): Promise<void>;
  replace(notebook: NotebookSnapshot): Promise<void>;
  starter(): Promise<NotebookSnapshot>;
}

interface Options {
  exerciseId: string;
  store: AttemptStore;
  contents: ContentsLike;
  notebook: RepNotebook;
  onBusy?: (busy: boolean) => void;
  beforeChange?: () => Promise<void>;
  onPhase?: (kind: RepChange['kind'], id: string, phase: RepPhase) => Promise<void>;
}
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/**
 * Owns the rep visible in this page, independently of another tab's selection.
 * Switching is exclusive; persistence and recovery finish before saving resumes.
 * Captures never change identity, so late checks remain bound to their own rep.
 */
export class RepLifecycle {
  private activeId?: string;
  private generation = 0;
  private busy = false;
  constructor(private readonly options: Options) {}

  get id(): string | undefined { return this.activeId; }
  get version(): number { return this.generation; }
  get locked(): boolean { return this.busy; }
  context(): RepContext { return { id: this.activeId, version: this.generation }; }
  isCurrent(context: RepContext): boolean {
    return !this.busy && context.id === this.activeId && context.version === this.generation;
  }

  async initialize(notebook: NotebookSnapshot): Promise<void> {
    const attempt = await this.options.store.getOrCreateAttempt(this.options.exerciseId, notebook);
    await this.load(attempt);
    this.activeId = attempt.id;
  }

  private async persist(id: string, notebook: NotebookSnapshot): Promise<AttemptRecord> {
    const attempt = await this.options.store.saveNotebook(id, notebook);
    await saveAttemptNotebook(this.options.contents, attempt);
    return attempt;
  }

  private async load(attempt: AttemptRecord): Promise<void> {
    let notebook = attempt.notebook;
    try {
      notebook = await restoreAttemptNotebook(this.options.contents, attempt);
    } catch {
      await saveAttemptNotebook(this.options.contents, attempt);
    }
    await this.options.notebook.replace(notebook);
  }

  /** Save and capture one rep's visible answer before running a check or revealing help. */
  async capture(): Promise<{ attempt: AttemptRecord; notebook: NotebookSnapshot }> {
    const context = this.context();
    if (!context.id || !this.isCurrent(context)) throw new Error('Wait for the rep to finish opening.');
    const notebook = await this.options.notebook.read();
    if (!notebook) throw new Error('Open the workout notebook first.');
    if (!this.isCurrent(context)) throw new Error('The rep changed. Check the answer in the open rep.');
    const attempt = await this.persist(context.id, notebook);
    if (!this.isCurrent(context)) throw new Error('The rep changed. Check the answer in the open rep.');
    return { attempt, notebook };
  }

  /** Autosave skips stale captures, including a switch during asynchronous hashing. */
  async save(): Promise<{ saved: boolean; hash?: string }> {
    const context = this.context();
    if (!context.id || !this.isCurrent(context)) return { saved: false };
    const notebook = await this.options.notebook.read();
    if (!notebook || !this.isCurrent(context)) return { saved: false };
    await this.persist(context.id, notebook);
    if (!this.isCurrent(context)) return { saved: false };
    let hash: string | undefined;
    try { hash = (await captureSolution(notebook, this.options.exerciseId)).sha256; } catch { /* An unfinished answer can still be saved. */ }
    return this.isCurrent(context) ? { saved: true, hash } : { saved: false };
  }

  /** Keep a scratch-cell insertion bound to this rep while its kernel responds. */
  async editNotebook(edit: () => Promise<void>): Promise<void> {
    if (this.busy) throw new Error('Wait for the current rep change to finish.');
    this.busy = true;
    this.generation += 1;
    this.options.onBusy?.(true);
    try {
      await this.options.beforeChange?.();
      await edit();
      const notebook = await this.options.notebook.read();
      if (notebook && this.activeId) await this.persist(this.activeId, notebook);
    } finally {
      this.generation += 1;
      this.busy = false;
      this.options.onBusy?.(false);
    }
  }

  async change(change: RepChange): Promise<RepChangeResult> {
    if (this.busy) return { status: 'ignored' };
    this.busy = true;
    this.generation += 1;
    this.options.onBusy?.(true);
    const previousId = this.activeId;
    let previousNotebook: NotebookSnapshot | undefined;
    let importedIds: string[] = [];
    let oldMapping: Record<string, string> | undefined;
    let recover = false;
    let recovered = true;
    let failure: string | undefined;
    let activated: AttemptRecord | undefined;
    const { store, notebook, contents, exerciseId } = this.options;
    try {
      // An archive is also a recovery path after browser storage was cleared.
      const missingOutgoing = change.kind === 'import' && previousId && !await store.getAttempt(previousId);
      if (!missingOutgoing) await this.options.beforeChange?.();
      const visible = await notebook.read();
      previousNotebook = visible ? clone(visible) : undefined;
      if (previousId && previousNotebook && !missingOutgoing) await this.persist(previousId, previousNotebook);
      let target: AttemptRecord;
      if (change.kind === 'new') {
        target = await store.newAttempt(exerciseId, clone(await notebook.starter()), { activate: false });
        await saveAttemptNotebook(contents, target);
      } else if (change.kind === 'resume') {
        const found = await store.getAttempt(change.id);
        if (!found || found.exerciseId !== exerciseId) throw new Error('That rep does not belong to this workout.');
        target = found;
      } else {
        oldMapping = { ...store.snapshot().activeByExercise };
        importedIds = await store.importArchive(change.archive);
        recover = true;
        const imported = (await Promise.all(importedIds.map(id => store.getAttempt(id))))
          .filter((attempt): attempt is AttemptRecord => Boolean(attempt));
        for (const attempt of imported) await saveAttemptNotebook(contents, attempt);
        const storedActive = await store.getOrCreateAttempt(exerciseId, await notebook.starter());
        target = imported.length === 1 && imported[0].exerciseId === exerciseId ? imported[0] : storedActive;
      }
      recover = true;
      await this.options.onPhase?.(change.kind, target.id, 'before-swap');
      await notebook.open();
      await this.load(target);
      await this.options.onPhase?.(change.kind, target.id, 'after-load');
      activated = await store.setActiveAttempt(target.id);
      this.activeId = activated.id;
      await this.options.onPhase?.(change.kind, target.id, 'after-activation');
    } catch (error) {
      failure = message(error);
      if (recover) {
        try {
          // Roll back only this import. Keep concurrent receipts and other tabs' changes.
          if (importedIds.length && oldMapping) store.rollbackImportedAttempts(importedIds, oldMapping);
          if (previousId) {
            const previous = change.kind === 'import'
              ? await store.getAttempt(previousId) : await store.setActiveAttempt(previousId);
            if (!previous) throw new Error(`Rep not found: ${previousId}`);
            if (previousNotebook) {
              await this.persist(previousId, previousNotebook);
              await notebook.replace(previousNotebook);
            } else await this.load(previous);
          } else if (previousNotebook) await notebook.replace(previousNotebook);
          this.activeId = previousId;
          failure = `Rep change failed: ${failure}\nPrevious rep restored.`;
        } catch (error) {
          recovered = false;
          failure = `Rep change failed: ${failure}\nRecovery error: ${message(error)}\nAutosave is disabled to avoid overwriting saved reps.`;
        }
      }
    } finally {
      this.generation += 1;
      this.busy = !recovered;
      this.options.onBusy?.(this.busy);
    }
    const context = this.context();
    return failure !== undefined
      ? { status: 'failed', message: failure, recovered, context }
      : { status: 'changed', attempt: activated!, imported: importedIds.length, context };
  }
}
