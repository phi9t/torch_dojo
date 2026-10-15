import type { CheckResult } from './checker';
import type { NotebookSnapshot } from './source';

export interface AssistanceEvent {
  kind: 'hint' | 'solution';
  level: number;
  revealedAt: string;
}

export interface CheckReceipt {
  id: string;
  exerciseId: string;
  checkedAt: string;
  source: string;
  sha256: string;
  /**
   * Cell indices (0-based, document order) in the notebook snapshot that the
   * concatenated `source` was built from. Present on every receipt so the
   * learner can always trace the checked source to the exact notebook cells.
   * Stop/timeout receipts are "source-bound": they preserve this mapping even
   * though the worker never returned verifier results.
   */
  sourceBound: number[];
  status: 'passed' | 'failed' | 'timeout' | 'stopped';
  checkerVersion: string;
  runtime: string;
  packageIdentity: Record<string, string>;
  fixtureId: string;
  result: CheckResult;
  assistance: AssistanceEvent[];
  /** The rep's active time when this check ran. Absent on receipts saved before timing. */
  activeSecondsAtCheck?: number;
}

export interface AttemptRecord {
  id: string;
  exerciseId: string;
  createdAt: string;
  updatedAt: string;
  notebookPath: string;
  notebook: NotebookSnapshot;
  notes: string;
  assistance: AssistanceEvent[];
  receipts: CheckReceipt[];
  /** Active time spent on this rep (visible page, not idle). 0 for reps saved before timing. */
  activeSeconds: number;
  /** Drill predictions, keyed by the drill's index in the workout. */
  predictions: Record<string, string>;
}

export interface ReceiptDescription {
  receipt: CheckReceipt;
  isPassingReceipt: boolean;
  isStale: boolean;
  isReferenceAssisted: boolean;
  classification: 'assisted-pass' | 'independent-pass' | 'reference-assisted-pass' | 'edited-since-check' | 'failed';
  label: string;
}

type StorageLike = Storage | Map<string, string>;

interface AttemptState {
  attempts: AttemptRecord[];
  activeByExercise: Record<string, string>;
}

export type AttemptStoreSnapshot = AttemptState;

interface AttemptStoreOptions {
  storage?: StorageLike;
  key?: string;
  now?: () => Date;
  createId?: () => string;
}

interface NewAttemptOptions {
  activate?: boolean;
}

export interface ContentsLike {
  get(path: string, options?: { type?: string; format?: 'json' | 'text' | 'base64' | null; content?: boolean }): Promise<{ content?: unknown }>;
  save(path: string, model: { type: string; format: 'json' | 'text' | 'base64' | null; content: unknown }): Promise<unknown>;
  rename?(path: string, newPath: string): Promise<unknown>;
  newUntitled?(options?: { path?: string; type?: string }): Promise<{ path: string }>;
}

const DEFAULT_KEY = 'torch-dojo.browser-gym.attempts.v1';
const ARCHIVE_FORMAT = 'torch-dojo.browser-gym.attempts';
const ARCHIVE_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function getStored(storage: StorageLike, key: string): string | null {
  if (storage instanceof Map) return storage.get(key) ?? null;
  return storage.getItem(key);
}

function setStored(storage: StorageLike, key: string, value: string): void {
  if (storage instanceof Map) storage.set(key, value);
  else storage.setItem(key, value);
}

function browserStorage(): StorageLike {
  if (typeof window !== 'undefined' && window.localStorage) return window.localStorage;
  return new Map<string, string>();
}

function createDefaultId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `attempt-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function notebookPath(exerciseId: string, attemptId: string): string {
  return `attempts/${exerciseId}/${attemptId}.ipynb`;
}

function parentDirectories(path: string): string[] {
  const parts = path.split('/').slice(0, -1);
  return parts.map((_, index) => parts.slice(0, index + 1).join('/'));
}

function sourceText(source: string | string[]): string {
  return Array.isArray(source) ? source.join('') : source;
}

function resultUsesReferenceAssistance(result: CheckResult): boolean {
  if ((result.reference_overrides?.length ?? 0) > 0) return true;
  if ((result.substitutions?.length ?? 0) > 0) return true;
  if (result.owned_checkpoint_passed === false) return true;
  return result.checks.some(check => {
    const evidence = check.evidence;
    return typeof evidence === 'object' && evidence !== null &&
      'independent_credit' in evidence &&
      (evidence as { independent_credit?: unknown }).independent_credit === false;
  });
}

function deriveSolution(notebook: NotebookSnapshot): string {
  return notebook.cells
    .filter(cell => cell.cell_type === 'code' && cell.metadata?.dojo?.role === 'solution')
    .map(cell => sourceText(cell.source))
    .join('\n\n');
}

async function sha256(source: string): Promise<string> {
  const bytes = new TextEncoder().encode(source);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function validateReceiptSourceHash(receipt: CheckReceipt): Promise<void> {
  const actual = await sha256(receipt.source);
  if (actual !== receipt.sha256) {
    throw new Error(`Receipt ${receipt.id} source hash mismatch: expected ${receipt.sha256}, computed ${actual}`);
  }
}

function readActiveTime(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error('Malformed active time in rep store.');
  }
  return value;
}

function readPredictions(value: unknown): Record<string, string> {
  if (value === undefined) return {};
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.values(value).some(text => typeof text !== 'string')) {
    throw new Error('Malformed drill predictions in rep store.');
  }
  return { ...(value as Record<string, string>) };
}

function normalizeReceipt(receipt: Partial<CheckReceipt>): CheckReceipt {
  if (!receipt.id || !receipt.exerciseId || !receipt.checkedAt || receipt.source === undefined ||
      !receipt.sha256 || !receipt.status || !receipt.checkerVersion || !receipt.runtime ||
      !receipt.fixtureId || !receipt.result) {
    throw new Error('Malformed receipt in rep store.');
  }
  return {
    id: receipt.id,
    exerciseId: receipt.exerciseId,
    checkedAt: receipt.checkedAt,
    source: receipt.source,
    sha256: receipt.sha256,
    sourceBound: Array.isArray(receipt.sourceBound) ? [...receipt.sourceBound] : [],
    status: receipt.status,
    checkerVersion: receipt.checkerVersion,
    runtime: receipt.runtime,
    packageIdentity: receipt.packageIdentity ?? {},
    fixtureId: receipt.fixtureId,
    result: receipt.result,
    assistance: clone(receipt.assistance ?? []),
    ...(receipt.activeSecondsAtCheck === undefined ? {} : { activeSecondsAtCheck: readActiveTime(receipt.activeSecondsAtCheck) })
  };
}

function normalizeAttempt(attempt: Partial<AttemptRecord>): AttemptRecord {
  if (!attempt.id || !attempt.exerciseId || !attempt.createdAt || !attempt.updatedAt || !attempt.notebook) {
    throw new Error('Malformed rep in rep store.');
  }
  return {
    id: attempt.id,
    exerciseId: attempt.exerciseId,
    createdAt: attempt.createdAt,
    updatedAt: attempt.updatedAt,
    notebookPath: attempt.notebookPath ?? notebookPath(attempt.exerciseId, attempt.id),
    notebook: clone(attempt.notebook),
    notes: attempt.notes ?? '',
    assistance: clone(attempt.assistance ?? []),
    receipts: (attempt.receipts ?? []).map(receipt => normalizeReceipt(receipt)),
    activeSeconds: readActiveTime(attempt.activeSeconds) ?? 0,
    predictions: readPredictions(attempt.predictions)
  };
}

interface AttemptArchive {
  format: typeof ARCHIVE_FORMAT;
  version: typeof ARCHIVE_VERSION;
  exportedAt: string;
  activeByExercise: Record<string, string>;
  attempts: AttemptRecord[];
  files: Record<string, string>;
}

function archiveAttemptFiles(attempt: AttemptRecord): Record<string, string> {
  const prefix = `attempts/${attempt.id}`;
  return {
    [`${prefix}/notebook.ipynb`]: JSON.stringify(attempt.notebook, null, 2) + '\n',
    [`${prefix}/solution.py`]: deriveSolution(attempt.notebook),
    [`${prefix}/notes.md`]: attempt.notes,
    [`${prefix}/assistance.json`]: JSON.stringify(attempt.assistance, null, 2) + '\n',
    [`${prefix}/receipts.json`]: JSON.stringify(attempt.receipts, null, 2) + '\n'
  };
}

async function archiveBlob(archive: AttemptArchive): Promise<Blob> {
  return new Blob([JSON.stringify(archive, null, 2) + '\n'], { type: 'application/json' });
}

async function ensureContentsDirectory(contents: ContentsLike, path: string): Promise<void> {
  try {
    await contents.get(path, { type: 'directory', content: false });
    return;
  } catch {
    // Fall through to creation.
  }

  try {
    await contents.save(path, { type: 'directory', format: 'json', content: null });
    return;
  } catch (saveError) {
    if (!contents.newUntitled || !contents.rename) throw saveError;
  }

  const parent = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';
  const created = await contents.newUntitled?.({ path: parent, type: 'directory' });
  if (!created?.path || created.path === path) return;
  await contents.rename?.(created.path, path);
}

const notebookWriteQueues = new Map<string, Promise<void>>();

async function writeAttemptNotebook(contents: ContentsLike, path: string, notebook: NotebookSnapshot): Promise<void> {
  for (const directory of parentDirectories(path)) {
    await ensureContentsDirectory(contents, directory);
  }
  await contents.save(path, {
    type: 'notebook',
    format: 'json',
    content: clone(notebook)
  });
}

export async function saveAttemptNotebook(contents: ContentsLike, attempt: AttemptRecord): Promise<void> {
  const path = attempt.notebookPath;
  const notebook = clone(attempt.notebook);
  const previous = notebookWriteQueues.get(path) ?? Promise.resolve();
  const current = previous.catch(() => undefined).then(() => {
    return writeAttemptNotebook(contents, path, notebook);
  });
  notebookWriteQueues.set(path, current);
  try {
    await current;
  } finally {
    if (notebookWriteQueues.get(path) === current) {
      notebookWriteQueues.delete(path);
    }
  }
}

export async function restoreAttemptNotebook(contents: ContentsLike, attempt: AttemptRecord): Promise<NotebookSnapshot> {
  await notebookWriteQueues.get(attempt.notebookPath);
  const model = await contents.get(attempt.notebookPath, { type: 'notebook', format: 'json', content: true });
  if (!model.content || typeof model.content !== 'object') {
    throw new Error(`Rep notebook is missing: ${attempt.notebookPath}`);
  }
  return clone(model.content as NotebookSnapshot);
}

export class AttemptStore {
  private readonly storage: StorageLike;
  private readonly key: string;
  private readonly now: () => Date;
  private readonly createId: () => string;

  constructor(options: AttemptStoreOptions = {}) {
    this.storage = options.storage ?? browserStorage();
    this.key = options.key ?? DEFAULT_KEY;
    this.now = options.now ?? (() => new Date());
    this.createId = options.createId ?? createDefaultId;
  }

  async newAttempt(exerciseId: string, notebook: NotebookSnapshot, options: NewAttemptOptions = {}): Promise<AttemptRecord> {
    const state = this.load();
    const timestamp = this.now().toISOString();
    const id = this.createId();
    const attempt: AttemptRecord = {
      id,
      exerciseId,
      createdAt: timestamp,
      updatedAt: timestamp,
      notebookPath: notebookPath(exerciseId, id),
      notebook: clone(notebook),
      notes: '',
      assistance: [],
      receipts: [],
      activeSeconds: 0,
      predictions: {}
    };
    state.attempts.push(attempt);
    if (options.activate ?? true) state.activeByExercise[exerciseId] = id;
    this.save(state);
    return clone(attempt);
  }

  async getOrCreateAttempt(exerciseId: string, notebook: NotebookSnapshot): Promise<AttemptRecord> {
    const state = this.load();
    const activeId = state.activeByExercise[exerciseId];
    const existing = state.attempts.find(attempt => attempt.id === activeId);
    if (existing) return clone(existing);

    return this.newAttempt(exerciseId, notebook);
  }

  async getAttempt(attemptId: string): Promise<AttemptRecord | undefined> {
    const attempt = this.load().attempts.find(item => item.id === attemptId);
    return attempt ? clone(attempt) : undefined;
  }

  async setActiveAttempt(attemptId: string): Promise<AttemptRecord> {
    const state = this.load();
    const attempt = state.attempts.find(item => item.id === attemptId);
    if (!attempt) throw new Error(`Rep not found: ${attemptId}`);
    state.activeByExercise[attempt.exerciseId] = attempt.id;
    this.save(state);
    return clone(attempt);
  }

  async listAttempts(exerciseId?: string): Promise<AttemptRecord[]> {
    const attempts = this.load().attempts;
    return clone(exerciseId ? attempts.filter(attempt => attempt.exerciseId === exerciseId) : attempts);
  }

  async saveNotebook(attemptId: string, notebook: NotebookSnapshot): Promise<AttemptRecord> {
    return this.updateAttempt(attemptId, attempt => {
      attempt.notebook = clone(notebook);
    });
  }

  async saveNotes(attemptId: string, notes: string): Promise<AttemptRecord> {
    return this.updateAttempt(attemptId, attempt => {
      attempt.notes = notes;
    });
  }

  /**
   * Records the rep's active time. Time only moves forward, except that a
   * finished rep's time is set to its finish time (`final`).
   */
  async saveActiveSeconds(attemptId: string, seconds: number, options: { final?: boolean } = {}): Promise<AttemptRecord> {
    const value = readActiveTime(seconds) ?? 0;
    return this.updateAttempt(attemptId, attempt => {
      attempt.activeSeconds = options.final ? value : Math.max(attempt.activeSeconds, value);
    });
  }

  async savePrediction(attemptId: string, drill: number, text: string): Promise<AttemptRecord> {
    return this.updateAttempt(attemptId, attempt => {
      attempt.predictions = { ...attempt.predictions, [drill]: text };
    });
  }

  async appendReceipt(attemptId: string, receipt: Omit<CheckReceipt, 'assistance'> & { assistance?: AssistanceEvent[] }): Promise<AttemptRecord> {
    const normalized = normalizeReceipt(receipt);
    await validateReceiptSourceHash(normalized);
    return this.updateAttempt(attemptId, attempt => {
      if (attempt.receipts.some(existing => existing.id === normalized.id)) {
        throw new Error(`Receipt already exists: ${normalized.id}`);
      }
      normalized.assistance = clone(attempt.assistance);
      attempt.receipts.push(normalized);
    });
  }

  async recordAssistance(attemptId: string, kind: AssistanceEvent['kind'], level: number): Promise<AttemptRecord> {
    return this.updateAttempt(attemptId, attempt => {
      attempt.assistance.push({ kind, level, revealedAt: this.now().toISOString() });
    });
  }

  describeReceipt(attemptId: string, receiptId: string, currentSourceHash: string): ReceiptDescription {
    const receipt = this.load().attempts.find(attempt => attempt.id === attemptId)
      ?.receipts.find(item => item.id === receiptId);
    if (!receipt) throw new Error(`Receipt not found: ${receiptId}`);
    const cloned = clone(receipt);
    const isPassingReceipt = cloned.status === 'passed';
    // Only *passing* receipts become stale when the source hash diverges.
    // Timeout, stopped, and failed receipts describe a past event tied to
    // their captured source, not an endorsement of the current notebook state.
    const isStale = isPassingReceipt && cloned.sha256 !== currentSourceHash;
    const isReferenceAssisted = isPassingReceipt && resultUsesReferenceAssistance(cloned.result);
    const classification = !isPassingReceipt
      ? 'failed'
      : isStale
        ? 'edited-since-check'
        : isReferenceAssisted
          ? 'reference-assisted-pass'
          : cloned.assistance.length
            ? 'assisted-pass'
            : 'independent-pass';
    const label = !isPassingReceipt
      ? cloned.status.toUpperCase()
      : isStale
        ? isReferenceAssisted
          ? 'STALE PASS (REFERENCE-ASSISTED CONTINUATION)'
          : 'STALE PASS'
        : isReferenceAssisted
          ? 'PASSED (REFERENCE-ASSISTED CONTINUATION)'
          : cloned.assistance.length
            ? 'PASSED (ASSISTED PASS)'
            : 'PASSED (INDEPENDENT PASS)';
    return {
      receipt: cloned,
      isPassingReceipt,
      isStale,
      isReferenceAssisted,
      classification,
      label
    };
  }

  async exportAttempt(attemptId: string): Promise<Blob> {
    const state = this.load();
    const attempt = state.attempts.find(item => item.id === attemptId);
    if (!attempt) throw new Error(`Rep not found: ${attemptId}`);
    return this.exportAttempts([attempt], state.activeByExercise);
  }

  async exportAll(): Promise<Blob> {
    const state = this.load();
    return this.exportAttempts(state.attempts, state.activeByExercise);
  }

  async importArchive(blob: Blob): Promise<string[]> {
    let parsed: unknown;
    try {
      parsed = JSON.parse(await blob.text());
    } catch (error) {
      throw new Error(`Malformed rep archive: ${error instanceof Error ? error.message : String(error)}`);
    }
    const archive = parsed as Partial<AttemptArchive>;
    if (archive.format !== ARCHIVE_FORMAT || archive.version !== ARCHIVE_VERSION ||
        !Array.isArray(archive.attempts) || !archive.files || typeof archive.files !== 'object') {
      throw new Error('Malformed rep archive.');
    }
    const imported = archive.attempts.map(attempt => normalizeAttempt(attempt));
    for (const attempt of imported) {
      for (const receipt of attempt.receipts) {
        await validateReceiptSourceHash(receipt);
      }
      const files = archiveAttemptFiles(attempt);
      for (const [path, contents] of Object.entries(files)) {
        if (archive.files[path] !== contents) {
          throw new Error(`Archive file mismatch: ${path}`);
        }
      }
    }

    const state = this.load();
    const usedIds = new Set(state.attempts.map(attempt => attempt.id));
    const importedIdMap = new Map<string, string>();
    const remapped = imported.map(attempt => {
      const originalId = attempt.id;
      let next = clone(attempt);
      if (usedIds.has(next.id)) {
        let replacement = this.createId();
        while (usedIds.has(replacement)) replacement = this.createId();
        next = {
          ...next,
          id: replacement,
          notebookPath: notebookPath(next.exerciseId, replacement)
        };
      }
      usedIds.add(next.id);
      importedIdMap.set(originalId, next.id);
      return next;
    });
    const finalImportedIds = new Set(remapped.map(attempt => attempt.id));
    state.attempts = state.attempts.concat(clone(remapped));
    state.activeByExercise = { ...state.activeByExercise };
    const archiveActiveExercises = new Set<string>();
    for (const [exerciseId, attemptId] of Object.entries(archive.activeByExercise ?? {})) {
      const remappedId = importedIdMap.get(attemptId);
      if (remappedId && finalImportedIds.has(remappedId)) {
        state.activeByExercise[exerciseId] = remappedId;
        archiveActiveExercises.add(exerciseId);
      }
    }
    for (const attempt of remapped) {
      if (!archiveActiveExercises.has(attempt.exerciseId)) state.activeByExercise[attempt.exerciseId] = attempt.id;
    }
    this.save(state);
    return remapped.map(attempt => attempt.id);
  }

  snapshot(): AttemptStoreSnapshot {
    return clone(this.load());
  }

  restore(snapshot: AttemptStoreSnapshot): void {
    this.save(clone(snapshot));
  }

  rollbackImportedAttempts(
    importedIds: string[],
    previousActiveByExercise: Record<string, string>
  ): AttemptStoreSnapshot {
    const imported = new Set(importedIds);
    const state = this.load();
    state.attempts = state.attempts.filter(attempt => !imported.has(attempt.id));
    const remainingIds = new Set(state.attempts.map(attempt => attempt.id));
    state.activeByExercise = { ...state.activeByExercise };
    for (const [exerciseId, attemptId] of Object.entries(state.activeByExercise)) {
      if (!imported.has(attemptId)) continue;
      const previousActiveId = previousActiveByExercise[exerciseId];
      if (previousActiveId && remainingIds.has(previousActiveId)) {
        state.activeByExercise[exerciseId] = previousActiveId;
      } else {
        delete state.activeByExercise[exerciseId];
      }
    }
    this.save(state);
    return clone(state);
  }

  private async exportAttempts(attempts: AttemptRecord[], activeByExercise: Record<string, string>): Promise<Blob> {
    const cloned = clone(attempts);
    const attemptIds = new Set(cloned.map(attempt => attempt.id));
    const exportedActive = Object.fromEntries(
      Object.entries(activeByExercise).filter(([, attemptId]) => attemptIds.has(attemptId))
    );
    const files: Record<string, string> = {};
    for (const attempt of cloned) {
      for (const receipt of attempt.receipts) {
        await validateReceiptSourceHash(receipt);
      }
      Object.assign(files, archiveAttemptFiles(attempt));
    }
    return archiveBlob({
      format: ARCHIVE_FORMAT,
      version: ARCHIVE_VERSION,
      exportedAt: this.now().toISOString(),
      activeByExercise: exportedActive,
      attempts: cloned,
      files
    });
  }

  private updateAttempt(attemptId: string, update: (attempt: AttemptRecord) => void): AttemptRecord {
    const state = this.load();
    const attempt = state.attempts.find(item => item.id === attemptId);
    if (!attempt) throw new Error(`Rep not found: ${attemptId}`);
    update(attempt);
    attempt.updatedAt = this.now().toISOString();
    this.save(state);
    return clone(attempt);
  }

  private load(): AttemptState {
    const raw = getStored(this.storage, this.key);
    if (!raw) return { attempts: [], activeByExercise: {} };
    const parsed = JSON.parse(raw) as Partial<AttemptState>;
    return {
      attempts: Array.isArray(parsed.attempts) ? parsed.attempts.map(attempt => normalizeAttempt(attempt)) : [],
      activeByExercise: parsed.activeByExercise ?? {}
    };
  }

  private save(state: AttemptState): void {
    setStored(this.storage, this.key, JSON.stringify(state));
  }
}
