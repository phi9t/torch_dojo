import { describe, expect, it, vi } from 'vitest';
import { AttemptStore, type ContentsLike } from './attempts';
import { RepLifecycle, type RepNotebook, type RepPhase } from './repLifecycle';
import type { NotebookSnapshot } from './source';

const notebook = (source: string): NotebookSnapshot => ({
  metadata: { dojo: { exercise_id: '0059' } },
  cells: [{ cell_type: 'code', source, metadata: { dojo: { role: 'solution', exercise_id: '0059' } } }]
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
async function practice() {
  let sequence = 0;
  const storage = new Map<string, string>();
  const store = new AttemptStore({ storage, createId: () => `rep-${++sequence}` });
  const files = new Map<string, unknown>();
  const contents: ContentsLike = {
    async get(path) {
      if (!files.has(path)) throw new Error('Missing file');
      return { content: files.get(path) };
    },
    async save(path, model) { files.set(path, structuredClone(model.content)); }
  };
  let visible = notebook('starter');
  const view: RepNotebook = {
    read: vi.fn(async () => structuredClone(visible)),
    open: vi.fn(async () => {}),
    replace: vi.fn(async next => { visible = structuredClone(next); }),
    starter: async () => notebook('starter')
  };
  const onPhase = vi.fn(async (_kind: string, _id: string, _phase: RepPhase) => {});
  const onBusy = vi.fn();
  const rep = new RepLifecycle({ exerciseId: '0059', store, contents, notebook: view, onPhase, onBusy });
  await rep.initialize(visible);
  return { rep, store, storage, view, onPhase, onBusy, edit: (source: string) => { visible = notebook(source); } };
}

describe('RepLifecycle', () => {
  it('preserves outgoing edits and opens a fresh rep before activating it', async () => {
    const { rep, store, view, edit, onPhase } = await practice();
    const previous = rep.id!;
    edit('my unsaved answer');
    onPhase.mockImplementation(async (kind, _id, phase) => {
      if (kind === 'new' && phase === 'after-load') expect(rep.id).toBe(previous);
    });
    expect((await rep.change({ kind: 'new' })).status).toBe('changed');
    expect((await store.getAttempt(previous))?.notebook).toEqual(notebook('my unsaved answer'));
    expect(await view.read()).toEqual(notebook('starter'));
    const next = rep.id!;
    await rep.change({ kind: 'resume', id: previous });
    expect(await view.read()).toEqual(notebook('my unsaved answer'));
    expect(rep.id).not.toBe(next);
  });

  it('ignores overlapping switches and autosave while loading', async () => {
    const { rep, onPhase } = await practice();
    const entered = deferred<void>();
    const release = deferred<void>();
    onPhase.mockImplementation(async (_kind, _id, phase) => {
      if (phase === 'after-load') { entered.resolve(); await release.promise; }
    });
    const switching = rep.change({ kind: 'new' });
    await entered.promise;
    expect(await rep.change({ kind: 'new' })).toEqual({ status: 'ignored' });
    expect(await rep.save()).toEqual({ saved: false });
    release.resolve();
    expect((await switching).status).toBe('changed');
    expect(rep.locked).toBe(false);
  });

  it.each(['after-load', 'after-activation'] as RepPhase[])('restores the previous rep after %s fails', async phase => {
    const { rep, store, view, edit, onPhase } = await practice();
    const previous = rep.id;
    edit('keep this answer');
    onPhase.mockImplementation(async (_kind, _id, current) => { if (current === phase) throw new Error('swap failed'); });
    const outcome = await rep.change({ kind: 'new' });
    expect(outcome).toMatchObject({ status: 'failed', recovered: true });
    expect(rep.id).toBe(previous);
    expect(rep.locked).toBe(false);
    expect(await view.read()).toEqual(notebook('keep this answer'));
    expect((await store.getOrCreateAttempt('0059', notebook('unused'))).id).toBe(previous);
  });

  it('leaves saving disabled when restoring the notebook fails', async () => {
    const { rep, view } = await practice();
    vi.mocked(view.replace).mockRejectedValue(new Error('notebook unavailable'));
    expect(await rep.change({ kind: 'new' })).toMatchObject({ status: 'failed', recovered: false });
    expect(rep.locked).toBe(true);
    expect(await rep.save()).toEqual({ saved: false });
    expect(await rep.change({ kind: 'new' })).toEqual({ status: 'ignored' });
  });

  it('a delayed check capture cannot reset the newly selected rep', async () => {
    const { rep, view, edit } = await practice();
    edit('old answer');
    const pending = deferred<NotebookSnapshot>();
    vi.mocked(view.read).mockImplementationOnce(() => pending.promise);
    const capture = rep.capture();
    const rejected = expect(capture).rejects.toThrow('rep changed');
    await rep.change({ kind: 'new' });
    const current = rep.id;
    pending.resolve(notebook('old answer'));
    await rejected;
    expect(rep.id).toBe(current);
    expect(await view.read()).toEqual(notebook('starter'));
  });

  it('discards an autosave captured before a switch', async () => {
    const { rep, view } = await practice();
    const pending = deferred<NotebookSnapshot>();
    vi.mocked(view.read).mockImplementationOnce(() => pending.promise);
    const saving = rep.save();
    await rep.change({ kind: 'new' });
    pending.resolve(notebook('old answer'));
    expect(await saving).toEqual({ saved: false });
    expect(await view.read()).toEqual(notebook('starter'));
  });

  it('failed import retains concurrent notes and another tab’s active mapping', async () => {
    const { rep, store, onPhase } = await practice();
    const previous = rep.id!;
    const other = await store.newAttempt('0059', notebook('other tab'), { activate: false });
    const remote = new AttemptStore({ storage: new Map(), createId: () => 'imported' });
    await remote.newAttempt('0059', notebook('imported answer'));
    onPhase.mockImplementation(async (_kind, _id, phase) => {
      if (phase !== 'after-load') return;
      await store.saveNotes(previous, 'written while importing');
      await store.setActiveAttempt(other.id);
      throw new Error('load failed');
    });
    expect(await rep.change({ kind: 'import', archive: await remote.exportAll() }))
      .toMatchObject({ status: 'failed', recovered: true });
    expect(rep.id).toBe(previous);
    expect(await store.getAttempt('imported')).toBeUndefined();
    expect((await store.getAttempt(previous))?.notes).toBe('written while importing');
    expect((await store.getOrCreateAttempt('0059', notebook('unused'))).id).toBe(other.id);
    expect((await rep.capture()).attempt.id).toBe(previous);
  });

  it('can recover an archive after browser storage was cleared', async () => {
    const { rep, store, storage, view, edit } = await practice();
    edit('saved in archive');
    await rep.save();
    const archive = await store.exportAll();
    storage.clear();
    expect(await rep.change({ kind: 'import', archive })).toMatchObject({ status: 'changed', imported: 1 });
    expect(await view.read()).toEqual(notebook('saved in archive'));
    expect((await rep.capture()).attempt.id).toBe(rep.id);
  });

  it('blocks autosave if import rollback itself fails', async () => {
    const { rep, store, onPhase } = await practice();
    const archive = await store.exportAll();
    onPhase.mockRejectedValue(new Error('swap failed'));
    vi.spyOn(store, 'rollbackImportedAttempts').mockImplementation(() => { throw new Error('storage unavailable'); });
    expect(await rep.change({ kind: 'import', archive })).toMatchObject({ status: 'failed', recovered: false });
    expect(await rep.save()).toEqual({ saved: false });
  });
});
