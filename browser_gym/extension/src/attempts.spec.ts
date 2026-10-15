import { describe, expect, it } from 'vitest';
import { AttemptStore, saveAttemptNotebook, restoreAttemptNotebook, type CheckReceipt } from './attempts';
import type { NotebookSnapshot } from './source';
import archiveBeforeRepRename from './fixtures/archive-before-rep-rename.json';

function notebook(source: string): NotebookSnapshot {
  return {
    metadata: { dojo: { exercise_id: '0059' } },
    cells: [
      { cell_type: 'markdown', source: 'Prompt' },
      { cell_type: 'code', source, metadata: { dojo: { role: 'solution', exercise_id: '0059' } } }
    ]
  };
}

function receipt(overrides: Partial<CheckReceipt> = {}): CheckReceipt {
  return {
    id: overrides.id ?? 'receipt-1',
    exerciseId: '0059',
    checkedAt: '2026-10-06T00:00:00.000Z',
    source: 'import numpy as np\n\ndef answer():\n    return 1\n',
    sha256: 'a1c90121714ceec89453f7eb493ccf9dd439b11e146c97cd91832e946a62afa8',
    sourceBound: overrides.sourceBound ?? [1],
    status: 'passed',
    checkerVersion: 'checker-a',
    runtime: 'pyodide-0.28.3',
    packageIdentity: { numpy: '2.1.3' },
    fixtureId: 'canonical-0059',
    assistance: [],
    result: {
      status: 'passed',
      checks: [{ name: 'verify_baseline', status: 'passed', elapsed_ms: 1 }],
      error: null,
      checker_version: 'checker-a',
      runtime: 'pyodide-0.28.3',
      fixture_id: 'canonical-0059'
    },
    ...overrides
  };
}

describe('AttemptStore', () => {
  it('creates_distinct_notebooks_for_attempts', async () => {
    const storage = new Map<string, string>();
    const ids = ['attempt-a', 'attempt-b'];
    const store = new AttemptStore({ storage, createId: () => ids.shift() ?? 'extra-id' });

    const first = await store.newAttempt('0059', notebook('first = 1'));
    await store.saveNotes(first.id, 'first notes');
    const second = await store.newAttempt('0059', notebook('second = 2'));

    expect(first.id).not.toBe(second.id);
    expect(first.notebookPath).toMatch(/0059\/attempt-a\.ipynb$/);
    expect(second.notebookPath).toMatch(/0059\/attempt-b\.ipynb$/);
    expect(first.notebookPath).not.toBe(second.notebookPath);

    const reloaded = new AttemptStore({ storage });
    expect((await reloaded.getAttempt(first.id))?.notebook).toEqual(notebook('first = 1'));
    expect((await reloaded.getAttempt(first.id))?.notes).toBe('first notes');
    expect((await reloaded.getAttempt(second.id))?.notebook).toEqual(notebook('second = 2'));
    expect((await reloaded.getOrCreateAttempt('0059', notebook('unused'))).id).toBe(second.id);
    await reloaded.setActiveAttempt(first.id);
    expect((await reloaded.getOrCreateAttempt('0059', notebook('unused'))).id).toBe(first.id);
    const inactive = await reloaded.newAttempt('0059', notebook('inactive = 3'), { activate: false });
    expect((await reloaded.getAttempt(inactive.id))?.notebook).toEqual(notebook('inactive = 3'));
    expect((await reloaded.getOrCreateAttempt('0059', notebook('unused'))).id).toBe(first.id);
  });

  it('keeps_reveal_history_after_collapse', async () => {
    const store = new AttemptStore({ storage: new Map(), createId: (() => {
      const ids = ['assisted-attempt', 'independent-attempt', 'hint-event', 'solution-event'];
      return () => ids.shift() ?? 'extra-id';
    })() });
    const assisted = await store.newAttempt('0059', notebook('def answer():\n    return 2\n'));

    await store.recordAssistance(assisted.id, 'hint', 1);
    await store.recordAssistance(assisted.id, 'solution', 0);
    await store.appendReceipt(assisted.id, receipt({
      id: 'assisted-pass',
      source: 'def answer():\n    return 2\n',
      sha256: 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306'
    }));

    const afterCollapse = await store.getAttempt(assisted.id);
    expect(afterCollapse?.notebook).toEqual(notebook('def answer():\n    return 2\n'));
    expect(afterCollapse?.assistance.map(event => ({ kind: event.kind, level: event.level }))).toEqual([
      { kind: 'hint', level: 1 },
      { kind: 'solution', level: 0 }
    ]);
    expect(afterCollapse?.receipts[0].assistance.map(event => event.kind)).toEqual(['hint', 'solution']);
    expect(store.describeReceipt(assisted.id, 'assisted-pass',
      'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306')).toMatchObject({
      classification: 'assisted-pass',
      isPassingReceipt: true,
      isStale: false
    });

    const independent = await store.newAttempt('0059', notebook('def answer():\n    return 3\n'));
    await store.appendReceipt(independent.id, receipt({
      id: 'independent-pass',
      source: 'def answer():\n    return 3\n',
      sha256: 'a4f7fe11cfc0d59da8707c96a9ee1588f6f431b57b5f2f30b60c3efd2489646c'
    }));
    expect(store.describeReceipt(independent.id, 'independent-pass',
      'a4f7fe11cfc0d59da8707c96a9ee1588f6f431b57b5f2f30b60c3efd2489646c')).toMatchObject({
      classification: 'independent-pass',
      isPassingReceipt: true,
      isStale: false
    });
    expect(store.describeReceipt(independent.id, 'independent-pass', 'edited-hash')).toMatchObject({
      classification: 'edited-since-check',
      isStale: true
    });
  });

  it('classifies reference substitutions as assisted continuations after export and import', async () => {
    const storage = new Map<string, string>();
    const store = new AttemptStore({ storage, createId: () => 'substituted-attempt' });
    const attempt = await store.newAttempt('mlp', notebook('REFERENCE_OVERRIDES = {"loss_and_grads"}'));
    const substituted = receipt({
      id: 'substituted-pass',
      exerciseId: 'mlp',
      source: 'REFERENCE_OVERRIDES = {"loss_and_grads"}',
      sha256: 'd21affe56771f4036c50501463175c08944ede85562670720b1b339fe0110a6e',
      result: {
        status: 'passed',
        checks: [
          {
            name: 'substitution_loss_and_grads',
            status: 'passed',
            evidence: { function: 'loss_and_grads', checkpoint: 'backward', independent_credit: false }
          },
          { name: 'checkpoint_backward', status: 'passed' }
        ],
        error: null,
        checker_version: 'checker-a',
        runtime: 'pyodide-0.28.3',
        fixture_id: 'canonical-mlp',
        stage: 'backward',
        checkpoint: 'backward',
        reference_overrides: ['loss_and_grads'],
        substitutions: [{ function: 'loss_and_grads', checkpoint: 'backward', independent_credit: false }],
        owned_checkpoint_passed: false
      }
    });

    await store.appendReceipt(attempt.id, substituted);
    expect(store.describeReceipt(attempt.id, 'substituted-pass', substituted.sha256)).toMatchObject({
      classification: 'reference-assisted-pass',
      label: 'PASSED (REFERENCE-ASSISTED CONTINUATION)',
      isPassingReceipt: true,
      isStale: false
    });

    const imported = new AttemptStore({ storage: new Map(), createId: () => 'imported-substituted-attempt' });
    await imported.importArchive(await store.exportAll());
    expect(imported.describeReceipt(attempt.id, 'substituted-pass', substituted.sha256)).toMatchObject({
      classification: 'reference-assisted-pass',
      label: 'PASSED (REFERENCE-ASSISTED CONTINUATION)'
    });
  });

  it('rejects_corrupt_archive_atomically', async () => {
    const storage = new Map<string, string>();
    const ids = ['first', 'second', 'hint'];
    const store = new AttemptStore({ storage, createId: () => ids.shift() ?? 'extra-id' });
    const first = await store.newAttempt('0059', notebook('def answer():\n    return 2\n'));
    await store.saveNotes(first.id, 'first notes');
    await store.recordAssistance(first.id, 'hint', 1);
    await store.appendReceipt(first.id, receipt({
      id: 'first-pass',
      source: 'def answer():\n    return 2\n',
      sha256: 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306'
    }));
    const second = await store.newAttempt('0059', notebook('def answer():\n    return 3\n'));

    const singleArchive = JSON.parse(await (await store.exportAttempt(first.id)).text());
    expect(Object.keys(singleArchive.files).sort()).toEqual([
      'attempts/first/assistance.json',
      'attempts/first/notebook.ipynb',
      'attempts/first/notes.md',
      'attempts/first/receipts.json',
      'attempts/first/solution.py'
    ]);
    expect(singleArchive.files['attempts/first/solution.py']).toBe('def answer():\n    return 2\n');

    const allArchive = JSON.parse(await (await store.exportAll()).text());
    expect(allArchive.attempts.map((attempt: { id: string }) => attempt.id)).toEqual(['first', 'second']);

    const imported = new AttemptStore({ storage: new Map() });
    await expect(imported.importArchive(new Blob([JSON.stringify(allArchive)]))).resolves.toEqual(['first', 'second']);
    expect((await imported.getAttempt(first.id))?.receipts[0].sha256)
      .toBe('f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306');
    expect((await imported.getAttempt(first.id))?.notes).toBe('first notes');

    const beforeCorrupt = await store.getAttempt(first.id);
    allArchive.attempts[0].receipts[0].sha256 = '0'.repeat(64);
    await expect(store.importArchive(new Blob([JSON.stringify(allArchive)]))).rejects.toThrow(/source hash/i);
    expect(await store.getAttempt(first.id)).toEqual(beforeCorrupt);
    expect(await store.getAttempt(second.id)).toEqual(second);
  });

  it('imports_conflicting_attempts_without_overwriting_local_history', async () => {
    const storage = new Map<string, string>();
    const ids = ['shared', 'imported-shared'];
    const store = new AttemptStore({ storage, createId: () => ids.shift() ?? 'extra-id' });
    const local = await store.newAttempt('0059', notebook('local notebook'));
    await store.saveNotes(local.id, 'local notes');
    await store.appendReceipt(local.id, receipt({
      id: 'local-pass',
      source: 'def answer():\n    return 2\n',
      sha256: 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306'
    }));

    const archived = await (await store.exportAttempt(local.id)).text();
    const archive = JSON.parse(archived);
    archive.attempts[0].notes = 'older imported notes';
    archive.attempts[0].notebook = notebook('imported notebook');
    archive.files[`attempts/${local.id}/notebook.ipynb`] = JSON.stringify(notebook('imported notebook'), null, 2) + '\n';
    archive.files[`attempts/${local.id}/solution.py`] = 'imported notebook';
    archive.files[`attempts/${local.id}/notes.md`] = 'older imported notes';
    const localBefore = await store.getAttempt(local.id);

    await expect(store.importArchive(new Blob([JSON.stringify(archive)]))).resolves.toEqual(['imported-shared']);
    expect(await store.getAttempt(local.id)).toEqual(localBefore);
    const imported = await store.getAttempt('imported-shared');
    expect(imported?.notes).toBe('older imported notes');
    expect(imported?.notebookPath).toMatch(/0059\/imported-shared\.ipynb$/);
  });

  it('restores a prior attempt store snapshot after a failed UI transition', async () => {
    const storage = new Map<string, string>();
    const ids = ['local-attempt', 'imported-attempt'];
    const store = new AttemptStore({ storage, createId: () => ids.shift() ?? 'extra-id' });
    const local = await store.newAttempt('0059', notebook('local notebook'));
    await store.saveNotes(local.id, 'local notes');
    const beforeImport = store.snapshot();

    const importedStore = new AttemptStore({ storage: new Map(), createId: () => 'imported-attempt' });
    const imported = await importedStore.newAttempt('0059', notebook('imported notebook'));
    const archive = await importedStore.exportAttempt(imported.id);
    await store.importArchive(archive);
    expect((await store.listAttempts('0059')).map(attempt => attempt.id)).toEqual(['local-attempt', 'imported-attempt']);
    expect((await store.getOrCreateAttempt('0059', notebook('unused'))).id).toBe('imported-attempt');

    store.restore(beforeImport);

    expect(await store.listAttempts('0059')).toEqual(beforeImport.attempts);
    expect((await store.getOrCreateAttempt('0059', notebook('unused'))).id).toBe('local-attempt');
  });

  it('rolls back failed imports without erasing concurrent same-origin updates', async () => {
    const storage = new Map<string, string>();
    const localIds = ['local-0059', 'local-0060', 'local-0060-concurrent'];
    const store = new AttemptStore({ storage, createId: () => localIds.shift() ?? 'extra-local-id' });
    const local0059 = await store.newAttempt('0059', notebook('local 0059 source'));
    const local0060 = await store.newAttempt('0060', notebook('local 0060 source'));
    const concurrent0060 = await store.newAttempt('0060', notebook('concurrent 0060 source'), { activate: false });
    const beforeImport = store.snapshot();

    const importIds = ['imported-0059', 'imported-0060'];
    const importStore = new AttemptStore({ storage: new Map(), createId: () => importIds.shift() ?? 'extra-import-id' });
    await importStore.newAttempt('0059', notebook('imported 0059 source'));
    await importStore.newAttempt('0060', notebook('imported 0060 source'));
    const importedIds = await store.importArchive(await importStore.exportAll());

    const otherTabStore = new AttemptStore({ storage });
    await otherTabStore.saveNotes(local0059.id, 'other tab notes');
    await otherTabStore.appendReceipt(local0059.id, receipt({ id: 'other-tab-receipt' }));
    await otherTabStore.setActiveAttempt(concurrent0060.id);

    store.rollbackImportedAttempts(importedIds, beforeImport.activeByExercise);

    expect(await store.getAttempt(importedIds[0])).toBeUndefined();
    expect(await store.getAttempt(importedIds[1])).toBeUndefined();
    const localAfterRollback = await store.getAttempt(local0059.id);
    expect(localAfterRollback?.notes).toBe('other tab notes');
    expect(localAfterRollback?.receipts.map(item => item.id)).toEqual(['other-tab-receipt']);
    const archive = JSON.parse(await (await store.exportAll()).text());
    expect(archive.attempts.map((attempt: { id: string }) => attempt.id)).toEqual([
      local0059.id,
      local0060.id,
      concurrent0060.id
    ]);
    expect(archive.activeByExercise['0059']).toBe(local0059.id);
    expect(archive.activeByExercise['0060']).toBe(concurrent0060.id);
  });

  it('rolls back failed imports without overwriting a concurrent same-exercise active selection', async () => {
    const storage = new Map<string, string>();
    const localIds = ['local-a', 'local-b'];
    const store = new AttemptStore({ storage, createId: () => localIds.shift() ?? 'extra-local-id' });
    const localA = await store.newAttempt('0059', notebook('local A source'));
    const localB = await store.newAttempt('0059', notebook('local B source'), { activate: false });
    await store.setActiveAttempt(localA.id);
    const beforeImport = store.snapshot();

    const importStore = new AttemptStore({ storage: new Map(), createId: () => 'imported-a' });
    const imported = await importStore.newAttempt('0059', notebook('imported source'));
    const importedIds = await store.importArchive(await importStore.exportAttempt(imported.id));
    expect(store.snapshot().activeByExercise['0059']).toBe(importedIds[0]);

    const otherTabStore = new AttemptStore({ storage });
    await otherTabStore.setActiveAttempt(localB.id);
    await otherTabStore.saveNotes(localB.id, 'local B notes from another tab');
    await otherTabStore.appendReceipt(localB.id, receipt({
      id: 'local-b-receipt',
      source: 'def answer():\n    return 2\n',
      sha256: 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306'
    }));

    store.rollbackImportedAttempts(importedIds, beforeImport.activeByExercise);

    expect(await store.getAttempt(importedIds[0])).toBeUndefined();
    const localAAfterRollback = await store.getAttempt(localA.id);
    expect(localAAfterRollback?.notebook).toEqual(notebook('local A source'));
    const localBAfterRollback = await store.getAttempt(localB.id);
    expect(localBAfterRollback?.notebook).toEqual(notebook('local B source'));
    expect(localBAfterRollback?.notes).toBe('local B notes from another tab');
    expect(localBAfterRollback?.receipts.map(item => item.id)).toEqual(['local-b-receipt']);
    expect(store.snapshot().activeByExercise['0059']).toBe(localB.id);
  });

  it('saves_attempt_notebooks_to_contents_paths', async () => {
    const saved = new Map<string, unknown>();
    const contents = {
      async save(path: string, model: { type: string; format: string | null; content: unknown }) {
        saved.set(path, model.content);
      },
      async get(path: string) {
        return { content: saved.get(path) };
      }
    };
    const store = new AttemptStore({ storage: new Map(), createId: () => 'attempt-a' });
    const attempt = await store.newAttempt('0059', notebook('first = 1'));

    await saveAttemptNotebook(contents, attempt);

    expect(saved.get('attempts/0059/attempt-a.ipynb')).toEqual(notebook('first = 1'));
    await expect(restoreAttemptNotebook(contents, attempt)).resolves.toEqual(notebook('first = 1'));
  });

  it('serializes attempt notebook writes so an older delayed contents save cannot overwrite a newer snapshot', async () => {
    let releaseOldNotebookWrite: (() => void) | undefined;
    const oldNotebookWriteStarted = new Promise<void>(resolve => {
      const saved = resolve;
      const complete = () => {
        saved();
      };
      releaseOldNotebookWrite = complete;
    });
    let recordNewNotebookWrite: (() => void) | undefined;
    const newNotebookWriteCompleted = new Promise<void>(resolve => {
      recordNewNotebookWrite = resolve;
    });
    let oldWriteGateResolve: (() => void) | undefined;
    const oldWriteGate = new Promise<void>(resolve => {
      oldWriteGateResolve = resolve;
    });
    const saved = new Map<string, unknown>();
    const contents = {
      async save(path: string, model: { type: string; format: string | null; content: unknown }) {
        if (path.endsWith('.ipynb') && model.content &&
            JSON.stringify(model.content).includes('older autosave source')) {
          releaseOldNotebookWrite?.();
          await oldWriteGate;
        }
        saved.set(path, model.content);
        if (path.endsWith('.ipynb') && model.content &&
            JSON.stringify(model.content).includes('newer transition source')) {
          recordNewNotebookWrite?.();
        }
      },
      async get(path: string) {
        return { content: saved.get(path) };
      }
    };
    const store = new AttemptStore({ storage: new Map(), createId: () => 'attempt-a' });
    const older = await store.newAttempt('0059', notebook('older autosave source'));
    const newer = { ...older, notebook: notebook('newer transition source') };

    const olderSave = saveAttemptNotebook(contents, older);
    await oldNotebookWriteStarted;
    const newerSave = saveAttemptNotebook(contents, newer);
    await Promise.race([
      newNotebookWriteCompleted,
      new Promise(resolve => setTimeout(resolve, 0))
    ]);

    oldWriteGateResolve?.();
    await Promise.all([olderSave, newerSave]);

    expect(saved.get('attempts/0059/attempt-a.ipynb')).toEqual(notebook('newer transition source'));
    await expect(restoreAttemptNotebook(contents, newer)).resolves.toEqual(notebook('newer transition source'));
  });

  it('keeps append-only receipts with exact checked source snapshots after reload', async () => {
    const storage = new Map<string, string>();
    const store = new AttemptStore({ storage });
    const attempt = await store.getOrCreateAttempt('0059', notebook('old = 1'));

    const first = receipt();
    await store.saveNotebook(attempt.id, notebook('edited after check'));
    await store.saveNotes(attempt.id, 'Remember the max-subtraction trick.');
    await store.appendReceipt(attempt.id, first);

    first.source = 'mutated';
    first.sha256 = 'mutated';
    const current = await store.getAttempt(attempt.id);
    expect(current?.receipts).toEqual([receipt()]);
    expect(current?.notebook).toEqual(notebook('edited after check'));
    expect(current?.notes).toBe('Remember the max-subtraction trick.');

    await store.appendReceipt(attempt.id, receipt({
      id: 'receipt-2',
      source: 'second source',
      sha256: '943dc3ca1c82562ec66512bd7b155c63f312d1185c889cc8667f9b7d026147e0',
      checkedAt: '2026-10-06T00:00:01.000Z'
    }));
    await store.saveNotebook(attempt.id, notebook('edited again'));

    const reloaded = new AttemptStore({ storage });
    const persisted = await reloaded.getAttempt(attempt.id);
    expect(persisted?.receipts).toEqual([
      receipt(),
      receipt({
        id: 'receipt-2',
        source: 'second source',
        sha256: '943dc3ca1c82562ec66512bd7b155c63f312d1185c889cc8667f9b7d026147e0',
        checkedAt: '2026-10-06T00:00:01.000Z'
      })
    ]);
    expect(persisted?.notebook).toEqual(notebook('edited again'));
    expect(persisted?.notes).toBe('Remember the max-subtraction trick.');
  });

  it('marks a passing receipt stale when the current source hash changes', async () => {
    const store = new AttemptStore({ storage: new Map() });
    const attempt = await store.getOrCreateAttempt('0059', notebook('answer'));
    await store.appendReceipt(attempt.id, receipt({
      status: 'passed',
      source: 'def answer():\n    return 2\n',
      sha256: 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306'
    }));

    expect(store.describeReceipt(attempt.id, 'receipt-1', 'f760ecee8d16c1da2171e50cb1143db3102e4f71e0e307d28c85331dd0a07306')).toMatchObject({
      isPassingReceipt: true,
      isStale: false
    });
    expect(store.describeReceipt(attempt.id, 'receipt-1', 'edited-hash')).toMatchObject({
      isPassingReceipt: true,
      isStale: true
    });
  });

  it('persists source-bound cell indices on every receipt and never marks non-passing statuses stale', async () => {
    const storage = new Map<string, string>();
    const store = new AttemptStore({ storage });
    const attempt = await store.getOrCreateAttempt('0059', notebook('src'));
    const defaultSourceHash = 'a1c90121714ceec89453f7eb493ccf9dd439b11e146c97cd91832e946a62afa8';
    await store.appendReceipt(attempt.id, receipt({
      id: 'r-passed', status: 'passed', sha256: defaultSourceHash, sourceBound: [1]
    }));
    await store.appendReceipt(attempt.id, receipt({
      id: 'r-timeout', status: 'timeout', sha256: defaultSourceHash, sourceBound: [1]
    }));
    await store.appendReceipt(attempt.id, receipt({
      id: 'r-stopped',
      status: 'stopped',
      source: 'def answer():\n    return 3\n',
      sha256: 'a4f7fe11cfc0d59da8707c96a9ee1588f6f431b57b5f2f30b60c3efd2489646c',
      sourceBound: [1, 3]
    }));
    await store.appendReceipt(attempt.id, receipt({
      id: 'r-failed', status: 'failed', sha256: defaultSourceHash, sourceBound: [1]
    }));

    const reloaded = new AttemptStore({ storage });
    const loaded = await reloaded.getAttempt(attempt.id);
    expect(loaded?.receipts.find(r => r.id === 'r-passed')?.sourceBound).toEqual([1]);
    expect(loaded?.receipts.find(r => r.id === 'r-timeout')?.sourceBound).toEqual([1]);
    expect(loaded?.receipts.find(r => r.id === 'r-stopped')?.sourceBound).toEqual([1, 3]);
    expect(loaded?.receipts.find(r => r.id === 'r-failed')?.sourceBound).toEqual([1]);

    // Only passing receipts become stale; timeout/stopped/failed never do.
    const diffHash = 'totally-different-sha256-hash-value-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
    expect(reloaded.describeReceipt(attempt.id, 'r-passed', diffHash)).toMatchObject({
      isPassingReceipt: true, isStale: true
    });
    expect(reloaded.describeReceipt(attempt.id, 'r-timeout', diffHash)).toMatchObject({
      isPassingReceipt: false, isStale: false
    });
    expect(reloaded.describeReceipt(attempt.id, 'r-stopped', diffHash)).toMatchObject({
      isPassingReceipt: false, isStale: false
    });
    expect(reloaded.describeReceipt(attempt.id, 'r-failed', diffHash)).toMatchObject({
      isPassingReceipt: false, isStale: false
    });
  });

  it('imports an archive exported before the learner-facing rep rename', async () => {
    const store = new AttemptStore({ storage: new Map() });
    const blob = new Blob([JSON.stringify(archiveBeforeRepRename)], { type: 'application/json' });

    const [id] = await store.importArchive(blob);

    const rep = await store.getAttempt(id);
    expect(rep?.notes).toBe('notes written before the rep rename');
    expect(rep?.receipts.map(item => item.id)).toEqual(['receipt-pre-rename']);
  });

  it('exports archives an older build can still import', async () => {
    const store = new AttemptStore({ storage: new Map(), createId: () => 'attempt-after-rename' });
    await store.newAttempt('0059', notebook('after = 1'));

    const exported = JSON.parse(await (await store.exportAll()).text());

    expect(exported.format).toBe(archiveBeforeRepRename.format);
    expect(exported.version).toBe(archiveBeforeRepRename.version);
    expect(Object.keys(exported).sort()).toEqual(Object.keys(archiveBeforeRepRename).sort());
    // An older build reads the fields it knows and ignores the timing fields added since.
    expect(Object.keys(exported.attempts[0])).toEqual(expect.arrayContaining(Object.keys(archiveBeforeRepRename.attempts[0])));
    expect(Object.keys(exported.files).every(path => path.startsWith('attempts/'))).toBe(true);
  });

  it('loads reps saved before timing with no active time, no predictions and unknown splits', async () => {
    const store = new AttemptStore({ storage: new Map() });
    const blob = new Blob([JSON.stringify(archiveBeforeRepRename)], { type: 'application/json' });

    const [id] = await store.importArchive(blob);

    const rep = await store.getAttempt(id);
    expect(rep?.activeSeconds).toBe(0);
    expect(rep?.predictions).toEqual({});
    expect(rep?.receipts[0].activeSecondsAtCheck).toBeUndefined();
  });

  it('keeps active time, predictions and receipt timing through export and import', async () => {
    const store = new AttemptStore({ storage: new Map(), createId: () => 'timed-rep' });
    const rep = await store.newAttempt('0059', notebook('timed = 1'));
    await store.saveActiveSeconds(rep.id, 95.5);
    await store.savePrediction(rep.id, 0, 'shape (2, 3)');
    const source = 'import numpy as np\n\ndef answer():\n    return 1\n';
    await store.appendReceipt(rep.id, receipt({ source, activeSecondsAtCheck: 90 }));

    const reloaded = new AttemptStore({ storage: new Map() });
    await reloaded.importArchive(await store.exportAll());
    const imported = (await reloaded.listAttempts('0059'))[0];

    expect(imported.activeSeconds).toBe(95.5);
    expect(imported.predictions).toEqual({ 0: 'shape (2, 3)' });
    expect(imported.receipts[0].activeSecondsAtCheck).toBe(90);
  });

  it('never moves active time backwards', async () => {
    const store = new AttemptStore({ storage: new Map(), createId: () => 'monotonic-rep' });
    const rep = await store.newAttempt('0059', notebook('timed = 1'));
    await store.saveActiveSeconds(rep.id, 40);
    await store.saveActiveSeconds(rep.id, 12);
    expect((await store.getAttempt(rep.id))?.activeSeconds).toBe(40);
  });

  it('rejects an archive whose timing is not a non-negative number', async () => {
    const store = new AttemptStore({ storage: new Map(), createId: () => 'bad-timing' });
    await store.newAttempt('0059', notebook('timed = 1'));
    const archive = JSON.parse(await (await store.exportAll()).text());
    archive.attempts[0].activeSeconds = -5;

    const fresh = new AttemptStore({ storage: new Map() });
    await expect(fresh.importArchive(new Blob([JSON.stringify(archive)]))).rejects.toThrow(/active time/);
    expect(await fresh.listAttempts()).toEqual([]);
  });
});
