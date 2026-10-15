import type { CheckSample } from './checker';

export function replayCode(cases: unknown[], sourceHash: string): string {
  const payload = JSON.stringify(JSON.stringify({ cases, sourceHash }));
  return `import json as _gym_json
import copy as _gym_copy
import numpy as _gym_np

def _gym_decode(v):
    if not isinstance(v, dict) or 'kind' not in v:
        return v
    k = v['kind']
    if k == 'array':
        return _gym_np.array(_gym_decode(v['data']), dtype=v['dtype']).reshape(v['shape'])
    if k == 'float':
        return float(v['value'])
    if k == 'int':
        return int(v['value'])
    if k in ('tuple', 'list'):
        items = [_gym_decode(x) for x in v['items']]
        return tuple(items) if k == 'tuple' else items
    if k == 'dict':
        return {key: _gym_decode(x) for key, x in v['items'].items()}
    return v

_gym_payload = _gym_json.loads(${payload})
gym_source_hash = _gym_payload['sourceHash']
gym_cases = [{key: _gym_decode(value) for key, value in case.items()} for case in _gym_payload['cases']]
gym_passed = [case for case in gym_cases if case['check_status'] == 'passed']
gym_failed = [case for case in gym_cases if case['check_status'] == 'failed']

def gym_case(index=0):
    """Return a fresh copy of a captured example, including NumPy arrays."""
    return _gym_copy.deepcopy(gym_cases[index])

def gym_try(index=0):
    """Run the current notebook function on a fresh copy of captured inputs."""
    case = gym_case(index)
    if not case.get('replayable'):
        raise ValueError('This example has preview-only inputs; it cannot be replayed.')
    fn = globals().get(case['function'])
    if not callable(fn):
        raise NameError('Run your solution cell first to define ' + case['function'])
    return fn(*case['args'], **case['kwargs'])

print(f"Loaded {len(gym_cases)} captured examples. Use gym_case(0), gym_try(0), gym_passed, or gym_failed.")
`;
}

export function element<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, text?: string): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

function formatValue(value: unknown): string {
  if (value === undefined) return 'No value returned';
  if (value === null || typeof value !== 'object') return String(value);
  const v = value as Record<string, any>;
  if (v.kind === 'array') return `ndarray · shape (${v.shape.join(', ') + (v.shape.length === 1 ? ',' : '')}) · ${v.dtype}\n${formatValue(v.data)}`;
  if (v.kind === 'list' || v.kind === 'tuple') {
    const parts = v.items.map(formatValue);
    return `${v.kind === 'tuple' ? '(' : '['}${parts.join(parts.some((p: string) => p.includes('\n')) ? ',\n\n' : ', ')}${v.kind === 'tuple' ? ')' : ']'}`;
  }
  if (v.kind === 'dict') return Object.entries(v.items).map(([key, item]) => `${key}: ${formatValue(item)}`).join('\n') || '{}';
  if (v.kind === 'preview') return v.text;
  if (v.kind === 'float' || v.kind === 'int') return v.value;
  return JSON.stringify(value, null, 2);
}

/** One captured call: its inputs, the expected value, and what the learner's code returned. */
export function sampleCard(sample: CheckSample, index: number): HTMLElement {
  const card = element('div', 'dojo-sample');
  card.dataset.testid = 'dojo-sample';
  card.append(element('div', 'dojo-sample-title', `Example ${index} · call ${sample.call} · ${sample.replayable ? `gym_case(${index})` : 'preview only'}`));
  const grid = element('div', 'dojo-sample-grid');
  const args = sample.args as { kind?: string; items?: unknown[] };
  const namedArgs = args?.kind === 'tuple' && Array.isArray(args.items)
    ? args.items.map((value, i) => `${sample.argument_names?.[i] ?? `arg ${i}`}\n${formatValue(value)}`).join('\n\n')
    : formatValue(sample.args);
  const input = `${namedArgs}${formatValue(sample.kwargs) === '{}' ? '' : '\nkwargs: ' + formatValue(sample.kwargs)}`;
  for (const [label, content] of [
    ['Inputs', input],
    ['Expected', sample.expected_error ?? (sample.expected === undefined ? 'Expected output unavailable for this sample' : formatValue(sample.expected))],
    ['Your output', sample.error ?? formatValue(sample.actual)]
  ]) {
    const column = element('div', 'dojo-sample-column');
    column.append(element('h4', '', label), element('pre', '', content));
    grid.append(column);
  }
  card.append(grid);
  return card;
}
