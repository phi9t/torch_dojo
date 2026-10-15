"""Learner-facing exploration and replay of concrete eval cases."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import torch
import numpy as np
from torch_judge import EvalCase, compare, get_eval_data, replay


class EvalDataTests(TestCase):
    def test_wrong_answer_can_be_inspected_and_replayed_after_a_fix(self):
        data = get_eval_data('0002_softmax_eval.py', stage='baseline')
        case = data['extreme_logits']
        bad = compare(case, torch.ones_like(case.inputs['x']))
        self.assertFalse(bad.passed)
        self.assertTrue(bad.diagnostics)
        self.assertTrue(torch.equal(bad.inputs['x'], case.inputs['x']))
        results = replay('0002_softmax_eval.py', lambda x, dim=-1: torch.softmax(x, dim), names=['extreme_logits'])
        self.assertTrue(results['extreme_logits'].passed)
        self.assertTrue(all(c.stage == 'baseline' for c in data.values()))

    def test_manual_comparison_rejects_changed_inputs_and_protects_expected_snapshot(self):
        case = EvalCase('literal', 'baseline', 'A worked example.', {'x': torch.tensor([1., 2.])}, torch.tensor([2., 4.]))
        case.expected.zero_()
        self.assertTrue(compare(case, torch.tensor([2., 4.])).passed)
        case.inputs['x'].add_(1)
        result = compare(case, torch.tensor([2., 4.]))
        self.assertFalse(result.passed)
        self.assertIn('rebuild', result.diagnostics[0])
        self.assertTrue(torch.equal(result.expectation_inputs['x'], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(result.inputs['x'], torch.tensor([2., 3.])))

    def test_retrieval_isolated_and_replay_keeps_errors_and_fresh_aliases(self):
        with TemporaryDirectory(dir=Path.cwd() / 'target') as directory:
            path = Path(directory) / '0000_isolated_eval.py'
            path.write_text('''from torch_judge import EvalCase
import torch
# %%
raise RuntimeError('reference ran')
# %% [verifiers]
def make_eval_data():
    x = torch.tensor([2.], requires_grad=True)
    return {
        'alias': EvalCase('alias', 'baseline', 'Same object twice.', {'x': x, 'same': x}, torch.tensor([4.])),
        'reject': EvalCase('reject', 'followup', 'Expected rejection.', {'x': -1}, expected_exception=ValueError),
    }
def verify_baseline_dummy(answer): pass
# %%
raise RuntimeError('runner ran')
''')
            case = get_eval_data(path)['alias']
            self.assertIs(case.inputs['x'], case.inputs['same'])
            def answer(x, same=None):
                if isinstance(x, int):
                    raise ValueError('negative')
                self.assertIs(x, same)
                self.assertIsNone(x.grad)
                (x * 2).sum().backward()
                return x * 2
            for _ in range(2):
                results = replay(path, answer)
                self.assertTrue(all(result.passed for result in results.values()))
                self.assertIs(results['alias'].inputs['x'], results['alias'].inputs['same'])
                self.assertIsInstance(results['reject'].exception, ValueError)
            result = replay(path, lambda **kwargs: 1 / 0, names=['alias'])['alias']
            self.assertFalse(result.passed)
            self.assertIn('Unexpected ZeroDivisionError', result.diagnostics[0])
            with self.assertRaisesRegex(KeyError, 'unknown or excluded'):
                replay(path, answer, names=['reject'], stage='baseline')
            with self.assertRaisesRegex(ValueError, 'unknown stage'):
                get_eval_data(path, stage='bogus')

    def test_mutating_answer_does_not_corrupt_replay_and_shapes_are_diagnostic(self):
        def mutating(x, dim=-1):
            x.zero_()
            return torch.softmax(x, dim)
        result = replay('0002_softmax_eval.py', mutating, names=['extreme_logits'])['extreme_logits']
        self.assertFalse(result.passed)
        self.assertIn('Inputs changed', result.diagnostics[0])
        clean = replay('0002_softmax_eval.py', lambda x, dim=-1: torch.softmax(x, dim), names=['extreme_logits'])
        self.assertTrue(clean['extreme_logits'].passed)
        case = get_eval_data('0002_softmax_eval.py')['equal_logits']
        bad = compare(case, torch.ones(3, dtype=torch.float64))
        self.assertTrue(any('shape' in item for item in bad.diagnostics))
        self.assertTrue(any('dtype' in item for item in bad.diagnostics))

    def test_replay_rebuilds_module_state_and_saved_answer_skips_exploration(self):
        from torch_judge import evaluate
        with TemporaryDirectory(dir=Path.cwd() / 'target') as directory:
            path = Path(directory) / '0001_modules_eval.py'
            path.write_text('''import torch
from torch_judge import EvalCase
# %% [verifiers]
def make_eval_data():
    expert = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): expert.weight.fill_(2)
    x = torch.tensor([[3.]])
    return {'module': EvalCase('module', 'production', 'Fresh parameters.', {'x': x, 'experts': [expert, expert]}, torch.tensor([[6.]]))}
def verify_baseline_answer(answer):
    case = make_eval_data()['module']
    assert torch.equal(answer(**case.inputs), case.expected)
# %%
raise RuntimeError('runner ran')
''')
            path.with_name('0001_modules.py').write_text('''import torch
# %% [answer]
def answer(x, experts): return experts[0](x)
# %%
raise RuntimeError('exploration ran')
''')
            self.assertTrue(evaluate(path, None, candidate=True))
            def mutating(x, experts):
                self.assertIs(experts[0], experts[1])
                result = experts[0](x)
                with torch.no_grad(): experts[0].weight.add_(10)
                return result
            self.assertFalse(replay(path, mutating)['module'].passed)
            clean = replay(path, lambda x, experts: experts[0](x))['module']
            self.assertTrue(clean.passed)
            self.assertIs(clean.inputs['experts'][0], clean.inputs['experts'][1])

    def test_view_aliases_and_exact_integral_and_complex_comparison(self):
        base = torch.arange(6.)
        case = EvalCase('views', 'baseline', 'Overlapping views.',
                        {'base': base, 'view': base[1:4]}, torch.tensor([2**60], dtype=torch.int64))
        result = compare(case, torch.tensor([2**60 + 1], dtype=torch.int64))
        self.assertFalse(result.passed)
        result.inputs['base'][1] = -5
        self.assertEqual(result.inputs['view'][0].item(), -5)
        complex_case = EvalCase('complex', 'baseline', 'Complex values.', {}, torch.tensor([1+2j]))
        self.assertFalse(compare(complex_case, torch.tensor([1+3j])).passed)

    def test_scalar_counts_and_flags_compare_exactly_and_detect_stale_inputs(self):
        for expected, actual in ((2**60, 2**60 + 1),
                                 (np.int64(2**60), np.int64(2**60 + 1)),
                                 (True, 1), (1, True), (1, 1.), (1., 1),
                                 (np.bool_(True), np.int64(1))):
            with self.subTest(expected=expected, actual=actual):
                case = EvalCase('scalar', 'baseline', 'Discrete scalar contract.', {}, expected)
                self.assertFalse(compare(case, actual).passed)
        for scalar in (2**60, np.int64(2**60)):
            case = EvalCase('count', 'baseline', 'Count edits invalidate expectations.',
                            {'count': scalar}, 0)
            case.inputs['count'] += 1
            result = compare(case, 0)
            self.assertFalse(result.passed)
            self.assertIn('Inputs changed', result.diagnostics[0])
        case = EvalCase('count', 'baseline', 'Python and NumPy integers agree.', {}, 2**60)
        self.assertTrue(compare(case, np.int64(2**60)).passed)
        case = EvalCase('flag', 'baseline', 'Python and NumPy flags agree.', {}, True)
        self.assertTrue(compare(case, np.bool_(True)).passed)
        case = EvalCase('float', 'baseline', 'Floating tolerance is retained.', {}, 1.)
        self.assertTrue(compare(case, 1. + 1e-7).passed)
