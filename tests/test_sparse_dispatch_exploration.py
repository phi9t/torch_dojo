"""The MoE exploration loop is exercised through public learner interfaces."""
from unittest import TestCase
import torch
from torch_judge import compare, get_eval_data, replay


class SparseDispatchExplorationTests(TestCase):
    def test_duplicate_overwrite_is_inspectable_and_fresh_replay_recovers(self):
        case = get_eval_data('0101_sparse_dispatch_eval.py')['duplicate_routes']
        def overwrite(x, expert_ids, weights, experts):
            output = torch.zeros_like(x)
            for token in range(len(x)):
                for slot in range(expert_ids.shape[1]):
                    output[token] = weights[token, slot] * experts[int(expert_ids[token, slot])](x[token:token+1])[0]
            return output, torch.argsort(expert_ids.flatten(), stable=True)
        result = compare(case, overwrite(**case.inputs))
        self.assertFalse(result.passed)
        self.assertTrue(any('output[0]' in error for error in result.diagnostics))
        self.assertEqual(result.actual[0].shape, result.expected[0].shape)
        # Explicit arithmetic oracle for these diagonal, deterministic experts.
        def fixed(x, expert_ids, weights, experts):
            factors = expert_ids.to(x.dtype) + 1
            return x * (weights * factors).sum(1, keepdim=True), torch.argsort(expert_ids.flatten(), stable=True)
        self.assertTrue(replay('0101_sparse_dispatch_eval.py', fixed, names=['duplicate_routes'])['duplicate_routes'].passed)

    def test_modules_gradients_and_aliases_are_fresh_after_a_bad_attempt(self):
        original = get_eval_data('0101_sparse_dispatch_eval.py')['shared_expert']
        self.assertIs(original.inputs['experts'][0], original.inputs['experts'][1])
        with torch.no_grad():
            original.inputs['x'].fill_(99)
            original.inputs['experts'][0].weight.fill_(99)
        original.inputs['experts'][0].weight.grad = torch.ones_like(original.inputs['experts'][0].weight)
        def answer(x, expert_ids, weights, experts):
            self.assertIs(experts[0], experts[1])
            self.assertIsNone(x.grad)
            self.assertIsNone(weights.grad)
            self.assertIsNone(experts[0].weight.grad)
            output = experts[0](x) * weights.sum(1, keepdim=True)
            output.square().sum().backward()
            return output, torch.argsort(expert_ids.flatten(), stable=True)
        for _ in range(2):
            result = replay('0101_sparse_dispatch_eval.py', answer, names=['shared_expert'])['shared_expert']
            self.assertTrue(result.passed, result.diagnostics)
            self.assertIs(result.inputs['experts'][0], result.inputs['experts'][1])

    def test_detached_forward_is_rejected_by_followup_verifiers(self):
        from torch_judge import evaluate
        def detached(x, expert_ids, weights, experts):
            rows = []
            for token in range(len(x)):
                row = [weights[token, slot] * experts[int(expert_ids[token, slot])](x[token:token+1])[0]
                       for slot in range(expert_ids.shape[1])]
                rows.append(torch.stack(row).sum(0))
            return torch.stack(rows).detach(), torch.argsort(expert_ids.flatten(), stable=True)
        self.assertFalse(evaluate('0101_sparse_dispatch_eval.py', detached, stage='followup'))
