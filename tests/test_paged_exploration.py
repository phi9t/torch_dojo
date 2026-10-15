"""Paged attention's learner-facing inspect and replay workflow."""
from unittest import TestCase
import torch
from torch_judge import get_eval_data, replay


class PagedExplorationTests(TestCase):
    def test_wrong_partial_tail_is_retained_and_fresh_replay_fixes_it(self):
        data = get_eval_data('0093_paged_attention_eval.py')
        case = data['unused_nonfinite_tail']
        self.assertTrue(torch.isnan(case.inputs['key_blocks']).any())
        def wrong(query, key_blocks, value_blocks, block_table, context_length):
            keys = key_blocks[block_table].flatten(0, 1)
            values = value_blocks[block_table].flatten(0, 1)
            scores = torch.einsum('hd,thd->ht', query, keys) / query.shape[-1] ** .5
            return torch.einsum('ht,thv->hv', scores.softmax(-1), values)
        bad = replay('0093_paged_attention_eval.py', wrong, names=[case.name])[case.name]
        self.assertFalse(bad.passed)
        self.assertTrue(bad.diagnostics)
        self.assertIsNotNone(bad.actual)
        def correct(query, key_blocks, value_blocks, block_table, context_length):
            if not context_length:
                return query.new_zeros((query.shape[0], value_blocks.shape[-1]))
            keys = key_blocks[block_table].flatten(0, 1)[:context_length]
            values = value_blocks[block_table].flatten(0, 1)[:context_length]
            return torch.einsum('ht,thv->hv', (torch.einsum('hd,thd->ht', query, keys) / query.shape[-1] ** .5).softmax(-1), values)
        for _ in range(2):
            self.assertTrue(replay('0093_paged_attention_eval.py', correct, names=[case.name])[case.name].passed)
        self.assertFalse(get_eval_data('0093_paged_attention_eval.py', stage='baseline')['two_full_pages'].inputs['query'].requires_grad)

    def test_gradient_replay_starts_with_fresh_leaf_tensors(self):
        seen = []
        def answer(query, key_blocks, value_blocks, block_table, context_length):
            for tensor in (query, key_blocks, value_blocks):
                self.assertIsNone(tensor.grad)
                self.assertTrue(tensor.is_leaf)
            keys = key_blocks[block_table].flatten(0, 1)[:context_length]
            values = value_blocks[block_table].flatten(0, 1)[:context_length]
            output = torch.einsum('ht,thv->hv', (torch.einsum('hd,thd->ht', query, keys) / query.shape[-1] ** .5).softmax(-1), values)
            output.square().sum().backward()
            seen.append(query)
            return output
        for _ in range(2):
            result = replay('0093_paged_attention_eval.py', answer, names=['gradient_pages'])['gradient_pages']
            self.assertTrue(result.passed)
        self.assertIsNot(seen[0], seen[1])
