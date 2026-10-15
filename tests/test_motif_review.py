"""The native review boundary: schedules, portable persistence and real probes."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch_judge.review import ReviewSession, motifs, schedule, validate_archive

ROOT = Path(__file__).resolve().parents[1]


class MotifReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'progress.json'
        self.review = ReviewSession(self.path)

    def test_every_motif_has_real_workouts_and_a_runnable_probe(self):
        cards = motifs()
        self.assertEqual(len({m['id'] for m in cards}), len(cards))
        for m in cards:
            with self.subTest(motif=m['id']):
                for key in ('title', 'topic', 'prompt', 'answer', 'probe', 'transfer'):
                    self.assertTrue(m[key])
                for w in m['workouts']:
                    self.assertTrue(list((ROOT / 'dojo').glob(w + '_*.py')), w)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(m['probe'], m['id'], 'exec'), {})
                self.assertTrue(output.getvalue())

    def test_reveal_then_rate_and_reload_without_double_rating(self):
        card = self.review.next()
        self.assertNotIn(card.motif['answer'], card._repr_html_())
        with self.assertRaises(ValueError):
            self.review.rate('good')
        self.review.reveal()
        with patch('torch_judge.review.time.time', return_value=1000):
            self.review.rate('good')
        with self.assertRaises(ValueError):
            self.review.rate('good')
        restored = ReviewSession(self.path)
        self.assertNotIn(card.motif, restored.queue(1001))
        self.assertEqual(restored.queue(1000 + 3 * 86400)[0], card.motif)

    def test_due_before_new_and_interval_cap(self):
        first, second = motifs()[:2]
        for card, rating, now in [(first, 'good', 0), (second, 'again', 100)]:
            self.review.next(card['id']); self.review.reveal()
            with patch('torch_judge.review.time.time', return_value=now):
                self.review.rate(rating)
        self.assertEqual(self.review.queue(800)[0], second)
        state = None
        for _ in range(12):
            state = schedule(state, 'good', 0)
        self.assertEqual(state['interval'], 90 * 86400)
        self.assertEqual(schedule(state, 'hard', 0)['interval'], 86400)
        self.assertEqual(schedule(state, 'again', 0)['interval'], 600)

    def test_import_merge_and_invalid_archive_leave_existing_file_untouched(self):
        self.review.next(); self.review.reveal()
        with patch('torch_judge.review.time.time', return_value=1000):
            self.review.rate('hard')
        exported = Path(self.tmp.name) / 'export.json'
        self.review.export(exported)
        other = ReviewSession(Path(self.tmp.name) / 'other.json')
        other.import_archive(exported)
        self.assertEqual(other._load(), self.review._load())
        incoming = json.loads(exported.read_text())
        key = next(iter(incoming['cards']))
        incoming['cards'][key] = schedule(None, 'again', 0)
        exported.write_text(json.dumps(incoming)); other.import_archive(exported)
        self.assertEqual(other._load(), self.review._load())
        before = other.path.read_bytes()
        incoming['cards'][key]['due'] = -1
        exported.write_text(json.dumps(incoming))
        with self.assertRaises(ValueError):
            other.import_archive(exported)
        self.assertEqual(other.path.read_bytes(), before)

    def test_corrupt_storage_is_not_overwritten(self):
        self.path.write_text('{broken')
        with self.assertRaises(ValueError):
            self.review.next()
        self.assertEqual(self.path.read_text(), '{broken')

    def test_invalid_schedules_and_unknown_ids_are_rejected(self):
        for value in [None, {'version': True, 'cards': {}}, {'version': 2, 'cards': {}}, {'version': 1, 'cards': {'unknown': {}}}]:
            with self.assertRaises(ValueError):
                validate_archive(value)
        for invalid in [float('inf'), float('nan'), True, -1]:
            state = schedule(None, 'good', 0); state['due'] = invalid
            with self.assertRaises(ValueError):
                validate_archive({'version': 1, 'cards': {motifs()[0]['id']: state}})

    def test_two_sessions_preserve_updates_to_other_cards(self):
        other = ReviewSession(self.path)
        first, second = motifs()[:2]
        self.review.next(first['id']); other.next(second['id'])
        self.review.reveal(); other.reveal()
        self.review.rate('good'); other.rate('hard')
        self.assertEqual(len(self.review._load()['cards']), 2)

    def test_browser_contract_fixtures(self):
        fixture = json.loads((ROOT / 'browser_gym/extension/src/fixtures/review.json').read_text())
        for case in fixture:
            self.assertEqual(schedule(case['previous'], case['rating'], case['now']), case['expected'])
