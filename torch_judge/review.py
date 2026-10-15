"""Motif recall for notebooks and terminals; no kernel or UI dependencies.

Use ``ReviewSession().next()``, then ``reveal()`` and ``rate('good')``.
The portable v1 archive is shared with the browser gym.
"""
from __future__ import annotations

import html
import json
import math
import os
import tempfile
import time
from pathlib import Path

RATINGS = {'again': 600, 'hard': 86400, 'good': 3 * 86400}
MAX_INTERVAL = 90 * 86400


def motifs() -> list[dict]:
    return json.loads(Path(__file__).with_name('motifs.json').read_text())


def validate_archive(value: object) -> dict:
    if not isinstance(value, dict) or value.get('version') != 1 or isinstance(value.get('version'), bool) or not isinstance(value.get('cards'), dict):
        raise ValueError('Expected a motif review archive (version 1).')
    known = {card['id'] for card in motifs()}
    for key, state in value['cards'].items():
        if key not in known or not isinstance(state, dict):
            raise ValueError(f'Unknown or invalid recall card: {key}')
        for name in ('due', 'reviewed', 'interval'):
            number = state.get(name)
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
                raise ValueError(f'Invalid {name} for {key}')
        if state['interval'] > MAX_INTERVAL or state['due'] < state['reviewed']:
            raise ValueError(f'Invalid schedule for {key}')
        if not isinstance(state.get('rating'), str) or state['rating'] not in RATINGS:
            raise ValueError(f'Invalid rating for {key}')
    return value


def schedule(previous: dict | None, rating: str, now: float) -> dict:
    if rating not in RATINGS:
        raise ValueError('Choose again, hard, or good.')
    interval = RATINGS[rating]
    if rating == 'good' and previous:
        interval = min(MAX_INTERVAL, max(interval, previous['interval'] * 2))
    return {'due': now + interval, 'reviewed': now, 'interval': interval, 'rating': rating}


class Card:
    """A safe rich display in Jupyter, with a readable terminal fallback."""
    def __init__(self, motif: dict, revealed: bool = False):
        self.motif, self.revealed = motif, revealed

    def __str__(self):
        m = self.motif
        text = f"{m['title']} · {m['topic']}\n\n{m['prompt']}"
        if self.revealed:
            text += f"\n\n{m['answer']}\n\n{m['probe']}\n\n{m['transfer']}"
        text += '\n\nWorkouts: ' + ', '.join(m['workouts'])
        return text

    __repr__ = __str__

    def _repr_html_(self):
        m = self.motif
        esc = html.escape
        content = f"<small>Motif · {esc(m['topic'])}</small><h2>{esc(m['title'])}</h2><p>{esc(m['prompt'])}</p>"
        if self.revealed:
            content += f"<hr><p>{esc(m['answer'])}</p><pre>{esc(m['probe'])}</pre><p>{esc(m['transfer'])}</p>"
        content += '<p>Workouts: ' + ', '.join(esc(w) for w in m['workouts']) + '</p>'
        hint = "review.rate('again'), review.rate('hard'), or review.rate('good')" if self.revealed else 'Make a prediction, then run review.reveal()'
        return f'<section style="max-width:760px;padding:24px;border:1px solid #888;border-radius:12px">{content}<p><code>{esc(hint)}</code></p></section>'


class ReviewSession:
    """Persistent self-rated recall. Revealing and rating never execute a probe.

    ``next()`` selects overdue cards before new cards; ``next(id)`` allows
    voluntary early practice. Saving rereads disk so other cards updated by
    another session are retained. A corrupt archive is never overwritten.
    """
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path.home() / '.local/share/torch-dojo/motif-review.json'
        self.current: dict | None = None
        self.revealed = False

    def _load(self) -> dict:
        if not self.path.exists():
            return {'version': 1, 'cards': {}}
        return validate_archive(json.loads(self.path.read_text()))

    def _save(self, archive: dict):
        validate_archive(archive)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.path.parent, prefix='.motif-review-')
        try:
            with os.fdopen(handle, 'w') as stream:
                json.dump(archive, stream, indent=2)
                stream.write('\n')
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def library(self) -> list[dict]:
        """Browse titles, topics, IDs and linked workout source paths."""
        root = Path(__file__).resolve().parent.parent
        return [{k: m[k] for k in ('id', 'title', 'topic', 'workouts')} | {
            'sources': [str(p) for w in m['workouts'] for p in sorted((root / 'dojo').glob(w + '_*.py')) if not p.stem.endswith('_eval')]
        } for m in motifs()]

    def queue(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        states = self._load()['cards']
        due = [m for m in motifs() if m['id'] in states and states[m['id']]['due'] <= now]
        due.sort(key=lambda m: states[m['id']]['due'])
        return due + [m for m in motifs() if m['id'] not in states]

    def next(self, motif_id: str | None = None) -> Card | str:
        candidates = [m for m in motifs() if m['id'] == motif_id] if motif_id else self.queue()
        if motif_id and not candidates:
            raise ValueError(f'Unknown motif: {motif_id}')
        self.current = candidates[0] if candidates else None
        self.revealed = False
        if self.current is None:
            return 'You are caught up. Stop here, or choose a motif from review.library() for extra practice.'
        return Card(self.current)

    def reveal(self) -> Card:
        if self.current is None:
            raise ValueError('Choose a card with review.next() first.')
        self.revealed = True
        return Card(self.current, True)

    def rate(self, rating: str) -> str:
        if self.current is None or not self.revealed:
            raise ValueError('Reveal the explanation before rating your recall.')
        archive = self._load()
        key = self.current['id']
        state = schedule(archive['cards'].get(key), rating, time.time())
        archive['cards'][key] = state
        self._save(archive)
        self.current = None
        self.revealed = False
        delay = '10 minutes' if rating == 'again' else f"{state['interval'] / 86400:g} days"
        return f'Saved. Returns in {delay}. Run review.next() when you want another.'

    def export(self, path: str | Path):
        """Write a portable archive for import in the browser or another notebook."""
        Path(path).write_text(json.dumps(self._load(), indent=2) + '\n')

    def import_archive(self, path: str | Path):
        """Merge by latest review time; validate the whole file before writing."""
        incoming = validate_archive(json.loads(Path(path).read_text()))
        archive = self._load()
        for key, value in incoming['cards'].items():
            if key not in archive['cards'] or value['reviewed'] > archive['cards'][key]['reviewed']:
                archive['cards'][key] = value
        self._save(archive)
