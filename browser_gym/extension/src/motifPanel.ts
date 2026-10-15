import { ReviewStore, reviewQueue, type Motif, type Rating } from './review';

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  return node;
}
function button(text: string, action: () => void): HTMLButtonElement {
  const node = element('button', text);
  node.type = 'button';
  node.onclick = action;
  return node;
}

/** The library and review queue share one card renderer and persistence boundary. */
export class MotifPanel {
  readonly node = element('section');
  private readonly status = element('p');
  private readonly card = element('div');
  private readonly count = element('p');
  private readonly store: ReviewStore;
  private reviewed = 0;
  constructor(private motifs: Motif[], private links: Map<string, { title: string; href: string }>) {
    this.store = new ReviewStore(motifs, window.localStorage);
    this.node.className = 'dojo-motifs';
    this.node.id = 'motifs';
    this.node.setAttribute('aria-label', 'Motif library and review');
    this.status.setAttribute('role', 'status');
    this.node.append(element('h2', 'Patterns worth keeping'), element('p',
      'A few minutes of recall. Predict, reveal, then try the probe in a scratch cell. Your rating sets the next visit; it is not a workout check.'), this.count);
    const actions = element('div');
    actions.className = 'dojo-motif-actions';
    actions.append(button('Review next motif', () => this.next()), button('Export motif reviews', () => this.guard(() => {
      const blob = new Blob([JSON.stringify(this.store.load(), null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = element('a'); link.href = url; link.download = 'motif-reviews.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    })));
    const upload = element('input'); upload.type = 'file'; upload.accept = '.json,application/json'; upload.hidden = true;
    upload.setAttribute('aria-label', 'Import motif review archive');
    upload.onchange = async () => {
      const file = upload.files?.[0];
      if (!file) return;
      try {
        this.store.import(JSON.parse(await file.text()));
        this.card.replaceChildren(); this.refresh();
        this.status.textContent = 'Reviews merged. The most recent rating for each card wins.';
      } catch (error) { this.error(error); }
      upload.value = '';
    };
    actions.append(button('Import motif reviews', () => upload.click()), upload);
    this.node.append(actions, this.status, this.card);
    const library = element('details');
    library.append(element('summary', `Browse all ${motifs.length} motifs`));
    const search = element('input'); search.type = 'search'; search.placeholder = 'Find an idea, topic, or workout number';
    search.setAttribute('aria-label', 'Find a motif');
    const list = element('ul');
    const renderList = () => {
      list.replaceChildren();
      const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
      for (const motif of motifs.filter(m => words.every(word => `${m.title} ${m.topic} ${m.transfer} ${m.workouts.join(' ')}`.toLowerCase().includes(word)))) {
        const item = element('li');
        item.append(button(motif.title, () => this.guard(() => this.show(motif))), element('span', ` · ${motif.topic} · ${motif.workouts.join(', ')}`));
        list.append(item);
      }
      if (!list.children.length) list.append(element('li', 'No matching motifs. Try a topic such as indexing or geometry.'));
    };
    search.oninput = renderList; renderList();
    library.append(search, list); this.node.append(library);
    this.guard(() => this.refresh());
  }
  private error(error: unknown): void {
    this.status.textContent = `Reviews could not be saved or loaded: ${error instanceof Error ? error.message : String(error)} Your existing archive has not been replaced.`;
  }
  private guard(action: () => void): void { try { action(); } catch (error) { this.error(error); } }
  private refresh(): void {
    const archive = this.store.load();
    const now = Date.now() / 1000;
    const due = this.motifs.filter(m => archive.cards[m.id]?.due <= now).length;
    const fresh = this.motifs.filter(m => !archive.cards[m.id]).length;
    this.count.textContent = `${due} due · ${fresh} new · ${this.reviewed} reviewed this visit. Stop whenever you like.`;
  }
  private next(): void {
    this.guard(() => {
      this.refresh();
      const archive = this.store.load();
      const next = reviewQueue(this.motifs, archive, Date.now() / 1000)[0];
      if (next) this.show(next);
      else {
        this.card.replaceChildren();
        const due = Math.min(...Object.values(archive.cards).map(s => s.due));
        this.status.textContent = `You are caught up. Next review ${new Date(due * 1000).toLocaleString()}. You can stop here or browse for extra practice.`;
      }
    });
  }
  private show(motif: Motif): void {
    const previous = this.store.load().cards[motif.id];
    this.status.textContent = previous && previous.due > Date.now() / 1000
      ? 'Extra practice: rating this card will update its next review.' : '';
    const heading = element('h3', motif.title); heading.tabIndex = -1;
    this.card.className = 'dojo-motif-card';
    this.card.replaceChildren(element('small', `Motif · ${motif.topic}`), heading, element('p', motif.prompt));
    const answer = element('div'); answer.hidden = true; answer.tabIndex = -1;
    const probe = element('pre', motif.probe);
    answer.append(element('p', motif.answer), probe, element('p', motif.transfer));
    answer.append(button('Copy probe', () => {
      navigator.clipboard.writeText(motif.probe).then(() => { this.status.textContent = 'Probe copied. Paste into a scratch cell to run it.'; }, () => { this.status.textContent = 'Copy was unavailable. Select the probe text and copy it manually.'; });
    }));
    const related = element('p', 'Try it in a workout: ');
    for (const id of motif.workouts) {
      const workout = this.links.get(id);
      if (workout) {
        const link = element('a', `${id} · ${workout.title}`); link.href = workout.href;
        related.append(link, document.createTextNode(' · '));
      } else related.append(element('span', `${id} (Jupyter / PyTorch) · `));
    }
    answer.append(related);
    const ratings = element('div'); ratings.className = 'dojo-motif-actions';
    for (const [rating, label] of [['again', 'Again'], ['hard', 'With effort'], ['good', 'Comfortable']] as [Rating, string][]) {
      ratings.append(button(label, () => this.guard(() => {
        const state = this.store.rate(motif.id, rating, Date.now() / 1000);
        this.reviewed += 1;
        this.card.replaceChildren(); this.refresh();
        const delay = rating === 'again' ? '10 minutes' : `${state.interval / 86400} day${state.interval === 86400 ? '' : 's'}`;
        this.status.textContent = `Saved. This motif returns in ${delay}.`;
        const next = button('Review another motif', () => this.next());
        this.card.append(next); next.focus();
      })));
    }
    answer.append(element('p', 'How readily did the idea come back? Again: 10 minutes · With effort: 1 day · Comfortable: 3 days, then doubling up to 90.'), ratings);
    const reveal = button('Reveal explanation', () => { answer.hidden = false; reveal.hidden = true; answer.focus(); });
    this.card.append(reveal, answer); heading.focus();
  }
}
