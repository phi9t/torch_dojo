import { Widget } from '@lumino/widgets';
import { nextUp, sinceLastRep, topicOf, type WorkoutStats } from './home';
import { formatDuration } from './splits';

export interface HomeWorkout {
  id: string;
  title: string;
  topics: string[];
  stats: WorkoutStats;
  href: string;
}

/** The parts of a workout's standing worth a line; none for a workout never repped. */
function statsLine(stats: WorkoutStats, now: Date): string {
  if (!stats.reps) return '';
  return [
    stats.best !== undefined ? `best ${formatDuration(stats.best)}` : '',
    `${stats.reps} rep${stats.reps === 1 ? '' : 's'}`,
    stats.highest ? `passed ${stats.highest}` : '',
    `last rep ${sinceLastRep(stats.lastRepAt, now)}`
  ].filter(Boolean).join(' · ');
}

/**
 * The site root: every workout grouped by topic, one calm column, with a
 * Next up suggestion first. Topics are each workout's first catalog concept.
 */
export class HomePage extends Widget {
  constructor(workouts: HomeWorkout[], learningPath: string[], controls: HTMLElement[]) {
    super();
    this.id = 'torch-dojo-home';
    this.title.label = 'Workouts';
    // The gym's own controls (buttons, focus) look the same here as on the Bench.
    this.addClass('dojo-gym-ui');
    this.addClass('dojo-home');
    const now = new Date();
    // JupyterLab's dock panel is already the main landmark; the page is a region named by its heading.
    const page = document.createElement('section');
    page.className = 'dojo-home-page';
    page.setAttribute('aria-labelledby', 'torch-dojo-home-heading');

    const header = document.createElement('header');
    header.className = 'dojo-home-header';
    const brand = document.createElement('div');
    brand.className = 'dojo-eyebrow';
    brand.textContent = 'Torch Dojo · gym';
    const heading = document.createElement('h1');
    heading.id = 'torch-dojo-home-heading';
    heading.textContent = 'Workouts';
    const tip = document.createElement('p');
    tip.className = 'dojo-home-tip';
    tip.textContent = 'Press ⌘K or Ctrl+K anywhere to jump to a workout.';
    header.append(brand, heading, tip, ...controls);
    page.append(header);

    const suggested = nextUp(workouts.map(workout => ({ ...workout, lastRepAt: workout.stats.lastRepAt })), learningPath);
    if (suggested) {
      const next = document.createElement('section');
      next.className = 'dojo-home-next';
      next.setAttribute('aria-labelledby', 'torch-dojo-next-up');
      const label = document.createElement('h2');
      label.id = 'torch-dojo-next-up';
      label.textContent = 'Next up';
      const link = document.createElement('a');
      link.href = suggested.href;
      link.textContent = `${suggested.id} · ${suggested.title}`;
      const why = document.createElement('p');
      why.textContent = suggested.stats.lastRepAt
        ? `Your last rep was ${sinceLastRep(suggested.stats.lastRepAt, now)}, longer ago than any other workout.`
        : 'You have not done a rep of this one yet.';
      next.append(label, link, why);
      page.append(next);
    }

    const topics = [...new Set(workouts.map(topicOf))];
    for (const topic of topics) {
      const section = document.createElement('section');
      section.className = 'dojo-home-topic';
      const title = document.createElement('h2');
      title.id = `torch-dojo-topic-${topic.replace(/\W+/g, '-')}`;
      title.textContent = topic[0].toUpperCase() + topic.slice(1);
      section.setAttribute('aria-labelledby', title.id);
      const list = document.createElement('ul');
      for (const workout of workouts.filter(item => topicOf(item) === topic)) {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = workout.href;
        link.textContent = `${workout.id} · ${workout.title}`;
        item.append(link);
        const line = statsLine(workout.stats, now);
        if (line) {
          const stats = document.createElement('span');
          stats.className = 'dojo-home-stats';
          stats.textContent = line;
          item.append(stats);
        }
        list.append(item);
      }
      section.append(title, list);
      page.append(section);
    }
    this.node.append(page);
  }
}
