import { passedStages, stageName, type StageOrder, type StagedResult } from './seals';
import { finishTime } from './splits';

/** What the home page needs from a rep. */
export interface HomeRep {
  createdAt: string;
  activeSeconds?: number;
  receipts: Array<{ status: string; checkedAt?: string; result?: StagedResult; activeSecondsAtCheck?: number }>;
}

/**
 * Opening a workout starts a rep, but it only counts once the learner has
 * worked on it: a check, or some active time. Autosaves don't date it; its
 * latest check does, or else its start.
 */
function countsAsRep(rep: HomeRep): boolean {
  return rep.receipts.length > 0 || (rep.activeSeconds ?? 0) > 0;
}

function repDate(rep: HomeRep): string {
  return rep.receipts.map(receipt => receipt.checkedAt ?? '').concat(rep.createdAt).sort().pop()!;
}

/** A workout's standing across its reps. Workouts never repped have only `reps: 0`, never zeros. */
export interface WorkoutStats {
  reps: number;
  best?: number;
  highest?: string;
  lastRepAt?: string;
}

export function workoutStats(allReps: HomeRep[], stages: string[], order: StageOrder): WorkoutStats {
  const reps = allReps.filter(countsAsRep);
  if (!reps.length) return { reps: 0 };
  const stats: WorkoutStats = { reps: reps.length };
  const times = reps.map(rep => finishTime(rep, stages, order)).filter((time): time is number => typeof time === 'number');
  if (times.length) stats.best = Math.min(...times);
  const passed = passedStages(stages, reps.flatMap(rep => rep.receipts), order);
  if (passed.size) {
    stats.highest = order === 'independent'
      ? `${passed.size} of ${stages.length} checkpoints`
      : stageName(stages.filter(stage => passed.has(stage)).pop()!);
  }
  stats.lastRepAt = reps.map(repDate).sort().pop();
  return stats;
}

/**
 * The workout to practice next: one never repped, earliest in the learning
 * path and then in catalog order; otherwise the one whose last rep is oldest.
 */
export function nextUp<T extends { id: string; lastRepAt?: string }>(
  workouts: T[],
  learningPath: string[],
  current?: string
): T | undefined {
  const rank = (id: string) => {
    const index = learningPath.indexOf(id);
    return index < 0 ? learningPath.length : index;
  };
  const candidates = workouts.map((workout, order) => ({ workout, order })).filter(({ workout }) => workout.id !== current);
  const fresh = candidates.filter(({ workout }) => !workout.lastRepAt);
  if (fresh.length) return fresh.sort((a, b) => rank(a.workout.id) - rank(b.workout.id) || a.order - b.order)[0].workout;
  return candidates.sort((a, b) => a.workout.lastRepAt!.localeCompare(b.workout.lastRepAt!))[0]?.workout;
}

/** The home page groups workouts by their first catalog concept. */
export function topicOf(workout: { topics: string[] }): string {
  return workout.topics[0] ?? 'other';
}

const DAY = 24 * 60 * 60 * 1000;

export function sinceLastRep(lastRepAt: string | undefined, now = new Date()): string {
  if (!lastRepAt) return 'never';
  const days = Math.floor((now.getTime() - new Date(lastRepAt).getTime()) / DAY);
  if (days < 1) return 'today';
  if (days < 2) return 'yesterday';
  if (days < 14) return `${days} days ago`;
  if (days < 60) return `${Math.floor(days / 7)} weeks ago`;
  return `${Math.floor(days / 30)} months ago`;
}

/** Whether every typed word appears in the workout's id, title or topics. */
export function matchesWorkout(workout: { id: string; title: string; topics: string[] }, query: string): boolean {
  const haystack = [workout.id, workout.title, ...workout.topics].join(' ').toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every(word => haystack.includes(word));
}
