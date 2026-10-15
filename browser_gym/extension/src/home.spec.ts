import { describe, expect, it } from 'vitest';
import { matchesWorkout, nextUp, sinceLastRep, workoutStats, type HomeRep } from './home';

const STAGES = ['baseline', 'followup', 'production'];

function rep(createdAt: string, ...passes: Array<[string, number | undefined]>): HomeRep {
  return {
    createdAt,
    activeSeconds: 30,
    receipts: passes.map(([stage, activeSecondsAtCheck]) => ({ status: 'passed', checkedAt: createdAt, result: { stage }, activeSecondsAtCheck }))
  };
}

describe('workoutStats', () => {
  it('has no stats for a workout never repped', () => {
    expect(workoutStats([], STAGES, 'cumulative')).toEqual({ reps: 0 });
  });

  it('does not count a rep that was only opened', () => {
    expect(workoutStats([{ createdAt: '2026-10-01T10:00:00Z', receipts: [] }], STAGES, 'cumulative')).toEqual({ reps: 0 });
  });

  it('dates a rep by its latest check, not by autosaves', () => {
    const checked = { createdAt: '2026-10-01T10:00:00Z', receipts: [{ status: 'failed', checkedAt: '2026-10-03T10:00:00Z' }] };
    expect(workoutStats([checked], STAGES, 'cumulative').lastRepAt).toBe('2026-10-03T10:00:00Z');
  });

  it('counts reps, the best finish, the highest stage passed and the last rep', () => {
    const stats = workoutStats(
      [rep('2026-10-01T10:00:00Z', ['production', 400]), rep('2026-10-05T10:00:00Z', ['production', 300]), rep('2026-10-07T10:00:00Z', ['baseline', 50])],
      STAGES,
      'cumulative'
    );
    expect(stats).toEqual({ reps: 3, best: 300, highest: 'Production', lastRepAt: '2026-10-07T10:00:00Z' });
  });

  it('leaves the best time out until a rep finishes with timing', () => {
    expect(workoutStats([rep('2026-10-01T10:00:00Z', ['followup', 90])], STAGES, 'cumulative'))
      .toEqual({ reps: 1, highest: 'Follow-up', lastRepAt: '2026-10-01T10:00:00Z' });
    expect(workoutStats([rep('2026-10-01T10:00:00Z', ['production', undefined])], STAGES, 'cumulative').best).toBeUndefined();
  });

  it('counts independent checkpoints passed in any rep', () => {
    const stats = workoutStats(
      [rep('2026-10-01T10:00:00Z', ['init', 5]), rep('2026-10-02T10:00:00Z', ['forward', 9])],
      ['init', 'forward', 'backward'],
      'independent'
    );
    expect(stats.highest).toBe('2 of 3 checkpoints');
  });
});

describe('nextUp', () => {
  const learningPath = ['0059', '0060', 'mlp'];
  const workouts = (last: Record<string, string>) => ['0059', '0060', '0061', 'mlp'].map(id => ({ id, lastRepAt: last[id] }));

  it('suggests workouts never repped first, in learning-path order, then the rest in catalog order', () => {
    expect(nextUp(workouts({ '0059': '2026-10-01T00:00:00Z' }), learningPath)?.id).toBe('0060');
    expect(nextUp(workouts({ '0059': 'x', '0060': 'x', mlp: 'x' }), learningPath)?.id).toBe('0061');
  });

  it('suggests the workout whose last rep is oldest once every workout has a rep', () => {
    const last = { '0059': '2026-10-05T00:00:00Z', '0060': '2026-10-01T00:00:00Z', '0061': '2026-10-03T00:00:00Z', mlp: '2026-10-04T00:00:00Z' };
    expect(nextUp(workouts(last), learningPath)?.id).toBe('0060');
  });

  it('skips the workout the learner is in', () => {
    expect(nextUp(workouts({}), learningPath, '0059')?.id).toBe('0060');
  });
});

describe('sinceLastRep', () => {
  const now = new Date('2026-10-09T12:00:00Z');

  it('reads like a person would say it', () => {
    expect(sinceLastRep(undefined, now)).toBe('never');
    expect(sinceLastRep('2026-10-09T08:00:00Z', now)).toBe('today');
    expect(sinceLastRep('2026-10-08T08:00:00Z', now)).toBe('yesterday');
    expect(sinceLastRep('2026-10-04T12:00:00Z', now)).toBe('5 days ago');
    expect(sinceLastRep('2026-09-20T12:00:00Z', now)).toBe('2 weeks ago');
    expect(sinceLastRep('2026-07-01T12:00:00Z', now)).toBe('3 months ago');
  });
});

describe('matchesWorkout', () => {
  const workout = { id: '0059', title: 'Stable Softmax Cross-Entropy', topics: ['training', 'stability'] };

  it('matches every typed word against the id, title or topics, ignoring case', () => {
    expect(matchesWorkout(workout, '')).toBe(true);
    expect(matchesWorkout(workout, '59')).toBe(true);
    expect(matchesWorkout(workout, 'softmax')).toBe(true);
    expect(matchesWorkout(workout, 'STAB cross')).toBe(true);
    expect(matchesWorkout(workout, 'training')).toBe(true);
    expect(matchesWorkout(workout, 'softmax geometry')).toBe(false);
  });
});
