import { expect, test, type Page } from '@playwright/test';
import { replaceAnswer } from './bench';

/**
 * Qualification for NumPy workouts without explore panels: in a fresh worker
 * the reference passes every stage, the starter fails at baseline, and one
 * plausible wrong answer fails at the verifier it is meant to trip.
 */
interface Qualification {
  id: string;
  /** Turns the reference answer into a plausible wrong one. */
  mutate: (reference: string) => string;
  /** The verifier the wrong answer must fail. */
  failingVerifier: string;
}

function swap(from: string, to: string) {
  return (reference: string) => {
    if (!reference.includes(from)) throw new Error(`mutation target not found: ${from}`);
    return reference.replace(from, to);
  };
}

const WORKOUTS: Qualification[] = [
  { id: '0068', mutate: swap('end = max(end, hi)', 'end = hi'), failingVerifier: 'verify_baseline_empty_and_nested' },
  { id: "0067", mutate: swap("answer is None and index < end", "answer is None and index <= end"), failingVerifier: "verify_baseline_interior_and_boundary" },
  { id: "0071", mutate: swap("(r, c+1)):", "(r, c+1), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)):"), failingVerifier: "verify_baseline_diagonal_and_cross" },
  { id: "0076", mutate: swap("row * sh + index // kw", "index // kw"), failingVerifier: "verify_baseline_values_and_original_coordinates" },
  { id: "0077", mutate: swap("np.add.at(out, self.row, self.value * x[self.col])", "out[self.row] += self.value * x[self.col]"), failingVerifier: "verify_followup_duplicate_entries_accumulate_in_both_directions" },
  { id: "0035", mutate: swap("best = min(counts, key=lambda pair: (-counts[pair], pair))", "best = max(counts, key=lambda pair: (counts[pair], pair))"), failingVerifier: "verify_followup_ties_and_overlap" },
  { id: "0072", mutate: swap("if active or row or field or state != \"start\":", "if field:"), failingVerifier: "verify_baseline_trailing_empty_field" },
  { id: "0078", mutate: swap("enumerate(rows[1:], start=2)", "enumerate(rows[1:], start=1)"), failingVerifier: "verify_baseline_valid_row_and_record_number" },
  { id: "0079", mutate: swap("return total + term", "return total"), failingVerifier: "verify_baseline_precedence" },
  { id: "0082", mutate: swap("updated[terminal] = 0", "values[terminal] = 0"), failingVerifier: "verify_baseline_one_step_and_absorbing_state" },
  { id: "0083", mutate: swap("bootstrap = 0. if done else gamma", "bootstrap = gamma"), failingVerifier: "verify_followup_terminal_discards_bootstrap" },
  { id: "0091", mutate: swap("discount = 1.", "discount = gamma"), failingVerifier: "verify_baseline_hand_trace" },
  { id: "0030", mutate: swap("progress = (step - warmup_steps) / (total_steps - warmup_steps)", "progress = (step - warmup_steps) / total_steps"), failingVerifier: "verify_baseline_landmarks" },
  { id: "0050", mutate: swap("gradients[right] += incoming * nodes[left][2]", "gradients[right] = incoming * nodes[left][2]"), failingVerifier: "verify_baseline_repeated_parent" },
  { id: "0075", mutate: swap("index = int(rng.choice(np.flatnonzero(~used)))", "index = int(rng.choice(len(X), p=nearest_d2 / nearest_d2.sum()))"), failingVerifier: "verify_followup_all_identical_points_and_distinct_indices" },
  { id: "0100", mutate: swap("for pos in np.argsort(-values.ravel(), kind=\"stable\"):", "for pos in np.argsort(values.ravel(), kind=\"stable\")[::-1]:"), failingVerifier: "verify_baseline_tie_favors_first_route" },
  { id: "0094", mutate: swap("return sigmoid(a*query_feature + b)", "return a*query_feature + b"), failingVerifier: "verify_baseline_platt_known_symmetric_optimum" },
  { id: "0080", mutate: swap("base = min(1., 1. / bumps_per_second, 1. / speed)", "base = min(1, 1 // bumps_per_second, 1 // speed)"), failingVerifier: "verify_baseline_four_reported_examples" }
];

async function check(page: Page, stage: 'baseline' | 'production') {
  await page.getByLabel('Check stage').selectOption(stage);
  await page.getByRole('button', { name: `Check ${stage}` }).click();
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).toContainText(/(PASSED|FAILED) · source SHA-256/, { timeout: 120_000 });
  return status;
}

for (const workout of WORKOUTS) {
  test(`${workout.id} qualifies: reference passes, starter and a plausible mistake fail`, async ({ page }) => {
    const catalog = await (await page.request.get('gym-assets/catalog.json')).json();
    const reference: string = catalog.find((entry: { id: string }) => entry.id === workout.id).solution;
    await page.goto(`lab/?exercise=${workout.id}`);
    await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });

    // The starter, as the learner first sees it.
    await expect(await check(page, 'baseline')).toContainText('FAILED · source SHA-256');

    await replaceAnswer(page, reference);
    await expect(await check(page, 'production')).toContainText('PASSED · source SHA-256');
    await expect(page.getByRole('region', { name: 'Check results' }).getByRole('button', { name: /: failed$/ })).toHaveCount(0);

    await replaceAnswer(page, workout.mutate(reference));
    const status = await check(page, 'production');
    await expect(status).toContainText('FAILED · source SHA-256');
    await expect(status).toContainText(`✗ ${workout.failingVerifier}`);
  });
}

export { WORKOUTS };
