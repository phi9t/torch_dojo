import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';

interface ExerciseCase {
  id: string;
  title: string;
  candidate: string;
  questionFile: string;
  evalFile: string;
  cardName: string;
  sliderName: string;
  sliderValueAfterStep: string;
  outputTestId: string;
  plotId: string;
  extraPlotId?: string;
  runCountTestId: string;
  leakedProbeName: string;
  starterFailure: string;
  mutantFailure: string;
  mutate(reference: string): string;
}

function canonicalReference(evalFile: string, candidate: string): string {
  const evalSource = readFileSync(resolve(__dirname, '../../../../dojo', evalFile), 'utf8');
  const match = evalSource.match(new RegExp(`def ${candidate}\\([\\s\\S]*?(?=\\n\\n# %% \\[verifiers\\])`));
  if (!match) throw new Error(`Could not find canonical reference function ${candidate}`);
  return `import numpy as np\n${match[0]}`;
}

function starterSource(questionFile: string): string {
  const questionSource = readFileSync(resolve(__dirname, '../../../../dojo', questionFile), 'utf8');
  const match = questionSource.match(/# %% \[answer\]\n([\s\S]*?)(?=\n# %%)/);
  if (!match) throw new Error(`Could not find answer cell in ${questionFile}`);
  return `import numpy as np\n${match[1].trim()}\n`;
}

const exercises: ExerciseCase[] = [
  {
    id: '0069',
    title: 'Multiclass calibration metrics',
    candidate: 'calibration_metrics',
    questionFile: '0069_calibration_metrics.py',
    evalFile: '0069_calibration_metrics_eval.py',
    cardName: 'Tensor card for calibration_probabilities_probe',
    sliderName: 'Calibration bins',
    sliderValueAfterStep: '6',
    outputTestId: 'dojo-calibration-summary',
    plotId: 'dojo-reliability-plot',
    extraPlotId: 'dojo-confidence-histogram',
    runCountTestId: 'dojo-calibration-probe-runs',
    leakedProbeName: 'calibration_probabilities_probe',
    starterFailure: 'verify_baseline_perfect_and_wrong',
    mutantFailure: 'verify_followup_bin_edges_and_empty_bins',
    mutate(reference: string): string {
      return reference.replace(
        'bucket = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)',
        'bucket = np.clip(np.floor(confidence * bins).astype(int), 0, bins - 1)'
      );
    }
  },
  {
    id: '0073',
    title: 'Paired trip bootstrap',
    candidate: 'paired_trip_bootstrap',
    questionFile: '0073_paired_trip_bootstrap.py',
    evalFile: '0073_paired_trip_bootstrap_eval.py',
    cardName: 'Tensor card for trip_improvements_probe',
    sliderName: 'Bootstrap iterations',
    sliderValueAfterStep: '250',
    outputTestId: 'dojo-bootstrap-ci-value',
    plotId: 'dojo-bootstrap-plot',
    runCountTestId: 'dojo-bootstrap-probe-runs',
    leakedProbeName: 'trip_improvements_probe',
    starterFailure: 'verify_baseline_constant_improvement',
    mutantFailure: 'verify_baseline_point_estimate_and_sign',
    mutate(reference: string): string {
      return reference.replace(
        'differences = baseline.astype(np.float64) - candidate.astype(np.float64)',
        'differences = candidate.astype(np.float64) - baseline.astype(np.float64)'
      );
    }
  },
  {
    id: '0074',
    title: 'Detection average precision',
    candidate: 'detection_ap',
    questionFile: '0074_detection_ap.py',
    evalFile: '0074_detection_ap_eval.py',
    cardName: 'Tensor card for precision_recall_points',
    sliderName: 'False-positive score',
    sliderValueAfterStep: '0.8',
    outputTestId: 'dojo-ap-value',
    plotId: 'dojo-pr-plot',
    runCountTestId: 'dojo-pr-probe-runs',
    leakedProbeName: 'precision_recall_points',
    starterFailure: 'verify_baseline_perfect_and_empty',
    mutantFailure: 'verify_baseline_missed_gt_denominator',
    mutate(reference: string): string {
      return reference.replace(
        'recall = true_positives / total_gt',
        'recall = true_positives / max(len(claimed), 1)'
      );
    }
  }
];

for (const exercise of exercises) {
  test(`${exercise.id} qualifies reference, starter, mutant, and visual probe`, async ({ page }) => {
    const reference = canonicalReference(exercise.evalFile, exercise.candidate);
    const starter = starterSource(exercise.questionFile);

    await page.goto(`lab/?exercise=${exercise.id}`);
    await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText(`${exercise.id} · ${exercise.title}`);
    await page.getByRole('button', { name: 'Hint', exact: true }).click();
    await expect(page.locator('#torch-dojo-gym ol')).toBeVisible();

    const probeCell = await runExplorePanel(page);

    const card = page.getByRole('group', { name: exercise.cardName });
    await expect(card).toContainText('shape', { timeout: 60_000 });
    await expect(card).toContainText('dtype');
    await card.focus();
    await expect(card).toBeFocused();

    const output = page.getByTestId(exercise.outputTestId);
    await expect(output).toBeVisible({ timeout: 60_000 });
    const firstOutput = await output.textContent();
    const answer = answerEditor(page);
    const initialAnswer = await answer.textContent();

    const slider = page.getByRole('slider', { name: exercise.sliderName });
    await slider.focus();
    await expect(slider).toBeFocused();
    await slider.press('ArrowRight');
    await expect(slider).toHaveValue(exercise.sliderValueAfterStep);
    await expect(output).not.toHaveText(firstOutput ?? '');
    await expect(page.getByTestId(exercise.runCountTestId)).toContainText('Probe runs: 2');
    if (exercise.id === '0069') {
      expect(await answer.textContent()).toBe(initialAnswer);
      expect(initialAnswer).toContain('def calibration_metrics');
      expect(initialAnswer).toContain('pass');
    }

    const plot = page.locator(`#${exercise.plotId}`);
    await expect(plot).toHaveAttribute('role', 'img');
    await plot.focus();
    await expect(plot).toBeFocused();
    if (exercise.extraPlotId) {
      await expect(page.locator(`#${exercise.extraPlotId}`)).toHaveAttribute('role', 'img');
    }
    await page.setViewportSize({ width: 390, height: 760 });
    await expect(card).toBeVisible();
    await expect(slider).toBeVisible();
    await expect(plot).toBeVisible();
    if (exercise.extraPlotId) {
      await expect(page.locator(`#${exercise.extraPlotId}`)).toBeVisible();
    }
    await page.setViewportSize({ width: 1280, height: 720 });

    async function replaceAnswer(source: string) {
      await answer.click();
      await page.keyboard.press('ControlOrMeta+A');
      await page.keyboard.insertText(source);
    }

    const stage = page.getByLabel('Check stage');
    await stage.selectOption('production');
    const status = page.locator('#torch-dojo-gym [role="status"]');

    await replaceAnswer(starter);
    await page.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
    await expect(status).toContainText(`✗ ${exercise.starterFailure}`);
    await expect(status).toContainText('Receipt source SHA-256');

    await replaceAnswer(`print(${exercise.leakedProbeName})\n${reference}`);
    await page.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('NameError', { timeout: 120_000 });
    await expect(status).toContainText('Receipt source SHA-256');

    await replaceAnswer(reference);
    await page.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
    await expect(status).toContainText('Assistance recorded: hint 1');
    await expect(status).toContainText('Receipt source SHA-256');

    const mutant = exercise.mutate(reference);
    expect(mutant).not.toBe(reference);
    await replaceAnswer(mutant);
    await page.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
    await expect(status).toContainText(`✗ ${exercise.mutantFailure}`);
    await expect(status).toContainText('Receipt source SHA-256');
  });
}
