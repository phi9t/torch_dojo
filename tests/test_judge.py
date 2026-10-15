"""Public behavior of the interactive dojo judge."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import sys

from torch_judge import check, evaluate


class JudgeTests(TestCase):
    def test_named_stages_are_cumulative_and_legacy_checks_still_run(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0009_stages_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_legacy(answer): assert answer(1) == 2\n"
                "def verify_baseline_positive(answer): assert answer(2) == 4\n"
                "def verify_followup_negative(answer): assert answer(-3) == -6\n"
                "def verify_production_zero(answer): assert answer(0) == 0\n"
            )
            with redirect_stdout(StringIO()) as output:
                self.assertTrue(check(evaluation, lambda x: 2 * x if x >= 0 else 0, stage="baseline"))
                self.assertFalse(check(evaluation, lambda x: 2 * x if x >= 0 else 0, stage="followup"))
                self.assertFalse(check(evaluation, lambda x: 2 * x + (x == 0), stage="production"))
                self.assertTrue(check(evaluation, lambda x: 2 * x, stage="production"))
            self.assertIn("2 checks", output.getvalue())
            self.assertIn("3 checks", output.getvalue())
            self.assertIn("4 checks", output.getvalue())
            with self.assertRaisesRegex(ValueError, "unknown stage"):
                check(evaluation, lambda x: 2 * x, stage="shipping")

    def test_candidate_defaults_to_baseline_and_reference_runs_all_stages(self) -> None:
        with TemporaryDirectory() as directory:
            question = Path(directory) / "0010_stages.py"
            question.write_text(
                "# %% [answer]\n"
                "def answer(value): return value * 2 if value > 0 else 0\n"
                "# %%\n"
                "raise RuntimeError('teaching probe ran')\n"
            )
            evaluation = Path(directory) / "0010_stages_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_baseline_positive(answer): assert answer(2) == 4\n"
                "def verify_followup_negative(answer): assert answer(-1) == -2\n"
            )
            with redirect_stdout(StringIO()):
                self.assertTrue(evaluate(evaluation, lambda value: 0, candidate=True))
                self.assertFalse(evaluate(evaluation, lambda value: 0, candidate=True, stage="followup"))
                self.assertTrue(evaluate(evaluation, lambda value: 2 * value))
                with patch.object(sys, "argv", [str(evaluation), "--candidate", "--stage", "followup"]):
                    self.assertFalse(evaluate(evaluation, lambda value: 0))

    def test_stage_without_baseline_checks_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0011_no_baseline_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_production_only(answer): assert answer(1) == 2\n"
            )
            with self.assertRaisesRegex(ValueError, "no verifiers available"):
                check(evaluation, lambda value: 2 * value, stage="baseline")

    def test_verifiers_run_without_importing_reference(self) -> None:
        with TemporaryDirectory() as directory:
            eval_path = Path(directory) / "0001_double_eval.py"
            marker = Path(directory) / "imported.txt"
            eval_path.write_text(
                "import math\n"
                "# %% [verifiers]\n"
                "BASE = math.sqrt(16); EXPECTED = BASE\n"
                "def expected(): return EXPECTED\n"
                "def verify_doubles_two(double):\n"
                "    assert double(2) == expected()\n"
                "def verify_doubles_three(double):\n"
                "    assert double(3) == 6\n"
                "# %%\n"
                f"open({str(marker)!r}, 'w').write('imported')\n"
            )

            with redirect_stdout(StringIO()) as output:
                self.assertTrue(check(eval_path, lambda value: value * 2))
                self.assertFalse(check(eval_path, lambda value: value + 1))
                live_namespace = {"check": check, "eval_path": eval_path}
                exec("def double(value): return value * 2\nresult = check(eval_path)", live_namespace)
                self.assertTrue(live_namespace["result"])
            self.assertIn("verify_doubles_two", output.getvalue())
            self.assertFalse(marker.exists())

    def test_candidate_eval_loads_definition_without_running_probes(self) -> None:
        with TemporaryDirectory() as directory:
            question = Path(directory) / "0002_double.py"
            question.write_text(
                "import math\n"
                "raise_if_run = 'teaching prose stays out of the judge'\n"
                "# %% [setup]\n"
                "BASE = math.sqrt(4); FACTOR = BASE\n"
                "# %% [setup]\n"
                "FACTOR *= 1\n"
                "# %%\n"
                "raise RuntimeError('probe should not run')\n"
                "# %% [answer]\n"
                "def double(value):\n    return value * FACTOR\n"
                "# %%\n"
                "raise RuntimeError('probe should not run')\n"
            )
            evaluation = Path(directory) / "0002_double_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_doubles_two(double):\n"
                "    assert double(2) == 4\n"
            )

            with redirect_stdout(StringIO()):
                self.assertTrue(evaluate(evaluation, lambda value: 0, candidate=True))

    def test_verifiers_must_agree_on_candidate_name(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0003_bad_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_one(first):\n    pass\n"
                "def verify_two(second):\n    pass\n"
            )
            with self.assertRaisesRegex(ValueError, "same candidate parameter"):
                check(evaluation, lambda value: value)

    def test_failure_notes_are_shown(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0004_note_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_property(answer):\n"
                "    error = AssertionError('wrong result')\n"
                "    error.add_note('Falsifying example: x=0')\n"
                "    raise error\n"
            )
            with redirect_stdout(StringIO()) as output:
                self.assertFalse(check(evaluation, lambda value: value))
            self.assertIn("Falsifying example: x=0", output.getvalue())

    def test_missing_or_duplicate_sections_have_file_and_line_errors(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0005_missing_eval.py"
            evaluation.write_text("def verify_one(answer): pass\n")
            with self.assertRaisesRegex(ValueError, r"0005_missing_eval.py.*verifiers"):
                check(evaluation, lambda value: value)

            question = Path(directory) / "0005_missing.py"
            question.write_text(
                "# %% [answer]\n"
                "def answer(value): return value\n"
                "# %% [answer]\n"
                "def answer(value): return value * 2\n"
            )
            evaluation.write_text("# %% [verifiers]\ndef verify_one(answer): pass\n")
            with self.assertRaisesRegex(ValueError, r"0005_missing.py:3.*answer"):
                evaluate(evaluation, lambda value: value, candidate=True)

    def test_failure_shows_failing_source_line(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0006_feedback_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_example(answer):\n"
                "    assert answer(2) == 4\n"
            )
            with redirect_stdout(StringIO()) as output:
                self.assertFalse(check(evaluation, lambda value: value))
            self.assertIn("assert answer(2) == 4", output.getvalue())

    def test_verifier_section_must_end_before_runner(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation = Path(directory) / "0007_runner_eval.py"
            evaluation.write_text(
                "# %% [verifiers]\n"
                "def verify_example(answer):\n    assert answer(2) == 4\n"
                "from torch_judge import evaluate\n"
                "if not evaluate('0007_runner_eval.py', lambda value: value):\n"
                "    raise SystemExit(1)\n"
            )
            with self.assertRaisesRegex(ValueError, r"0007_runner_eval.py:4.*close"):
                check(evaluation, lambda value: value)

    def test_missing_answer_names_the_section(self) -> None:
        with TemporaryDirectory() as directory:
            question = Path(directory) / "0008_missing.py"
            question.write_text("# %% [answer]\ndef another(value): return value\n")
            evaluation = Path(directory) / "0008_missing_eval.py"
            evaluation.write_text("# %% [verifiers]\ndef verify_example(answer): pass\n")
            with self.assertRaisesRegex(ValueError, r"0008_missing.py:1.*answer section.*answer"):
                evaluate(evaluation, lambda value: value, candidate=True)
