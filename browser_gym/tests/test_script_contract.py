"""Native and browser interpretation share the authoring contract."""
from contextlib import redirect_stdout
from io import StringIO

import pytest

from browser_gym.build_catalog import _bundle
from browser_gym.check_runtime import run_check_bundle
from torch_judge import check, evaluate
from torch_judge.scripts import TaggedScript


@pytest.mark.parametrize('declaration', [
    'def verify_baseline_one(answer, *, scale=1): pass',
    'def verify_baseline_one(answer=1): pass',
    'def verify_baseline_one(answer, /): pass',
    'def verify_baseline_one(answer, *args): pass',
    'def verify_baseline_one(answer, **kwargs): pass',
])
def test_native_and_browser_reject_the_same_verifier_declarations(tmp_path, declaration):
    path = tmp_path / 'sample_eval.py'
    source = '# %% [verifiers]\n' + declaration + '\n'
    path.write_text(source)
    with pytest.raises(ValueError) as native:
        check(path, lambda: 1)
    with pytest.raises(ValueError) as browser:
        _bundle(source, path.name)
    assert str(native.value) == str(browser.value)


def test_helpers_decorators_and_setup_keep_their_order_in_both_workflows(tmp_path):
    question = tmp_path / 'sample.py'
    question.write_text('''# %% [setup]
FACTOR = 2
# %%
raise RuntimeError('teaching probe must not run')
# %% [answer]
def answer(value): return value * FACTOR
''')
    evaluation = tmp_path / 'sample_eval.py'
    evaluation.write_text('''raise RuntimeError('reference must not run')
# %% [verifiers]
def decorate(fn):
    def wrapped(answer):
        return fn(answer)
    return wrapped
@decorate
def verify_baseline_first(answer):
    assert answer(3) == 6
alias = verify_baseline_first
def verify_followup_alias(answer):
    alias(answer)
# %%
raise RuntimeError('runner must not run')
''')
    with redirect_stdout(StringIO()):
        assert evaluate(evaluation, None, candidate=True, stage='production')
    bundle = _bundle(evaluation.read_text(), evaluation.name)
    bundle.update(exercise_id='sample', checker_version='test')
    result = run_check_bundle(TaggedScript(question.read_text()).answer().source(), bundle, 'production')
    assert result['status'] == 'passed', result
    assert len(result['checks']) == 2


@pytest.mark.parametrize('source', [
    '# %% [verifiers]\ndef verify_one(answer):\n# %%\n    pass\n',
    'def decorate(f): return f\n@decorate\n# %% [verifiers]\ndef verify_one(answer): pass\n',
    '# %% [verifiers]\ndef verify_one(answer): pass\nfrom torch_judge import evaluate\n',
    '# %% [verifiers]\ndef verify_one(answer): pass\n# %% [verifiers]\n',
])
def test_native_and_browser_reject_the_same_invalid_sections(tmp_path, source):
    path = tmp_path / 'sample_eval.py'
    path.write_text(source)
    with pytest.raises(ValueError) as native:
        check(path, lambda: 1)
    with pytest.raises(ValueError) as browser:
        _bundle(source, path.name)
    assert str(native.value) == str(browser.value)


def test_empty_baseline_cannot_be_reported_as_a_pass(tmp_path):
    source = '# %% [verifiers]\ndef verify_production_only(answer): assert answer() == 1\n'
    path = tmp_path / 'sample_eval.py'
    path.write_text(source)
    with pytest.raises(ValueError, match='no verifiers available'):
        check(path, lambda: 1, stage='baseline')
    bundle = _bundle(source)
    bundle.update(exercise_id='sample', checker_version='test')
    result = run_check_bundle('def answer(): return 1', bundle)
    assert result['status'] == 'failed'
    assert 'no verifiers available' in result['error']
