"""The facade builds REAL pydantic_evals objects (it's a translation layer, not a 2nd engine)."""

from __future__ import annotations

from easy_evals import eval_suite
from easy_evals.core import case
from fake_models import qa_agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, Equals, LLMJudge, MaxDuration


def test_case_translates_to_real_pydantic_case():
    c = case('Capital of France?', expect='Paris', equals=None, max_duration=2, judge='is correct')
    pyd = c.to_case()
    assert isinstance(pyd, Case)
    types = {type(e) for e in pyd.evaluators}
    assert Contains in types and MaxDuration in types and LLMJudge in types


def test_equals_maps_to_equals_evaluator():
    pyd = case('x', equals='y').to_case()
    assert any(isinstance(e, Equals) for e in pyd.evaluators)


def test_suite_to_dataset_is_real_dataset():
    suite = eval_suite(qa_agent())
    suite.case('Capital of France?', expect='Paris')
    ds = suite.to_dataset()
    assert isinstance(ds, Dataset)
    assert len(ds.cases) == 1


def test_case_names_are_unique_and_stable():
    suite = eval_suite(qa_agent())
    suite.case('same prompt')
    suite.case('same prompt')
    assert suite.case_names == ['1. same prompt', '2. same prompt']


def test_structured_expect_does_not_leak_dict_into_expected_output():
    pyd = case('x', expect={'a': 1}).to_case()
    assert pyd.expected_output is None
