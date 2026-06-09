"""Each friendly matcher: a passing case and a failing case, through the real engine."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from easy_evals import check, eval_suite
from fake_models import judge_model, person_agent, qa_agent, tool_agent
from pydantic_ai import Agent
from pydantic_evals.reporting import EvaluationReport

JUDGE = judge_model()


@pytest.fixture
def agent() -> Agent[None, str]:
    return qa_agent()


def _fails(fn: Callable[[], None]) -> str:
    with pytest.raises(AssertionError) as exc:
        fn()
    return str(exc.value)


def test_expect_forgiving(agent: Agent[None, str]) -> None:
    check(agent, 'What is the capital of France?', expect='Paris')  # "...is Paris." still matches
    assert 'expect' in _fails(lambda: check(agent, 'What is the capital of France?', expect='Berlin'))


def test_equals_strict(agent: Agent[None, str]) -> None:
    check(agent, 'What is 2 + 2?', equals='2 + 2 = 4.')
    assert 'equals' in _fails(lambda: check(agent, 'What is 2 + 2?', equals='4'))


def test_contains_and_excludes(agent: Agent[None, str]) -> None:
    check(agent, 'What is the capital of Japan?', contains=['Tokyo', 'capital'], excludes='Paris')
    assert 'excludes' in _fails(lambda: check(agent, 'What is the capital of France?', excludes='Paris'))


def test_one_of(agent: Agent[None, str]) -> None:
    check(agent, 'Sentiment of "I love this"?', one_of=['positive', 'negative'])
    assert 'one_of' in _fails(lambda: check(agent, 'What is the capital of France?', one_of=['yes', 'no']))


def test_matches(agent: Agent[None, str]) -> None:
    check(agent, 'What is 2 + 2?', matches=r'\d')
    assert 'matches' in _fails(lambda: check(agent, 'What is 2 + 2?', matches=r'\d{5}'))


def test_max_words_and_chars(agent: Agent[None, str]) -> None:
    check(agent, 'What is 2 + 2?', max_words=10, max_chars=50)
    assert 'max_length' in _fails(lambda: check(agent, 'What is the capital of France?', max_words=2))


def test_judge(agent: Agent[None, str]) -> None:
    check(agent, 'Write a haiku about the sea.', judge='is a haiku', judge_model=JUDGE)
    assert 'judge' in _fails(lambda: check(agent, 'What is 2 + 2?', judge='is a haiku', judge_model=JUDGE))


def test_passes_escape_hatch(agent: Agent[None, str]) -> None:
    check(agent, 'What is 2 + 2?', passes=lambda out: '4' in out)
    assert _fails(lambda: check(agent, 'What is 2 + 2?', passes=lambda out: 'nope' in out))


def test_structured_expect() -> None:
    check(person_agent(), 'Make Ada aged 36', expect={'name': 'Ada', 'age': 36})
    assert 'age' in _fails(lambda: check(person_agent(), 'Make Ada aged 36', expect={'name': 'Ada', 'age': 99}))


def test_calls_tool() -> None:
    check(tool_agent(), 'Weather in Paris?', expect='sunny', calls_tool='get_weather')
    assert 'calls_send_email' in _fails(lambda: check(tool_agent(), 'Weather in Paris?', calls_tool='send_email'))


def test_combined_matchers_on_one_case(agent: Agent[None, str]) -> None:
    # matchers compose; all must pass
    check(agent, 'What is 2 + 2?', contains='4', excludes='five', max_words=10, matches=r'=')


def test_suite_run_reports_pass_rate(agent: Agent[None, str]) -> None:
    suite = eval_suite(agent, judge_model=JUDGE)
    suite.case('What is the capital of France?', expect='Paris')
    suite.case('What is the capital of France?', expect='Berlin')  # fails
    assert suite.run() is False
    assert _all_values(suite.report()) == [True, False]


def _all_values(report: EvaluationReport[str, str, None]) -> list[bool]:
    return [all(a.value for a in c.assertions.values()) for c in report.cases]
