"""Integration test for the pytest plugin: as_pytest() emits one passing item per case.

When pytest collects this module it should produce:
    test_agent[1. ...]  test_agent[2. ...]  test_agent[3. ...]
all backed by a SINGLE session-scoped run of the suite.
"""

from __future__ import annotations

from easy_evals import eval_suite
from easy_evals.pytest_plugin import as_pytest
from fake_models import judge_model, qa_agent

suite = eval_suite(qa_agent(), judge_model=judge_model())
suite.case('What is the capital of France?', expect='Paris')
suite.case('What is 2 + 2?', contains='4', max_words=10)
suite.case('Write a haiku about the sea.', judge='is a haiku (three lines)')

test_agent = as_pytest(suite)
