"""WITHOUT easy_evals -- evals as pytest tests: wrap the agent and assert by hand.

Run:  pytest samples/06_pytest_today.py
"""

import pytest

from fake_models import qa_agent

agent = qa_agent()


@pytest.mark.parametrize(
    'question,expected',
    [('What is the capital of France?', 'Paris'), ('What is the capital of Japan?', 'Tokyo')],
)
def test_qa(question: str, expected: str) -> None:
    output = agent.run_sync(question).output
    assert expected.lower() in output.lower()  # forgiving match, written out by hand
