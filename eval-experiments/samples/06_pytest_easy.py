"""WITH easy_evals -- a suite becomes per-case pytest items (one concurrent run).

Run:  pytest samples/06_pytest_easy.py
"""

from easy_evals import eval_suite
from easy_evals.pytest_plugin import as_pytest

from fake_models import qa_agent

suite = eval_suite(qa_agent())
suite.case('What is the capital of France?', expect='Paris')
suite.case('What is the capital of Japan?', expect='Tokyo')

test_qa = as_pytest(suite)
