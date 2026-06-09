"""easy_evals -- the easiest way to write and run evals for a Pydantic AI agent.

A thin, beginner-friendly FACADE over `pydantic_evals`. It adds no evaluation
logic of its own: friendly kwargs are translated into real `pydantic_evals`
evaluators, cases are real `Case`s, runs go through the real `Dataset.evaluate`,
and pass/fail is read from the real `EvaluationReport`.

Three ways to use it, one vocabulary:

    # 1. immediate (great in pytest or a script)
    from easy_evals import check
    check(agent, 'Capital of France?', expect='Paris')

    # 2. a suite (batch run, concurrency + report handled for you)
    from easy_evals import eval_suite
    suite = eval_suite(agent)
    suite.case('Capital of France?', expect='Paris')
    suite.case('Write a haiku', judge='is a haiku')
    suite.run()

    # 3. drop down to pydantic_evals Case/Dataset/Evaluator any time.
"""

from __future__ import annotations

from .core import EvalSuite, case, case_failures, check, eval_suite
from .evaluators import HasFields, Matches, MaxLength, NotContains, OneOf

__all__ = (
    'check',
    'eval_suite',
    'EvalSuite',
    'case',
    'case_failures',
    # proposed new built-in evaluators (usable standalone in pydantic_evals)
    'NotContains',
    'OneOf',
    'Matches',
    'MaxLength',
    'HasFields',
)
