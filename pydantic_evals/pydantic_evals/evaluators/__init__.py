from .common import (
    Contains,
    Equals,
    EqualsExpected,
    HasMatchingSpan,
    IsInstance,
    LLMJudge,
    MaxDuration,
    OutputConfig,
)
from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationResult, Evaluator, EvaluatorFailure, EvaluatorOutput, EvaluatorSpec

__all__ = (
    # common
    'Equals',
    'EqualsExpected',
    'Contains',
    'IsInstance',
    'MaxDuration',
    'LLMJudge',
    'HasMatchingSpan',
    'OutputConfig',
    # context
    'EvaluatorContext',
    # evaluator
    'Evaluator',
    'EvaluationReason',
    'EvaluatorFailure',
    'EvaluatorOutput',
    'EvaluatorSpec',
    'EvaluationReason',
    'EvaluationResult',
)


def __getattr__(name: str):
    if name == 'Python':
        raise ImportError(
            'The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.'
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
