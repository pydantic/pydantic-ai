from . import ragas
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
from .quality import (
    GembaScore,
    GEval,
)
from .report_common import (
    ConfusionMatrixEvaluator,
    KolmogorovSmirnovEvaluator,
    PrecisionRecallEvaluator,
    ROCAUCEvaluator,
)
from .report_evaluator import ReportEvaluator, ReportEvaluatorContext

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
    'EvaluationResult',
    # quality (LLM-backed metrics named after their methods)
    'GEval',
    'GembaScore',
    # ragas (rubric presets modeled on Ragas RAG metrics; see the `ragas` submodule)
    'ragas',
    # report evaluators
    'ReportEvaluator',
    'ReportEvaluatorContext',
    'ConfusionMatrixEvaluator',
    'KolmogorovSmirnovEvaluator',
    'PrecisionRecallEvaluator',
    'ROCAUCEvaluator',
)


def __getattr__(name: str):
    if name == 'Python':
        raise ImportError(
            'The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.'
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
