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
from .conversation import ConversationGoalAchievement, RoleAdherence
from .evaluator import EvaluationReason, EvaluationResult, Evaluator, EvaluatorFailure, EvaluatorOutput, EvaluatorSpec
from .extraction import ConversationTurn, extract_conversation_turns, format_transcript
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
    # conversation
    'ConversationGoalAchievement',
    'RoleAdherence',
    # extraction (reusable primitives for building custom conversation evaluators)
    'ConversationTurn',
    'extract_conversation_turns',
    'format_transcript',
    # context
    'EvaluatorContext',
    # evaluator
    'Evaluator',
    'EvaluationReason',
    'EvaluatorFailure',
    'EvaluatorOutput',
    'EvaluatorSpec',
    'EvaluationResult',
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
