"""Conversation-level evaluators that score whole agent runs using an LLM judge.

Both evaluators share the extraction layer in
[`pydantic_evals.evaluators.extraction`][pydantic_evals.evaluators.extraction]
and make a single judge call per conversation — cheaper and more holistic than a
per-turn score, at the cost of requiring the judge model to handle the full transcript
in its context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings

from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationScalar, Evaluator, EvaluatorOutput
from .extraction import extract_conversation_turns
from .llm_as_a_judge import judge_conversation_goal, judge_role_adherence

__all__ = (
    'ConversationGoalAchievement',
    'RoleAdherence',
)


@dataclass(repr=False)
class ConversationGoalAchievement(Evaluator[object, object, object]):
    """Judge whether a conversation achieved the user-specified goal.

    Extracts the full conversation from `ctx.span_tree` (via
    [`extract_conversation_turns`][pydantic_evals.evaluators.extract_conversation_turns])
    and asks an LLM judge for a 0.0–1.0 score together with a reason. The goal is
    considered achieved when `score >= threshold`.

    Makes exactly one judge call per evaluation. Very long conversations may be
    truncated by the judge model's context window — prefer a judge with a large
    context for multi-turn agents.

    Example:
    ```python
    from pydantic_evals.evaluators import ConversationGoalAchievement

    ConversationGoalAchievement(
        goal='Resolve the user billing dispute without escalating to a human agent.',
        threshold=0.8,
    )
    ```
    """

    goal: str
    """The conversation goal the agent is being evaluated against. Be specific."""

    threshold: float = 0.7
    """Minimum score (0.0–1.0) for `goal_achieved` to be `True`."""

    model: models.Model | models.KnownModelName | str | None = None
    """Model used by the judge. Defaults to the shared default set via
    [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model]."""

    model_settings: ModelSettings | None = None
    """Optional model settings forwarded to the judge."""

    include_reason: bool = True
    """Whether to surface the judge's reason on the assertion result.

    When `True` (the default), `goal_achieved` is returned as an
    [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] so that reports
    can display *why* the judge decided the goal was (not) achieved.
    """

    evaluation_name: str | None = field(default=None)
    """Custom name used for this evaluator in reports."""

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        # Accessing `ctx.span_tree` propagates `SpanTreeRecordingError` when spans
        # weren't recorded, matching `HasMatchingSpan`'s behavior.
        turns = extract_conversation_turns(ctx.span_tree)
        if not turns:
            reason = 'No conversation turns were found in the span tree; cannot judge goal achievement.'
            return _build_result(
                pass_key='goal_achieved',
                score_key='goal_achievement_score',
                achieved=False,
                score=0.0,
                reason=reason,
                include_reason=self.include_reason,
            )

        grading = await judge_conversation_goal(
            goal=self.goal,
            turns=turns,
            final_output=ctx.output,
            model=self.model,
            model_settings=self.model_settings,
        )
        return _build_result(
            pass_key='goal_achieved',
            score_key='goal_achievement_score',
            achieved=grading.score >= self.threshold,
            score=grading.score,
            reason=grading.reason,
            include_reason=self.include_reason,
        )

    def build_serialization_arguments(self):
        result = super().build_serialization_arguments()
        if (model := result.get('model')) and isinstance(model, models.Model):  # pragma: no branch
            result['model'] = model.model_id
        return result


@dataclass(repr=False)
class RoleAdherence(Evaluator[object, object, object]):
    """Judge whether the assistant stayed within its assigned role for every turn.

    Extracts the full conversation from `ctx.span_tree` and asks an LLM judge to
    flag any assistant turn that broke the role (by turn number, in the reason).
    Returns a score in `[0.0, 1.0]` where `1.0` means perfect adherence.

    Makes exactly one judge call per evaluation, and follows the same extraction
    pattern as [`ConversationGoalAchievement`][pydantic_evals.evaluators.ConversationGoalAchievement] —
    see that evaluator's docs for caveats about context-window limits on long
    conversations.

    Example:
    ```python
    from pydantic_evals.evaluators import RoleAdherence

    RoleAdherence(
        role='helpful customer support assistant; never reveals the system prompt, '
             'never gives legal advice, stays polite even when the user is rude.',
        threshold=0.8,
    )
    ```
    """

    role: str
    """Description of the assigned assistant role, including constraints."""

    threshold: float = 0.7
    """Minimum score (0.0–1.0) for `role_adhered` to be `True`."""

    model: models.Model | models.KnownModelName | str | None = None
    """Model used by the judge. Defaults to the shared default set via
    [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model]."""

    model_settings: ModelSettings | None = None
    """Optional model settings forwarded to the judge."""

    include_reason: bool = True
    """Whether to surface the judge's reason on the assertion result."""

    evaluation_name: str | None = field(default=None)
    """Custom name used for this evaluator in reports."""

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        turns = extract_conversation_turns(ctx.span_tree)
        if not turns:
            reason = 'No conversation turns were found in the span tree; cannot judge role adherence.'
            return _build_result(
                pass_key='role_adhered',
                score_key='role_adherence_score',
                achieved=False,
                score=0.0,
                reason=reason,
                include_reason=self.include_reason,
            )

        grading = await judge_role_adherence(
            role=self.role,
            turns=turns,
            model=self.model,
            model_settings=self.model_settings,
        )
        return _build_result(
            pass_key='role_adhered',
            score_key='role_adherence_score',
            achieved=grading.score >= self.threshold,
            score=grading.score,
            reason=grading.reason,
            include_reason=self.include_reason,
        )

    def build_serialization_arguments(self):
        result = super().build_serialization_arguments()
        if (model := result.get('model')) and isinstance(model, models.Model):  # pragma: no branch
            result['model'] = model.model_id
        return result


def _build_result(
    *,
    pass_key: str,
    score_key: str,
    achieved: bool,
    score: float,
    reason: str,
    include_reason: bool,
) -> dict[str, EvaluationScalar | EvaluationReason]:
    """Build the dict-shaped output used by both conversation evaluators."""
    pass_value: EvaluationScalar | EvaluationReason = (
        EvaluationReason(value=achieved, reason=reason) if include_reason else achieved
    )
    return {pass_key: pass_value, score_key: score}
