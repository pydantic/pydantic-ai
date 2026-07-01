from __future__ import annotations

from dataclasses import dataclass

from core import ConversationRun, ConversationScenario

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass(repr=False)
class ConversationCompleted(Evaluator[ConversationScenario, ConversationRun, object]):
    """Pass when the simulator marks the scenario goal as completed."""

    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        completed = ctx.output.stop_reason == 'simulator_done'
        reason = None if completed else f'Conversation stopped after {ctx.output.turn_count} turns.'
        return EvaluationReason(value=completed, reason=reason)


@dataclass(repr=False)
class MaxTargetTurns(Evaluator[ConversationScenario, ConversationRun, object]):
    """Pass when the target reaches the simulator goal within the desired turn budget."""

    max_turns: int

    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        passed = ctx.output.turn_count <= self.max_turns
        reason = None if passed else f'Conversation took {ctx.output.turn_count} turns; expected at most {self.max_turns}.'
        return EvaluationReason(value=passed, reason=reason)


@dataclass(repr=False)
class MaxAverageTargetTurnDuration(Evaluator[ConversationScenario, ConversationRun, object]):
    """Pass when the target agent's average turn latency stays below the threshold."""

    max_seconds: float

    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        passed = ctx.output.average_target_turn_duration <= self.max_seconds
        reason = (
            None
            if passed
            else (
                'Average target-agent turn duration was '
                f'{ctx.output.average_target_turn_duration:.2f}s; expected at most {self.max_seconds:.2f}s.'
            )
        )
        return EvaluationReason(value=passed, reason=reason)
