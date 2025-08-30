from dataclasses import dataclass
from datetime import timedelta

from pydantic_ai_examples.evals.models import (
    TimeRangeBuilderSuccess,
    TimeRangeInputs,
    TimeRangeResponse,
)
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)
from pydantic_evals.otel import SpanQuery


@dataclass
class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    def evaluate(
        self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]
    ) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            window_end = ctx.output.max_timestamp_with_offset
            window_size = window_end - ctx.output.min_timestamp_with_offset
            return {
                'window_is_not_too_long': window_size <= timedelta(days=30),
                'window_is_not_in_the_future': window_end <= ctx.inputs['now'],
            }

        return {}  # No evaluation needed for errors


@dataclass
class UserMessageIsConcise(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    async def evaluate(
        self,
        ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse],
    ) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            user_facing_message = ctx.output.explanation
        else:
            user_facing_message = ctx.output.error_message

        if user_facing_message is not None:
            return len(user_facing_message.split()) < 50
        else:
            return {}


@dataclass
class AgentCalledTool(Evaluator[object, object, object]):
    agent_name: str
    tool_name: str

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return ctx.span_tree.any(
            SpanQuery(
                name_equals='agent run',
                has_attributes={'agent_name': self.agent_name},
                stop_recursing_when=SpanQuery(name_equals='agent run'),
                some_descendant_has=SpanQuery(
                    name_equals='running tool',
                    has_attributes={'gen_ai.tool.name': self.tool_name},
                ),
            )
        )


CUSTOM_EVALUATOR_TYPES = (ValidateTimeRange, UserMessageIsConcise, AgentCalledTool)
