"""Select successive models after accepted output retries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from pydantic_ai import CallToolsNode, CapabilityEvent, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, AgentNode, ModelSelector, NodeResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryFeedbackPart, RetryInfo, ToolReturnPart
from pydantic_ai.models import Model, ModelRequestContext, ModelSelectionContext
from pydantic_ai.tools import AgentDepsT, RunContext


@dataclass(frozen=True)
class EscalationLevel:
    """A model and the number of accepted output failures before selecting the next level."""

    model: Model | str
    """A model instance or an ID resolved through the agent's model resolver."""

    _: KW_ONLY

    after_failures: int | None = None
    """Positive transition threshold. Only the final level omits it."""

    def __post_init__(self) -> None:
        if self.after_failures is not None and self.after_failures < 1:
            raise UserError('`after_failures` must be positive or `None`.')


class EscalationLevelSpec(TypedDict):
    """A serializable escalation level, using a model ID."""

    model: str
    after_failures: NotRequired[int | None]


@dataclass(kw_only=True)
class OutputRetryEvent(CapabilityEvent, namespace='escalate_on_output_retry', name='output_retry'):
    """An output retry accepted by core while processing a model response.

    The response identifies the actual producer, including when request fallback served the step.
    `tool_name` and `tool_call_id` identify an output tool when `source` is `'output_tool'`.
    """

    source: Literal['response_output', 'output_tool']
    response: ModelResponse
    retry: RetryInfo[str]


@dataclass
class EscalateOnOutputRetry(AbstractCapability[AgentDepsT]):
    """Move to successive models when output validation keeps failing within a run.

    Each accepted output retry counts once. Function-tool and model-request-hook retries do not
    advance the ladder. Several failed output calls in one response can cross several thresholds.
    The final model keeps serving until core's existing retry or usage limits end the run.

    Set the agent's output retry budget explicitly; `required_output_retries` is the minimum to
    reach the final level on a serial output path. Per-output-tool overrides remain authoritative.
    """

    levels: Sequence[EscalationLevel]
    """Ordered models. Every level except the last supplies `after_failures`."""

    _failures: int = field(default=0, init=False, repr=False, compare=False)
    _selected_level: int = field(default=0, init=False, repr=False, compare=False)
    _reported_level: int = field(default=0, init=False, repr=False, compare=False)
    _last_retry: OutputRetryEvent | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.levels = tuple(self.levels)
        if not self.levels:
            raise UserError('`EscalateOnOutputRetry` requires at least one level.')
        if any(level.after_failures is None for level in self.levels[:-1]):
            raise UserError('Every non-final escalation level requires `after_failures`.')
        if self.levels[-1].after_failures is not None:
            raise UserError('The final escalation level must omit `after_failures`.')

    @classmethod
    def from_spec(
        cls,
        levels: Sequence[EscalationLevelSpec],
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> EscalateOnOutputRetry[AgentDepsT]:
        """Construct from model IDs and transition thresholds in an agent spec."""
        for level in levels:
            if unknown := set(level) - {'model', 'after_failures'}:
                raise UserError(f'Unknown escalation level fields: {sorted(unknown)}.')
        return cls(
            levels=[EscalationLevel(**level) for level in levels],
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    @property
    def required_output_retries(self) -> int:
        """Minimum retry allowance to reach the final level, excluding other budget consumers.

        Request-hook retries, usage limits, and restrictive output-tool overrides can stop the run
        earlier. This property does not change the agent's configured limits.
        """
        return sum(level.after_failures or 0 for level in self.levels)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> EscalateOnOutputRetry[AgentDepsT]:
        """Keep the ladder's state separate for each run."""
        return replace(self)

    def get_model(self) -> ModelSelector[AgentDepsT]:
        return self.select_model

    def select_model(self, ctx: ModelSelectionContext[AgentDepsT]) -> Model | str:
        """Select from accepted output failures, independently of request-hook retry counts."""
        remaining = self._failures
        self._selected_level = len(self.levels) - 1
        for index, level in enumerate(self.levels[:-1]):
            assert level.after_failures is not None
            if remaining < level.after_failures:
                self._selected_level = index
                break
            remaining -= level.after_failures
        return self.levels[self._selected_level].model

    async def after_node_run(
        self, ctx: RunContext[AgentDepsT], *, node: AgentNode[AgentDepsT], result: NodeResult[AgentDepsT]
    ) -> NodeResult[AgentDepsT]:
        """Observe settled response processing, after core has accepted its retry charges."""
        if not isinstance(node, CallToolsNode):
            return result
        # Model-request hooks can also consume output retries. Only response-processing nodes
        # establish that these parts came from output validation rather than request hooks.
        request = result.request if isinstance(result, ModelRequestNode) else ctx.messages[-1]
        if not isinstance(request, ModelRequest):
            return result
        for part in request.parts:
            source: Literal['response_output', 'output_tool']
            if isinstance(part, RetryFeedbackPart):
                source = 'response_output'
            elif isinstance(part, ToolReturnPart) and part.outcome == 'retried' and part.retry is not None:
                definition = ctx.tools.get(part.tool_name)
                if definition is None or definition.kind != 'output':
                    continue
                source = 'output_tool'
            else:
                continue
            retry = part.retry
            if retry is None or retry.attempt is None or retry.max_retries is None:
                continue
            self._failures += 1
            event = OutputRetryEvent(
                source=source,
                response=node.model_response,
                retry=retry,
                tool_name=part.tool_name if isinstance(part, ToolReturnPart) else None,
                tool_call_id=part.tool_call_id if isinstance(part, ToolReturnPart) else None,
            )
            self._last_retry = event
            await ctx.emit(event)
        return result

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Trace transitions when this selector actually contributes a new request's model."""
        if self._selected_level != self._reported_level:
            span = ctx.tracer.start_span('escalate_on_output_retry')
            if span.is_recording():
                span.set_attributes(
                    {
                        'escalate_on_output_retry.from_level': self._reported_level,
                        'escalate_on_output_retry.to_level': self._selected_level,
                        'escalate_on_output_retry.failures': self._failures,
                        'escalate_on_output_retry.model': ctx.model.model_name,
                    }
                )
                if self._last_retry is not None:
                    span.set_attribute('escalate_on_output_retry.source', self._last_retry.source)
                    if self._last_retry.response.model_name is not None:
                        span.set_attribute('escalate_on_output_retry.producer', self._last_retry.response.model_name)
            span.end()
            self._reported_level = self._selected_level
        return request_context
