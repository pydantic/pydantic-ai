"""Online evaluation capability for pydantic-ai agents.

Provides an `OnlineEvaluation` capability that attaches evaluators to agent runs,
dispatching them asynchronously in the background after each run completes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Self

import logfire_api
from pydantic import BaseModel, PlainSerializer, TypeAdapter, model_validator
from typing_extensions import TypedDict

from pydantic_ai.capabilities import durable_operation
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from . import _online as _online_internal, _task_run
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import Evaluator
from .online import (
    DEFAULT_CONFIG,
    OnlineEvalConfig,
    OnlineEvaluator,
    SpanReference,
)
from .otel._errors import SpanTreeRecordingError
from .otel.span_tree import SpanTree

__all__ = ('OnlineEvaluation',)


def _parse_traceparent(traceparent: str | None) -> SpanReference | None:
    """Parse a W3C traceparent string into a SpanReference.

    Format: `00-{trace_id}-{span_id}-{flags}`
    Returns None if the string is missing, malformed, or has zero IDs.
    """
    if traceparent is None:
        return None
    parts = traceparent.split('-')
    if len(parts) != 4:
        return None
    trace_id, span_id = parts[1], parts[2]
    if not trace_id or trace_id == '0' * 32:
        return None
    if not span_id or span_id == '0' * 16:
        return None
    return SpanReference(trace_id=trace_id, span_id=span_id)


@dataclass(kw_only=True)
class OnlineEvaluation(AbstractCapability[AgentDepsT]):
    """Capability that runs online evaluators on agent run results.

    Dispatches evaluators asynchronously in the background after each completed
    agent run. Non-blocking — the agent run returns without waiting for evaluators
    to finish.

    !!! note
        [`OnlineEvaluation`][pydantic_evals.online_capability.OnlineEvaluation]
        wraps [`agent.run()`][pydantic_ai.agent.AbstractAgent.run],
        [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], and
        [`agent.iter()`][pydantic_ai.agent.Agent.iter] when the run reaches a
        final result.
        For streaming runs, evaluators are dispatched only after the final
        result is available and the surrounding context manager exits.

    Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/),
    sampling and dispatch run as one durable operation, so a replay doesn't evaluate the run again.
    Pass an explicit `id` to use it with a durability capability.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_ai import Agent
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online_capability import OnlineEvaluation


    @dataclass
    class OutputNotEmpty(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return bool(ctx.output)


    agent = Agent(
        'openai:gpt-5.2',
        name='assistant',
        capabilities=[OnlineEvaluation(evaluators=[OutputNotEmpty()])],
    )
    ```
    """

    evaluators: Sequence[Evaluator | OnlineEvaluator]
    """Evaluators to run after each agent run."""

    config: OnlineEvalConfig | None = None
    """Optional config override. Defaults to the global `DEFAULT_CONFIG`."""

    _online_evaluators: list[OnlineEvaluator] = field(init=False, repr=False)
    _resolved_config: OnlineEvalConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._online_evaluators = [
            e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e) for e in self.evaluators
        ]
        self._resolved_config = self.config if self.config is not None else DEFAULT_CONFIG

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        config = self._resolved_config

        # Skip if disabled or already inside an evaluation context (e.g. Dataset.evaluate)
        if not config.should_evaluate():
            return await handler()

        # Use the raw prompt so sampling and evaluation see the same inputs value.
        inputs = ctx.prompt

        # Workflow code replays, so a durable run leaves sampling and dispatch to
        # `_dispatch_durably`, which the engine journals once per run.
        durable = ctx.in_durable_context
        sampled: list[OnlineEvaluator] = []
        if durable:
            if not any(_online_internal.may_sample(online_eval, config) for online_eval in self._online_evaluators):
                return await handler()
        else:
            # Determine which evaluators are sampled (before running the agent)
            sampled = _online_internal.sample_evaluators(self._online_evaluators, config, inputs)
            if not sampled:
                return await handler()

        # Merge config and run metadata
        metadata: dict[str, Any] | None = None
        if config.metadata is not None or ctx.metadata is not None:
            metadata = {**(config.metadata or {}), **(ctx.metadata or {})}

        # Use the agent's declared name when available so evaluation events can be
        # filtered per-agent. Fall back to the generic 'agent' label when unset.
        agent_name = ctx.agent.name if ctx.agent is not None else None
        target = agent_name or 'agent'
        span_reference = _parse_traceparent(logfire_api.get_context().get('traceparent'))

        async def dispatch(output: Any, eval_context_kwargs: dict[str, Any], *, failed: bool) -> None:
            if durable:
                await self._dispatch_durably(
                    ctx,
                    _DurableDispatch(
                        name=ctx.run_id,
                        inputs=inputs,
                        output=output,
                        failed=failed,
                        metadata=metadata,
                        duration=eval_context_kwargs['duration'],
                        span_tree=eval_context_kwargs['_span_tree'],
                        attributes=eval_context_kwargs['attributes'],
                        metrics=eval_context_kwargs['metrics'],
                        span_reference=span_reference,
                        target=target,
                    ),
                )
                return
            evaluators = [ev for ev in sampled if ev.run_on_errors] if failed else sampled
            context = EvaluatorContext(
                name=ctx.run_id,
                inputs=inputs,
                output=output,
                expected_output=None,
                metadata=metadata,
                **eval_context_kwargs,
            )
            _online_internal.dispatch_async(
                _online_internal.dispatch_evaluators(evaluators, context, span_reference, target, config)
            )

        # Run the agent with span tree capture and attribute/metric tracking.
        # `get_eval_context_kwargs` is bound by the `with` once `run_task` enters; pre-init
        # to `None` only to satisfy pyright's flow analysis on the except path.
        get_eval_context_kwargs: Callable[[], dict[str, Any]] | None = None
        try:
            with _task_run.run_task() as get_eval_context_kwargs:
                result = await handler()
        except Exception as e:
            candidates = self._online_evaluators if durable else sampled
            if get_eval_context_kwargs is not None and any(ev.run_on_errors for ev in candidates):
                try:
                    await dispatch(e, get_eval_context_kwargs(), failed=True)
                except Exception as dispatch_error:
                    # The run's own exception is what the caller needs to see.
                    warnings.warn(f'Online evaluation of the failed agent run was not dispatched: {dispatch_error!r}')
            raise

        await dispatch(result.output, get_eval_context_kwargs(), failed=False)
        return result

    @durable_operation(name='dispatch')
    async def _dispatch_durably(self, ctx: RunContext[AgentDepsT], dispatch: _DurableDispatch) -> None:
        """Sample and start the evaluators of a durable run, once: the engine journals this call."""
        config = self._resolved_config
        sampled = _online_internal.sample_evaluators(self._online_evaluators, config, dispatch.inputs)
        if dispatch.failed:
            sampled = [ev for ev in sampled if ev.run_on_errors]
        if not sampled:
            return
        context = EvaluatorContext(
            name=dispatch.name,
            inputs=dispatch.inputs,
            output=dispatch.output,
            expected_output=None,
            metadata=dispatch.metadata,
            duration=dispatch.duration,
            _span_tree=dispatch.span_tree
            if isinstance(dispatch.span_tree, SpanTree | SpanTreeRecordingError)
            else SpanTreeRecordingError(
                "The span tree is not available because the run's evaluation context crossed a durable "
                'execution boundary that cannot carry it.'
            ),
            attributes=dispatch.attributes,
            metrics=dispatch.metrics,
        )
        _online_internal.dispatch_async(
            _online_internal.dispatch_evaluators(sampled, context, dispatch.span_reference, dispatch.target, config)
        )


class _ErrorPayload(TypedDict):
    type: str
    message: str


_error_payload_adapter = TypeAdapter(_ErrorPayload)


def _json_safe_output(output: Any) -> Any:
    """Reduce an exception to its type name and message when the dispatch is serialized."""
    if isinstance(output, BaseException):
        return _ErrorPayload(type=type(output).__name__, message=str(output))
    return output


class _DurableDispatch(BaseModel):
    """The evaluation context of a durable run, as it crosses into `_dispatch_durably`.

    Engines whose durable units run in the run's own process pass it through unchanged, so the
    evaluators see the live values. Serialized to JSON (as Temporal does), the values arrive in
    their JSON form, an exception as an `Exception` naming its original type, and no span tree.
    """

    name: str | None
    inputs: Any
    output: Annotated[Any, PlainSerializer(_json_safe_output, when_used='json')]
    failed: bool
    metadata: dict[str, Any] | None
    duration: float
    span_tree: Annotated[Any, PlainSerializer(lambda _: None, when_used='json')]
    """A `SpanTree` or `SpanTreeRecordingError`; `None` once serialized.

    Typed `Any` because Pydantic can't build a schema for `SpanTree`.
    """
    attributes: dict[str, Any]
    metrics: dict[str, int | float]
    span_reference: SpanReference | None
    target: str

    @model_validator(mode='after')
    def _rebuild_exception(self) -> Self:
        if self.failed and not isinstance(self.output, BaseException):
            error = _error_payload_adapter.validate_python(self.output)
            self.output = Exception(f'{error["type"]}: {error["message"]}')
        return self
