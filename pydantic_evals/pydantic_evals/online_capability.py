"""Online evaluation capability for pydantic-ai agents.

Provides an `OnlineEvaluation` capability that attaches evaluators to agent runs,
dispatching them asynchronously in the background after each run completes.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from . import _online as _online_internal, _task_run
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import Evaluator
from .online import (
    DEFAULT_CONFIG,
    EvaluationTarget,
    OnlineEvalConfig,
    OnlineEvaluator,
    SamplingContext,
    SpanReference,
)
from .otel._context_subtree import context_subtree
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
        wraps [`agent.run()`][pydantic_ai.Agent.run],
        [`agent.run_stream()`][pydantic_ai.Agent.run_stream], and
        [`agent.iter()`][pydantic_ai.Agent.iter] when the run reaches a
        final result.
        For streaming runs, evaluators are dispatched only after the final
        result is available and the surrounding context manager exits.

    Example:
    ```python {lint="skip"}
    from dataclasses import dataclass

    from pydantic_ai import Agent
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import OnlineEvalConfig
    from pydantic_evals.online_capability import OnlineEvaluation

    @dataclass
    class OutputNotEmpty(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return bool(ctx.output)

    agent = Agent(
        'openai:gpt-5.2',
        capabilities=[
            OnlineEvaluation(
                evaluators=[OutputNotEmpty()],
                config=OnlineEvalConfig(default_sink=lambda results, failures, context: None),
            ),
        ],
    )
    ```
    """

    evaluators: Sequence[Evaluator | OnlineEvaluator]
    """Evaluators to run after each agent run."""

    config: OnlineEvalConfig | None = None
    """Optional config override. Defaults to the global `DEFAULT_CONFIG`."""

    name: str | None = None
    """Optional name for the EvaluatorContext. Defaults to the agent run's `run_id`."""

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

    @staticmethod
    def _build_sampling_context(
        evaluator: Evaluator,
        inputs: Any,
        metadata: dict[str, Any] | None,
        call_seed: float,
    ) -> SamplingContext:
        return SamplingContext(
            evaluator=evaluator,
            inputs=inputs,
            metadata=metadata,
            call_seed=call_seed,
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        config = self._resolved_config

        # Skip if disabled or already inside an evaluation context (e.g. Dataset.evaluate)
        if (
            not config.enabled
            or _online_internal.EVALUATION_DISABLED.get()
            or _task_run.CURRENT_TASK_RUN.get() is not None
        ):
            return await handler()

        # Use the raw prompt so sampling and evaluation see the same inputs value.
        inputs = ctx.prompt

        # Determine which evaluators are sampled (before running the agent)
        sampled = _online_internal.sample_evaluators(
            self._online_evaluators,
            config,
            inputs,
            build_sampling_context=self._build_sampling_context,
        )
        if not sampled:
            return await handler()

        # Run the agent with span tree capture and attribute/metric tracking
        task_run = _task_run.TaskRun()
        token = _task_run.CURRENT_TASK_RUN.set(task_run)
        try:
            with context_subtree() as span_tree:
                t0 = time.perf_counter()
                result = await handler()
                duration = time.perf_counter() - t0
        finally:  # pragma: no branch
            _task_run.CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _task_run.extract_span_tree_metrics(task_run, span_tree)

        # Merge config and run metadata
        metadata: dict[str, Any] | None = None
        if config.metadata is not None or ctx.metadata is not None:
            metadata = {**(config.metadata or {}), **(ctx.metadata or {})}

        context = EvaluatorContext(
            name=self.name or ctx.run_id or 'agent',
            inputs=inputs,
            output=result.output,
            expected_output=None,
            metadata=metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes=task_run.attributes,
            metrics=task_run.metrics,
        )

        span_reference = _parse_traceparent(result._traceparent(required=False))  # pyright: ignore[reportPrivateUsage]

        target = EvaluationTarget(name=self.name or 'agent', type='agent')
        _online_internal.dispatch_async(
            _online_internal.dispatch_evaluators(sampled, context, span_reference, target, config)
        )

        return result
