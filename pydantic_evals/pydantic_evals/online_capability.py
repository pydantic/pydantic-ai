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

from .dataset import (
    _CURRENT_TASK_RUN as _CURRENT_TASK_RUN,  # pyright: ignore[reportPrivateUsage]
    _extract_span_tree_metrics as _extract_span_tree_metrics,  # pyright: ignore[reportPrivateUsage]
    _TaskRun as _TaskRun,  # pyright: ignore[reportPrivateUsage]
)
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import Evaluator
from .online import (
    _EVALUATION_DISABLED,  # pyright: ignore[reportPrivateUsage]
    DEFAULT_CONFIG,
    OnlineEvalConfig,
    OnlineEvaluator,
    SpanReference,
    _dispatch_async,  # pyright: ignore[reportPrivateUsage]
    _dispatch_evaluators,  # pyright: ignore[reportPrivateUsage]
    _sample_evaluators,  # pyright: ignore[reportPrivateUsage]
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
    if len(parts) != 4:  # pragma: no cover
        return None
    trace_id, span_id = parts[1], parts[2]
    if not trace_id or trace_id == '0' * 32:  # pragma: no cover
        return None
    if not span_id or span_id == '0' * 16:  # pragma: no cover
        return None
    return SpanReference(trace_id=trace_id, span_id=span_id)


@dataclass(kw_only=True)
class OnlineEvaluation(AbstractCapability[AgentDepsT]):
    """Capability that runs online evaluators on agent run results.

    Dispatches evaluators asynchronously in the background after each
    [`agent.run()`][pydantic_ai.Agent.run] completes. Non-blocking — the agent run returns
    immediately and evaluators run concurrently.

    !!! note
        Only [`agent.run()`][pydantic_ai.Agent.run] is supported.
        Streaming via [`agent.run_stream()`][pydantic_ai.Agent.run_stream] does not trigger
        evaluators since the final result is not available until the stream completes.

    Example:
    ```python {test="skip" lint="skip"}
    from pydantic_ai import Agent
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import OnlineEvalConfig
    from pydantic_evals.online_capability import OnlineEvaluation

    agent = Agent(
        'test',
        capabilities=[
            OnlineEvaluation(
                evaluators=[MyEvaluator()],
                config=OnlineEvalConfig(default_sink=my_sink),
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

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        config = self._resolved_config

        # Skip if disabled or already inside an evaluation context (e.g. Dataset.evaluate)
        if not config.enabled or _EVALUATION_DISABLED.get() or _CURRENT_TASK_RUN.get() is not None:
            return await handler()

        # Build a synthetic inputs dict for sampling context
        inputs: dict[str, Any] = {'prompt': ctx.prompt}

        # Determine which evaluators are sampled (before running the agent)
        sampled = _sample_evaluators(self._online_evaluators, config, inputs)
        if not sampled:
            return await handler()

        # Run the agent with span tree capture and attribute/metric tracking
        task_run = _TaskRun()
        token = _CURRENT_TASK_RUN.set(task_run)
        try:
            with context_subtree() as span_tree:
                t0 = time.perf_counter()
                result = await handler()
                duration = time.perf_counter() - t0
        finally:
            _CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _extract_span_tree_metrics(task_run, span_tree)

        # Merge config and run metadata
        metadata: dict[str, Any] | None = None
        if config.metadata or ctx.metadata:
            metadata = {**(config.metadata or {}), **(ctx.metadata or {})}
        elif config.metadata is not None:
            metadata = dict(config.metadata)

        context = EvaluatorContext(
            name=self.name or ctx.run_id or 'agent',
            inputs=ctx.prompt,
            output=result.output,
            expected_output=None,
            metadata=metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes=task_run.attributes,
            metrics=task_run.metrics,
        )

        span_reference = _parse_traceparent(result._traceparent(required=False))  # pyright: ignore[reportPrivateUsage]

        _dispatch_async(_dispatch_evaluators(sampled, context, span_reference, config))

        return result
