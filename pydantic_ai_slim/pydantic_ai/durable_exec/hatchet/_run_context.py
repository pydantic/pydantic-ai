from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import NoOpTracer
from typing_extensions import TypeVar

from pydantic_ai.durable_exec._run_context import DurableRunContext
from pydantic_ai.durable_exec._utils import agent_instrumentation_settings
from pydantic_ai.tools import RunContext

from ._spec import HATCHET_ENGINE_SPEC

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)


class HatchetRunContext(DurableRunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] used inside the Hatchet tasks a durable agent run spawns.

    A task may run on another worker, so only the fields
    [`DurableRunContext`][pydantic_ai.durable_exec.DurableRunContext] serializes cross into it, and reading
    any other field raises a `UserError` explaining how to make it available. `agent` is re-attached from the
    worker's agent instance, and `tracer` is resolved from it.

    To make another attribute available, create a `HatchetRunContext` subclass with a custom
    `serialize_run_context` class method that returns a dictionary that includes the attribute, and pass it
    as the `run_context_type` argument to [`HatchetDurability`][pydantic_ai.durable_exec.hatchet.HatchetDurability].
    """

    def unavailable_field_message(self, name: str) -> str:
        return (
            f'{name!r} is not available on {self.__class__.__name__!r} inside a Hatchet task. '
            'To make the attribute available, create a `HatchetRunContext` subclass with a custom '
            '`serialize_run_context` class method that returns a dictionary that includes the attribute and pass it '
            'as the `run_context_type` argument to `HatchetDurability`.'
        )

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> HatchetRunContext[Any]:
        return cls(**ctx, deps=deps)


def deserialize_run_context(
    run_context_type: type[HatchetRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
    """Rebuild a task's run context, resolving `tracer` from the worker's agent unless the subclass set it."""
    ctx = run_context_type.restore(serialized, deps=deps, agent=agent, engine_spec=HATCHET_ENGINE_SPEC)
    if 'tracer' not in ctx.__dict__:
        settings = agent_instrumentation_settings(agent) if agent is not None else None
        ctx.__dict__['tracer'] = settings.tracer if settings is not None else NoOpTracer()
        assert isinstance(ctx, DurableRunContext)
        ctx._expose_field('tracer')  # pyright: ignore[reportPrivateUsage]
    return ctx
