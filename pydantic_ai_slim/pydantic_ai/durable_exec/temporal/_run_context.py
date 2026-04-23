from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


class TemporalRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` attributes will be available.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent].
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        self.__dict__.setdefault('agent', None)
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:  # pragma: no cover
            if name in RunContext.__dataclass_fields__:
                raise UserError(
                    f'{self.__class__.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to `TemporalAgent`.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        return {
            'run_id': ctx.run_id,
            'metadata': ctx.metadata,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'tool_call_metadata': ctx.tool_call_metadata,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'partial_output': ctx.partial_output,
            'usage': ctx.usage,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> TemporalRunContext[Any]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)


def deserialize_run_context(
    run_context_type: type[TemporalRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
    """Deserialize a run context and attach the agent instance.

    This is a helper used internally by the Temporal wrappers. It calls the
    (potentially user-overridden) `TemporalRunContext.deserialize_run_context`
    and then sets `agent` on the result, so custom subclasses don't need to
    know about the `agent` parameter.
    """
    ctx = run_context_type.deserialize_run_context(serialized, deps=deps)
    if agent is not None:
        ctx.__dict__['agent'] = agent
    return ctx
