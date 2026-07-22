from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from typing_extensions import TypeVar

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""

# The serialized run context crosses the activity boundary as untyped JSON (`Any`, so
# `TemporalRunContext` subclasses can add their own fields), which means structured values
# arrive back as plain dicts. Rehydrate the ones with behavior the framework relies on
# inside activities — `usage`/`usage_limits` drive the mid-chain continuation usage check.
_run_usage_ta = TypeAdapter(RunUsage)
_usage_limits_ta = TypeAdapter(UsageLimits)


class TemporalRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`, `partial_output`, `loaded_capability_ids`, `discovered_tool_names`, and `capability_loaded` attributes will be available.

    The `capabilities` registry is intentionally excluded: it holds live capability objects (toolsets, hooks, callables) that aren't serializable across the activity boundary, like `tool_manager`. As a result `available_capability_ids` (which reads `capabilities`) is unavailable inside an activity, while `available_tool_names` still works via its `discovered_tool_names` fallback.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent].
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        self.__dict__.setdefault('agent', None)
        if isinstance(usage := self.__dict__.get('usage'), dict):
            self.__dict__['usage'] = _run_usage_ta.validate_python(usage)
        if isinstance(usage_limits := self.__dict__.get('usage_limits'), dict):
            self.__dict__['usage_limits'] = _usage_limits_ta.validate_python(usage_limits)
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
            'usage_limits': ctx.usage_limits,
            'loaded_capability_ids': ctx.loaded_capability_ids,
            'discovered_tool_names': ctx.discovered_tool_names,
            'capability_loaded': ctx.capability_loaded,
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
    and then sets `agent` and `root_capability` on the result so custom subclasses
    don't need to know about either parameter. Setting `root_capability` lets the
    durability capability fire the capability chain against the live model stream
    inside the activity, which is required for capabilities like
    `ProcessEventStream` to see real (non-replayed) events.
    """
    ctx = run_context_type.deserialize_run_context(serialized, deps=deps)
    if agent is not None:
        ctx.__dict__['agent'] = agent
        ctx.__dict__['root_capability'] = agent.root_capability
    return ctx
