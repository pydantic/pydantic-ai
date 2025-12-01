from __future__ import annotations

from typing import Any

from temporalio import activity
from typing_extensions import TypeVar

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext

AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""

ActivityDepsT = TypeVar('ActivityDepsT', default=None)
"""Type variable for activity-level dependencies."""

# Registry for activity-level dependencies, keyed by agent name.
# These are set at worker startup via AgentPlugin and are not serialized.
_activity_deps_registry: dict[str, Any] = {}


def get_activity_deps() -> Any:
    """Get activity-level dependencies for the current agent.

    This function retrieves non-serializable dependencies that were registered
    at worker startup via [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin].

    Must be called from within a Temporal activity (e.g., inside a tool function
    that's being executed as part of a durable agent run).

    Returns:
        The activity dependencies object registered for the current agent,
        or `None` if no dependencies were registered.

    Example:
    ```python
    from pydantic_ai.durable_exec.temporal import get_activity_deps

    @agent.tool
    async def my_tool(ctx: RunContext[MyDeps]) -> str:
        activity_deps = get_activity_deps()
        return await activity_deps.db_pool.fetch("...")
    ```
    """
    if not activity.in_activity():
        return None

    activity_type = activity.info().activity_type
    # Activity names follow the pattern: agent__{agent_name}__{activity_type}
    # e.g., "agent__myagent__model_request" or "agent__myagent__toolset__mytoolset__call_tool"
    parts = activity_type.split('__')
    if len(parts) >= 2 and parts[0] == 'agent':
        agent_name = parts[1]
        return _activity_deps_registry.get(agent_name)
    return None


def register_activity_deps(agent_name: str, deps: Any) -> None:
    """Register activity-level dependencies for an agent.

    This is called internally by [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin] at worker startup.

    Args:
        agent_name: The name of the agent.
        deps: The activity dependencies object.
    """
    _activity_deps_registry[agent_name] = deps


def unregister_activity_deps(agent_name: str) -> None:
    """Unregister activity-level dependencies for an agent.

    This is primarily useful for testing cleanup.

    Args:
        agent_name: The name of the agent.
    """
    _activity_deps_registry.pop(agent_name, None)


class TemporalRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` attributes will be available.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent].

    Additionally, `activity_deps` provides access to non-serializable dependencies registered at worker startup
    via [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin].
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
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

    @property
    def activity_deps(self) -> Any:
        """Get activity-level dependencies for the current agent.

        This property provides access to non-serializable dependencies that were registered
        at worker startup via [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin].

        Returns:
            The activity dependencies object registered for the current agent,
            or `None` if no dependencies were registered or not in an activity.

        Example:
        ```python
        @agent.tool
        async def my_tool(ctx: RunContext[MyDeps]) -> str:
            # Access activity deps via the context
            return await ctx.activity_deps.db_pool.fetch("...")
        ```
        """
        return get_activity_deps()

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        return {
            'run_id': ctx.run_id,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
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
