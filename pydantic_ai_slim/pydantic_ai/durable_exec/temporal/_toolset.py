from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ConfigDict, with_config
from pydantic.errors import PydanticUserError
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig
from typing_extensions import Self

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableToolsetBase,
    resolve_tool_durable_config,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._run_context import TemporalRunContext

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class GetToolsParams:
    serialized_run_context: Any


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition | None


class TemporalWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    @property
    def id(self) -> str:
        # An error is raised in `TemporalAgent` if no `id` is set.
        assert self.wrapped.id is not None
        return self.wrapped.id

    @property
    @abstractmethod
    def temporal_activities(self) -> list[Callable[..., Any]]:
        raise NotImplementedError

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Temporal-wrapped toolsets manage their wrapped toolset's lifecycle
        # per-activity (inside activities), not per-run.
        return self  # pragma: no cover

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Temporal-wrapped toolsets manage their wrapped toolset's lifecycle
        # per-activity (inside activities), not per-run-step.
        return self

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Temporalized toolsets cannot be swapped out after the fact.
        return self  # pragma: no cover

    async def __aenter__(self) -> Self:
        if not workflow.in_workflow():
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if not workflow.in_workflow():
            return await self.wrapped.__aexit__(*args)
        return None

    async def _wrap_call_tool_result(self, coro: Awaitable[Any]) -> CallToolResult:
        return await wrap_tool_call_result(coro)

    def _unwrap_call_tool_result(self, result: CallToolResult) -> Any:
        return unwrap_tool_call_result(result)


def with_non_retryable_errors(retry_policy: RetryPolicy | None) -> RetryPolicy:
    """Return a copy of `retry_policy` with the framework's non-retryable errors ensured."""
    retry_policy = copy.copy(retry_policy) if retry_policy else RetryPolicy()
    existing = retry_policy.non_retryable_error_types or []
    additional = [UserError.__name__, PydanticUserError.__name__, UnexpectedModelBehavior.__name__]
    retry_policy.non_retryable_error_types = [*existing, *(name for name in additional if name not in existing)]
    return retry_policy


def resolve_tool_activity_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    tool_activity_config: Mapping[str, ActivityConfig | Literal[False]],
) -> ActivityConfig | Literal[False]:
    """Resolve per-tool Temporal activity config.

    Reads `tool.tool_def.metadata['temporal']` first, then falls back to the explicit
    `tool_activity_config` dict keyed by tool name. Returns an `ActivityConfig` dict
    (possibly empty), or `False` to skip activity wrapping.
    """
    config = cast(
        'ActivityConfig | Literal[False]',
        resolve_tool_durable_config(
            tool,
            tool_name,
            tool_activity_config,
            metadata_key='temporal',
            config_type_label='ActivityConfig',
        ),
    )
    if config is False:
        return False
    config = copy.copy(config)
    if 'retry_policy' in config:
        config['retry_policy'] = with_non_retryable_errors(config.get('retry_policy'))
    return config


def toolset_temporal_activities(toolset: AbstractToolset[Any]) -> list[Callable[..., Any]]:
    """The Temporal activities a durable-wrapped toolset needs registered with the worker."""
    if isinstance(toolset, DurableToolsetBase):
        return toolset.durable_registrations
    if isinstance(toolset, TemporalWrapperToolset):
        return toolset.temporal_activities
    return []


async def call_tool_in_activity(
    toolset: AbstractToolset[AgentDepsT],
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> CallToolResult:
    args = tool.args_validator.validate_python(tool_args)
    return await wrap_tool_call_result(toolset.call_tool(name, args, ctx, tool))


def temporalize_toolset(
    toolset: AbstractToolset[AgentDepsT],
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> AbstractToolset[AgentDepsT]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        activity_name_prefix: Prefix for Temporal activity names.
        activity_config: The Temporal activity config to use.
        tool_activity_config: The Temporal activity config to use for specific tools identified by tool name.
        deps_type: The type of agent's dependencies object. It needs to be serializable using Pydantic's `TypeAdapter`.
        run_context_type: The `TemporalRunContext` (sub)class that's used to serialize and deserialize the run context.
        agent: The agent instance to attach to deserialized run contexts in activities.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import temporalize_function_toolset

        return temporalize_function_toolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    if isinstance(toolset, DynamicToolset):
        from ._dynamic_toolset import temporalize_dynamic_toolset

        return temporalize_dynamic_toolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    try:
        from pydantic_ai.mcp import MCPToolset

        from ._mcp_toolset import temporalize_mcp_toolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return temporalize_mcp_toolset(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activity_config=activity_config,
                tool_activity_config=tool_activity_config,
                deps_type=deps_type,
                run_context_type=run_context_type,
                agent=agent,
            )

    return toolset
