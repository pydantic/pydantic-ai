from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Discriminator, Tag, with_config
from temporalio import workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Self, assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.messages import ToolReturn, ToolReturnContent
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


@dataclass
class _ApprovalRequired:
    metadata: dict[str, Any] | None = None
    kind: Literal['approval_required'] = 'approval_required'


@dataclass
class _CallDeferred:
    metadata: dict[str, Any] | None = None
    kind: Literal['call_deferred'] = 'call_deferred'


@dataclass
class _ModelRetry:
    message: str
    kind: Literal['model_retry'] = 'model_retry'


def _result_discriminator(v: Any) -> str:
    if isinstance(v, ToolReturn) or (isinstance(v, dict) and v.get('kind') == 'tool-return'):  # pyright: ignore[reportUnknownMemberType]
        return 'tool-return'
    return 'content'


# Defined at module level so Pydantic resolves the Annotated metadata at runtime,
# not as a string annotation (which would lose the discriminator under `from __future__ import annotations`).
_ToolReturnResult = Annotated[
    Annotated[ToolReturn, Tag('tool-return')] | Annotated[ToolReturnContent, Tag('content')],
    Discriminator(_result_discriminator),
]


@dataclass
class _ToolReturn:
    result: _ToolReturnResult
    kind: Literal['tool_return'] = 'tool_return'


CallToolResult = Annotated[
    _ApprovalRequired | _CallDeferred | _ModelRetry | _ToolReturn,
    Discriminator('kind'),
]


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
        return self

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Temporal-wrapped toolsets manage their wrapped toolset's lifecycle
        # per-activity (inside activities), not per-run-step.
        return self

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Temporalized toolsets cannot be swapped out after the fact.
        return self

    async def __aenter__(self) -> Self:
        if not workflow.in_workflow():  # pragma: no cover
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if not workflow.in_workflow():  # pragma: no cover
            return await self.wrapped.__aexit__(*args)
        return None

    async def _wrap_call_tool_result(self, coro: Awaitable[Any]) -> CallToolResult:
        try:
            result = await coro
            return _ToolReturn(result=result)
        except ApprovalRequired as e:
            return _ApprovalRequired(metadata=e.metadata)
        except CallDeferred as e:
            return _CallDeferred(metadata=e.metadata)
        except ModelRetry as e:
            return _ModelRetry(message=e.message)

    def _unwrap_call_tool_result(self, result: CallToolResult) -> Any:
        if isinstance(result, _ToolReturn):
            return result.result
        elif isinstance(result, _ApprovalRequired):
            raise ApprovalRequired(metadata=result.metadata)
        elif isinstance(result, _CallDeferred):
            raise CallDeferred(metadata=result.metadata)
        elif isinstance(result, _ModelRetry):
            raise ModelRetry(result.message)
        else:
            assert_never(result)

    async def _call_tool_in_activity(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        *,
        toolset: AbstractToolset[AgentDepsT] | None = None,
    ) -> CallToolResult:
        """Call a tool inside an activity, re-validating args that were deserialized.

        The tool args will already have been validated into their proper types in the `ToolManager`,
        but `execute_activity` would have turned them into simple Python types again, so we need to re-validate them.

        Args:
            name: The name of the tool to call.
            tool_args: The raw tool arguments to re-validate and pass.
            ctx: The run context.
            tool: The tool definition.
            toolset: The toolset to call the tool on. Defaults to `self.wrapped`.
        """
        toolset = toolset or self.wrapped
        args_dict = tool.args_validator.validate_python(tool_args)
        return await self._wrap_call_tool_result(toolset.call_tool(name, args_dict, ctx, tool))


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
        from ._function_toolset import TemporalFunctionToolset

        return TemporalFunctionToolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    if isinstance(toolset, DynamicToolset):
        from ._dynamic_toolset import TemporalDynamicToolset

        return TemporalDynamicToolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    try:
        from pydantic_ai.mcp import MCPServer

        from ._mcp_server import TemporalMCPServer
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPServer):
            return TemporalMCPServer(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activity_config=activity_config,
                tool_activity_config=tool_activity_config,
                deps_type=deps_type,
                run_context_type=run_context_type,
                agent=agent,
            )

    try:
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        from ._fastmcp_toolset import TemporalFastMCPToolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, FastMCPToolset):
            return TemporalFastMCPToolset(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activity_config=activity_config,
                tool_activity_config=tool_activity_config,
                deps_type=deps_type,
                run_context_type=run_context_type,
                agent=agent,
            )

    return toolset
