from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Discriminator, with_config
from typing_extensions import assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.messages import Return, ToolReturn
from pydantic_ai.tools import AgentDepsT, ToolDefinition

if TYPE_CHECKING:
    from ._agent import TemporalizeToolsetContext


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition | None


@dataclass
class _ApprovalRequired:
    kind: Literal['approval_required'] = 'approval_required'


@dataclass
class _CallDeferred:
    kind: Literal['call_deferred'] = 'call_deferred'


@dataclass
class _ModelRetry:
    message: str
    kind: Literal['model_retry'] = 'model_retry'


@dataclass
class _ToolReturn:
    result: ToolReturn[Any] | Any
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

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Temporalized toolsets cannot be swapped out after the fact.
        return self

    async def _wrap_call_tool_result(self, coro: Awaitable[Any]) -> CallToolResult:
        try:
            result = await coro
            if type(result) is Return:
                # We don't use `isinstance` because `ToolReturn` is a subclass of `Return` with additional fields, which should be returned in full.
                result = result.return_value
            return _ToolReturn(result=result)
        except ApprovalRequired:
            return _ApprovalRequired()
        except CallDeferred:
            return _CallDeferred()
        except ModelRetry as e:
            return _ModelRetry(message=e.message)

    def _unwrap_call_tool_result(self, result: CallToolResult) -> Any:
        if isinstance(result, _ToolReturn):
            return result.result
        elif isinstance(result, _ApprovalRequired):
            raise ApprovalRequired()
        elif isinstance(result, _CallDeferred):
            raise CallDeferred()
        elif isinstance(result, _ModelRetry):
            raise ModelRetry(result.message)
        else:
            assert_never(result)


def temporalize_toolset(
    toolset: AbstractToolset[AgentDepsT], context: TemporalizeToolsetContext[AgentDepsT]
) -> AbstractToolset[AgentDepsT]:
    """Temporalize a toolset."""
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import TemporalFunctionToolset

        return TemporalFunctionToolset(
            toolset,
            activity_name_prefix=context.activity_name_prefix,
            activity_config=context.activity_config,
            tool_activity_config=context.tool_activity_config,
            deps_type=context.deps_type,
            run_context_type=context.run_context_type,
            event_stream_handler=context.event_stream_handler,
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
                activity_name_prefix=context.activity_name_prefix,
                activity_config=context.activity_config,
                tool_activity_config=context.tool_activity_config,
                deps_type=context.deps_type,
                run_context_type=context.run_context_type,
            )

    return toolset
