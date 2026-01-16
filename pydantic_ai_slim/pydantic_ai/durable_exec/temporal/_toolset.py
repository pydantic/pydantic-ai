from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from pydantic import BeforeValidator, ConfigDict, Discriminator, PlainSerializer, TypeAdapter, with_config
from pydantic.dataclasses import dataclass as pydantic_dataclass
from temporalio import workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Self, assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._run_context import TemporalRunContext


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


@dataclass
class _ToolReturn:
    result: Any
    kind: Literal['tool_return'] = 'tool_return'


CallToolResult = Annotated[
    _ApprovalRequired | _CallDeferred | _ModelRetry | _ToolReturn,
    Discriminator('kind'),
]


@pydantic_dataclass
class SerializableBinaryContent:
    """Temporal-serializable version of BinaryContent with base64-encoded data."""

    data: Annotated[
        bytes,
        BeforeValidator(lambda v: base64.b64decode(v) if isinstance(v, str) else v),
        PlainSerializer(lambda v: base64.b64encode(v).decode('ascii'), return_type=str),
    ]
    media_type: str
    identifier: str | None = None
    vendor_metadata: dict[str, Any] | None = None
    kind: Literal['binary'] = 'binary'


serializable_binary_content_ta = TypeAdapter(SerializableBinaryContent)


def to_serializable(result: Any) -> Any:
    """Convert BinaryContent to SerializableBinaryContent for Temporal serialization."""
    if isinstance(result, BinaryContent):
        return SerializableBinaryContent(
            data=result.data,
            media_type=result.media_type,
            identifier=result.identifier,
            vendor_metadata=result.vendor_metadata,
        )
    else:
        return result


def from_serializable(result: Any) -> BinaryContent | Any:
    """Convert SerializableBinaryContent or dict back to BinaryContent after Temporal deserialization."""
    if isinstance(result, SerializableBinaryContent):
        return BinaryContent(
            data=result.data,
            media_type=result.media_type,
            identifier=result.identifier,
            vendor_metadata=result.vendor_metadata,
        )
    elif isinstance(result, dict):
        result = cast(dict[str, Any], result)
        if result.get('kind') != 'binary':
            return result
        else:
            sbc = serializable_binary_content_ta.validate_python(result)
            return BinaryContent(
                data=sbc.data,
                media_type=sbc.media_type,
                identifier=sbc.identifier,
                vendor_metadata=sbc.vendor_metadata,
            )
    else:
        return result


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
            return _ToolReturn(result=to_serializable(result))
        except ApprovalRequired as e:
            return _ApprovalRequired(metadata=e.metadata)
        except CallDeferred as e:
            return _CallDeferred(metadata=e.metadata)
        except ModelRetry as e:
            return _ModelRetry(message=e.message)

    def _unwrap_call_tool_result(self, result: CallToolResult) -> Any:
        if isinstance(result, _ToolReturn):
            return from_serializable(result.result)
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
    ) -> CallToolResult:
        """Call a tool inside an activity, re-validating args that were deserialized.

        The tool args will already have been validated into their proper types in the `ToolManager`,
        but `execute_activity` would have turned them into simple Python types again, so we need to re-validate them.
        """
        args_dict = tool.args_validator.validate_python(tool_args)
        return await self._wrap_call_tool_result(self.wrapped.call_tool(name, args_dict, ctx, tool))


def temporalize_toolset(
    toolset: AbstractToolset[AgentDepsT],
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
) -> AbstractToolset[AgentDepsT]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        activity_name_prefix: Prefix for Temporal activity names.
        activity_config: The Temporal activity config to use.
        tool_activity_config: The Temporal activity config to use for specific tools identified by tool name.
        deps_type: The type of agent's dependencies object. It needs to be serializable using Pydantic's `TypeAdapter`.
        run_context_type: The `TemporalRunContext` (sub)class that's used to serialize and deserialize the run context.
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
            )

    return toolset
