from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Discriminator, with_config
from temporalio.workflow import ActivityConfig
from typing_extensions import assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from ._run_context import TemporalRunContext


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class GetToolsParamsData:
    serialized_run_context: Any


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CallToolParamsData:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition | None


@dataclass
class ApprovalRequiredData:
    kind: Literal['approval_required'] = 'approval_required'


@dataclass
class CallDeferredData:
    kind: Literal['call_deferred'] = 'call_deferred'


@dataclass
class ModelRetryData:
    message: str
    kind: Literal['model_retry'] = 'model_retry'


@dataclass
class ToolReturnData:
    result: Any
    kind: Literal['tool_return'] = 'tool_return'


CallToolResultData = Annotated[
    ApprovalRequiredData | CallDeferredData | ModelRetryData | ToolReturnData,
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


def remap_exception_to_dataclass(e: Exception) -> CallToolResultData:
    try:
        raise e
    except ApprovalRequired:
        return ApprovalRequiredData()
    except CallDeferred:
        return CallDeferredData()
    except ModelRetry as e:
        return ModelRetryData(message=e.message)


def remap_dataclass_to_exception(o: CallToolResultData):
    if isinstance(o, ApprovalRequiredData):
        raise ApprovalRequired()
    elif isinstance(o, CallDeferredData):
        raise CallDeferred()
    elif isinstance(o, ModelRetryData):
        raise ModelRetry(o.message)
    elif isinstance(o, ToolReturnData):
        return o.result
    else:
        assert_never(o)


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

    return toolset
