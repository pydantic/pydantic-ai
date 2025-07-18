from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from mcp import types as mcp_types
from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ._run_context import AgentDepsT, RunContext
from .messages import (
    ModelMessage,
    ModelResponse,
)
from .models import Model, ModelRequestParameters, StreamedResponse
from .settings import ModelSettings
from .toolsets import ToolsetTool


@dataclass
class TemporalSettings:
    task_queue: str | None = None
    schedule_to_close_timeout: timedelta | None = None
    schedule_to_start_timeout: timedelta | None = None
    start_to_close_timeout: timedelta | None = None
    heartbeat_timeout: timedelta | None = None
    retry_policy: RetryPolicy | None = None
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL
    activity_id: str | None = None
    versioning_intent: VersioningIntent | None = None
    summary: str | None = None
    priority: Priority = Priority.default


def initialize_temporal():
    from pydantic_ai.messages import (  # noqa F401
        ModelResponse,  # pyright: ignore[reportUnusedImport]
        ImageUrl,  # pyright: ignore[reportUnusedImport]
        AudioUrl,  # pyright: ignore[reportUnusedImport]
        DocumentUrl,  # pyright: ignore[reportUnusedImport]
        VideoUrl,  # pyright: ignore[reportUnusedImport]
        BinaryContent,  # pyright: ignore[reportUnusedImport]
        UserContent,  # pyright: ignore[reportUnusedImport]
    )


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class ModelRequestParams:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


def temporalize_model(model: Model, temporal_settings: TemporalSettings | None = None) -> list[Callable[..., Any]]:
    if activities := getattr(model, '__temporal_activities', None):
        return activities

    temporal_settings = temporal_settings or TemporalSettings()

    original_request = model.request

    @activity.defn(name='model_request')
    async def request_activity(params: ModelRequestParams) -> ModelResponse:
        return await original_request(params.messages, params.model_settings, params.model_request_parameters)

    async def request(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=request_activity,
            arg=ModelRequestParams(
                messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
            ),
            **temporal_settings.__dict__,
        )

    @asynccontextmanager
    async def request_stream(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        raise NotImplementedError('Cannot stream with temporal yet')
        yield

    model.request = request
    model.request_stream = request_stream

    activities = [request_activity]
    setattr(model, '__temporal_activities', activities)
    return activities


# @dataclass
# class TemporalModel(WrapperModel):
#     temporal_settings: TemporalSettings

#     def __init__(
#         self,
#         wrapped: Model | KnownModelName,
#         temporal_settings: TemporalSettings | None = None,
#     ) -> None:
#         super().__init__(wrapped)
#         self.temporal_settings = temporal_settings or TemporalSettings()

#         @activity.defn
#         async def request_activity(params: ModelRequestParams) -> ModelResponse:
#             return await self.wrapped.request(params.messages, params.model_settings, params.model_request_parameters)

#         self.request_activity = request_activity

#     async def request(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> ModelResponse:
#         return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
#             activity=self.request_activity,
#             arg=ModelRequestParams(
#                 messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
#             ),
#             **self.temporal_settings.__dict__,
#         )

#     @asynccontextmanager
#     async def request_stream(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> AsyncIterator[StreamedResponse]:
#         raise NotImplementedError('Cannot stream with temporal yet')
#         yield


class TemporalRunContext(RunContext[AgentDepsT]):
    _data: dict[str, Any]

    def __init__(self, **kwargs: Any):
        self._data = kwargs
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in kwargs},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            data = super().__getattribute__('_data')
            if name in data:
                return data[name]
            raise e  # TODO: Explain how to make a new run context attribute available

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[AgentDepsT]) -> dict[str, Any]:
        return {
            'deps': ctx.deps,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any]) -> RunContext[AgentDepsT]:
        return cls(**ctx)


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class MCPCallToolParams:
    name: str
    tool_args: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class FunctionCallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any


def temporalize_mcp_server(
    server: MCPServer, temporal_settings: TemporalSettings | None = None
) -> list[Callable[..., Any]]:
    if activities := getattr(server, '__temporal_activities', None):
        return activities

    temporal_settings = temporal_settings or TemporalSettings()

    original_list_tools = server.list_tools
    original_direct_call_tool = server.direct_call_tool

    @activity.defn(
        name='mcp_server_list_tools'
    )  # TODO: Require a name to be passed to TemporalMCPServer? If we get toolsets from a lib, what do we do? Strongly recommend setting a name?
    async def list_tools_activity() -> list[mcp_types.Tool]:
        return await original_list_tools()

    @activity.defn(name='mcp_server_call_tool')
    async def call_tool_activity(params: MCPCallToolParams) -> ToolResult:
        return await original_direct_call_tool(params.name, params.tool_args, params.metadata)

    async def list_tools() -> list[mcp_types.Tool]:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            activity=list_tools_activity,
            **temporal_settings.__dict__,
        )

    async def direct_call_tool(
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=MCPCallToolParams(name=name, tool_args=args, metadata=metadata),
            **temporal_settings.__dict__,
        )

    server.list_tools = list_tools
    server.direct_call_tool = direct_call_tool

    activities = [list_tools_activity, call_tool_activity]
    setattr(server, '__temporal_activities', activities)
    return activities


# class TemporalMCPServer(WrapperToolset[Any]):
#     temporal_settings: TemporalSettings

#     @property
#     def wrapped_server(self) -> MCPServer:
#         assert isinstance(self.wrapped, MCPServer)
#         return self.wrapped

#     def __init__(self, wrapped: MCPServer, temporal_settings: TemporalSettings | None = None):
#         assert isinstance(self.wrapped, MCPServer)
#         super().__init__(wrapped)
#         self.temporal_settings = temporal_settings or TemporalSettings()

#         @activity.defn(name='mcp_server_list_tools')
#         async def list_tools_activity() -> list[mcp_types.Tool]:
#             return await self.wrapped_server.list_tools()

#         self.list_tools_activity = list_tools_activity

#         @activity.defn(name='mcp_server_call_tool')
#         async def call_tool_activity(params: MCPCallToolParams) -> ToolResult:
#             return await self.wrapped_server.direct_call_tool(params.name, params.tool_args, params.metadata)

#         self.call_tool_activity = call_tool_activity

#     async def list_tools(self) -> list[mcp_types.Tool]:
#         return await workflow.execute_activity(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
#             activity=self.list_tools_activity,
#             **self.temporal_settings.__dict__,
#         )

#     async def direct_call_tool(
#         self,
#         name: str,
#         args: dict[str, Any],
#         metadata: dict[str, Any] | None = None,
#     ) -> ToolResult:
#         return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
#             activity=self.call_tool_activity,
#             arg=MCPCallToolParams(name=name, tool_args=args, metadata=metadata),
#             **self.temporal_settings.__dict__,
#         )


def temporalize_function_toolset(
    toolset: FunctionToolset[AgentDepsT],
    temporal_settings: TemporalSettings | None = None,
    serialize_run_context: Callable[[RunContext[AgentDepsT]], Any] | None = None,
    deserialize_run_context: Callable[[Any], RunContext[AgentDepsT]] | None = None,
) -> list[Callable[..., Any]]:
    if activities := getattr(toolset, '__temporal_activities', None):
        return activities

    temporal_settings = temporal_settings or TemporalSettings()
    # TODO: Settings per tool name
    serialize_run_context = serialize_run_context or TemporalRunContext[AgentDepsT].serialize_run_context
    deserialize_run_context = deserialize_run_context or TemporalRunContext[AgentDepsT].deserialize_run_context

    original_call_tool = toolset.call_tool

    @activity.defn(name='function_toolset_call_tool')
    async def call_tool_activity(params: FunctionCallToolParams) -> Any:
        ctx = deserialize_run_context(params.serialized_run_context)
        tool = (await toolset.get_tools(ctx))[params.name]
        return await original_call_tool(params.name, params.tool_args, ctx, tool)

    async def call_tool(
        name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        serialized_run_context = serialize_run_context(ctx)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=FunctionCallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
            **temporal_settings.__dict__,
        )

    toolset.call_tool = call_tool

    activities = [call_tool_activity]
    setattr(toolset, '__temporal_activities', activities)
    return activities


# class TemporalFunctionToolset(FunctionToolset[AgentDepsT]):
#     def __init__(
#         self,
#         tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
#         max_retries: int = 1,
#         temporal_settings: TemporalSettings | None = None,
#         serialize_run_context: Callable[[RunContext[AgentDepsT]], Any] | None = None,
#         deserialize_run_context: Callable[[Any], RunContext[AgentDepsT]] | None = None,
#     ):
#         super().__init__(tools, max_retries)
#         self.temporal_settings = temporal_settings or TemporalSettings()
#         self.serialize_run_context = serialize_run_context or TemporalRunContext[AgentDepsT].serialize_run_context
#         self.deserialize_run_context = deserialize_run_context or TemporalRunContext[AgentDepsT].deserialize_run_context

#         @activity.defn(name='function_toolset_call_tool')
#         async def call_tool_activity(params: FunctionCallToolParams) -> Any:
#             ctx = self.deserialize_run_context(params.serialized_run_context)
#             tool = (await self.get_tools(ctx))[params.name]
#             return await FunctionToolset[AgentDepsT].call_tool(self, params.name, params.tool_args, ctx, tool)

#         self.call_tool_activity = call_tool_activity

#     async def call_tool(
#         self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
#     ) -> Any:
#         serialized_run_context = self.serialize_run_context(ctx)
#         return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
#             activity=self.call_tool_activity,
#             arg=FunctionCallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
#             **self.temporal_settings.__dict__,
#         )


def temporalize_agent(agent: Agent, temporal_settings: TemporalSettings | None = None) -> list[Callable[..., Any]]:
    if existing_activities := getattr(agent, '__temporal_activities', None):
        return existing_activities

    temporal_settings = temporal_settings or TemporalSettings()

    activities: list[Callable[..., Any]] = []
    if isinstance(agent.model, Model):
        # Doesn't work when model is not set already
        activities.extend(temporalize_model(agent.model, temporal_settings))

    # TODO : Make TemporalMCPServer a wrapper
    if toolset := agent._get_toolset():
        # Doesn't consider toolsets passed at iter time
        def temporalize_toolset(toolset: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(toolset, FunctionToolset):
                activities.extend(temporalize_function_toolset(toolset, temporal_settings))
            elif isinstance(toolset, MCPServer):
                activities.extend(temporalize_mcp_server(toolset, temporal_settings))

        toolset.apply(temporalize_toolset)

    setattr(agent, '__temporal_activities', activities)
    return activities
