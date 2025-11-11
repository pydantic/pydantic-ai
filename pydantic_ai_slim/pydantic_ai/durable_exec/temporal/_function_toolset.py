from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any, Literal

import anyio
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CustomEventDataT, HandleResponseEvent
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._run_context import TemporalRunContext
from ._toolset import (
    CallToolParams,
    CallToolResult,
    TemporalWrapperToolset,
)


class TemporalFunctionToolset(TemporalWrapperToolset[AgentDepsT]):
    def __init__(
        self,
        toolset: FunctionToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        event_stream_handler: EventStreamHandler[AgentDepsT, CustomEventDataT] | None = None,
    ):
        super().__init__(toolset)
        self.activity_config = activity_config
        self.tool_activity_config = tool_activity_config
        self.run_context_type = run_context_type
        self.event_stream_handler = event_stream_handler

        async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
            name = params.name
            # TODO (DouweM): RunContext.event_stream -> call event_stream_handler directly?
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            try:
                tool = (await toolset.get_tools(ctx))[name]
            except KeyError as e:  # pragma: no cover
                raise UserError(
                    f'Tool {name!r} not found in toolset {self.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Temporal.'
                ) from e

            # The tool args will already have been validated into their proper types in the `ToolManager`,
            # but `execute_activity` would have turned them into simple Python types again, so we need to re-validate them.
            args_dict = tool.args_validator.validate_python(params.tool_args)

            async def _call_tool(ctx: RunContext[AgentDepsT]) -> CallToolResult:
                return await self._wrap_call_tool_result(self.wrapped.call_tool(name, args_dict, ctx, tool))

            if self.event_stream_handler is not None:
                send_stream, receive_stream = anyio.create_memory_object_stream[HandleResponseEvent[CustomEventDataT]]()

                async def _call_tool_with_stream() -> CallToolResult:
                    async with send_stream:
                        ctx_with_stream = replace(ctx, event_stream=send_stream)
                        return await _call_tool(ctx_with_stream)

                task = asyncio.create_task(_call_tool_with_stream())

                async def _receive_events() -> AsyncIterator[HandleResponseEvent[CustomEventDataT]]:
                    async with receive_stream:
                        async for event in receive_stream:
                            yield event

                await self.event_stream_handler(ctx, _receive_events())

                return await task
            else:
                return await _call_tool(ctx)

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        call_tool_activity.__annotations__['deps'] = deps_type

        self.call_tool_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{self.id}__call_tool')(
            call_tool_activity
        )

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.call_tool_activity]

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not workflow.in_workflow():
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.tool_activity_config.get(name, {})
        if tool_activity_config is False:
            assert isinstance(tool, FunctionToolsetTool)
            if not tool.is_async:
                raise UserError(
                    f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
                    'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'
                )
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.activity_config | tool_activity_config
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        return self._unwrap_call_tool_result(
            await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
                activity=self.call_tool_activity,
                args=[
                    CallToolParams(
                        name=name,
                        tool_args=tool_args,
                        serialized_run_context=serialized_run_context,
                        tool_def=None,
                    ),
                    ctx.deps,
                ],
                **tool_activity_config,
            )
        )
