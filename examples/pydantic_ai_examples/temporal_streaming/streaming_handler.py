"""Event streaming handler for the Temporal workflow.

This module handles streaming events from the agent during execution,
processing various event types and sending them to the workflow via signals.
"""

from collections.abc import AsyncIterable

from temporalio import activity
from temporalio.client import WorkflowHandle

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart, RunContext,
)
from .datamodels import AgentDependencies, EventKind, EventStream


async def streaming_handler(
        ctx: RunContext[AgentDependencies],
        event_stream_events: AsyncIterable[AgentStreamEvent],
) -> None:
    """
    Handle streaming events from the agent.

    This function processes events from the agent's execution stream, including
    tool calls, LLM responses, and streaming results. It aggregates the events
    and sends them to the workflow via signals.

    Args:
        ctx: The run context containing dependencies.
        event_stream_events: Async iterable of agent stream events.
    """
    if not activity.in_activity():
        return

    output: str = ''
    output_tool_delta: dict[str, str] = dict(
        tool_call_id='',
        tool_name_delta='',
        args_delta='',
    )

    # Process all events from the stream
    async for event in event_stream_events:
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart):
                output += f'{event.part.content}'
            elif isinstance(event.part, ToolCallPart):
                output += f'\nTool Call Id: {event.part.tool_call_id}'
                output += f'\nTool Name: {event.part.tool_name}'
                output += f'\nTool Args: {event.part.args}'
            else:
                pass
        elif isinstance(event, FunctionToolCallEvent):
            output += f'\nTool Call Id: {event.part.tool_call_id}'
            output += f'\nTool Name: {event.part.tool_name}'
            output += f'\nTool Args: {event.part.args}'
        elif isinstance(event, FunctionToolResultEvent):
            output += f'\nTool Call Id: {event.result.tool_call_id}'
            output += f'\nTool Name: {event.result.tool_name}'
            output += f'\nContent: {event.result.content}'
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta) or isinstance(event.delta, ThinkingPartDelta):
                output += f'{event.delta.content_delta}'
            else:
                if len(output_tool_delta['tool_call_id']) == 0:
                    output_tool_delta['tool_call_id'] += event.delta.tool_call_id or ''
                output_tool_delta['tool_name_delta'] += event.delta.tool_name_delta or ''
                # Handle args_delta which can be str or dict
                args_delta = event.delta.args_delta
                if isinstance(args_delta, str):
                    output_tool_delta['args_delta'] += args_delta
                elif isinstance(args_delta, dict):
                    output_tool_delta['args_delta'] += str(args_delta)

    # Add accumulated tool delta output if present
    if len(output_tool_delta['tool_call_id']):
        output += f'\nTool Call Id: {output_tool_delta["tool_call_id"]}'
        output += f'\nTool Name: {output_tool_delta["tool_name_delta"]}'
        args_delta_str = str(output_tool_delta["args_delta"])
        output += f'\nTool Args: {args_delta_str}'

    # Send events to workflow if running in an activity
    deps: AgentDependencies = ctx.deps

    workflow_id: str = deps.workflow_id
    run_id: str = deps.run_id
    from typing import Any
    workflow_handle: WorkflowHandle[Any, Any] = activity.client().get_workflow_handle(workflow_id=workflow_id, run_id=run_id)  # type: ignore[misc]
    await workflow_handle.signal('append_event', arg=EventStream(kind=EventKind.EVENT, content=output))
