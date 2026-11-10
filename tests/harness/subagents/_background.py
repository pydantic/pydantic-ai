"""A parent model that delegates to `worker` in the background, ends its turn, then answers with the task's report."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import (
    CapabilityEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.subagents import is_task_report

WORKER_CALLS: list[str] = []
"""The worker's model requests, one entry per request."""

_turn_ended = asyncio.Event()


def reset_worker() -> None:
    """Forget the worker's requests, and hold its next one until the parent ends its turn."""
    global _turn_ended
    WORKER_CALLS.clear()
    _turn_ended = asyncio.Event()


def _parent(messages: list[ModelMessage], background: bool) -> ModelResponse:
    reports = [
        ''.join(item.content for item in part.content if isinstance(item, TextContent))
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if is_task_report(part)
    ]
    if reports:
        return ModelResponse(parts=[TextPart(reports[-1])])
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        _turn_ended.set()
        return ModelResponse(parts=[TextPart('Waiting for the report.')])
    args = {'agent_name': 'worker', 'task': 'Work on it', 'background': background}
    return ModelResponse(parts=[ToolCallPart('delegate_task', args)])


def background_parent(*, background: bool = True) -> FunctionModel:
    """Without `background`, the task only moves to the background with `auto_background_after`.

    Streams only, as a run whose capabilities handle events streams its model requests.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        [part] = _parent(messages, background).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return FunctionModel(stream_function=stream)


async def background_worker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    WORKER_CALLS.append('worker')
    await _turn_ended.wait()
    return ModelResponse(parts=[TextPart('from the worker')])


@dataclass
class EventKinds(AbstractCapability[Any]):
    """Records the kind of every task event the run emits."""

    kinds: list[str] = field(default_factory=list[str])

    @on_event(CapabilityEvent)
    async def record(self, ctx: RunContext[Any], event: CapabilityEvent) -> None:
        if event.kind.startswith('sub_agents.task_'):
            self.kinds.append(event.kind)
