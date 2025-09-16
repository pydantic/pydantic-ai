from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic_core import to_json

from .. import messages
from ..agent import Agent
from ..run import AgentRunResultEvent
from ..tools import AgentDepsT
from . import response_types as _t

__all__ = 'sse_stream', 'VERCEL_AI_ELEMENTS_HEADERS', 'EventStreamer'
# no idea if this is important, but vercel sends it, therefore so am I
VERCEL_AI_ELEMENTS_HEADERS = {'x-vercel-ai-ui-message-stream': 'v1'}


async def sse_stream(agent: Agent[AgentDepsT], user_prompt: str, deps: Any) -> AsyncIterator[str]:
    """Stream events from an agent run as Vercel AI Elements events.

    Args:
        agent: The agent to run.
        user_prompt: The user prompt to run the agent with.
        deps: The dependencies to pass to the agent.

    Yields:
        An async iterator text lines to stream over SSE.
    """
    event_streamer = EventStreamer()
    async for event in agent.run_stream_events(user_prompt, deps=deps):
        if not isinstance(event, AgentRunResultEvent):
            async for chunk in event_streamer.event_to_chunks(event):
                yield chunk.sse()
    async for chunk in event_streamer.finish():
        yield chunk.sse()


@dataclass
class EventStreamer:
    """Logic for mapping pydantic-ai events to Vercel AI Elements events which can be streamed to a client over SSE."""

    message_id: str = field(default_factory=lambda: uuid4().hex)
    _final_result_tool_id: str | None = field(default=None, init=False)

    async def event_to_chunks(self, event: messages.AgentStreamEvent) -> AsyncIterator[_t.AbstractSSEChunk]:  # noqa C901
        """Convert pydantic-ai events to Vercel AI Elements events which can be streamed to a client over SSE.

        Args:
            event: The pydantic-ai event to convert.

        Yields:
            An async iterator of Vercel AI Elements events.
        """
        match event:
            case messages.PartStartEvent(part=part):
                match part:
                    case messages.TextPart(content=content):
                        yield _t.TextStartChunk(id=self.message_id)
                        yield _t.TextDeltaChunk(id=self.message_id, delta=content)
                    case (
                        messages.ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                        | messages.BuiltinToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                    ):
                        yield _t.ToolInputStartChunk(tool_call_id=tool_call_id, tool_name=tool_name)
                        if isinstance(args, str):
                            yield _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=args)
                        elif args is not None:
                            yield (
                                _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=_json_dumps(args))
                            )

                    case messages.BuiltinToolReturnPart(
                        tool_name=tool_name, tool_call_id=tool_call_id, content=content
                    ):
                        yield _t.ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=content)

                    case messages.ThinkingPart(content=content):
                        yield _t.ReasoningStartChunk(id=self.message_id)
                        yield _t.ReasoningDeltaChunk(id=self.message_id, delta=content)

            case messages.PartDeltaEvent(delta=delta):
                match delta:
                    case messages.TextPartDelta(content_delta=content_delta):
                        yield _t.TextDeltaChunk(id=self.message_id, delta=content_delta)
                    case messages.ThinkingPartDelta(content_delta=content_delta):
                        if content_delta:
                            yield _t.ReasoningDeltaChunk(id=self.message_id, delta=content_delta)
                    case messages.ToolCallPartDelta(args_delta=args, tool_call_id=tool_call_id):
                        tool_call_id = tool_call_id or ''
                        if isinstance(args, str):
                            yield _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=args)
                        elif args is not None:
                            yield (
                                _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=_json_dumps(args))
                            )
            case messages.FinalResultEvent(tool_name=tool_name, tool_call_id=tool_call_id):
                if tool_call_id and tool_name:
                    self._final_result_tool_id = tool_call_id
                    yield _t.ToolInputStartChunk(tool_call_id=tool_call_id, tool_name=tool_name)
            case messages.FunctionToolCallEvent():
                pass
                # print(f'TODO FunctionToolCallEvent {part}')
            case messages.FunctionToolResultEvent(result=result):
                match result:
                    case messages.ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content):
                        yield _t.ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=content)
                    case messages.RetryPromptPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content):
                        yield _t.ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=content)
            case messages.BuiltinToolCallEvent(part=part):
                tool_call_id = part.tool_call_id
                tool_name = part.tool_name
                args = part.args
                yield _t.ToolInputStartChunk(tool_call_id=tool_call_id, tool_name=tool_name)
                if isinstance(args, str):
                    yield _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=args)
                elif args is not None:
                    yield _t.ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=_json_dumps(args))
            case messages.BuiltinToolResultEvent(result=result):
                yield _t.ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

    async def finish(self) -> AsyncIterator[_t.AbstractSSEChunk | DoneChunk]:
        """Send extra messages required to close off the stream."""
        if tool_call_id := self._final_result_tool_id:
            yield _t.ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=None)
        yield _t.FinishChunk()
        yield DoneChunk()


class DoneChunk:
    def sse(self) -> str:
        return '[DONE]'

    def __str__(self) -> str:
        return 'DoneChunk<marker for the end of sse stream message>'


def _json_dumps(obj: Any) -> str:
    return to_json(obj).decode('utf-8')
