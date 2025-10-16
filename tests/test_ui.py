from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models.function import AgentInfo, BuiltinToolCallsReturns, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset
from pydantic_ai.ui.adapter import BaseAdapter
from pydantic_ai.ui.event_stream import BaseEventStream, SourceEvent

from .conftest import try_import

with try_import() as starlette_import_successful:
    from starlette.requests import Request

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


class UIRequest(BaseModel):
    messages: list[ModelMessage] = field(default_factory=list)
    tool_defs: list[ToolDefinition] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class UIState(BaseModel):
    country: str | None = None


@dataclass
class UIDeps:
    state: UIState


class UIAdapter(BaseAdapter[UIRequest, ModelMessage, str, UIDeps]):
    @classmethod
    async def validate_request(cls, request: Request) -> UIRequest:
        return UIRequest.model_validate(await request.json())

    @classmethod
    def load_messages(cls, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        return list(messages)

    @property
    def event_stream(self) -> UIEventStream:
        return UIEventStream(self.request)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        return self.request.messages

    @cached_property
    def state(self) -> dict[str, Any] | None:
        return self.request.state

    @cached_property
    def toolset(self) -> AbstractToolset[UIDeps] | None:
        return ExternalToolset(self.request.tool_defs) if self.request.tool_defs else None


@dataclass(kw_only=True)
class UIEventStream(BaseEventStream[UIRequest, str, UIDeps]):
    def encode_event(self, event: str, accept: str | None = None) -> str:
        return event

    async def handle_event(self, event: SourceEvent) -> AsyncIterator[str]:
        # yield f'[{event.event_kind}]'
        async for e in super().handle_event(event):
            yield e

    async def handle_part_start(self, event: PartStartEvent) -> AsyncIterator[str]:
        # yield f'[{event.part.part_kind}]'
        async for e in super().handle_part_start(event):
            yield e

    async def handle_part_delta(self, event: PartDeltaEvent) -> AsyncIterator[str]:
        # yield f'[>{event.delta.part_delta_kind}]'
        async for e in super().handle_part_delta(event):
            yield e

    async def handle_part_end(self, event: PartEndEvent) -> AsyncIterator[str]:
        # yield f'[/{event.part.part_kind}]'
        async for e in super().handle_part_end(event):
            yield e

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[str]:
        yield f'<text>{part.content}'
        async for e in super().handle_text_start(part, follows_text):
            yield e

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[str]:
        yield delta.content_delta
        async for e in super().handle_text_delta(delta):
            yield e

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[str]:
        yield '</text>'
        async for e in super().handle_text_end(part, followed_by_text):
            yield e

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[str]:
        yield f'<thinking>{part.content}'
        async for e in super().handle_thinking_start(part, follows_thinking):
            yield e

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[str]:
        yield str(delta.content_delta)
        async for e in super().handle_thinking_delta(delta):
            yield e

    async def handle_thinking_end(self, part: ThinkingPart, followed_by_thinking: bool = False) -> AsyncIterator[str]:
        yield '</thinking>'
        async for e in super().handle_thinking_end(part, followed_by_thinking):
            yield e

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[str]:
        yield f'<tool-call name={part.tool_name!r}>{part.args}'
        async for e in super().handle_tool_call_start(part):
            yield e

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[str]:
        yield str(delta.args_delta)
        async for e in super().handle_tool_call_delta(delta):
            yield e

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[str]:
        yield f'</tool-call name={part.tool_name!r}>'
        async for e in super().handle_tool_call_end(part):
            yield e

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[str]:
        yield f'<builtin-tool-call name={part.tool_name!r}>{part.args}'
        async for e in super().handle_builtin_tool_call_start(part):
            yield e

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[str]:
        yield f'</builtin-tool-call name={part.tool_name!r}>'
        async for e in super().handle_builtin_tool_call_end(part):
            yield e

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[str]:
        yield f'<builtin-tool-return name={part.tool_name!r}>{part.content}</builtin-tool-return>'
        async for e in super().handle_builtin_tool_return(part):
            yield e

    async def handle_file(self, part: FilePart) -> AsyncIterator[str]:
        yield f'<file media_type={part.content.media_type!r} />'
        async for e in super().handle_file(part):
            yield e

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[str]:
        yield '<final-result />'
        async for e in super().handle_final_result(event):
            yield e

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[str]:
        yield f'<function-tool-call name={event.part.tool_name!r}>{event.part.args}</function-tool-call>'
        async for e in super().handle_function_tool_call(event):
            yield e

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[str]:
        yield f'<function-tool-result name={event.result.tool_name!r}>{event.result.content}</function-tool-result>'
        async for e in super().handle_function_tool_result(event):
            yield e

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[str]:
        yield f'<run-result>{event.result.output}</run-result>'
        async for e in super().handle_run_result(event):
            yield e

    async def before_request(self) -> AsyncIterator[str]:
        yield '<request>'
        async for e in super().before_request():
            yield e

    async def after_request(self) -> AsyncIterator[str]:
        yield '</request>'
        async for e in super().after_request():
            yield e

    async def before_response(self) -> AsyncIterator[str]:
        yield '<response>'
        async for e in super().before_response():
            yield e

    async def after_response(self) -> AsyncIterator[str]:
        yield '</response>'
        async for e in super().after_response():
            yield e

    async def before_stream(self) -> AsyncIterator[str]:
        yield '<stream>'
        async for e in super().before_stream():
            yield e

    async def after_stream(self) -> AsyncIterator[str]:
        yield '</stream>'
        async for e in super().after_stream():
            yield e

    async def on_error(self, error: Exception) -> AsyncIterator[str]:
        yield f'on_error({error.__class__.__name__}({str(error)!r}))'
        async for e in super().on_error(error):
            yield e


async def test_event_stream_text():
    # text
    #  - back to back
    # thinking
    #  - back to back
    # tool call
    # builtin tool call
    # file
    # error
    # output tool
    pass


async def test_event_stream_builtin_tool_call():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        yield {
            0: BuiltinToolCallPart(
                tool_name=WebSearchTool.kind,
                args='{"query":',
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield {
            0: DeltaToolCall(
                json_args='"Hello world"}',
                tool_call_id='search_1',
            )
        }
        yield {
            1: BuiltinToolReturnPart(
                tool_name=WebSearchTool.kind,
                content={
                    'results': [
                        {
                            'title': '"Hello, World!" program',
                            'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                        }
                    ]
                },
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". '

    agent = Agent(model=FunctionModel(stream_function=stream_function), deps_type=UIDeps)

    request = UIRequest(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    deps = UIDeps(state=UIState())

    adapter = UIAdapter(agent, request)
    events = [event async for event in adapter.run_stream(deps=deps)]

    assert events == snapshot(
        [
            '<stream>',
            '<request>',
            '<builtin-tool-call name=\'web_search\'>{"query":',
            '"Hello world"}',
            "</builtin-tool-call name='web_search'>",
            "<builtin-tool-return name='web_search'>{'results': [{'title': '\"Hello, World!\" program', 'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program'}]}</builtin-tool-return>",
            '<text>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
            '<final-result />',
            '</text>',
            '</request>',
            '<run-result>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". </run-result>',
            '</stream>',
        ]
    )
