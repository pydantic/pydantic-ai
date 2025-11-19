from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    BinaryImage,
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
from pydantic_ai.models.function import (
    AgentInfo,
    BuiltinToolCallsReturns,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset
from pydantic_ai.ui import NativeEvent, UIAdapter, UIEventStream

from .conftest import try_import

with try_import() as starlette_import_successful:
    from starlette.requests import Request
    from starlette.responses import StreamingResponse


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


class DummyUIRunInput(BaseModel):
    messages: list[ModelMessage] = field(default_factory=list)
    tool_defs: list[ToolDefinition] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class DummyUIState(BaseModel):
    country: str | None = None


@dataclass
class DummyUIDeps:
    state: DummyUIState


class DummyUIAdapter(UIAdapter[DummyUIRunInput, ModelMessage, str, AgentDepsT, OutputDataT]):
    @classmethod
    def build_run_input(cls, body: bytes) -> DummyUIRunInput:
        return DummyUIRunInput.model_validate_json(body)

    @classmethod
    def load_messages(cls, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        return list(messages)

    def build_event_stream(self) -> UIEventStream[DummyUIRunInput, str, AgentDepsT, OutputDataT]:
        return DummyUIEventStream[AgentDepsT, OutputDataT](self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        return self.load_messages(self.run_input.messages)

    @cached_property
    def state(self) -> dict[str, Any] | None:
        return self.run_input.state

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return ExternalToolset(self.run_input.tool_defs) if self.run_input.tool_defs else None


class DummyUIEventStream(UIEventStream[DummyUIRunInput, str, AgentDepsT, OutputDataT]):
    @property
    def response_headers(self) -> dict[str, str]:
        return {'x-test': 'test'}

    def encode_event(self, event: str) -> str:
        return event

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[str]:
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
        yield f'<text follows_text={follows_text!r}>{part.content}'

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[str]:
        yield delta.content_delta

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[str]:
        yield f'</text followed_by_text={followed_by_text!r}>'

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[str]:
        yield f'<thinking follows_thinking={follows_thinking!r}>{part.content}'

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[str]:
        yield str(delta.content_delta)

    async def handle_thinking_end(self, part: ThinkingPart, followed_by_thinking: bool = False) -> AsyncIterator[str]:
        yield f'</thinking followed_by_thinking={followed_by_thinking!r}>'

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[str]:
        yield f'<tool-call name={part.tool_name!r}>{part.args}'

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[str]:
        yield str(delta.args_delta)

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[str]:
        yield f'</tool-call name={part.tool_name!r}>'

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[str]:
        yield f'<builtin-tool-call name={part.tool_name!r}>{part.args}'

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[str]:
        yield f'</builtin-tool-call name={part.tool_name!r}>'

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[str]:
        yield f'<builtin-tool-return name={part.tool_name!r}>{part.content}</builtin-tool-return>'

    async def handle_file(self, part: FilePart) -> AsyncIterator[str]:
        yield f'<file media_type={part.content.media_type!r} />'

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[str]:
        yield f'<final-result tool_name={event.tool_name!r} />'

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[str]:
        yield f'<function-tool-call name={event.part.tool_name!r}>{event.part.args}</function-tool-call>'

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[str]:
        yield f'<function-tool-result name={event.result.tool_name!r}>{event.result.content}</function-tool-result>'

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[str]:
        yield f'<run-result>{event.result.output}</run-result>'

    async def before_stream(self) -> AsyncIterator[str]:
        yield '<stream>'

    async def before_response(self) -> AsyncIterator[str]:
        yield '<response>'

    async def after_response(self) -> AsyncIterator[str]:
        yield '</response>'

    async def before_request(self) -> AsyncIterator[str]:
        yield '<request>'

    async def after_request(self) -> AsyncIterator[str]:
        yield '</request>'

    async def after_stream(self) -> AsyncIterator[str]:
        yield '</stream>'

    async def on_error(self, error: Exception) -> AsyncIterator[str]:
        yield f'<error type={error.__class__.__name__!r}>{str(error)}</error>'


async def test_run_stream_text_and_thinking():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Half of ')}
        yield {0: DeltaThinkingPart(content='a thought')}
        yield {1: DeltaThinkingPart(content='Another thought')}
        yield {2: DeltaThinkingPart(content='And one more')}
        yield 'Half of '
        yield 'some text'
        yield {5: DeltaThinkingPart(content='More thinking')}

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<thinking follows_thinking=False>Half of ',
            'a thought',
            '</thinking followed_by_thinking=True>',
            '<thinking follows_thinking=True>Another thought',
            '</thinking followed_by_thinking=True>',
            '<thinking follows_thinking=True>And one more',
            '</thinking followed_by_thinking=False>',
            '<text follows_text=False>Half of ',
            '<final-result tool_name=None />',
            'some text',
            '</text followed_by_text=False>',
            '<thinking follows_thinking=False>More thinking',
            '</thinking followed_by_thinking=False>',
            '</response>',
            '<run-result>Half of some text</run-result>',
            '</stream>',
        ]
    )


async def test_event_stream_back_to_back_text():
    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'), next_part_kind='text')
        yield PartStartEvent(index=1, part=TextPart(content='Goodbye'), previous_part_kind='text')
        yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=1, part=TextPart(content='Goodbye world'))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])
    event_stream = DummyUIEventStream(run_input=request)
    events = [event async for event in event_stream.transform_stream(event_generator())]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<text follows_text=False>Hello',
            ' world',
            '</text followed_by_text=True>',
            '<text follows_text=True>Goodbye',
            ' world',
            '</text followed_by_text=False>',
            '</response>',
            '</stream>',
        ]
    )


async def test_run_stream_builtin_tool_call():
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

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<builtin-tool-call name=\'web_search\'>{"query":',
            '"Hello world"}',
            "</builtin-tool-call name='web_search'>",
            "<builtin-tool-return name='web_search'>{'results': [{'title': '\"Hello, World!\" program', 'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program'}]}</builtin-tool-return>",
            '<text follows_text=False>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
            '<final-result tool_name=None />',
            '</text followed_by_text=False>',
            '</response>',
            '<run-result>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". </run-result>',
            '</stream>',
        ]
    )


async def test_run_stream_tool_call():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='web_search',
                    json_args='{"query":',
                    tool_call_id='search_1',
                )
            }
            yield {
                0: DeltaToolCall(
                    json_args='"Hello world"}',
                    tool_call_id='search_1',
                )
            }
        else:
            yield 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". '

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    async def web_search(query: str) -> dict[str, list[dict[str, str]]]:
        return {
            'results': [
                {
                    'title': '"Hello, World!" program',
                    'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                }
            ]
        }

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<tool-call name=\'web_search\'>{"query":',
            '"Hello world"}',
            "</tool-call name='web_search'>",
            '</response>',
            '<request>',
            '<function-tool-call name=\'web_search\'>{"query":"Hello world"}</function-tool-call>',
            "<function-tool-result name='web_search'>{'results': [{'title': '\"Hello, World!\" program', 'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program'}]}</function-tool-result>",
            '</request>',
            '<response>',
            '<text follows_text=False>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
            '<final-result tool_name=None />',
            '</text followed_by_text=False>',
            '</response>',
            '<run-result>A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". </run-result>',
            '</stream>',
        ]
    )


async def test_event_stream_file():
    async def event_generator():
        yield PartStartEvent(index=0, part=FilePart(content=BinaryImage(data=b'fake', media_type='image/png')))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])
    event_stream = DummyUIEventStream(run_input=request)
    events = [event async for event in event_stream.transform_stream(event_generator())]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            "<file media_type='image/png' />",
            '</response>',
            '</stream>',
        ]
    )


async def test_run_stream_external_tools():
    agent = Agent(model=TestModel())

    request = DummyUIRunInput(
        messages=[ModelRequest.user_text_prompt('Call a tool')],
        tool_defs=[ToolDefinition(name='external_tool')],
    )
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            "<tool-call name='external_tool'>{}",
            '<final-result tool_name=None />',
            "</tool-call name='external_tool'>",
            '</response>',
            '<request>',
            "<function-tool-call name='external_tool'>{}</function-tool-call>",
            '</request>',
            "<run-result>DeferredToolRequests(calls=[ToolCallPart(tool_name='external_tool', args={}, tool_call_id='pyd_ai_tool_call_id__external_tool')], approvals=[], metadata={})</run-result>",
            '</stream>',
        ]
    )


async def test_run_stream_output_tool():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {
            0: DeltaToolCall(
                name='final_result',
                json_args='{"query":',
                tool_call_id='search_1',
            )
        }
        yield {
            0: DeltaToolCall(
                json_args='"Hello world"}',
                tool_call_id='search_1',
            )
        }

    def web_search(query: str) -> dict[str, list[dict[str, str]]]:
        return {
            'results': [
                {
                    'title': '"Hello, World!" program',
                    'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                }
            ]
        }

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=web_search)

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<tool-call name=\'final_result\'>{"query":',
            "<final-result tool_name='final_result' />",
            '"Hello world"}',
            "</tool-call name='final_result'>",
            '</response>',
            '<request>',
            "<function-tool-result name='final_result'>Final result processed.</function-tool-result>",
            '</request>',
            "<run-result>{'results': [{'title': '\"Hello, World!\" program', 'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program'}]}</run-result>",
            '</stream>',
        ]
    )


async def test_run_stream_response_error():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {
            0: DeltaToolCall(
                name='unknown_tool',
            )
        }

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Tell me about Hello World')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            "<tool-call name='unknown_tool'>None",
            "</tool-call name='unknown_tool'>",
            '</response>',
            '<request>',
            "<function-tool-call name='unknown_tool'>None</function-tool-call>",
            "<function-tool-result name='unknown_tool'>Unknown tool name: 'unknown_tool'. No tools available.</function-tool-result>",
            '</request>',
            '<response>',
            "<tool-call name='unknown_tool'>None",
            "</tool-call name='unknown_tool'>",
            "<error type='UnexpectedModelBehavior'>Exceeded maximum retries (1) for output validation</error>",
            '</response>',
            '</stream>',
        ]
    )


async def test_run_stream_request_error():
    agent = Agent(model=TestModel())

    @agent.tool_plain
    async def tool(query: str) -> str:
        raise ValueError('Unknown tool')

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            "<tool-call name='tool'>{'query': 'a'}",
            "</tool-call name='tool'>",
            '</response>',
            '<request>',
            "<function-tool-call name='tool'>{'query': 'a'}</function-tool-call>",
            "<error type='ValueError'>Unknown tool</error>",
            '</request>',
            '</stream>',
        ]
    )


async def test_run_stream_on_complete_error():
    agent = Agent(model=TestModel())

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])

    def raise_error(run_result: AgentRunResult[Any]) -> None:
        raise ValueError('Faulty on_complete')

    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream(on_complete=raise_error)]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<text follows_text=False>',
            '<final-result tool_name=None />',
            'success ',
            '(no ',
            'tool ',
            'calls)',
            '</text followed_by_text=False>',
            '</response>',
            "<error type='ValueError'>Faulty on_complete</error>",
            '</stream>',
        ]
    )


async def test_run_stream_on_complete():
    agent = Agent(model=TestModel())

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])

    async def on_complete(run_result: AgentRunResult[Any]) -> AsyncIterator[str]:
        yield '<custom>'

    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream(on_complete=on_complete)]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<text follows_text=False>',
            '<final-result tool_name=None />',
            'success ',
            '(no ',
            'tool ',
            'calls)',
            '</text followed_by_text=False>',
            '</response>',
            '<custom>',
            '<run-result>success (no tool calls)</run-result>',
            '</stream>',
        ]
    )


@pytest.mark.skipif(not starlette_import_successful, reason='Starlette is not installed')
async def test_adapter_dispatch_request():
    agent = Agent(model=TestModel())
    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': request.model_dump_json().encode('utf-8')}

    starlette_request = Request(
        scope={
            'type': 'http',
            'method': 'POST',
            'headers': [
                (b'content-type', b'application/json'),
            ],
        },
        receive=receive,
    )

    response = await DummyUIAdapter.dispatch_request(starlette_request, agent=agent)

    assert isinstance(response, StreamingResponse)

    chunks: list[MutableMapping[str, Any]] = []

    async def send(data: MutableMapping[str, Any]) -> None:
        chunks.append(data)

    await response.stream_response(send)

    assert chunks == snapshot(
        [
            {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'x-test', b'test'), (b'content-type', b'text/event-stream; charset=utf-8')],
            },
            {'type': 'http.response.body', 'body': b'<stream>', 'more_body': True},
            {'type': 'http.response.body', 'body': b'<response>', 'more_body': True},
            {'type': 'http.response.body', 'body': b'<text follows_text=False>', 'more_body': True},
            {'type': 'http.response.body', 'body': b'<final-result tool_name=None />', 'more_body': True},
            {'type': 'http.response.body', 'body': b'success ', 'more_body': True},
            {'type': 'http.response.body', 'body': b'(no ', 'more_body': True},
            {'type': 'http.response.body', 'body': b'tool ', 'more_body': True},
            {'type': 'http.response.body', 'body': b'calls)', 'more_body': True},
            {'type': 'http.response.body', 'body': b'</text followed_by_text=False>', 'more_body': True},
            {'type': 'http.response.body', 'body': b'</response>', 'more_body': True},
            {
                'type': 'http.response.body',
                'body': b'<run-result>success (no tool calls)</run-result>',
                'more_body': True,
            },
            {'type': 'http.response.body', 'body': b'</stream>', 'more_body': True},
            {'type': 'http.response.body', 'body': b'', 'more_body': False},
        ]
    )
