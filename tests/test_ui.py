from __future__ import annotations

import inspect
import warnings
from collections.abc import AsyncIterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.messages import (
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UserPromptPart,
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
from pydantic_ai.tools import DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset

from ._inline_snapshot import snapshot
from .conftest import IsDatetime

pytest.importorskip('starlette')

from starlette.requests import Request
from starlette.responses import StreamingResponse

from pydantic_ai.ui import NativeEvent, UIAdapter, UIEventStream

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
    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    tool_defs: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    state: dict[str, Any] = field(default_factory=dict[str, Any])


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
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        return list(messages)

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
            '</response>',
            '<request>',
            "<function-tool-call name='unknown_tool'>None</function-tool-call>",
            "<function-tool-result name='unknown_tool'>Tool execution was interrupted by an error.</function-tool-result>",
            "<error type='UnexpectedModelBehavior'>Tool 'unknown_tool' exceeded max retries count of 1</error>",
            '</request>',
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
            "<function-tool-result name='tool'>Tool execution was interrupted by an error.</function-tool-result>",
            "<error type='ValueError'>Unknown tool</error>",
            '</request>',
            '</stream>',
        ]
    )


async def test_run_stream_output_tool_error():
    """Output tool errors should close the pending tool call via _final_result_event drain."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {
            0: DeltaToolCall(
                name='final_result',
                json_args='{"value": "bad"}',
                tool_call_id='out_1',
            )
        }

    def bad_output(value: str) -> str:
        raise ValueError('Output validation failed')

    agent = Agent(
        model=FunctionModel(stream_function=stream_function), output_type=bad_output, tool_retries=0, output_retries=0
    )

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])
    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream()]

    assert events == snapshot(
        [
            '<stream>',
            '<response>',
            '<tool-call name=\'final_result\'>{"value": "bad"}',
            "<final-result tool_name='final_result' />",
            "</tool-call name='final_result'>",
            '</response>',
            '<request>',
            "<function-tool-result name='final_result'>Tool execution was interrupted by an error.</function-tool-result>",
            "<error type='ValueError'>Output validation failed</error>",
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


async def test_run_stream_metadata_forwarded():
    agent = Agent(model=TestModel(custom_output_text='meta'))

    request = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])
    captured_metadata: list[dict[str, Any] | None] = []

    def on_complete(run_result: AgentRunResult[Any]) -> None:
        captured_metadata.append(run_result.metadata)

    adapter = DummyUIAdapter(agent, request)
    events = [event async for event in adapter.run_stream(metadata={'ui': 'adapter'}, on_complete=on_complete)]

    assert captured_metadata == [{'ui': 'adapter'}]
    assert events[-2:] == ['<run-result>meta</run-result>', '</stream>']


async def test_run_stream_native_metadata_forwarded():
    agent = Agent(model=TestModel(custom_output_text='native meta'))
    adapter = DummyUIAdapter(agent, DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')]))

    events = [event async for event in adapter.run_stream_native(metadata={'ui': 'native'})]
    run_result_event = next(event for event in events if isinstance(event, AgentRunResultEvent))

    assert run_result_event.result.metadata == {'ui': 'native'}


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

    captured_metadata: list[dict[str, Any] | None] = []

    def on_complete(run_result: AgentRunResult[Any]) -> None:
        captured_metadata.append(run_result.metadata)

    response = await DummyUIAdapter.dispatch_request(
        starlette_request, agent=agent, metadata={'ui': 'dispatch'}, on_complete=on_complete
    )

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
    assert captured_metadata == [{'ui': 'dispatch'}]


def test_manage_system_prompt_visible_in_base_adapter_signatures():
    from_request_parameters = inspect.signature(DummyUIAdapter.from_request).parameters
    dispatch_request_parameters = inspect.signature(DummyUIAdapter.dispatch_request).parameters

    assert 'manage_system_prompt' in from_request_parameters
    assert from_request_parameters['manage_system_prompt'].default == 'server'
    assert 'manage_system_prompt' in dispatch_request_parameters
    assert dispatch_request_parameters['manage_system_prompt'].default == 'server'


def test_dummy_adapter_dump_messages():
    """Test that DummyUIAdapter.dump_messages returns messages as-is."""
    messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]
    result = DummyUIAdapter.dump_messages(messages)
    assert result == messages


async def test_reinject_system_prompt_capability_injects_when_history_missing():
    """The `ReinjectSystemPrompt` capability prepends the agent's configured system prompt
    to the first `ModelRequest` when no `SystemPromptPart` is present in the history.
    """
    agent = Agent(model=TestModel(), system_prompt='You are a helpful assistant')

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First message')]),
        ModelResponse(parts=[TextPart(content='First response')]),
    ]

    result = await agent.run(
        'Second message',
        message_history=history,
        capabilities=[ReinjectSystemPrompt()],
    )

    first_request = result.all_messages()[0]
    assert isinstance(first_request, ModelRequest)
    assert first_request.parts == snapshot(
        [
            SystemPromptPart(content='You are a helpful assistant', timestamp=IsDatetime()),
            UserPromptPart(content='First message', timestamp=IsDatetime()),
        ]
    )


async def test_reinject_system_prompt_capability_reaches_model_and_all_messages():
    """Regression guard: the injected `SystemPromptPart` must appear in *both* the messages
    actually sent to the model AND the stored `result.all_messages()` — they're the same
    list after `_agent_graph.py:835` syncs the hook's mutations back to canonical state.
    """
    captured: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured.append([m for m in messages])
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(respond), system_prompt='Server prompt')

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Earlier turn')]),
        ModelResponse(parts=[TextPart(content='Earlier reply')]),
    ]

    result = await agent.run(
        'Follow up',
        message_history=history,
        capabilities=[ReinjectSystemPrompt()],
    )

    # What the model received on its one call
    assert len(captured) == 1
    model_first_request = captured[0][0]
    assert isinstance(model_first_request, ModelRequest)
    assert [type(p).__name__ for p in model_first_request.parts] == ['SystemPromptPart', 'UserPromptPart']
    assert isinstance(model_first_request.parts[0], SystemPromptPart)
    assert model_first_request.parts[0].content == 'Server prompt'

    # What the caller sees via result.all_messages()
    stored_first_request = result.all_messages()[0]
    assert isinstance(stored_first_request, ModelRequest)
    assert [type(p).__name__ for p in stored_first_request.parts] == ['SystemPromptPart', 'UserPromptPart']
    assert isinstance(stored_first_request.parts[0], SystemPromptPart)
    assert stored_first_request.parts[0].content == 'Server prompt'


async def test_reinject_system_prompt_capability_does_not_mutate_input_history():
    """Regression guard: the capability must not mutate `ModelRequest` objects from the caller's
    `message_history` list. `request_context.messages` is a shallow copy, so mutating `.parts`
    on shared `ModelRequest` instances would leak back into the user's input.
    """
    agent = Agent(model=TestModel(), system_prompt='Server prompt')

    history_request = ModelRequest(parts=[SystemPromptPart(content='Client prompt'), UserPromptPart(content='Hi')])
    history_response = ModelResponse(parts=[TextPart(content='Hello')])
    original_parts = list(history_request.parts)
    history: list[ModelMessage] = [history_request, history_response]

    await agent.run(
        'Follow up',
        message_history=history,
        capabilities=[ReinjectSystemPrompt(replace_existing=True)],
    )

    assert history_request.parts == original_parts, 'capability mutated caller-owned ModelRequest'


async def test_reinject_system_prompt_capability_replace_drops_system_only_requests():
    """When `replace_existing=True` strips the only parts in a `ModelRequest`, the request is
    dropped from the history rather than left as an empty placeholder.
    """
    captured: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(respond), system_prompt='Server prompt')

    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='Caller-supplied client prompt')]),
        ModelResponse(parts=[TextPart(content='Earlier reply')]),
    ]

    await agent.run(
        'Follow up',
        message_history=history,
        capabilities=[ReinjectSystemPrompt(replace_existing=True)],
    )

    assert len(captured) == 1
    seen = captured[0]
    assert all(not (isinstance(m, ModelRequest) and not m.parts) for m in seen), (
        'capability left an empty ModelRequest in history'
    )
    assert any(
        isinstance(m, ModelRequest)
        and any(isinstance(p, SystemPromptPart) and p.content == 'Server prompt' for p in m.parts)
        for m in seen
    ), "agent's configured system prompt should be reinjected at the head"


async def test_reinject_system_prompt_capability_agent_without_model():
    """Regression guard: agent constructed without a model gets its model passed via `run(model=...)`.

    `ReinjectSystemPrompt.before_model_request` must resolve the system prompt using the
    run-time model from `RunContext`, not re-fetch from the agent (which would raise `UserError`).
    This is the default UIAdapter path for agents that delegate model choice to the server.
    """
    agent = Agent(system_prompt='You are a helpful assistant')

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First message')]),
        ModelResponse(parts=[TextPart(content='First response')]),
    ]

    result = await agent.run(
        'Second message',
        model=TestModel(),
        message_history=history,
        capabilities=[ReinjectSystemPrompt()],
    )

    first_request = result.all_messages()[0]
    assert isinstance(first_request, ModelRequest)
    assert first_request.parts == snapshot(
        [
            SystemPromptPart(content='You are a helpful assistant', timestamp=IsDatetime()),
            UserPromptPart(content='First message', timestamp=IsDatetime()),
        ]
    )


async def test_reinject_system_prompt_capability_preserves_existing():
    """The `ReinjectSystemPrompt` capability is a no-op if any `SystemPromptPart` is already
    in the history (e.g. from a prior agent). Multi-agent handoff keeps the original system
    prompt authoritative.
    """
    agent = Agent(model=TestModel(), system_prompt='Second agent')

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='First agent'),
                UserPromptPart(content='Hi'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='Hello')]),
    ]

    result = await agent.run(
        'Follow up',
        message_history=history,
        capabilities=[ReinjectSystemPrompt()],
    )

    first_request = result.all_messages()[0]
    assert isinstance(first_request, ModelRequest)
    sys_parts = [p for p in first_request.parts if isinstance(p, SystemPromptPart)]
    assert [p.content for p in sys_parts] == ['First agent']


def test_allowed_file_url_schemes_visible_in_base_adapter_signatures():
    from_request_parameters = inspect.signature(DummyUIAdapter.from_request).parameters
    dispatch_request_parameters = inspect.signature(DummyUIAdapter.dispatch_request).parameters

    assert 'allowed_file_url_schemes' in from_request_parameters
    assert from_request_parameters['allowed_file_url_schemes'].default == frozenset({'http', 'https'})
    assert 'allowed_file_url_schemes' in dispatch_request_parameters
    assert dispatch_request_parameters['allowed_file_url_schemes'].default == frozenset({'http', 'https'})


def _make_dummy_adapter(
    messages: list[ModelMessage],
    *,
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
) -> DummyUIAdapter[None, str]:
    agent: Agent[None, str] = Agent(model=TestModel())
    return DummyUIAdapter(
        agent=agent,
        run_input=DummyUIRunInput(messages=messages),
        allowed_file_url_schemes=allowed_file_url_schemes,
    )


def test_sanitize_messages_strips_file_urls_with_disallowed_schemes():
    """File URLs with schemes outside `allowed_file_url_schemes` are dropped with a warning."""
    adapter = _make_dummy_adapter(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Look at this:',
                            ImageUrl(url='s3://my-bucket/key.png'),
                            ImageUrl(url='https://example.com/ok.png'),
                            DocumentUrl(url='gs://my-bucket/doc.pdf'),
                        ]
                    )
                ]
            )
        ]
    )

    with pytest.warns(UserWarning, match=r"scheme\(s\).*'gs'.*'s3'"):
        sanitized = adapter.sanitize_messages(adapter.messages)

    assert len(sanitized) == 1
    request = sanitized[0]
    assert isinstance(request, ModelRequest)
    user_part = request.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert user_part.content == snapshot(['Look at this:', ImageUrl(url='https://example.com/ok.png')])


def test_sanitize_messages_leaves_string_user_content_alone():
    """Sanitization never modifies string-only user prompts."""
    adapter = _make_dummy_adapter([ModelRequest(parts=[UserPromptPart(content='Plain text')])])
    sanitized = adapter.sanitize_messages(adapter.messages)
    assert sanitized == snapshot([ModelRequest(parts=[UserPromptPart(content='Plain text', timestamp=IsDatetime())])])


def test_sanitize_messages_respects_custom_allowed_schemes():
    """Schemes explicitly added to `allowed_file_url_schemes` flow through unchanged."""
    adapter = _make_dummy_adapter(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            ImageUrl(url='s3://bucket/ok.png'),
                            ImageUrl(url='gs://bucket/blocked.png'),
                        ]
                    )
                ]
            )
        ],
        allowed_file_url_schemes=frozenset({'http', 'https', 's3'}),
    )

    with pytest.warns(UserWarning, match=r"scheme\(s\).*'gs'"):
        sanitized = adapter.sanitize_messages(adapter.messages)

    assert isinstance(sanitized[0], ModelRequest)
    user_part = sanitized[0].parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert user_part.content == snapshot([ImageUrl(url='s3://bucket/ok.png')])


def test_sanitize_messages_strips_dangling_tool_calls():
    """A trailing ModelResponse with unresolved ToolCallParts has them dropped with a warning."""
    adapter = _make_dummy_adapter(
        [
            ModelRequest(parts=[UserPromptPart(content='Run it')]),
            ModelResponse(
                parts=[
                    TextPart(content='Working on it'),
                    ToolCallPart(tool_name='refresh_cache', args={'key': 1}, tool_call_id='call-1'),
                ]
            ),
        ]
    )

    with pytest.warns(UserWarning, match=r'unresolved tool call.*refresh_cache'):
        sanitized = adapter.sanitize_messages(adapter.messages)

    assert len(sanitized) == 2
    response = sanitized[1]
    assert isinstance(response, ModelResponse)
    assert [type(p).__name__ for p in response.parts] == ['TextPart']


def test_sanitize_messages_keeps_tool_calls_resolved_by_deferred_results():
    """Tool calls matched by `deferred_tool_results` survive sanitization (HITL resumption)."""
    adapter = _make_dummy_adapter(
        [
            ModelRequest(parts=[UserPromptPart(content='Run it')]),
            ModelResponse(parts=[ToolCallPart(tool_name='refresh_cache', args={'key': 1}, tool_call_id='call-1')]),
        ]
    )

    deferred_tool_results = DeferredToolResults(approvals={'call-1': True})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sanitized = adapter.sanitize_messages(adapter.messages, deferred_tool_results=deferred_tool_results)

    response = sanitized[1]
    assert isinstance(response, ModelResponse)
    assert [type(p).__name__ for p in response.parts] == ['ToolCallPart']


def test_sanitize_messages_drops_response_left_empty_after_stripping():
    """If the last `ModelResponse` consists entirely of dangling tool calls, the whole
    response is dropped from history rather than being left as an empty placeholder.
    """
    adapter = _make_dummy_adapter(
        [
            ModelRequest(parts=[UserPromptPart(content='Run it')]),
            ModelResponse(parts=[ToolCallPart(tool_name='refresh_cache', args={'key': 'prod'}, tool_call_id='call-1')]),
        ]
    )

    with pytest.warns(UserWarning, match=r'unresolved tool call.*refresh_cache'):
        sanitized = adapter.sanitize_messages(adapter.messages)

    assert len(sanitized) == 1
    assert isinstance(sanitized[0], ModelRequest)


def test_sanitize_messages_strips_dangling_builtin_tool_calls():
    """Builtin tool calls are also model-emitted, so a dangling `BuiltinToolCallPart` at
    the end of client-supplied history is treated the same as a `ToolCallPart`.
    """
    adapter = _make_dummy_adapter(
        [
            ModelRequest(parts=[UserPromptPart(content='Run it')]),
            ModelResponse(
                parts=[
                    TextPart(content='Looking it up'),
                    BuiltinToolCallPart(
                        tool_name='code_execution', args={'code': 'print(1)'}, tool_call_id='builtin-1'
                    ),
                ]
            ),
        ]
    )

    with pytest.warns(UserWarning, match=r'unresolved tool call.*code_execution'):
        sanitized = adapter.sanitize_messages(adapter.messages)

    response = sanitized[1]
    assert isinstance(response, ModelResponse)
    assert [type(p).__name__ for p in response.parts] == ['TextPart']


def test_sanitize_messages_keeps_tool_calls_in_middle_of_history():
    """Only the *last* message is checked for dangling tool calls; completed tool exchanges earlier
    in the history are legitimate context and must be preserved verbatim.
    """
    adapter = _make_dummy_adapter(
        [
            ModelRequest(parts=[UserPromptPart(content='Run it')]),
            ModelResponse(parts=[ToolCallPart(tool_name='do_thing', args={'x': 1}, tool_call_id='earlier-1')]),
            ModelRequest(parts=[UserPromptPart(content='Follow up')]),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sanitized = adapter.sanitize_messages(adapter.messages)

    mid_response = sanitized[1]
    assert isinstance(mid_response, ModelResponse)
    assert [type(p).__name__ for p in mid_response.parts] == ['ToolCallPart']


async def test_run_stream_strips_dangling_tool_calls_from_client_history():
    """End-to-end: a client-submitted history ending in an unresolved tool call has
    that tool call stripped before the agent sees the history, so the agent never
    has the chance to execute it.
    """
    captured: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        captured.append(list(messages))
        yield 'done'

    agent: Agent[None, str] = Agent(model=FunctionModel(stream_function=stream_function))

    request = DummyUIRunInput(
        messages=[
            ModelRequest(parts=[UserPromptPart(content='Hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='refresh_cache', args={'key': 42}, tool_call_id='call-1')]),
        ]
    )
    adapter = DummyUIAdapter(agent=agent, run_input=request)

    with pytest.warns(UserWarning, match=r'unresolved tool call.*refresh_cache'):
        async for _ in adapter.run_stream():
            pass

    assert len(captured) == 1
    history_seen_by_model = captured[0]
    assert not any(
        isinstance(message, ModelResponse) and any(isinstance(part, ToolCallPart) for part in message.parts)
        for message in history_seen_by_model
    ), 'dangling client-submitted tool call leaked into the agent run'


async def test_run_stream_strips_file_urls_with_disallowed_schemes():
    """End-to-end: an s3:// URL in a client-submitted user prompt is dropped before the agent runs."""
    captured: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        captured.append(list(messages))
        yield 'ok'

    agent: Agent[None, str] = Agent(model=FunctionModel(stream_function=stream_function))

    request = DummyUIRunInput(
        messages=[
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'See attached',
                            ImageUrl(url='s3://some-bucket/internal.png'),
                            ImageUrl(url='https://example.com/public.png'),
                        ]
                    )
                ]
            )
        ]
    )
    adapter = DummyUIAdapter(agent=agent, run_input=request)

    with pytest.warns(UserWarning, match=r"scheme\(s\).*'s3'"):
        async for _ in adapter.run_stream():
            pass

    assert len(captured) == 1
    first_request = captured[0][0]
    assert isinstance(first_request, ModelRequest)
    user_part = first_request.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert user_part.content == ['See attached', ImageUrl(url='https://example.com/public.png')]


async def test_reinject_system_prompt_capability_with_pending_tool_calls():
    """History ending with pending tool calls early-returns in `UserPromptNode`, but the
    capability's `before_model_request` hook still runs on the subsequent model request (after
    tool results are collected), so the system prompt ends up in the first request.
    """

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), system_prompt='You are a helpful assistant')

    @agent.tool_plain
    def do_something(x: int) -> int:
        return x + 1

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Call the tool')]),
        ModelResponse(parts=[ToolCallPart(tool_name='do_something', args={'x': 1}, tool_call_id='call_1')]),
    ]

    result = await agent.run(message_history=history, capabilities=[ReinjectSystemPrompt()])

    first_request = result.all_messages()[0]
    assert isinstance(first_request, ModelRequest)
    assert first_request.parts == snapshot(
        [
            SystemPromptPart(content='You are a helpful assistant', timestamp=IsDatetime()),
            UserPromptPart(content='Call the tool', timestamp=IsDatetime()),
        ]
    )
