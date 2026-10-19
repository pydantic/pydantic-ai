"""`SystemPromptPart.metadata`: application data on a system prompt that stays with the part and never reaches the model.

The run tests use `FunctionModel`, as what they check is what the history keeps across runs and merges, not a
provider's answer. The wire test injects a transport instead of using cassettes: the cassette matchers ignore
the body, so a recording can't show that the metadata stays off the wire.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import AbstractCapability, ReinjectSystemPrompt
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

ORIGIN = {'origin': 'my_app.note'}


def marked_parts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.metadata == ORIGIN
    ]


@dataclass
class RecordMessages(AbstractCapability[Any]):
    """Records the merged messages a capability sees in `request_context.messages`."""

    seen: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.seen.append(list(request_context.messages))
        return request_context


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ok')])


async def test_metadata_survives_continuation_and_merge():
    """A system prompt enqueued as a request of its own keeps its metadata once it's merged with its neighbor.

    The enqueued request follows the tool results in the history, so the history the next run starts with merges
    the two. `ModelRequest.metadata` is lost there: the merged request keeps only `__pydantic_ai__`.
    """
    calls = 0

    def call_tool_then_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('note')])
        return ModelResponse(parts=[TextPart('ok')])

    first_agent = Agent(FunctionModel(call_tool_then_reply))

    @first_agent.tool
    def note(ctx: RunContext[Any]) -> str:
        ctx.enqueue(ModelRequest(parts=[SystemPromptPart('Noted.', metadata=ORIGIN)], metadata=ORIGIN))
        return 'done'

    first = await first_agent.run('hi')
    assert [part.content for part in marked_parts(first.all_messages())] == ['Noted.']

    second_recorder = RecordMessages()
    second = await Agent(FunctionModel(reply), capabilities=[second_recorder]).run(
        'more', message_history=first.all_messages()
    )

    merged_on_continuation = second.all_messages()[2]
    assert isinstance(merged_on_continuation, ModelRequest)
    assert [type(part).__name__ for part in merged_on_continuation.parts] == ['ToolReturnPart', 'SystemPromptPart']
    assert merged_on_continuation.metadata is None
    assert [part.content for part in marked_parts(second.all_messages())] == ['Noted.']
    assert [part.content for part in marked_parts(second_recorder.seen[0])] == ['Noted.']


async def test_dynamic_system_prompt_keeps_metadata_when_reevaluated():
    agent = Agent(FunctionModel(reply))

    @agent.system_prompt(dynamic=True)
    def today() -> str:
        return 'Today is Monday.'

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart('Today is Sunday.', dynamic_ref=today.__qualname__, metadata=ORIGIN),
                UserPromptPart('hi'),
            ]
        ),
        ModelResponse(parts=[TextPart('hello')]),
    ]
    result = await agent.run('again', message_history=history)

    assert marked_parts(result.all_messages()) == [
        SystemPromptPart('Today is Monday.', timestamp=IsDatetime(), dynamic_ref=today.__qualname__, metadata=ORIGIN)
    ]


async def test_reinjector_keeps_metadata_on_parts_added_during_the_run():
    """`replace_existing=True` replaces the history's system prompts, but keeps what a capability adds during the run."""

    @dataclass
    class AddNote(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            ctx.enqueue(SystemPromptPart('Noted.', metadata=ORIGIN))
            return request_context

    agent = Agent(
        FunctionModel(reply),
        system_prompt='Be brief.',
        capabilities=[ReinjectSystemPrompt(replace_existing=True), AddNote()],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('Stale.'), UserPromptPart('hi')]),
        ModelResponse(parts=[TextPart('hello')]),
    ]
    result = await agent.run('again', message_history=history)

    system_prompts = [
        (part.content, part.metadata)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    assert system_prompts == [('Be brief.', None), ('Noted.', ORIGIN)]


def test_serialization_round_trip():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('Be brief.', metadata={'origin': 'my_app', 'version': 2})]),
    ]

    assert ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages)) == messages
    assert ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages)) == messages
    assert ModelMessagesTypeAdapter.dump_python(messages, mode='json')[0]['parts'] == snapshot(
        [
            {
                'content': 'Be brief.',
                'timestamp': IsDatetime(iso_string=True),
                'dynamic_ref': None,
                'scope': 'conversation',
                'metadata': {'origin': 'my_app', 'version': 2},
                'part_kind': 'system-prompt',
            }
        ]
    )


def test_payload_without_metadata_loads():
    """A history dumped before the field existed loads with no metadata."""
    [message] = ModelMessagesTypeAdapter.validate_python(
        [
            {
                'kind': 'request',
                'parts': [
                    {
                        'content': 'Be brief.',
                        'timestamp': '2026-10-19T00:00:00Z',
                        'dynamic_ref': None,
                        'part_kind': 'system-prompt',
                    }
                ],
            }
        ]
    )
    assert isinstance(message, ModelRequest)
    [part] = message.parts
    assert isinstance(part, SystemPromptPart)
    assert part.metadata is None


def history(metadata: dict[str, Any] | None) -> list[ModelMessage]:
    """A standing prompt, a mid-conversation prompt and a request-scoped one, all carrying `metadata`."""
    return [
        ModelRequest(parts=[SystemPromptPart('Be brief.', metadata=metadata), UserPromptPart('hi')]),
        ModelResponse(parts=[TextPart('hello')]),
        ModelRequest(
            parts=[
                SystemPromptPart('The user is on mobile now.', metadata=metadata),
                SystemPromptPart('Reminder: one line.', scope='request', metadata=metadata),
                UserPromptPart('again'),
            ]
        ),
    ]


ModelFactory = Callable[[httpx2.AsyncClient], Model]


def openai_chat(client: httpx2.AsyncClient) -> Model:
    return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))


def openai_responses(client: httpx2.AsyncClient) -> Model:
    return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))


def anthropic(client: httpx2.AsyncClient) -> Model:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test', http_client=client))


def google(client: httpx2.AsyncClient) -> Model:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test', http_client=client))


def mistral(client: httpx2.AsyncClient) -> Model:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test', http_client=client))


@pytest.mark.parametrize(
    'make_model',
    [
        pytest.param(openai_chat, id='openai-chat', marks=pytest.mark.skipif(not openai_available(), reason='openai')),
        pytest.param(
            openai_responses,
            id='openai-responses',
            marks=pytest.mark.skipif(not openai_available(), reason='openai'),
        ),
        pytest.param(
            anthropic, id='anthropic', marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic')
        ),
        pytest.param(google, id='google', marks=pytest.mark.skipif(not google_available(), reason='google')),
        pytest.param(mistral, id='mistral', marks=pytest.mark.skipif(not mistral_available(), reason='mistral')),
    ],
)
async def test_metadata_is_not_sent(allow_model_requests: None, make_model: ModelFactory):
    """The request body is the same byte for byte with or without metadata on the system prompts."""
    bodies: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(request.read())
        # A 400 isn't retried, so each run sends exactly one request.
        return httpx2.Response(400, json={'error': {'message': 'captured'}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(make_model(client))
        for metadata in (None, {'origin': 'my_app.note', 'secret': 'not for the model'}):
            with pytest.raises(ModelHTTPError):
                await agent.run(message_history=history(metadata))

    without_metadata, with_metadata = bodies
    assert b'not for the model' not in with_metadata
    assert with_metadata == without_metadata
    assert b'The user is on mobile now.' in without_metadata
