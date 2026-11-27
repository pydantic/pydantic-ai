"""Instructions and mid-conversation system prompts on every provider-side continuation route.

A continuation route sends only the messages after the latest response the API stores, and the API puts back what it
holds. Each route is checked in the same scenario: an agent with `instructions` continues a stored conversation whose
requests carry a mid-conversation `SystemPromptPart`, one from before the continuation point (the API already holds
it) and one added after it (new since the last response). On the continued request:

1. The instructions reach the model exactly once, in the API's system channel: not dropped where the API doesn't
   keep them, and not sent again where it does.
2. The stored system prompt the continuation covers isn't sent again, and the API still holds it once. The new one is
   sent once, in the voice the model's profile names: a native system message where `supports_inline_system_prompts`
   is set, otherwise a `<system>` block in a user turn (`<system verified>` with `needs_sealed_system_blocks`). It is
   never plain user text and never folded into the instructions.

The routes:

- OpenAI Responses with `openai_previous_response_id='auto'`: the request continues from the latest response in the
  history. `instructions` isn't stored with a response, so it's sent on every request, except on a profile with
  `openai_responses_stable_instructions_as_input` (GPT-6.2), where the chain keeps them as its leading input.
- OpenAI Responses with an explicit `openai_previous_response_id`, with the `message_history` and without it (the seed
  alone, as the OpenAI docs show it).
- OpenAI Responses with `openai_conversation_id`, explicit and `'auto'`: the conversation holds every item, but not
  the `instructions`.
- Anthropic message threads (`anthropic_previous_message_id='auto'`, where `anthropic_supports_message_threads` is
  set): the thread keeps the `system` it was created with, and a continue request omits it.
- xAI with `xai_previous_response_id` and `xai_store_messages`: the stored response holds the messages it was sent
  with, the system messages included.

Google's Interactions API isn't among them: `GoogleInteractionsModel` sends `store=False` and replays the history on
every request, so there is no continuation for it to get wrong. The Responses WebSocket transport resolves its
`'auto'` default through the same code as the HTTP route covered here.

The parametrized tests use a mock transport (a fake gRPC client for xAI) that keeps what each route stores the way
the API does, because their claim is the exact request body and what the API adds to it, which cassette matching
doesn't check. The `_live` tests record the real exchange on OpenAI and Anthropic and check the same claims on the
bodies the live code sent.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx2
import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ..conftest import RequestCapture, try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .conftest import AnthropicModelFactory

with try_import() as xai_available:
    import xai_sdk.chat as xai_chat
    from google.protobuf.json_format import MessageToDict
    from xai_sdk import AsyncClient as XaiAsyncClient
    from xai_sdk.proto import chat_pb2, sample_pb2

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.usefixtures('allow_model_requests')

INSTRUCTIONS = 'End every answer with the code word ZEBRA.'
EARLIER = 'From now on, answer in French.'
LATER = 'From now on, answer in Spanish.'

Channel = Literal['instructions', 'system', 'user', 'assistant']
"""Where a text reaches the model: the API's top-level system field, a native system message, or a turn."""

Seen = list[tuple[Channel, str]]
"""The texts a request puts in front of the model, in order, each with its channel."""


def voice(model: Model, text: str) -> tuple[Channel, str]:
    """How a mid-conversation system prompt reaches `model`, as its profile says."""
    if model.profile.get('supports_inline_system_prompts', False):
        return 'system', text
    if model.profile.get('needs_sealed_system_blocks', False):
        return 'user', f'<system verified>{text}</system>'
    return 'user', f'<system>{text}</system>'


def openai_wire(body: dict[str, Any]) -> Seen:
    """What a Responses request body sends, without what the API holds."""
    seen: Seen = [('instructions', body['instructions'])] if 'instructions' in body else []
    for item in body['input']:
        role, content = item['role'], item['content']
        texts: list[str] = [content] if isinstance(content, str) else [p['text'] for p in content if 'text' in p]
        channel: Channel = 'system' if role in ('system', 'developer') else cast(Channel, role)
        seen.extend((channel, text) for text in texts)
    return seen


@dataclass
class OpenAIResponsesServer:
    """Keeps what the Responses API stores: each response's input and output, and each conversation's items.

    Neither keeps the top-level `instructions`, which apply to their own request only.
    """

    bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    seen: list[Seen] = field(default_factory=list[Seen])
    responses: dict[str, Seen] = field(default_factory=dict[str, Seen])
    conversations: dict[str, Seen] = field(default_factory=dict[str, Seen])

    wire = staticmethod(openai_wire)

    def record(self, body: dict[str, Any], response_id: str, reply: str) -> None:
        self.bodies.append(body)
        conversation = body.get('conversation')
        if conversation is not None:
            held = self.conversations.get(conversation, [])
        elif 'previous_response_id' in body:
            held = self.responses[body['previous_response_id']]
        else:
            held = []
        wire = openai_wire(body)
        instructions = [entry for entry in wire if entry[0] == 'instructions']
        sent = [entry for entry in wire if entry[0] != 'instructions']
        self.seen.append([*instructions, *held, *sent])
        stored: Seen = [*held, *sent, ('assistant', reply)]
        self.responses[response_id] = stored
        if conversation is not None:
            self.conversations[conversation] = stored

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        response_id, reply = f'resp_{len(self.bodies) + 1}', f'Answer {len(self.bodies) + 1}.'
        self.record(body, response_id, reply)
        response: dict[str, Any] = {
            'id': response_id,
            'object': 'response',
            'created_at': 0,
            'model': body['model'],
            'status': 'completed',
            'output': [
                {
                    'type': 'message',
                    'id': f'msg_{response_id}',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': reply, 'annotations': []}],
                }
            ],
            'parallel_tool_calls': True,
            'tool_choice': 'auto',
            'tools': [],
        }
        if 'conversation' in body:
            response['conversation'] = {'id': body['conversation']}
        return httpx2.Response(200, json=response)


def anthropic_wire(system: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Seen:
    seen: Seen = [('instructions', block['text']) for block in system]
    for message in messages:
        content = message['content']
        blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        seen += [(message['role'], block['text']) for block in blocks if block['type'] == 'text']
    return seen


@dataclass
class AnthropicThreadsServer:
    """Keeps what a message thread stores: the `system` of its create request and every message after it."""

    bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    seen: list[Seen] = field(default_factory=list[Seen])
    threads: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = field(
        default_factory=dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]
    )

    @staticmethod
    def wire(body: dict[str, Any]) -> Seen:
        return anthropic_wire(body.get('system', []), body['messages'])

    def record(self, body: dict[str, Any], message_id: str, reply: str) -> None:
        self.bodies.append(body)
        thread = body['thread']
        if thread['type'] == 'continue':
            # The thread keeps the `system` it was created with, so a continue request resending it is a regression.
            assert 'system' not in body
            system, held = self.threads[thread['previous_message_id']]
        else:
            system, held = body['system'], []
        messages = [*held, *body['messages']]
        self.seen.append(anthropic_wire(system, messages))
        self.threads[message_id] = (system, [*messages, {'role': 'assistant', 'content': reply}])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        message_id, reply = f'msg_{len(self.bodies) + 1}', f'Answer {len(self.bodies) + 1}.'
        self.record(body, message_id, reply)
        return httpx2.Response(
            200,
            json={
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': body['model'],
                'content': [{'type': 'text', 'text': reply}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 5, 'output_tokens': 1},
            },
        )


def xai_wire(body: dict[str, Any]) -> Seen:
    roles: dict[str, Channel] = {'ROLE_SYSTEM': 'system', 'ROLE_USER': 'user', 'ROLE_ASSISTANT': 'assistant'}
    return [
        (roles[message['role']], part['text'])
        for message in body['messages']
        for part in message.get('content', [])
        if 'text' in part
    ]


@dataclass
class XaiServer:
    """A fake xAI SDK client that keeps what `store_messages` stores: each response's messages and its reply."""

    bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    seen: list[Seen] = field(default_factory=list[Seen])
    responses: dict[str, Seen] = field(default_factory=dict[str, Seen])

    wire = staticmethod(xai_wire)

    @property
    def chat(self) -> XaiServer:
        return self

    def create(self, **kwargs: Any) -> _XaiChat:
        body: dict[str, Any] = {
            **kwargs,
            'messages': [MessageToDict(m, preserving_proto_field_name=True) for m in kwargs['messages']],
        }
        self.bodies.append(body)
        response_id, reply = f'resp_{len(self.bodies)}', f'Answer {len(self.bodies)}.'
        previous = body.get('previous_response_id')
        held = self.responses[previous] if previous else []
        sent = xai_wire(body)
        self.seen.append([*held, *sent])
        assert body['store_messages']
        self.responses[response_id] = [*held, *sent, ('assistant', reply)]
        return _XaiChat(response_id, reply)


@dataclass
class _XaiChat:
    response_id: str
    reply: str

    async def sample(self) -> xai_chat.Response:
        output = chat_pb2.CompletionOutput(
            index=0,
            finish_reason=sample_pb2.FinishReason.REASON_STOP,
            message=chat_pb2.CompletionMessage(content=self.reply, role=chat_pb2.MessageRole.ROLE_ASSISTANT),
        )
        proto = chat_pb2.GetChatCompletionResponse(id=self.response_id, model='grok-4.7', outputs=[output])
        proto.created.GetCurrentTime()
        return xai_chat.Response(proto, index=None)


Server = OpenAIResponsesServer | AnthropicThreadsServer | XaiServer


@dataclass(frozen=True)
class Case:
    id: str
    server: Callable[[], Server]
    model: Callable[[Server, httpx2.AsyncClient], Model]
    settings: ModelSettings
    """The settings of every run."""
    continue_settings: Callable[[ModelResponse], ModelSettings] = lambda _: {}
    """Settings added to the runs that continue from `response`, the latest one."""
    with_history: bool = True
    """Whether a continuing run passes the stored conversation, or only its own new request and a seed."""
    instructions_channel: Channel = 'instructions'
    """Where the instructions are held: the top-level field, or a leading system message the API stores."""
    instructions_xfail: str | None = None
    held_xfail: str | None = None
    new_xfail: str | None = None


def openai(model_name: str, profile: OpenAIModelProfile | None = None) -> Callable[[Server, httpx2.AsyncClient], Model]:
    return lambda _, client: OpenAIResponsesModel(
        model_name, provider=OpenAIProvider(api_key='test', http_client=client), profile=profile
    )


def anthropic(model_name: str) -> Callable[[Server, httpx2.AsyncClient], Model]:
    return lambda _, client: AnthropicModel(
        model_name, provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='test', http_client=client))
    )


def xai(model_name: str) -> Callable[[Server, httpx2.AsyncClient], Model]:
    return lambda server, _: XaiModel(model_name, provider=XaiProvider(xai_client=cast(XaiAsyncClient, server)))


def seed_openai(response: ModelResponse) -> ModelSettings:
    return cast(ModelSettings, {'openai_previous_response_id': response.provider_response_id})


def seed_xai(response: ModelResponse) -> ModelSettings:
    return cast(ModelSettings, {'xai_previous_response_id': response.provider_response_id})


AUTO = cast(ModelSettings, {'openai_previous_response_id': 'auto'})
CONVERSATION = cast(ModelSettings, {'openai_conversation_id': 'conv_1'})
THREADS = cast(ModelSettings, {'max_tokens': 512, 'anthropic_previous_message_id': 'auto'})
XAI_STORE = cast(ModelSettings, {'xai_store_messages': True})
XAI_RESENDS = (
    'Defect: `XaiModel` passes `xai_previous_response_id` through without trimming the history the stored response '
    'holds, so the API sees it twice (https://github.com/pydantic/pydantic-ai/issues/14837)'
)
XAI_REPEATS_INSTRUCTIONS = (
    'Defect: `XaiModel` sends the instructions as a system message on a request with `xai_previous_response_id`, '
    'and the stored response already holds them (https://github.com/pydantic/pydantic-ai/issues/14837)'
)
GPT_6_2_SEED = (
    'Defect: a seed without `message_history` on GPT-6.2 sends the instructions in the top-level field, next to the '
    'leading input message the seeded response stores them in (documented in `docs/models/openai.md`, part of '
    'https://github.com/pydantic/pydantic-ai/issues/13923)'
)

openai_case = pytest.mark.skipif(not openai_available(), reason='openai not installed')
anthropic_case = pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
xai_case = pytest.mark.skipif(not xai_available(), reason='xai_sdk not installed')

CASES: list[tuple[Case, pytest.MarkDecorator]] = [
    (Case('responses-auto', OpenAIResponsesServer, openai('gpt-5.4-mini'), AUTO), openai_case),
    (
        Case(
            'responses-auto-stable-instructions',
            OpenAIResponsesServer,
            openai('gpt-6.2-sol'),
            AUTO,
            instructions_channel='system',
        ),
        openai_case,
    ),
    (
        Case(
            'responses-auto-tagged',
            OpenAIResponsesServer,
            openai('gpt-5.4-mini', OpenAIModelProfile(supports_inline_system_prompts=False)),
            AUTO,
        ),
        openai_case,
    ),
    (Case('responses-seed', OpenAIResponsesServer, openai('gpt-5.4-mini'), {}, seed_openai), openai_case),
    (
        Case('responses-seed-stable-instructions', OpenAIResponsesServer, openai('gpt-6.2-sol'), {}, seed_openai),
        openai_case,
    ),
    (
        Case(
            'responses-seed-without-history',
            OpenAIResponsesServer,
            openai('gpt-5.4-mini'),
            {},
            seed_openai,
            with_history=False,
        ),
        openai_case,
    ),
    (
        Case(
            'responses-seed-without-history-stable-instructions',
            OpenAIResponsesServer,
            openai('gpt-6.2-sol'),
            {},
            seed_openai,
            with_history=False,
            instructions_xfail=GPT_6_2_SEED,
        ),
        openai_case,
    ),
    (Case('responses-conversation', OpenAIResponsesServer, openai('gpt-5.4-mini'), CONVERSATION), openai_case),
    (
        Case(
            'responses-conversation-auto',
            OpenAIResponsesServer,
            openai('gpt-5.4-mini'),
            CONVERSATION,
            lambda _: cast(ModelSettings, {'openai_conversation_id': 'auto'}),
        ),
        openai_case,
    ),
    (
        Case('responses-conversation-stable-instructions', OpenAIResponsesServer, openai('gpt-6.2-sol'), CONVERSATION),
        openai_case,
    ),
    (Case('anthropic-threads-native', AnthropicThreadsServer, anthropic('claude-fable-5-2'), THREADS), anthropic_case),
    (Case('anthropic-threads-tagged', AnthropicThreadsServer, anthropic('claude-sonnet-5'), THREADS), anthropic_case),
    (Case('anthropic-threads-sealed', AnthropicThreadsServer, anthropic('claude-haiku-5-5'), THREADS), anthropic_case),
    (
        Case(
            'xai-previous-response-id',
            XaiServer,
            xai('grok-4.7'),
            XAI_STORE,
            seed_xai,
            instructions_channel='system',
            instructions_xfail=XAI_RESENDS,
            held_xfail=XAI_RESENDS,
        ),
        xai_case,
    ),
    (
        Case(
            'xai-previous-response-id-without-history',
            XaiServer,
            xai('grok-4.7'),
            XAI_STORE,
            seed_xai,
            with_history=False,
            instructions_channel='system',
            instructions_xfail=XAI_REPEATS_INSTRUCTIONS,
        ),
        xai_case,
    ),
]


@dataclass
class Outcome:
    model: Model
    wire: Seen
    """What the continued request sent."""
    seen: Seen
    """What the continued request put in front of the model, with what the API holds."""

    def entries(self, text: str, where: Seen | None = None) -> Seen:
        return [entry for entry in (self.seen if where is None else where) if text in entry[1]]


def turn(prompt: str, system_prompt: str) -> ModelRequest:
    """A stored request with a mid-conversation system prompt after the user's words."""
    return ModelRequest(parts=[UserPromptPart(prompt), SystemPromptPart(system_prompt)])


async def run_scenario(case: Case) -> Outcome:
    """Store a conversation through the route, add a system prompt after its latest response, and continue it.

    The first run starts the conversation and the second adds `EARLIER`, so the API holds both by the third, which adds
    `LATER` and is the continued request.
    """
    server = case.server()
    handle = server.handle if isinstance(server, OpenAIResponsesServer | AnthropicThreadsServer) else None
    transport = httpx2.MockTransport(handle) if handle else None
    async with httpx2.AsyncClient(transport=transport) as client:
        model = case.model(server, client)
        agent = Agent(model, instructions=INSTRUCTIONS)
        history = (await agent.run('Hello.', model_settings=case.settings)).all_messages()
        for request in (turn('What is the weather like?', EARLIER), turn('And tomorrow?', LATER)):
            response = history[-1]
            assert isinstance(response, ModelResponse)
            settings: ModelSettings = {**case.settings, **case.continue_settings(response)}
            messages = [*history, request] if case.with_history else [request]
            history = (await agent.run(None, message_history=messages, model_settings=settings)).all_messages()
    return Outcome(model, server.wire(server.bodies[-1]), server.seen[-1])


def cases(claim: Callable[[Case], str | None]) -> list[Any]:
    return [
        pytest.param(
            case,
            id=case.id,
            marks=[skip] + ([pytest.mark.xfail(strict=True, reason=reason)] if (reason := claim(case)) else []),
        )
        for case, skip in CASES
    ]


def check_instructions(outcome: Outcome, channel: Channel) -> None:
    assert outcome.entries(INSTRUCTIONS) == [(channel, INSTRUCTIONS)]


def check_held_prompt(outcome: Outcome) -> None:
    assert outcome.entries(EARLIER, outcome.wire) == []
    assert outcome.entries(EARLIER) == [voice(outcome.model, EARLIER)]


def check_new_prompt(outcome: Outcome) -> None:
    assert outcome.entries(LATER, outcome.wire) == [voice(outcome.model, LATER)]
    assert outcome.entries(LATER) == [voice(outcome.model, LATER)]


@pytest.mark.parametrize('case', cases(lambda case: case.instructions_xfail))
async def test_instructions_reach_the_model_once(case: Case):
    """The continued request puts the instructions in front of the model once, in the system channel."""
    check_instructions(await run_scenario(case), case.instructions_channel)


@pytest.mark.parametrize('case', cases(lambda case: case.held_xfail))
async def test_held_system_prompt_is_not_resent(case: Case):
    """The system prompt from before the continuation point isn't sent again, and the API still holds it once."""
    check_held_prompt(await run_scenario(case))


@pytest.mark.parametrize('case', cases(lambda case: case.new_xfail))
async def test_new_system_prompt_is_sent_once_in_its_voice(case: Case):
    """The system prompt added after the continuation point is sent once, in the voice of the model's profile."""
    check_new_prompt(await run_scenario(case))


def check_live(server: OpenAIResponsesServer | AnthropicThreadsServer, model: Model) -> None:
    """The claims of the tests above, on the last request `server` replayed."""
    outcome = Outcome(model, server.wire(server.bodies[-1]), server.seen[-1])
    check_instructions(outcome, 'instructions')
    check_held_prompt(outcome)
    check_new_prompt(outcome)


async def run_live(model: Model, settings: ModelSettings) -> list[ModelResponse]:
    agent = Agent(model, instructions=INSTRUCTIONS, model_settings=settings)
    history = (await agent.run('Hello.')).all_messages()
    for request in (turn('What is the capital of France?', EARLIER), turn('And of Spain?', LATER)):
        history = (await agent.run(None, message_history=[*history, request])).all_messages()
    return [message for message in history if isinstance(message, ModelResponse)]


def reply(response: ModelResponse) -> str:
    return ''.join(part.content for part in response.parts if isinstance(part, TextPart))


@pytest.mark.vcr
@openai_case
async def test_openai_previous_response_id_live(openai_api_key: str, request_capture: RequestCapture):
    """OpenAI continues the chain from the last response, and the continued request keeps both claims.

    The recorded bodies are replayed through `OpenAIResponsesServer` to add what the API holds.
    """
    model = OpenAIResponsesModel(
        'gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    responses = await run_live(model, AUTO)
    server = OpenAIResponsesServer()
    for body, response in zip(request_capture.bodies('/responses'), responses, strict=True):
        assert response.provider_response_id is not None
        server.record(cast(dict[str, Any], body), response.provider_response_id, reply(response))
    assert server.bodies[-1]['previous_response_id'] == responses[-2].provider_response_id
    check_live(server, model)


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(
    reason='the framing announcement joins `system` on the first request with a mid-conversation system prompt'
)
@anthropic_case
async def test_anthropic_message_threads_live(anthropic_model: AnthropicModelFactory, request_capture: RequestCapture):
    """Anthropic continues the thread from the last response, and the continued request keeps both claims.

    The recorded bodies are replayed through `AnthropicThreadsServer` to add what the thread holds.
    """
    model = anthropic_model('claude-haiku-5-5', capture=True)
    responses = await run_live(model, THREADS)
    server = AnthropicThreadsServer()
    for body, response in zip(request_capture.bodies('/messages'), responses, strict=True):
        assert response.provider_response_id is not None
        server.record(cast(dict[str, Any], body), response.provider_response_id, reply(response))
    assert server.bodies[-1]['thread'] == {
        'type': 'continue',
        'previous_message_id': responses[-2].provider_response_id,
    }
    check_live(server, model)
