"""Mid-conversation system prompts on the OpenAI model classes.

OpenAI's own API takes a `system` message at any position, on Chat Completions and on Responses, so
`OpenAIProvider` sends a `SystemPromptPart` outside the first `ModelRequest` as one. Many OpenAI-compatible
servers take a `system` message only at the start and answer a later one with a bare 400
(https://github.com/pydantic/pydantic-ai/issues/14135), so the providers that build on `openai_model_profile`
send it as a `<system>` block in a user turn unless the model's profile sets `supports_inline_system_prompts`.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._framing import lookalike_note_announcement
from pydantic_ai.models import Model
from pydantic_ai.profiles import ModelProfile

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]

INSTRUCTIONS = 'You are a test agent.'
REMINDER = 'Reply with exactly the word BANANA.'
# What the system prompt says the `<system>` blocks mean, where a mid-conversation system prompt becomes one.
ANNOUNCEMENT = lookalike_note_announcement(system_role=False)


def ping() -> str:
    return 'pong'


def message_history() -> list[ModelMessage]:
    """A conversation whose last request carries a system prompt right after a tool result."""
    return [
        ModelRequest(parts=[UserPromptPart('Call ping, then tell me what it returned.')]),
        ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('ping', 'pong', tool_call_id='call_1'), SystemPromptPart(REMINDER)]),
    ]


LIVE_WIRE: dict[str, list[dict[str, Any]]] = {
    'chat': snapshot(
        [
            {'role': 'system', 'content': INSTRUCTIONS},
            {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
            {'role': 'assistant', 'content': None},
            {'role': 'tool', 'content': 'pong'},
            {'role': 'system', 'content': REMINDER},
        ]
    ),
    'responses': snapshot(
        [
            {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
            {'type': 'function_call'},
            {'type': 'function_call_output'},
            {'role': 'system', 'content': REMINDER},
        ]
    ),
}


@pytest.mark.parametrize(
    'api,model_name',
    [pytest.param('chat', 'gpt-4.1-mini', id='chat'), pytest.param('responses', 'gpt-6.1-terra', id='responses')],
)
async def test_openai_sends_a_native_system_message(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture, api: str, model_name: str
):
    """OpenAI's API accepts a `system` message after a tool result, and the model follows it.

    The recordings answer `BANANA`, which only the instruction in that message asks for.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    if api == 'chat':
        model: Model = OpenAIChatModel(model_name, provider=provider)
        body_key, path = 'messages', '/chat/completions'
    else:
        model = OpenAIResponsesModel(model_name, provider=provider)
        body_key, path = 'input', '/responses'

    result = await Agent(model, instructions=INSTRUCTIONS, tools=[ping]).run(message_history=message_history())

    assert result.output == 'BANANA'
    items = request_capture.body(path)[body_key]
    assert isinstance(items, list)
    assert [
        {key: item[key] for key in ('role', 'type', 'content') if key in item}
        for item in items
        if isinstance(item, dict)
    ] == LIVE_WIRE[api]


async def test_gpt_oss_on_a_compatible_endpoint_gets_the_announcement_first(
    allow_model_requests: None, deepinfra_api_key: str, request_capture: RequestCapture
):
    """gpt-oss on an OpenAI-compatible endpoint gets the announcement from a conversation's first request.

    Its profile's `default_framing_announcement='always'` applies once the endpoint has no mid-conversation system
    role, so even a plain request opens with two system messages: the announcement, then the instructions. The
    recording shows a server running gpt-oss behind a chat template accepting that.
    """
    provider = OpenAICompatibleProvider(
        'deepinfra',
        base_url='https://api.deepinfra.com/v1/openai',
        api_key=deepinfra_api_key,
        http_client=request_capture.client,
    )
    agent = Agent(OpenAIChatModel('openai/gpt-oss-120b', provider=provider), instructions=INSTRUCTIONS)

    result = await agent.run('Reply with one word: what color is the sky on a clear day?')

    assert result.output
    assert request_capture.body('/chat/completions')['messages'] == snapshot(
        [
            {'role': 'system', 'content': ANNOUNCEMENT},
            {'role': 'system', 'content': INSTRUCTIONS},
            {'role': 'user', 'content': 'Reply with one word: what color is the sky on a clear day?'},
        ]
    )


def chat_completion(request: httpx2.Request) -> httpx2.Response:
    if request.url.path.endswith('/chat/completions'):
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': json.loads(request.content)['model'],
                'choices': [
                    {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'BANANA'}}
                ],
            },
        )
    return httpx2.Response(
        200,
        json={
            'id': 'resp_1',
            'object': 'response',
            'created_at': 0,
            'model': json.loads(request.content)['model'],
            'status': 'completed',
            'output': [
                {
                    'type': 'message',
                    'id': 'msg_1',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'BANANA', 'annotations': []}],
                }
            ],
            'parallel_tool_calls': True,
            'tool_choice': 'auto',
            'tools': [],
        },
    )


def compatible_provider(client: httpx2.AsyncClient) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider('nim', base_url='https://nim.example/v1', api_key='test', http_client=client)


@dataclass(frozen=True)
class Case:
    id: str
    build: Callable[[httpx2.AsyncClient], Model]
    expected: dict[str, Any]


CASES = [
    Case(
        id='compatible-chat',
        build=lambda client: OpenAIChatModel('mistralai/mistral-large-4-1t-a52b', provider=compatible_provider(client)),
        expected=snapshot(
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': ANNOUNCEMENT,
                    },
                    {'role': 'system', 'content': 'You are a test agent.'},
                    {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'RksXkbv4v',
                                'type': 'function',
                                'function': {'name': 'ping', 'arguments': '{}'},
                            }
                        ],
                    },
                    {'role': 'tool', 'tool_call_id': 'RksXkbv4v', 'content': 'pong'},
                    {'role': 'user', 'content': '<system>Reply with exactly the word BANANA.</system>'},
                ]
            }
        ),
    ),
    Case(
        id='compatible-responses',
        build=lambda client: OpenAIResponsesModel(
            'mistralai/mistral-large-4-1t-a52b', provider=compatible_provider(client)
        ),
        expected=snapshot(
            {
                'instructions': 'You are a test agent.',
                'input': [
                    {
                        'role': 'system',
                        'content': ANNOUNCEMENT,
                    },
                    {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
                    {'name': 'ping', 'arguments': '{}', 'call_id': 'RksXkbv4v', 'type': 'function_call'},
                    {'type': 'function_call_output', 'call_id': 'RksXkbv4v', 'output': 'pong'},
                    {'role': 'user', 'content': '<system>Reply with exactly the word BANANA.</system>'},
                ],
            }
        ),
    ),
    Case(
        id='compatible-chat-opted-in',
        build=lambda client: OpenAIChatModel(
            'mistralai/mistral-large-4-1t-a52b',
            provider=compatible_provider(client),
            profile=ModelProfile(supports_inline_system_prompts=True),
        ),
        expected=snapshot(
            {
                'messages': [
                    {'role': 'system', 'content': 'You are a test agent.'},
                    {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'RksXkbv4v',
                                'type': 'function',
                                'function': {'name': 'ping', 'arguments': '{}'},
                            }
                        ],
                    },
                    {'role': 'tool', 'tool_call_id': 'RksXkbv4v', 'content': 'pong'},
                    {'role': 'system', 'content': 'Reply with exactly the word BANANA.'},
                ]
            }
        ),
    ),
    Case(
        id='openai-provider-custom-base-url',
        build=lambda client: OpenAIChatModel(
            'gpt-4.1-mini',
            provider=OpenAIProvider(base_url='https://proxy.example/v1', api_key='test', http_client=client),
        ),
        expected=snapshot(
            {
                'messages': [
                    {'role': 'system', 'content': 'You are a test agent.'},
                    {'role': 'user', 'content': 'Call ping, then tell me what it returned.'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'call_1',
                                'type': 'function',
                                'function': {'name': 'ping', 'arguments': '{}'},
                            }
                        ],
                    },
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'pong'},
                    {'role': 'system', 'content': 'Reply with exactly the word BANANA.'},
                ]
            }
        ),
    ),
]


# Without `ignore_hosts`, the module's `vcr` mark answers these hosts from a cassette before the mock transport sees them.
@pytest.mark.vcr(ignore_hosts=['nim.example', 'proxy.example'])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_mid_conversation_system_prompt_rendering(allow_model_requests: None, case: Case):
    """Where a system prompt after a tool result goes, per provider and profile.

    A mock transport stands in for a cassette because the claim is which rendering each provider picks, not how a
    particular server answers: the servers the default protects reject the request, and no key reaches them here.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_completion(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        await Agent(case.build(client), instructions=INSTRUCTIONS, tools=[ping]).run(message_history=message_history())

    [body] = bodies
    if 'messages' in body:
        rendered = {'messages': body['messages']}
    else:
        rendered = {'instructions': body.get('instructions'), 'input': body['input']}
    assert rendered == case.expected
