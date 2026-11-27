"""One leading system message for models whose profile sets `honors_only_first_system_message`.

Bedrock Mantle's Responses endpoint ignores the top-level `instructions` for `gpt-oss-120b` and follows only one
leading system item (https://github.com/pydantic/pydantic-ai/issues/14451), so `BedrockMantleProvider` sets the flag
for gpt-oss there: the standing system prompt, the framing announcement and the instructions go out as exactly one
leading system message. On Chat Completions, gpt-oss's family profile merges the leading system messages already, and
every other request is unchanged, including gpt-oss on other Responses hosts.

The tests before the last use a mock transport rather than recordings: their claim is the exact request body, which
cassette matching doesn't check, across a route and profile matrix that has no recordings. The last test records the
real exchange on Mantle.
"""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai._framing import lookalike_note_announcement
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
    from pydantic_ai.providers.vllm import VLLMProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

SYSTEM_PROMPT = 'You are a terse assistant.'
INSTRUCTIONS = 'End every answer with the code word ZEBRA.'
ANNOUNCEMENT = lookalike_note_announcement(system_role=False, sealed=True)


def chat_or_responses(request: httpx2.Request) -> httpx2.Response:
    model = json.loads(request.content)['model']
    if request.url.path.endswith('/chat/completions'):
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': model,
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Blue'}}],
            },
        )
    return httpx2.Response(
        200,
        json={
            'id': 'resp_1',
            'object': 'response',
            'created_at': 0,
            'model': model,
            'status': 'completed',
            'output': [
                {
                    'type': 'message',
                    'id': 'msg_1',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'Blue', 'annotations': []}],
                }
            ],
            'parallel_tool_calls': True,
            'tool_choice': 'auto',
            'tools': [],
        },
    )


def compatible_provider(client: httpx2.AsyncClient) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider('nim', base_url='https://nim.example/v1', api_key='test', http_client=client)


def mantle_provider(client: httpx2.AsyncClient) -> BedrockMantleProvider:
    return BedrockMantleProvider(
        region_name='us-east-1', api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'mock-api-key'), http_client=client
    )


GPT_OSS_ROUTES: dict[str, Callable[[httpx2.AsyncClient], Model]] = {
    'compatible-chat': lambda client: OpenAIChatModel('openai/gpt-oss-120b', provider=compatible_provider(client)),
    'mantle-responses': lambda client: infer_model(
        'bedrock-mantle:openai.gpt-oss-120b', lambda _: mantle_provider(client)
    ),
}


def flagged_compatible_responses(profile: OpenAIModelProfile) -> Callable[[httpx2.AsyncClient], Model]:
    """gpt-oss on a compatible Responses route with the flag set in `profile`, to combine it with other profile keys."""
    return lambda client: OpenAIResponsesModel(
        'openai/gpt-oss-120b',
        provider=compatible_provider(client),
        profile=OpenAIModelProfile(honors_only_first_system_message=True, **profile),
    )


async def run_and_capture(
    model: Callable[[httpx2.AsyncClient], Model],
    *,
    system_prompt: str | None,
    instructions: str | None,
    message_history: list[ModelMessage] | None = None,
    model_settings: ModelSettings | None = None,
) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(
            model(client),
            system_prompt=system_prompt or (),
            instructions=instructions,
            model_settings=model_settings,
        )
        await agent.run('What color is the sky?', message_history=message_history)
    [body] = bodies
    return body


def leading_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's messages or input items, with the system ones at the start."""
    return body['messages'] if 'messages' in body else body['input']


@pytest.mark.parametrize('route', list(GPT_OSS_ROUTES))
@pytest.mark.parametrize(
    'system_prompt,instructions,expected',
    [
        pytest.param(None, INSTRUCTIONS, f'{ANNOUNCEMENT}\n\n{INSTRUCTIONS}', id='instructions'),
        pytest.param(SYSTEM_PROMPT, None, f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}', id='system-prompt'),
        pytest.param(SYSTEM_PROMPT, INSTRUCTIONS, f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}', id='both'),
    ],
)
async def test_gpt_oss_gets_one_leading_system_message(
    route: str, system_prompt: str | None, instructions: str | None, expected: str
):
    """The announcement that gpt-oss gets from the first request, the system prompt and the instructions share it.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    body = await run_and_capture(GPT_OSS_ROUTES[route], system_prompt=system_prompt, instructions=instructions)

    assert 'instructions' not in body
    assert leading_items(body) == [
        {'role': 'system', 'content': expected},
        {'role': 'user', 'content': 'What color is the sky?'},
    ]


def history_with_a_mid_conversation_system_prompt() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[SystemPromptPart(SYSTEM_PROMPT), UserPromptPart('Hi')]),
        ModelResponse(parts=[TextPart('Hello')]),
        ModelRequest(parts=[SystemPromptPart('Answer in one word.'), UserPromptPart('Ready?')]),
        ModelResponse(parts=[TextPart('Yes')]),
    ]


@pytest.mark.parametrize('route', list(GPT_OSS_ROUTES))
async def test_the_when_needed_announcement_joins_the_one_system_message(route: str):
    """With `'when_needed'`, the announcement joins the same single message once a `<system>` block needs it.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    settings = ModelSettings(framing_announcement='when_needed')
    before = await run_and_capture(
        GPT_OSS_ROUTES[route], system_prompt=SYSTEM_PROMPT, instructions=INSTRUCTIONS, model_settings=settings
    )
    after = await run_and_capture(
        GPT_OSS_ROUTES[route],
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=settings,
        message_history=history_with_a_mid_conversation_system_prompt(),
    )

    assert 'instructions' not in before and 'instructions' not in after
    assert leading_items(before)[0] == {'role': 'system', 'content': f'{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}'}
    assert leading_items(after)[0] == {
        'role': 'system',
        'content': f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}',
    }
    assert [item.get('role') for item in leading_items(after)].count('system') == 1


@pytest.mark.parametrize('route', list(GPT_OSS_ROUTES))
async def test_the_single_system_message_keeps_the_prefix_stable(route: str):
    """Each request's input is a prefix of the next one's: the single message leads every request unchanged.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(GPT_OSS_ROUTES[route](client), system_prompt=SYSTEM_PROMPT, instructions=INSTRUCTIONS)
        first = await agent.run('What color is the sky?')
        await agent.run('And at night?', message_history=first.all_messages())

    earlier, later = (leading_items(body) for body in bodies)
    assert later[: len(earlier)] == earlier
    assert all('instructions' not in body for body in bodies)


async def test_a_dynamic_system_prompt_still_joins_the_one_system_message():
    """A dynamic system prompt keeps other models' instructions in the top-level field, but not gpt-oss's.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(GPT_OSS_ROUTES['mantle-responses'](client), instructions=INSTRUCTIONS)

        @agent.system_prompt(dynamic=True)
        def dynamic_prompt() -> str:
            return SYSTEM_PROMPT

        await agent.run('What color is the sky?')

    [body] = bodies
    assert 'instructions' not in body
    assert body['input'][0] == {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}'}


async def test_a_continued_response_keeps_the_instructions_in_the_top_level_field():
    """A request continuing a stored response sends its instructions in the top-level field, not a new system item.

    The stored input already starts with the leading system message. Not a VCR test: the claim is the request body,
    which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(GPT_OSS_ROUTES['mantle-responses'](client), instructions=INSTRUCTIONS)
        await agent.run(
            'What color is the sky?', model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='resp_0')
        )

    [body] = bodies
    assert body['instructions'] == INSTRUCTIONS
    assert body['previous_response_id'] == 'resp_0'
    assert body['input'] == [{'role': 'user', 'content': 'What color is the sky?'}]


async def test_adding_to_a_conversation_keeps_the_instructions_in_the_top_level_field():
    """A request adding to an `openai_conversation_id` conversation sends its instructions in the top-level field.

    The conversation may already hold the leading system message, so the instructions don't join a new one, while a
    leading system prompt still merges with the announcement. Not a VCR test: the claim is the request body, which
    cassette matching doesn't check.
    """
    body = await run_and_capture(
        GPT_OSS_ROUTES['mantle-responses'],
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=OpenAIResponsesModelSettings(openai_conversation_id='conv_1'),
    )

    assert body['instructions'] == INSTRUCTIONS
    assert body['conversation'] == 'conv_1'
    assert body['input'] == [
        {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}'},
        {'role': 'user', 'content': 'What color is the sky?'},
    ]


async def test_an_auto_chain_starts_with_the_one_system_message():
    """With `openai_previous_response_id='auto'`, the request that starts the chain holds the instructions in it.

    The next request continues the stored response, whose input already starts with them. Not a VCR test: the claim
    is the request body, which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        agent = Agent(GPT_OSS_ROUTES['mantle-responses'](client), instructions=INSTRUCTIONS, model_settings=settings)
        first = await agent.run('What color is the sky?')
        await agent.run('And at night?', message_history=first.all_messages())

    started, continued = bodies
    assert 'instructions' not in started and 'previous_response_id' not in started
    assert started['input'] == [
        {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{INSTRUCTIONS}'},
        {'role': 'user', 'content': 'What color is the sky?'},
    ]
    assert continued['previous_response_id'] == 'resp_1'
    assert continued['input'] == [{'role': 'user', 'content': 'And at night?'}]
    # The stored input holds them, and the host may ignore the top-level field anyway.
    assert 'instructions' not in continued


async def test_changed_instructions_restart_an_auto_chain():
    """An `'auto'` chain only continues while the instructions its stored input starts with are unchanged.

    Otherwise the whole history goes out again, led by the one system message with the new instructions. Not a VCR
    test: the claim is the request body, which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    new_instructions = 'End every answer with the code word YAK.'
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = GPT_OSS_ROUTES['mantle-responses'](client)
        first = await Agent(model, instructions=INSTRUCTIONS, model_settings=settings).run('What color is the sky?')
        await Agent(model, instructions=new_instructions, model_settings=settings).run(
            'And at night?', message_history=first.all_messages()
        )

    _, restarted = bodies
    assert 'instructions' not in restarted and 'previous_response_id' not in restarted
    assert restarted['input'][0] == {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{new_instructions}'}
    assert [item.get('role') for item in restarted['input']] == ['system', 'user', 'assistant', 'user']


@pytest.mark.parametrize(
    'profile,settings',
    [
        pytest.param(
            OpenAIModelProfile(openai_supports_prompt_cache_breakpoints=True),
            OpenAIResponsesModelSettings(openai_cache_instructions=True),
            id='cache-instructions',
        ),
        pytest.param(
            OpenAIModelProfile(openai_responses_stable_instructions_as_input=True),
            None,
            id='stable-instructions-as-input',
        ),
    ],
)
async def test_instruction_caching_leaves_the_one_system_message_whole(
    profile: OpenAIModelProfile, settings: OpenAIResponsesModelSettings | None
):
    """The instruction breakpoint and the stable/rewriting split both need a boundary the merge removes, so neither applies.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    body = await run_and_capture(
        flagged_compatible_responses(profile),
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=settings,
    )

    assert 'instructions' not in body
    assert body['input'] == [
        {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}'},
        {'role': 'user', 'content': 'What color is the sky?'},
    ]


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param(None, id='no-chain'),
        pytest.param(OpenAIResponsesModelSettings(openai_previous_response_id='auto'), id='auto-chain'),
    ],
)
async def test_the_user_system_prompt_role_merges_nothing(settings: OpenAIResponsesModelSettings | None):
    """A system prompt sent as a user message can't be told apart from a real one, so nothing is merged.

    Nor does an `'auto'` chain start with the instructions in the input. Not a VCR test: the claim is the request
    body, which cassette matching doesn't check.
    """
    body = await run_and_capture(
        flagged_compatible_responses(OpenAIModelProfile(openai_system_prompt_role='user')),
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=settings,
    )

    assert body['instructions'] == INSTRUCTIONS
    assert body['input'] == [
        {'role': 'user', 'content': ANNOUNCEMENT},
        {'role': 'user', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': 'What color is the sky?'},
    ]


async def test_a_leading_cache_point_keeps_its_breakpoint_on_the_merged_message():
    """A breakpoint a leading `CachePoint` puts on the system prompt goes at the end of the merged message.

    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_or_responses(request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = flagged_compatible_responses(OpenAIModelProfile(openai_supports_prompt_cache_breakpoints=True))(client)
        agent = Agent(model, system_prompt=SYSTEM_PROMPT, instructions=INSTRUCTIONS)
        await agent.run([CachePoint(), 'What color is the sky?'])

    [body] = bodies
    assert body['input'][0] == snapshot(
        {
            'role': 'system',
            'content': [
                {
                    'type': 'input_text',
                    'text': f'{ANNOUNCEMENT}\n\n{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}',
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                }
            ],
        }
    )
    assert [item.get('role') for item in body['input']].count('system') == 1


UNFLAGGED_ROUTES: dict[str, Callable[[httpx2.AsyncClient], Model]] = {
    'compatible-chat': lambda client: OpenAIChatModel(
        'mistralai/mistral-large-4-1t-a52b', provider=compatible_provider(client)
    ),
    'compatible-responses': lambda client: OpenAIResponsesModel(
        'mistralai/mistral-large-4-1t-a52b', provider=compatible_provider(client)
    ),
    'mantle-responses': lambda client: infer_model(
        'bedrock-mantle:openai.gpt-5.6-luna', lambda _: mantle_provider(client)
    ),
    'gpt-oss-compatible-responses': lambda client: OpenAIResponsesModel(
        'openai/gpt-oss-120b', provider=compatible_provider(client)
    ),
}


UNFLAGGED_BODIES: dict[str, dict[str, Any]] = {
    'compatible-chat': snapshot(
        {
            'model': 'mistralai/mistral-large-4-1t-a52b',
            'stream': False,
            'messages': [
                {'role': 'system', 'content': 'You are a terse assistant.'},
                {'role': 'system', 'content': 'End every answer with the code word ZEBRA.'},
                {'role': 'user', 'content': 'What color is the sky?'},
            ],
        }
    ),
    'compatible-responses': snapshot(
        {
            'model': 'mistralai/mistral-large-4-1t-a52b',
            'stream': False,
            'input': [
                {'role': 'system', 'content': 'You are a terse assistant.'},
                {'role': 'user', 'content': 'What color is the sky?'},
            ],
            'instructions': 'End every answer with the code word ZEBRA.',
        }
    ),
    'mantle-responses': snapshot(
        {
            'model': 'openai.gpt-5.6-luna',
            'stream': False,
            'include': ['reasoning.encrypted_content'],
            'input': [
                {'role': 'system', 'content': 'You are a terse assistant.'},
                {'role': 'user', 'content': 'What color is the sky?'},
            ],
            'instructions': 'End every answer with the code word ZEBRA.',
            'reasoning': {'context': 'all_turns'},
        }
    ),
    # gpt-oss's family profile merges on Chat Completions only, so its Responses request elsewhere is unchanged.
    'gpt-oss-compatible-responses': {
        'model': 'openai/gpt-oss-120b',
        'stream': False,
        'input': [
            {'role': 'system', 'content': ANNOUNCEMENT},
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'What color is the sky?'},
        ],
        'instructions': INSTRUCTIONS,
    },
}


@pytest.mark.parametrize('route', list(UNFLAGGED_ROUTES))
async def test_other_models_send_the_same_request(route: str):
    """Models without `honors_only_first_system_message` send what they sent before it existed.

    The system prompt and the instructions stay separate, and on Responses the instructions stay in the top-level field,
    including gpt-oss's on a Responses host other than Bedrock Mantle.
    Not a VCR test: the claim is the request body, which cassette matching doesn't check.
    """
    body = await run_and_capture(UNFLAGGED_ROUTES[route], system_prompt=SYSTEM_PROMPT, instructions=INSTRUCTIONS)

    assert body == UNFLAGGED_BODIES[route]


def vllm_provider(client: httpx2.AsyncClient) -> VLLMProvider:
    return VLLMProvider(base_url='http://localhost:8000/v1', api_key='test', http_client=client)


VLLM_QWEN_BODIES: dict[str, dict[str, Any]] = {
    'chat': snapshot(
        {
            'model': 'Qwen/Qwen3-32B',
            'stream': False,
            'messages': [
                {
                    'role': 'system',
                    'content': """\
You are a terse assistant.

End every answer with the code word ZEBRA.\
""",
                },
                {'role': 'user', 'content': 'What color is the sky?'},
            ],
        }
    ),
    'responses': snapshot(
        {
            'model': 'Qwen/Qwen3-32B',
            'stream': False,
            'input': [
                {'role': 'system', 'content': 'You are a terse assistant.'},
                {'role': 'user', 'content': 'What color is the sky?'},
            ],
            'instructions': 'End every answer with the code word ZEBRA.',
        }
    ),
}


@pytest.mark.parametrize('api', list(VLLM_QWEN_BODIES))
async def test_vllm_merges_other_models_on_chat_only(api: str):
    """`VLLMProvider` merges the leading system messages on Chat Completions for every model, as before.

    On Responses, the instructions stay in the top-level field. Not a VCR test: the claim is the request body, which
    cassette matching doesn't check.
    """
    model_class = OpenAIChatModel if api == 'chat' else OpenAIResponsesModel
    body = await run_and_capture(
        lambda client: model_class('Qwen/Qwen3-32B', provider=vllm_provider(client)),
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
    )

    assert body == VLLM_QWEN_BODIES[api]


@pytest.mark.vcr
async def test_mantle_gpt_oss_responses_follows_the_instructions(
    allow_model_requests: None, request_capture: RequestCapture
):
    """Bedrock Mantle's `/v1/responses` ignores the top-level `instructions` for `gpt-oss-120b` (#14451).

    Sent in the one leading system input item instead, they're followed: the recording answers with the code word
    only the instructions ask for.
    """
    model = infer_model('bedrock-mantle:openai.gpt-oss-120b', lambda _: mantle_provider(request_capture.client))
    agent = Agent(model, instructions=INSTRUCTIONS)

    result = await agent.run('Reply with one word: what color is the sky on a clear day?')

    assert 'ZEBRA' in result.output
    [path] = request_capture.paths
    assert path.endswith('/v1/responses')
    body = request_capture.body(path)
    assert 'instructions' not in body
    assert body['input'] == snapshot(
        [
            {'role': 'system', 'content': f'{ANNOUNCEMENT}\n\n{INSTRUCTIONS}'},
            {'role': 'user', 'content': 'Reply with one word: what color is the sky on a clear day?'},
        ]
    )
