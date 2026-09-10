"""Offline wire contracts for Kimi's native tool-addition channel.

The real SDK serializes every request into a recording transport. Comparing successive bodies
checks prefix stability, which a frozen response cassette alone cannot establish.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, BinaryContent, ModelAPIError, ModelRetry, Tool, ToolReturn, capture_run_messages
from pydantic_ai.capabilities import Toolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    RetryPromptPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ToolChoice, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.moonshotai import MoonshotAIModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai not installed')]


@dataclass
class MoonshotAPI:
    messages: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    fail_on_request: int | None = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if len(self.requests) == self.fail_on_request:
            raise httpx2.ReadTimeout('Response lost', request=request)
        message: dict[str, Any] = self.messages.pop(0) if self.messages else {'role': 'assistant', 'content': 'done'}
        finish_reason = 'tool_calls' if message.get('tool_calls') else 'stop'
        response = {
            'id': f'resp_{len(self.requests)}',
            'model': body['model'],
            'created': 1704067200,
            'object': 'chat.completion',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
        }
        if body.get('stream'):
            for index, call in enumerate(message.get('tool_calls', [])):
                call['index'] = index
            response['object'] = 'chat.completion.chunk'
            response['choices'] = [{'index': 0, 'delta': message, 'finish_reason': finish_reason}]
            return httpx2.Response(
                200,
                headers={'content-type': 'text/event-stream'},
                content=f'data: {json.dumps(response)}\n\ndata: [DONE]\n\n',
            )
        return httpx2.Response(200, json=response)


@pytest.fixture
def moonshot_api() -> MoonshotAPI:
    return MoonshotAPI()


@pytest.fixture
async def moonshot_provider(moonshot_api: MoonshotAPI) -> AsyncIterator[MoonshotAIProvider]:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(moonshot_api), trust_env=False) as client:
        yield MoonshotAIProvider(api_key='test-key', http_client=client)


def tool_call(name: str, arguments: str = '{}') -> dict[str, Any]:
    return {
        'role': 'assistant',
        'reasoning_content': 'Use the requested tool.',
        'tool_calls': [{'id': f'call_{name}', 'type': 'function', 'function': {'name': name, 'arguments': arguments}}],
    }


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('reveal', ['return', 'search', 'capability'])
async def test_tool_reveal_preserves_request_prefix(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    stream: bool,
    reveal: str,
):
    model = infer_model('moonshotai:kimi-k3', provider_factory=lambda _: moonshot_provider)
    assert isinstance(model, MoonshotAIModel)

    def weather() -> str:
        """Get the current weather."""
        return 'sunny'

    def load_weather() -> ToolReturn:
        """Load the weather tool."""
        return ToolReturn(return_value='Weather tool loaded.', tools=['weather'])

    if reveal == 'capability':
        agent = Agent(
            model,
            capabilities=[Toolset(FunctionToolset([weather]), id='weather', defer_loading=True)],
            instructions='Answer briefly.',
        )
        first_call = tool_call('load_capability', '{"id":"weather"}')
        eager_tools = ['load_capability']
    else:
        agent = Agent(model, tools=[load_weather, Tool(weather, defer_loading=True)], instructions='Answer briefly.')
        first_call = (
            tool_call('search_tools', '{"queries":["weather"]}') if reveal == 'search' else tool_call('load_weather')
        )
        eager_tools = ['load_weather', 'search_tools']
    moonshot_api.messages = [first_call, tool_call('weather')]
    if stream:
        async with agent.run_stream('Load weather and use it.') as result:
            assert await result.get_output() == 'done'
            history = result.all_messages()
    else:
        result = await agent.run('Load weather and use it.')
        assert result.output == 'done'
        history = result.all_messages()

    first, revealed, final = moonshot_api.requests
    assert first['tools'] == revealed['tools'] == final['tools']
    assert [tool['function']['name'] for tool in first['tools']] == eager_tools
    assert revealed['messages'][: len(first['messages'])] == first['messages']
    assert final['messages'][: len(revealed['messages'])] == revealed['messages']
    additions = [message for message in revealed['messages'] if 'tools' in message]
    assert additions == snapshot(
        [
            {
                'role': 'system',
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'description': 'Get the current weather.',
                            'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        },
                    }
                ],
            }
        ]
    )
    assert [message for message in final['messages'] if 'tools' in message] == additions
    assert any(message.get('role') == 'tool' and message.get('content') == 'sunny' for message in final['messages'])

    # A persisted conversation must reconstruct the same declaration at the same position.
    reloaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    await agent.run('Continue.', message_history=reloaded)
    assert moonshot_api.requests[-1]['messages'][: len(final['messages'])] == final['messages']


@pytest.mark.parametrize('search_return', [False, True])
async def test_reveals_deduplicate_and_respect_visibility(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    search_return: bool,
):
    model = MoonshotAIModel('kimi-k3', provider=moonshot_provider)
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='weather', defer_loading=True),
            ToolDefinition(name='hidden', defer_loading=True),
            ToolDefinition(name='eager'),
        ],
        revealed_tool_names={'weather'},
    )
    names = ['weather', 'weather', 'hidden', 'eager', 'missing']
    parts: list[ModelRequestPart] = [UserPromptPart('hello'), ToolAvailabilityDeltaPart(tools_added=names)]
    if search_return:
        parts.insert(
            1,
            ToolSearchReturnPart(
                tool_call_id='search',
                content={'discovered_tools': [{'name': name} for name in names]},
            ),
        )
    messages: list[ModelMessage] = [
        ModelRequest(parts=parts),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['weather']), UserPromptPart('next')]),
    ]
    original = deepcopy(messages)
    for _ in range(2):
        await model.request(messages, None, params)
    assert messages == original
    assert moonshot_api.requests[0] == moonshot_api.requests[1]
    body = moonshot_api.requests[0]
    additions = [message for message in body['messages'] if 'tools' in message]
    assert len(additions) == 1
    assert [tool['function']['name'] for tool in additions[0]['tools']] == ['weather']
    assert [tool['function']['name'] for tool in body['tools']] == ['eager']
    assert body['messages'][-1] == {'role': 'user', 'content': 'next'}


@pytest.mark.parametrize('tool_choice', ['none', [], ToolOrOutput([]), ToolOrOutput(['weather']), ['weather']])
async def test_tool_choice_applies_to_history_tools(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    tool_choice: ToolChoice,
):
    model = MoonshotAIModel(
        'kimi-k3', provider=moonshot_provider, profile=OpenAIModelProfile(openai_supports_tool_choice_required=True)
    )
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name=name, defer_loading=True) for name in ('weather', 'other')],
        revealed_tool_names={'weather', 'other'},
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hello')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['weather', 'other'])]),
    ]
    for choice in (None, tool_choice, 'auto'):
        await model.request(messages, {'tool_choice': choice, 'parallel_tool_calls': False}, params)
    before, body, after = moonshot_api.requests
    assert 'tools' not in body
    assert body['parallel_tool_calls'] is False
    if tool_choice == ['weather']:
        assert body['tool_choice'] == {'type': 'function', 'function': {'name': 'weather'}}
    else:
        assert body['tool_choice'] == ('auto' if tool_choice == ToolOrOutput(['weather']) else 'none')
    expected_names = ['weather', 'other'] if body['tool_choice'] == 'none' else ['weather']
    assert [tool['function']['name'] for tool in body['messages'][-1]['tools']] == expected_names
    if body['tool_choice'] == 'none':
        assert body['messages'] == before['messages'] == after['messages']


@pytest.mark.parametrize('tool_choice', ['none', ToolOrOutput([])])
@pytest.mark.parametrize('stream', [False, True])
async def test_disabling_and_reenabling_tools_preserves_request_prefix(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    tool_choice: ToolChoice,
    stream: bool,
):
    def weather() -> str:
        return 'sunny'

    def load_weather() -> ToolReturn:
        return ToolReturn(return_value='loaded', tools=['weather'])

    agent = Agent(
        MoonshotAIModel('kimi-k3', provider=moonshot_provider), tools=[load_weather, Tool(weather, defer_loading=True)]
    )
    history: list[ModelMessage] = []
    steps: list[tuple[str, ToolChoice, list[dict[str, Any]]]] = [
        ('Load weather and use it.', None, [tool_call('load_weather'), tool_call('weather')]),
        ('Summarize without tools.', tool_choice, []),
        ('Use weather again.', 'auto', [tool_call('weather')]),
    ]
    for prompt, choice, responses in steps:
        moonshot_api.messages = responses
        if stream:
            async with agent.run_stream(
                prompt, message_history=history, model_settings={'tool_choice': choice}
            ) as result:
                assert await result.get_output() == 'done'
                history = result.all_messages()
        else:
            result = await agent.run(prompt, message_history=history, model_settings={'tool_choice': choice})
            assert result.output == 'done'
            history = result.all_messages()
        history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))

    first, revealed, called, disabled, reenabled, final = moonshot_api.requests
    assert [body['tool_choice'] for body in moonshot_api.requests] == ['auto', 'auto', 'auto', 'none', 'auto', 'auto']
    assert [message for message in first['messages'] if 'tools' in message] == []
    additions = [message for message in revealed['messages'] if 'tools' in message]
    assert len(additions) == 1
    assert [tool['function']['name'] for tool in additions[0]['tools']] == ['weather']
    for body in (called, disabled, reenabled, final):
        assert [message for message in body['messages'] if 'tools' in message] == additions
    for before, after in zip(moonshot_api.requests, moonshot_api.requests[1:]):
        assert after['tools'] == before['tools']
        assert json.dumps(after['messages'][: len(before['messages'])]) == json.dumps(before['messages'])
    for body in (called, final):
        assert body['messages'][-1]['role'] == 'tool'
        assert body['messages'][-1]['content'] == 'sunny'


@pytest.mark.parametrize('model_name', ['kimi-k2.6', 'kimi-k3'])
async def test_plain_chat_mapping_is_unchanged(
    allow_model_requests: None, moonshot_provider: MoonshotAIProvider, moonshot_api: MoonshotAPI, model_name: str
):
    agent = Agent(MoonshotAIModel(model_name, provider=moonshot_provider), instructions='Be concise.')
    await agent.run('hello')
    await Agent(OpenAIChatModel(model_name, provider=moonshot_provider), instructions='Be concise.').run('hello')
    assert moonshot_api.requests[0] == moonshot_api.requests[1]


@pytest.mark.parametrize('tool_choice', ['none', [], ToolOrOutput([])])
async def test_disabling_function_tools_preserves_output_tools(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    tool_choice: ToolChoice,
):
    model = MoonshotAIModel(
        'kimi-k3', provider=moonshot_provider, profile=OpenAIModelProfile(openai_supports_tool_choice_required=True)
    )
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='weather', defer_loading=True)],
        output_tools=[ToolDefinition(name='final_result')],
        output_mode='tool',
        revealed_tool_names={'weather'},
    )
    await model.request(
        [ModelRequest(parts=[UserPromptPart('hello'), ToolAvailabilityDeltaPart(tools_added=['weather'])])],
        {'tool_choice': tool_choice},
        params,
    )
    body = moonshot_api.requests[0]
    assert body['tool_choice'] == 'auto'
    assert [tool['function']['name'] for tool in body['tools']] == ['final_result']
    assert not any('tools' in message for message in body['messages'])


def test_native_channel_is_provider_and_adapter_scoped(moonshot_provider: MoonshotAIProvider):
    assert MoonshotAIModel('kimi-k3', provider=moonshot_provider).tool_addition_mode == 'with_definitions'
    assert MoonshotAIModel('kimi-k2.6', provider=moonshot_provider).tool_addition_mode is None
    assert (
        MoonshotAIModel('kimi-k3', provider=moonshot_provider, profile={'tool_addition_mode': None}).tool_addition_mode
        is None
    )
    assert OpenAIChatModel('kimi-k3', provider=moonshot_provider).tool_addition_mode is None
    profile = OpenRouterProvider.model_profile('moonshotai/kimi-k3')
    assert profile is not None
    assert profile.get('tool_addition_mode') is None


@pytest.mark.parametrize('model_name,profile', [('kimi-k2.6', None), ('kimi-k3', {'tool_addition_mode': None})])
async def test_disabled_channel_keeps_the_chat_fallback(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    model_name: str,
    profile: ModelProfile | None,
):
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='weather', defer_loading=True)], revealed_tool_names={'weather'}
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hello'), ToolAvailabilityDeltaPart(tools_added=['weather'])])
    ]
    for model in (
        MoonshotAIModel(model_name, provider=moonshot_provider, profile=profile),
        OpenAIChatModel(model_name, provider=moonshot_provider),
    ):
        await model.request(model.prepare_messages(messages, params), None, params)
    assert moonshot_api.requests[0] == moonshot_api.requests[1]
    body = moonshot_api.requests[0]
    assert [tool['function']['name'] for tool in body['tools']] == ['weather']
    assert not any('tools' in message for message in body['messages'])


async def test_ignored_reveal_preserves_tool_media_grouping(
    allow_model_requests: None, moonshot_provider: MoonshotAIProvider, moonshot_api: MoonshotAPI
):
    """An unknown reveal must not move the user message holding tool-returned images."""
    image = BinaryContent(data=b'image', media_type='image/png')
    first = ToolReturnPart('first', content=[image], tool_call_id='first')
    second = ToolReturnPart('second', content=[image], tool_call_id='second')
    await MoonshotAIModel('kimi-k3', provider=moonshot_provider).request(
        [ModelRequest(parts=[first, ToolAvailabilityDeltaPart(tools_added=['missing']), second])],
        None,
        ModelRequestParameters(),
    )
    await OpenAIChatModel('kimi-k3', provider=moonshot_provider).request(
        [ModelRequest(parts=[first, second])], None, ModelRequestParameters()
    )
    assert moonshot_api.requests[0] == moonshot_api.requests[1]


@pytest.mark.parametrize('origin', ['anthropic', 'openai', 'local', 'legacy', 'delta'])
async def test_tool_reveal_history_replays_on_kimi(
    allow_model_requests: None, moonshot_provider: MoonshotAIProvider, moonshot_api: MoonshotAPI, origin: str
):
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('Find the weather tool.')])]
    if origin in ('anthropic', 'openai'):
        history.append(
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        args={'queries': ['weather']}, tool_call_id='search', provider_name=origin
                    ),
                    NativeToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'weather'}]},
                        tool_call_id='search',
                        provider_name=origin,
                    ),
                ],
                provider_name=origin,
            )
        )
    elif origin == 'delta':
        history.append(ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['weather'])]))
    else:
        call = (
            ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='search')
            if origin == 'local'
            else ToolCallPart('search_tools', {'queries': ['weather']}, tool_call_id='search')
        )
        result_part = (
            ToolSearchReturnPart(content={'discovered_tools': [{'name': 'weather'}]}, tool_call_id='search')
            if origin == 'local'
            else ToolReturnPart(
                'search_tools', 'Found weather.', tool_call_id='search', metadata={'discovered_tools': ['weather']}
            )
        )
        history.extend([ModelResponse(parts=[call]), ModelRequest(parts=[result_part])])
    reloaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))

    def weather() -> str:
        return 'sunny'

    moonshot_api.messages = [tool_call('weather')]
    agent = Agent(MoonshotAIModel('kimi-k3', provider=moonshot_provider), tools=[Tool(weather, defer_loading=True)])
    result = await agent.run('Use the discovered tool.', message_history=reloaded)
    assert result.output == 'done'
    body = moonshot_api.requests[0]
    additions = [message for message in body['messages'] if 'tools' in message]
    assert len(additions) == 1
    assert [tool['function']['name'] for tool in additions[0]['tools']] == ['weather']
    assert all(tool['function']['name'] != 'weather' for tool in body['tools'])


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('retry', [False, True])
async def test_reveal_preserves_parallel_tool_results_and_media(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    stream: bool,
    retry: bool,
):
    def weather() -> str:
        return 'sunny'

    def load_weather() -> ToolReturn:
        return ToolReturn(return_value=[BinaryContent(data=b'first image', media_type='image/png')], tools=['weather'])

    def second() -> ToolReturn:
        if retry:
            raise ModelRetry('Try again.')
        return ToolReturn(return_value=[BinaryContent(data=b'second image', media_type='image/png')])

    calls = tool_call('load_weather')
    calls['tool_calls'].extend(tool_call('second')['tool_calls'])
    moonshot_api.messages = [calls, tool_call('weather')]
    agent = Agent(
        MoonshotAIModel('kimi-k3', provider=moonshot_provider),
        tools=[load_weather, second, Tool(weather, defer_loading=True)],
    )
    if stream:
        async with agent.run_stream('Load weather and call second.') as result:
            assert await result.get_output() == 'done'
            history = result.all_messages()
    else:
        result = await agent.run('Load weather and call second.')
        assert result.output == 'done'
        history = result.all_messages()

    body = moonshot_api.requests[1]
    assert [message['role'] for message in body['messages']] == ['user', 'assistant', 'tool', 'tool', 'user', 'system']
    media = body['messages'][-2]['content']
    expected_images = ['data:image/png;base64,Zmlyc3QgaW1hZ2U=']
    if not retry:
        expected_images.append('data:image/png;base64,c2Vjb25kIGltYWdl')
    assert [part['image_url']['url'] for part in media if part['type'] == 'image_url'] == expected_images
    assert [tool['function']['name'] for tool in body['messages'][-1]['tools']] == ['weather']
    await agent.run('Continue.', message_history=history)
    assert moonshot_api.requests[-1]['messages'][: len(body['messages'])] == body['messages']


async def test_resume_after_lost_response_keeps_reveal_position(
    allow_model_requests: None, moonshot_provider: MoonshotAIProvider, moonshot_api: MoonshotAPI
):
    moonshot_provider.client.max_retries = 0
    moonshot_api.fail_on_request = 2
    moonshot_api.messages = [tool_call('load_weather'), tool_call('weather')]

    def weather() -> str:
        return 'sunny'

    def load_weather() -> ToolReturn:
        return ToolReturn(return_value='loaded', tools=['weather'])

    agent = Agent(
        MoonshotAIModel('kimi-k3', provider=moonshot_provider), tools=[load_weather, Tool(weather, defer_loading=True)]
    )
    with capture_run_messages() as history:
        with pytest.raises(ModelAPIError):
            await agent.run('Load weather.')
    before = moonshot_api.requests[-1]
    assert any('tools' in message for message in before['messages'])
    reloaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    result = await agent.run('Continue.', message_history=reloaded)
    assert result.output == 'done'
    after = moonshot_api.requests[-1]
    assert after['tools'] == before['tools']
    assert after['messages'][: len(before['messages'])] == before['messages']


@pytest.mark.parametrize('separator', [None, UserPromptPart('Continue.'), RetryPromptPart('Correct the output.')])
async def test_appending_a_reveal_preserves_prior_definitions(
    allow_model_requests: None,
    moonshot_provider: MoonshotAIProvider,
    moonshot_api: MoonshotAPI,
    separator: ModelRequestPart | None,
):
    model = MoonshotAIModel('kimi-k3', provider=moonshot_provider)
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name=name, defer_loading=True) for name in ('weather', 'clock')],
        revealed_tool_names={'weather'},
    )
    parts: list[ModelRequestPart] = [UserPromptPart('hello'), ToolAvailabilityDeltaPart(tools_added=['weather'])]
    await model.request([ModelRequest(parts=parts)], None, params)
    if separator is not None:
        parts.append(separator)
    parts.append(ToolAvailabilityDeltaPart(tools_added=['clock']))
    await model.request([ModelRequest(parts=parts)], None, replace(params, revealed_tool_names={'weather', 'clock'}))
    before, after = moonshot_api.requests
    assert after['messages'][: len(before['messages'])] == before['messages']
    assert [tool['function']['name'] for tool in after['messages'][-1]['tools']] == ['clock']
