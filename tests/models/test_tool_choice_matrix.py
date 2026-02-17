"""Cross-provider matrix tests for tool_choice functionality.

This module tests tool_choice behavior across all providers using a cartesian
product of test dimensions: provider, scenario.

Tests verify that the correct tool_choice value is sent to each provider's API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import assert_never

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as huggingface_available:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

Expectation = Literal['native', 'skip']
Scenario = Literal['auto', 'none', 'required', 'list_single', 'none_with_output', 'tools_plus_output']


SUPPORT_MATRIX: dict[tuple[str, Scenario], Expectation] = {
    ('openai', 'auto'): 'native',
    ('openai', 'none'): 'native',
    ('openai', 'required'): 'native',
    ('openai', 'list_single'): 'native',
    ('openai', 'none_with_output'): 'native',
    ('openai', 'tools_plus_output'): 'native',
    ('openai_responses', 'auto'): 'native',
    ('openai_responses', 'none'): 'native',
    ('openai_responses', 'required'): 'native',
    ('openai_responses', 'list_single'): 'native',
    ('openai_responses', 'none_with_output'): 'native',
    ('openai_responses', 'tools_plus_output'): 'native',
    ('anthropic', 'auto'): 'native',
    ('anthropic', 'none'): 'native',
    ('anthropic', 'required'): 'native',
    ('anthropic', 'list_single'): 'native',
    ('anthropic', 'none_with_output'): 'native',
    ('anthropic', 'tools_plus_output'): 'native',
    ('groq', 'auto'): 'native',
    ('groq', 'none'): 'native',
    ('groq', 'required'): 'native',
    ('groq', 'list_single'): 'native',
    ('groq', 'none_with_output'): 'native',
    ('groq', 'tools_plus_output'): 'native',
    ('mistral', 'auto'): 'native',
    ('mistral', 'none'): 'native',
    ('mistral', 'required'): 'native',
    ('mistral', 'list_single'): 'native',
    ('mistral', 'none_with_output'): 'native',
    ('mistral', 'tools_plus_output'): 'native',
    ('google', 'auto'): 'native',
    ('google', 'none'): 'native',
    ('google', 'required'): 'native',
    ('google', 'list_single'): 'native',
    ('google', 'none_with_output'): 'native',
    ('google', 'tools_plus_output'): 'native',
    ('bedrock', 'auto'): 'native',
    ('bedrock', 'none'): 'native',
    ('bedrock', 'required'): 'native',
    ('bedrock', 'list_single'): 'native',
    ('bedrock', 'none_with_output'): 'native',
    ('bedrock', 'tools_plus_output'): 'native',
    ('huggingface', 'auto'): 'skip',
    ('huggingface', 'none'): 'native',
    ('huggingface', 'required'): 'native',
    ('huggingface', 'list_single'): 'native',
    ('huggingface', 'none_with_output'): 'native',
    ('huggingface', 'tools_plus_output'): 'native',
    ('xai', 'auto'): 'native',
    ('xai', 'none'): 'native',
    ('xai', 'required'): 'native',
    ('xai', 'list_single'): 'native',
    ('xai', 'none_with_output'): 'native',
    ('xai', 'tools_plus_output'): 'native',
}


@dataclass
class SkipReason:
    reason: str


SKIP_REASONS: dict[tuple[str, Scenario], SkipReason] = {
    ('huggingface', 'auto'): SkipReason('Together backend 500s on tool continuation'),
}


MODEL_CONFIGS: dict[str, tuple[str, Any]] = {
    'openai': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_available),
    'anthropic': ('claude-sonnet-4-5', anthropic_available),
    'groq': ('meta-llama/llama-4-scout-17b-16e-instruct', groq_available),
    'mistral': ('mistral-large-latest', mistral_available),
    'google': ('gemini-2.5-flash', google_available),
    'bedrock': ('us.anthropic.claude-sonnet-4-5-20250929-v1:0', bedrock_available),
    'huggingface': ('meta-llama/Llama-4-Scout-17B-16E-Instruct', huggingface_available),
    'xai': ('grok-3-fast', xai_available),
}


def create_model(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any = None,
    xai_provider: Any = None,
) -> Model:
    model_name = MODEL_CONFIGS[provider][0]
    if provider == 'openai':
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'openai_responses':
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_keys['anthropic']))
    elif provider == 'groq':
        return GroqModel(model_name, provider=GroqProvider(api_key=api_keys['groq']))
    elif provider == 'mistral':
        return MistralModel(model_name, provider=MistralProvider(api_key=api_keys['mistral']))
    elif provider == 'google':
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'bedrock':
        assert bedrock_provider is not None
        return BedrockConverseModel(model_name, provider=bedrock_provider)
    elif provider == 'huggingface':
        return HuggingFaceModel(
            model_name, provider=HuggingFaceProvider(api_key=api_keys['huggingface'], provider_name='together')
        )
    elif provider == 'xai':
        assert xai_provider is not None
        return XaiModel(model_name, provider=xai_provider)
    else:  # pragma: no cover
        raise ValueError(f'Unknown provider: {provider}')


def is_provider_available(provider: str) -> bool:
    _, available = MODEL_CONFIGS.get(provider, (None, lambda: False))
    return bool(available() if callable(available) else available)


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'Sunny, 22C in {city}'


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f'14:30 in {timezone}'  # pragma: no cover


class CityInfo(BaseModel):
    city: str
    summary: str


def make_tool_def(name: str, description: str, param_name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            'properties': {param_name: {'type': 'string'}},
            'required': [param_name],
        },
    )


def get_tool_choice_from_cassette(cassette: Any, provider: str, xai_provider: Any = None) -> Any:
    """Extract tool_choice from cassette request body, handling provider differences."""
    if provider == 'xai':
        return _get_xai_tool_choice(xai_provider)

    if not cassette.requests:
        return None  # pragma: no cover

    request = None
    for req in cassette.requests:
        if req.method == 'POST':
            request = req
            break
    if request is None:  # pragma: no cover
        return None

    body_bytes = request.body
    if body_bytes is None:
        return None  # pragma: no cover

    try:
        body: dict[str, Any] = json.loads(body_bytes) if isinstance(body_bytes, (str, bytes)) else body_bytes
    except (json.JSONDecodeError, TypeError):  # pragma: no cover
        return None

    if provider == 'google':
        tool_config: dict[str, Any] = body.get('toolConfig', {})
        func_config: dict[str, Any] = tool_config.get('functionCallingConfig', {})
        return func_config.get('mode')
    elif provider == 'anthropic':
        tc = body.get('tool_choice', {})
        if isinstance(tc, dict):
            return tc.get('type')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return tc  # pragma: no cover
    elif provider == 'bedrock':
        tool_config = body.get('toolConfig', {})
        tool_choice = tool_config.get('toolChoice', {})
        if 'auto' in tool_choice:
            return 'auto'
        elif 'any' in tool_choice:
            return 'any'
        elif 'tool' in tool_choice:
            return tool_choice['tool'].get('name')
        return None
    else:
        return body.get('tool_choice')


def _get_xai_tool_choice(xai_provider: Any) -> Any:
    """Extract tool_choice from xAI provider's underlying client cassette.

    xAI uses protobuf format which MessageToDict converts to:
    - {'mode': 'TOOL_MODE_AUTO'} -> 'auto'
    - {'mode': 'TOOL_MODE_NONE'} -> 'none'
    - {'mode': 'TOOL_MODE_REQUIRED'} -> 'required'
    - {'function_name': 'X'} -> {'type': 'function', 'function': {'name': 'X'}}
    """
    if xai_provider is None:
        return None  # pragma: no cover

    client = xai_provider._client
    if hasattr(client, 'cassette') and client.cassette.interactions:
        interaction = client.cassette.interactions[0]
        if hasattr(interaction, 'request_json') and interaction.request_json:
            tc = interaction.request_json.get('tool_choice')
            if tc is None:
                return None  # pragma: no cover
            if isinstance(tc, dict):
                mode: str | None = tc.get('mode')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                if mode == 'TOOL_MODE_AUTO':
                    return 'auto'
                elif mode == 'TOOL_MODE_NONE':
                    return 'none'
                elif mode == 'TOOL_MODE_REQUIRED':
                    return 'required'
                fn_name: str | None = tc.get('function_name')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                if fn_name:
                    result: dict[str, Any] = {'type': 'function', 'function': {'name': fn_name}}
                    return result
            return tc  # pragma: no cover  # pyright: ignore[reportUnknownVariableType]
    return None  # pragma: no cover


@pytest.fixture
def api_keys(
    openai_api_key: str,
    anthropic_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    gemini_api_key: str,
    huggingface_api_key: str,
    xai_api_key: str,
) -> dict[str, str]:
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'groq': groq_api_key,
        'mistral': mistral_api_key,
        'google': gemini_api_key,
        'huggingface': huggingface_api_key,
        'xai': xai_api_key,
    }


EXPECTED_TOOL_CHOICE: dict[tuple[str, Scenario], Any] = {
    ('openai', 'auto'): snapshot('auto'),
    ('openai', 'none'): snapshot('none'),
    ('openai', 'required'): snapshot('required'),
    ('openai', 'list_single'): snapshot({'type': 'function', 'function': {'name': 'get_weather'}}),
    ('openai', 'none_with_output'): snapshot({'type': 'function', 'function': {'name': 'final_result'}}),
    ('openai', 'tools_plus_output'): snapshot('required'),
    ('openai_responses', 'auto'): snapshot('auto'),
    ('openai_responses', 'none'): snapshot('none'),
    ('openai_responses', 'required'): snapshot('required'),
    ('openai_responses', 'list_single'): snapshot({'type': 'function', 'name': 'get_weather'}),
    ('openai_responses', 'none_with_output'): snapshot({'type': 'function', 'name': 'final_result'}),
    ('openai_responses', 'tools_plus_output'): snapshot(
        {
            'type': 'allowed_tools',
            'mode': 'required',
            'tools': [{'type': 'function', 'name': 'final_result'}, {'type': 'function', 'name': 'get_weather'}],
        }
    ),
    ('anthropic', 'auto'): snapshot('auto'),
    ('anthropic', 'none'): snapshot('none'),
    ('anthropic', 'required'): snapshot('any'),
    ('anthropic', 'list_single'): snapshot('tool'),
    ('anthropic', 'none_with_output'): snapshot('tool'),
    ('anthropic', 'tools_plus_output'): snapshot('any'),
    ('groq', 'auto'): snapshot('auto'),
    ('groq', 'none'): snapshot('none'),
    ('groq', 'required'): snapshot('required'),
    ('groq', 'list_single'): snapshot({'type': 'function', 'function': {'name': 'get_weather'}}),
    ('groq', 'none_with_output'): snapshot({'type': 'function', 'function': {'name': 'final_result'}}),
    ('groq', 'tools_plus_output'): snapshot('required'),
    ('mistral', 'auto'): snapshot('auto'),
    ('mistral', 'none'): snapshot(None),
    ('mistral', 'required'): snapshot('any'),
    ('mistral', 'list_single'): snapshot('any'),
    ('mistral', 'none_with_output'): snapshot('any'),
    ('mistral', 'tools_plus_output'): snapshot('any'),
    ('google', 'auto'): snapshot('AUTO'),
    ('google', 'none'): snapshot('NONE'),
    ('google', 'required'): snapshot('ANY'),
    ('google', 'list_single'): snapshot('ANY'),
    ('google', 'none_with_output'): snapshot('ANY'),
    ('google', 'tools_plus_output'): snapshot('ANY'),
    ('bedrock', 'auto'): snapshot('auto'),
    ('bedrock', 'none'): snapshot(None),
    ('bedrock', 'required'): snapshot('any'),
    ('bedrock', 'list_single'): snapshot('get_weather'),
    ('bedrock', 'none_with_output'): snapshot('final_result'),
    ('bedrock', 'tools_plus_output'): snapshot('any'),
    ('huggingface', 'auto'): snapshot('auto'),
    ('huggingface', 'none'): snapshot('none'),
    ('huggingface', 'required'): snapshot('required'),
    ('huggingface', 'list_single'): snapshot({'function': {'name': 'get_weather'}}),
    ('huggingface', 'none_with_output'): snapshot({'function': {'name': 'final_result'}}),
    ('huggingface', 'tools_plus_output'): snapshot('required'),
    ('xai', 'auto'): snapshot('auto'),
    ('xai', 'none'): snapshot('none'),
    ('xai', 'required'): snapshot('required'),
    ('xai', 'list_single'): snapshot({'type': 'function', 'function': {'name': 'get_weather'}}),
    ('xai', 'none_with_output'): snapshot({'type': 'function', 'function': {'name': 'final_result'}}),
    ('xai', 'tools_plus_output'): snapshot('required'),
}


PROVIDERS = [
    pytest.param('openai', id='openai'),
    pytest.param('openai_responses', id='openai_responses'),
    pytest.param('anthropic', id='anthropic'),
    pytest.param('groq', id='groq'),
    pytest.param('mistral', id='mistral'),
    pytest.param('google', id='google'),
    pytest.param('bedrock', id='bedrock'),
    pytest.param('huggingface', id='huggingface'),
    pytest.param('xai', id='xai'),
]

SCENARIOS: list[Any] = [
    pytest.param('auto', id='auto'),
    pytest.param('none', id='none'),
    pytest.param('required', id='required'),
    pytest.param('list_single', id='list_single'),
    pytest.param('none_with_output', id='none_with_output'),
    pytest.param('tools_plus_output', id='tools_plus_output'),
]


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('scenario', SCENARIOS)
async def test_tool_choice_matrix(
    provider: str,
    scenario: Scenario,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    allow_model_requests: None,
    vcr: Any,
):
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    expectation = SUPPORT_MATRIX.get((provider, scenario))
    if expectation == 'skip':
        skip_info = SKIP_REASONS.get((provider, scenario))
        pytest.skip(skip_info.reason if skip_info else f'{provider}/{scenario} skipped')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    expected_tool_choice = EXPECTED_TOOL_CHOICE.get((provider, scenario))

    if scenario == 'auto':
        agent: Agent[None, str] = Agent(model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}
        await agent.run(
            "What's the weather in Paris?", model_settings=settings, usage_limits=UsageLimits(output_tokens_limit=5000)
        )

    elif scenario == 'none':
        agent = Agent(model, tools=[get_weather])
        settings = {'tool_choice': 'none'}
        prompt = 'Say hello' if provider == 'anthropic' else "What's the weather in Paris?"
        await agent.run(prompt, model_settings=settings, usage_limits=UsageLimits(output_tokens_limit=5000))

    elif scenario == 'required':
        tool_defs = [make_tool_def('get_weather', 'Get weather for a city', 'city')]
        params = ModelRequestParameters(function_tools=tool_defs, allow_text_output=True)
        settings = {'tool_choice': 'required'}
        await model.request([ModelRequest.user_text_prompt("What's the weather in Paris?")], settings, params)

    elif scenario == 'list_single':
        tool_defs = [
            make_tool_def('get_weather', 'Get weather for a city', 'city'),
            make_tool_def('get_time', 'Get time in a timezone', 'timezone'),
        ]
        params = ModelRequestParameters(function_tools=tool_defs, allow_text_output=True)
        settings = {'tool_choice': ['get_weather']}
        await model.request([ModelRequest.user_text_prompt("What's the weather in Paris?")], settings, params)

    elif scenario == 'none_with_output':
        agent_with_output: Agent[None, CityInfo] = Agent(model, tools=[get_weather], output_type=CityInfo)
        settings = {'tool_choice': 'none'}
        await agent_with_output.run(
            'Tell me about Paris', model_settings=settings, usage_limits=UsageLimits(output_tokens_limit=5000)
        )

    elif scenario == 'tools_plus_output':
        agent_tpo: Agent[None, CityInfo] = Agent(model, tools=[get_weather, get_time], output_type=CityInfo)
        settings = {'tool_choice': ToolOrOutput(function_tools=['get_weather'])}
        await agent_tpo.run(
            'Get weather for Paris and summarize',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=5000),
        )
    else:
        assert_never(scenario)

    actual_tool_choice = get_tool_choice_from_cassette(vcr, provider, xai_provider)
    assert actual_tool_choice == expected_tool_choice
