"""Unified `thinking='max'` on the wire, recorded against each provider's API.

`'max'` is sent as the native `max` effort only where the model accepts it on the API the request goes to,
and falls back to the highest level the model does accept everywhere else, so the request succeeds.
Each case asserts on the request the live code built (through `request_capture`), and the recording
proves the provider accepted it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, ThinkingEffort

from .conftest import RequestCapture, try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.vcr]

Provider = Literal['openai-responses', 'openai-chat', 'anthropic', 'deepseek', 'moonshotai', 'openrouter']

_SKIP_MARKS: dict[Provider, pytest.MarkDecorator] = {
    'openai-responses': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
    'openai-chat': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
    'anthropic': pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),
    'deepseek': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
    'moonshotai': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
    'openrouter': pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
}

_REQUEST_PATHS: dict[Provider, str] = {
    'openai-responses': '/responses',
    'openai-chat': '/chat/completions',
    'anthropic': '/v1/messages',
    'deepseek': '/chat/completions',
    'moonshotai': '/chat/completions',
    'openrouter': '/chat/completions',
}

_EFFORT_FIELDS: dict[Provider, tuple[str, ...]] = {
    'openai-responses': ('reasoning', 'effort'),
    'openai-chat': ('reasoning_effort',),
    'anthropic': ('output_config', 'effort'),
    'deepseek': ('reasoning_effort',),
    'moonshotai': ('reasoning_effort',),
    'openrouter': ('reasoning', 'effort'),
}


@dataclass(frozen=True)
class Case:
    id: str
    provider: Provider
    model_name: str
    sent_effort: str
    """The effort the request carries."""
    thinking: ThinkingEffort = 'max'


CASES = [
    Case(id='openai-responses-gpt-6.1-sol', provider='openai-responses', model_name='gpt-6.1-sol', sent_effort='max'),
    # GPT-5.5 rejects `max` ("Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'").
    Case(id='openai-responses-gpt-5.5', provider='openai-responses', model_name='gpt-5.5', sent_effort='xhigh'),
    # OpenAI's Chat Completions API rejects `max` on every model, including those whose Responses API takes it.
    Case(id='openai-chat-gpt-6.1-sol', provider='openai-chat', model_name='gpt-6.1-sol', sent_effort='xhigh'),
    Case(id='anthropic-opus-5-5', provider='anthropic', model_name='claude-opus-5-5', sent_effort='max'),
    # Sonnet 4.6 accepts `max` but not `xhigh`.
    Case(id='anthropic-sonnet-4-6', provider='anthropic', model_name='claude-sonnet-4-6', sent_effort='max'),
    # Opus 4.5 accepts only `low`, `medium` and `high`.
    Case(id='anthropic-opus-4-5', provider='anthropic', model_name='claude-opus-4-5', sent_effort='high'),
    Case(
        id='anthropic-opus-4-5-xhigh',
        provider='anthropic',
        model_name='claude-opus-4-5',
        thinking='xhigh',
        sent_effort='high',
    ),
    # DeepSeek maps `xhigh` to `high`, so `max` is the only way to its deepest reasoning.
    Case(id='deepseek-v4-pro', provider='deepseek', model_name='deepseek-v4-pro', sent_effort='max'),
    Case(id='moonshotai-kimi-k3', provider='moonshotai', model_name='kimi-k3', sent_effort='max'),
    # OpenRouter takes `max` for every model and snaps it to the highest effort the routed model supports,
    # here a model whose own API rejects `max`.
    Case(id='openrouter-gpt-5.5', provider='openrouter', model_name='openai/gpt-5.5', sent_effort='max'),
]


def _build_model(
    case: Case,
    capture: RequestCapture,
    *,
    openai_api_key: str,
    anthropic_api_key: str,
    deepseek_api_key: str,
    moonshotai_api_key: str,
    openrouter_api_key: str,
) -> Model:
    client = capture.client
    if case.provider == 'openai-responses':
        return OpenAIResponsesModel(
            case.model_name, provider=OpenAIProvider(api_key=openai_api_key, http_client=client)
        )
    if case.provider == 'openai-chat':
        return OpenAIChatModel(case.model_name, provider=OpenAIProvider(api_key=openai_api_key, http_client=client))
    if case.provider == 'anthropic':
        return AnthropicModel(
            case.model_name, provider=AnthropicProvider(api_key=anthropic_api_key, http_client=client)
        )
    if case.provider == 'deepseek':
        return OpenAIChatModel(case.model_name, provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=client))
    if case.provider == 'moonshotai':
        return OpenAIChatModel(
            case.model_name, provider=MoonshotAIProvider(api_key=moonshotai_api_key, http_client=client)
        )
    return OpenRouterModel(case.model_name, provider=OpenRouterProvider(api_key=openrouter_api_key, http_client=client))


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=_SKIP_MARKS[c.provider]) for c in CASES])
async def test_max_effort_wire(
    case: Case,
    allow_model_requests: None,
    request_capture: RequestCapture,
    openai_api_key: str,
    anthropic_api_key: str,
    deepseek_api_key: str,
    moonshotai_api_key: str,
    openrouter_api_key: str,
):
    """The unified effort reaches the wire as the native `max` or as the highest level the model accepts."""
    model = _build_model(
        case,
        request_capture,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        moonshotai_api_key=moonshotai_api_key,
        openrouter_api_key=openrouter_api_key,
    )
    agent = Agent(model, model_settings=ModelSettings(thinking=case.thinking))

    result = await agent.run('What is 2+2? Reply with just the number.')

    assert result.output
    node: JsonValue = request_capture.body(_REQUEST_PATHS[case.provider])
    for key in _EFFORT_FIELDS[case.provider]:
        assert isinstance(node, dict)
        node = node[key]
    assert node == case.sent_effort
