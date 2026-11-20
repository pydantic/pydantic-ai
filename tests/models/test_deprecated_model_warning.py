"""Which requests emit `DeprecatedModelWarning`, and which don't.

These aren't VCR tests: the warning is emitted while a request is prepared, before anything is sent, so no provider
answer plays a part in it. Each request goes to a transport that answers 404, as the APIs answer a model they no longer
serve, which ends the run right after the warning. The models are made up and put in the deprecation tables for the
test, so no test depends on a real model's schedule.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import date

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, DeprecatedModelWarning, ModelAPIError, ModelDeprecation
from pydantic_ai._unsupported_settings import suppress_warnings
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from .. import conftest as tests_conftest
from ..conftest import ignore_deprecated_models_in_recordings, try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers._model_deprecations import ANTHROPIC_MODEL_DEPRECATIONS, OPENAI_MODEL_DEPRECATIONS
    from pydantic_ai.providers.anthropic import AnthropicProvider, AsyncAnthropicClient
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic or openai not installed')

CLAUDE = 'claude-retiring'
GPT = 'gpt-retiring'
DEPRECATION = ModelDeprecation(
    source_url='https://example.com/deprecations',
    retirement_date=date(2026, 11, 30),
    replacement='current-model',
)


@pytest.fixture(autouse=True)
def deprecated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ANTHROPIC_MODEL_DEPRECATIONS, CLAUDE, DEPRECATION)
    monkeypatch.setitem(OPENAI_MODEL_DEPRECATIONS, GPT, DEPRECATION)


def not_found_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(404, json={'error': {'message': 'model not found'}}))
    )


def anthropic_direct() -> Model:
    client = AsyncAnthropic(api_key='mock-api-key', max_retries=0, http_client=not_found_client())
    return AnthropicModel(CLAUDE, provider=AnthropicProvider(anthropic_client=client))


def anthropic_gateway() -> Model:
    provider = gateway_provider('anthropic', api_key='pylf_v1_us_mock', http_client=not_found_client())
    provider.client.max_retries = 0
    return AnthropicModel(CLAUDE, provider=provider)


def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(
        openai_client=AsyncOpenAI(api_key='mock-api-key', max_retries=0, http_client=not_found_client())
    )


def openai_gateway() -> Provider[AsyncOpenAI]:
    provider = gateway_provider('openai', api_key='pylf_v1_us_mock', http_client=not_found_client())
    provider.client.max_retries = 0
    return provider


WARNING_MODELS: dict[str, Callable[[], Model]] = {
    'anthropic': anthropic_direct,
    'gateway/anthropic': anthropic_gateway,
    'anthropic-foundry': lambda: AnthropicModel(
        CLAUDE,
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicFoundry(
                api_key='mock-api-key', resource='resource', max_retries=0, http_client=not_found_client()
            )
        ),
    ),
    'openai-chat': lambda: OpenAIChatModel(GPT, provider=openai_provider()),
    'openai-responses': lambda: OpenAIResponsesModel(GPT, provider=openai_provider()),
    'gateway/openai-chat': lambda: OpenAIChatModel(GPT, provider=openai_gateway()),
    'gateway/openai-responses': lambda: OpenAIResponsesModel(GPT, provider=openai_gateway()),
}


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('model_factory', WARNING_MODELS.values(), ids=WARNING_MODELS.keys())
async def test_request_warns_once(allow_model_requests: None, model_factory: Callable[[], Model], stream: bool):
    """A request to a deprecated model on the Anthropic or OpenAI APIs, directly, through the Gateway or on Microsoft
    Foundry, which takes Anthropic's dates, warns once.

    Creating the model and resolving its profile don't warn, or `filterwarnings = error` would fail the test before the
    `pytest.warns` block. Only the request warns, once, although the agent prepares it more than once before sending
    it, to resolve the output mode and to describe it on a span.
    """
    model = model_factory()
    assert model.profile.get('deprecation') == DEPRECATION
    agent = Agent(model)

    with pytest.warns(DeprecatedModelWarning) as record, pytest.raises(ModelAPIError):
        if stream:
            async with agent.run_stream('hello'):
                pass  # pragma: no cover
        else:
            await agent.run('hello')

    assert [str(w.message) for w in record if w.category is DeprecatedModelWarning] == [
        f"Model '{model.model_id}' is deprecated by its provider, with a retirement date of 2026-11-30. "
        "It recommends 'current-model' instead. See https://example.com/deprecations"
    ]


def test_each_request_warns(allow_model_requests: None):
    """Each request warns, and Python's warning filters decide how often it's shown.

    The default filters don't show it once per model: `run_sync` enters `warnings.catch_warnings()`, which resets the
    record of warnings already shown.
    """
    agent = Agent(anthropic_direct())
    with pytest.warns(DeprecatedModelWarning) as record:
        for _ in range(2):
            with pytest.raises(ModelAPIError):
                agent.run_sync('hello')
    assert len([w for w in record if w.category is DeprecatedModelWarning]) == 2


def anthropic_on(client: AsyncAnthropicClient) -> Model:
    return AnthropicModel(CLAUDE, provider=AnthropicProvider(anthropic_client=client))


def bedrock_client() -> AsyncAnthropicBedrock:
    return AsyncAnthropicBedrock(aws_access_key='key', aws_secret_key='secret', aws_region='us-east-1')


def azure_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key='mock-api-key', azure_endpoint='https://example.openai.azure.com', api_version='2025-04-01'
    )


SILENT_MODELS: dict[str, Callable[[], Model]] = {
    'anthropic-bedrock': lambda: anthropic_on(bedrock_client()),
    'anthropic-bedrock-mantle': lambda: anthropic_on(
        AsyncAnthropicBedrockMantle(aws_access_key='key', aws_secret_key='secret', aws_region='us-east-1')
    ),
    'anthropic-vertex': lambda: anthropic_on(
        AsyncAnthropicVertex(region='us-east5', project_id='project', access_token='token')
    ),
    'openai-chat-azure-client': lambda: OpenAIChatModel(GPT, provider=OpenAIProvider(openai_client=azure_client())),
    'openai-responses-azure-client': lambda: OpenAIResponsesModel(
        GPT, provider=OpenAIProvider(openai_client=azure_client())
    ),
    'azure': lambda: OpenAIChatModel(
        GPT,
        provider=AzureProvider(
            api_key='mock-api-key', azure_endpoint='https://example.openai.azure.com', api_version='2025-04-01'
        ),
    ),
    'openrouter': lambda: OpenAIChatModel(f'openai/{GPT}', provider=OpenRouterProvider(api_key='mock-api-key')),
}


@pytest.mark.parametrize('model_factory', SILENT_MODELS.values(), ids=SILENT_MODELS.keys())
def test_other_platforms_dont_warn(model_factory: Callable[[], Model]):
    """Platforms that serve the same models on schedules of their own don't take the Anthropic or OpenAI API's dates.

    Preparing the request is where the warning would be emitted, and `filterwarnings = error` would fail on it.
    """
    model = model_factory()
    assert model.profile.get('deprecation') is None
    model.prepare_request(None, ModelRequestParameters())


def test_openai_subscription_providers():
    """ChatGPT plan usage goes to the OpenAI API and takes its dates; the Codex backend serves its own set of models."""
    assert (OpenAIChatGPTProvider.model_profile(GPT) or {}).get('deprecation') == DEPRECATION
    assert (OpenAICodexProvider.model_profile(GPT) or {}).get('deprecation') is None
    assert 'deprecation' not in (OpenAICodexProvider.model_profile('gpt-current') or {})


def test_profile_overrides():
    """The user's `profile=` can stop the warning for one model, or add one for a model served elsewhere."""
    silenced = AnthropicModel(CLAUDE, provider=AnthropicProvider(api_key='mock-api-key'), profile={'deprecation': None})
    silenced.prepare_request(None, ModelRequestParameters())

    custom = OpenAIChatModel(
        'openai/gpt-elsewhere',
        provider=OpenRouterProvider(api_key='mock-api-key'),
        profile={'deprecation': ModelDeprecation(source_url='https://example.com/openrouter')},
    )
    with pytest.warns(DeprecatedModelWarning) as record:
        custom.prepare_request(None, ModelRequestParameters())
    assert [str(w.message) for w in record] == snapshot(
        ["Model 'openrouter:openai/gpt-elsewhere' is deprecated by its provider. See https://example.com/openrouter"]
    )


PLATFORM_DROPS: dict[str, Callable[[ModelProfile], Model]] = {
    'anthropic-bedrock': lambda profile: AnthropicModel(
        CLAUDE, provider=AnthropicProvider(anthropic_client=bedrock_client()), profile=profile
    ),
    'openai-chat-azure-client': lambda profile: OpenAIChatModel(
        GPT, provider=OpenAIProvider(openai_client=azure_client()), profile=profile
    ),
    'openai-responses-azure-client': lambda profile: OpenAIResponsesModel(
        GPT, provider=OpenAIProvider(openai_client=azure_client()), profile=profile
    ),
}


@pytest.mark.parametrize('model_factory', PLATFORM_DROPS.values(), ids=PLATFORM_DROPS.keys())
def test_profile_override_survives_platform_drop(model_factory: Callable[[ModelProfile], Model]):
    """A deprecation the user sets with `profile=` survives the drop for a platform with a schedule of its own.

    The model drops the provider's deprecation for that platform before the user's `profile=` applies, which is what
    the docs tell users to rely on to warn about a model they serve there.
    """
    custom = ModelDeprecation(source_url='https://example.com/platform')
    model = model_factory(ModelProfile(deprecation=custom))
    assert model.profile.get('deprecation') == custom
    with pytest.warns(DeprecatedModelWarning):
        model.prepare_request(None, ModelRequestParameters())


async def test_unused_fallback_doesnt_warn(allow_model_requests: None):
    """`FallbackModel` prepares each candidate's request to check it, which mustn't warn for a candidate it never sends to."""
    agent = Agent(FallbackModel(TestModel(), anthropic_direct()))
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecatedModelWarning)
        result = await agent.run('hello')
    assert result.output == snapshot('success (no tool calls)')


def test_suppression_with_settings_keeps_the_warning():
    """`suppress_warnings(settings)` silences only those settings' warnings, for a request it still sends.

    A unit test, because the caller that passes settings, MCP sampling, needs a live MCP session: its request goes out,
    so it keeps the warning, unless framework code that only prepares the request already silenced everything.
    """
    model = anthropic_direct()
    with pytest.warns(DeprecatedModelWarning), suppress_warnings(['stop_sequences']):
        model.prepare_request(None, ModelRequestParameters())

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecatedModelWarning)
        with suppress_warnings(), suppress_warnings(['stop_sequences']):
            model.prepare_request(None, ModelRequestParameters())


def test_recording_a_deprecated_model_fails(monkeypatch: pytest.MonkeyPatch):
    """The suite ignores the warning for the deprecated models its recordings use only while it replays them.

    Recording a test against one of them fails, so the test moves to a current model before it's re-recorded.
    """
    monkeypatch.setattr(tests_conftest, 'DEPRECATED_MODELS_IN_RECORDINGS', (GPT,))
    model = OpenAIChatModel(GPT, provider=OpenAIProvider(api_key='mock-api-key'))

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ignore_deprecated_models_in_recordings('none')
        model.prepare_request(None, ModelRequestParameters())

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ignore_deprecated_models_in_recordings('rewrite')
        with pytest.raises(DeprecatedModelWarning, match="Model 'openai:gpt-retiring' is deprecated"):
            model.prepare_request(None, ModelRequestParameters())
