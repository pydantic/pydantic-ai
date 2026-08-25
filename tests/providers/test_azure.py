import os

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import BinaryContent, DocumentUrl
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import
from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_azure_provider():
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    assert isinstance(provider, AzureProvider)
    assert provider.name == 'azure'
    assert provider.base_url == snapshot('https://project-id.openai.azure.com/openai/')
    assert isinstance(provider.client, AsyncAzureOpenAI)


def test_azure_provider_with_openai_model():
    model = OpenAIChatModel(
        model_name='gpt-4o',
        provider=AzureProvider(
            azure_endpoint='https://project-id.openai.azure.com/',
            api_version='2023-03-15-preview',
            api_key='1234567890',
        ),
    )
    assert isinstance(model, OpenAIChatModel)
    assert isinstance(model.client, AsyncAzureOpenAI)


def test_azure_provider_with_azure_openai_client():
    client = AsyncAzureOpenAI(
        api_version='2024-12-01-preview',
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
    )
    provider = AzureProvider(openai_client=client)
    assert isinstance(provider.client, AsyncAzureOpenAI)


def test_azure_provider_api_key_required_when_absent():
    """`AzureProvider.api_key` raises when no API key is available (e.g. a client built for Entra auth)."""
    provider = AzureProvider(
        api_version='2024-12-01-preview',
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
    )
    provider._api_key = None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(UserError, match='has no API key'):
        _ = provider.api_key


def test_azure_provider_voice_live_api_key_required_when_absent():
    """`AzureProvider.voice_live_api_key` raises when no key is available (e.g. an Entra-only client)."""
    provider = AzureProvider(
        api_version='2024-12-01-preview',
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
    )
    provider._voice_live_api_key = None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(UserError, match='Voice Live requires API-key authentication'):
        _ = provider.voice_live_api_key


@pytest.mark.parametrize('auth', ['api-key-provider', 'token', 'token-provider'])
def test_azure_provider_realtime_rejects_entra_auth(auth: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)

    async def token_provider() -> str:  # pragma: no cover - the SDK stores it, nothing here calls it
        return 'token'

    if auth == 'api-key-provider':
        client = AsyncAzureOpenAI(
            api_version='2024-12-01-preview',
            azure_endpoint='https://project-id.openai.azure.com/',
            api_key=token_provider,
        )
    elif auth == 'token':
        client = AsyncAzureOpenAI(
            api_version='2024-12-01-preview',
            azure_endpoint='https://project-id.openai.azure.com/',
            azure_ad_token='token',
        )
    else:
        client = AsyncAzureOpenAI(
            api_version='2024-12-01-preview',
            azure_endpoint='https://project-id.openai.azure.com/',
            azure_ad_token_provider=token_provider,
        )
    provider = AzureProvider(openai_client=client)

    with pytest.raises(UserError, match='has no API key'):
        _ = provider.api_key


def test_azure_provider_with_http_client():
    import httpx2

    http_client = httpx2.AsyncClient()
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
        api_version='2024-12-01-preview',
        http_client=http_client,
    )
    assert isinstance(provider.client, AsyncAzureOpenAI)
    assert provider._own_http_client is None  # pyright: ignore[reportPrivateUsage]


async def test_azure_provider_call(allow_model_requests: None):
    api_key = os.getenv('AZURE_OPENAI_API_KEY', '1234567890')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

    provider = AzureProvider(
        api_key=api_key,
        azure_endpoint='https://pydanticai7521574644.openai.azure.com/',
        api_version=api_version,
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


def test_azure_provider_model_profile(mocker: MockerFixture):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )

    ns = 'pydantic_ai.providers.azure'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    grok_model_profile_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)

    meta_profile = provider.model_profile('Llama-4-Scout-17B-16E')
    meta_model_profile_mock.assert_called_with('llama-4-scout-17b-16e')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('Meta-Llama-3.1-405B-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('DeepSeek-R1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-medium-2505')
    mistral_model_profile_mock.assert_called_with('mistral-medium-2505')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistralai-Mixtral-8x22B-Instruct-v0-1')
    mistral_model_profile_mock.assert_called_with('mixtral-8x22b-instruct-v0-1')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere-command-a')
    cohere_model_profile_mock.assert_called_with('command-a')
    assert cohere_profile is not None
    assert cohere_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    grok_profile = provider.model_profile('grok-3')
    grok_model_profile_mock.assert_called_with('grok-3')
    assert grok_profile is not None
    assert grok_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    openai_profile = provider.model_profile('o4-mini')
    openai_model_profile_mock.assert_called_with('o4-mini')
    assert openai_profile is not None
    assert openai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    openai_model_profile_mock.assert_called_with('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


async def test_azure_document_input_not_supported(allow_model_requests: None):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match=r"Azure's Chat Completions API does not support document input.*OpenAIResponsesModel",
    ):
        await agent.run(
            [
                'Summarize this document',
                BinaryContent(data=b'%PDF-1.4 test', media_type='application/pdf'),
            ]
        )


async def test_azure_document_url_input_not_supported(allow_model_requests: None):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match=r"Azure's Chat Completions API does not support document input.*OpenAIResponsesModel",
    ):
        await agent.run(['Summarize this document', DocumentUrl(url='https://example.com/test.pdf')])


def test_azure_provider_foundry_serverless_endpoint():
    provider = AzureProvider(
        azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
        api_key='test-key-123',
    )
    assert provider.name == 'azure'
    # Serverless model endpoints reject the `api-version` query parameter, so we
    # must use plain AsyncOpenAI rather than AsyncAzureOpenAI.
    assert type(provider.client) is AsyncOpenAI
    assert provider.base_url == 'https://gpt-oss-120b.eastus2.models.ai.azure.com/v1/'


def test_azure_provider_v1_endpoint_rejects_api_version():
    with pytest.raises(UserError, match='`api_version` must not be set'):
        AzureProvider(
            azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
            api_version='2024-12-01-preview',
            api_key='test-key-123',
        )


def test_azure_provider_openai_v1_ga_endpoint():
    # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/openai/v1/',
        api_key='test-key-123',
    )
    assert type(provider.client) is AsyncOpenAI
    assert provider.base_url == 'https://project-id.openai.azure.com/openai/v1/'


def test_azure_provider_reads_voice_live_env_prefix(monkeypatch: pytest.MonkeyPatch):
    # Azure AI Voice Live is a distinct resource with `AZURE_VOICELIVE_*` credentials; the provider falls
    # back to that prefix (endpoint, key, and api-version) so a Voice Live user doesn't also need to set
    # the `AZURE_OPENAI_*` variables.
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://my-voice-live.cognitiveservices.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'voice-live-key')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-04-10')

    provider = AzureProvider()
    assert provider.azure_endpoint == 'https://my-voice-live.cognitiveservices.azure.com'
    assert provider.api_key == 'voice-live-key'
    assert provider.voice_live_endpoint == 'https://my-voice-live.cognitiveservices.azure.com'
    assert provider.voice_live_api_key == 'voice-live-key'
    assert provider.voice_live_api_version == '2026-04-10'


def test_azure_provider_openai_client_reads_voice_live_env_prefix(monkeypatch: pytest.MonkeyPatch):
    """The `openai_client` branch resolves Voice Live from `AZURE_VOICELIVE_*`, like the other branch.

    Regression: `AzureProvider(openai_client=...)` passed the `voice_live_*` arguments straight through
    without consulting the environment, so `AZURE_VOICELIVE_ENDPOINT` / `AZURE_VOICELIVE_API_KEY` were
    ignored and Voice Live silently reused the Azure OpenAI client's resource.
    """
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://my-voice-live.cognitiveservices.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'voice-live-key')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-04-10')

    client = AsyncAzureOpenAI(
        api_version='2024-12-01-preview',
        azure_endpoint='https://ga-resource.openai.azure.com/',
        api_key='ga-key',
    )
    provider = AzureProvider(openai_client=client)
    assert provider.voice_live_endpoint == 'https://my-voice-live.cognitiveservices.azure.com'
    assert provider.voice_live_api_key == 'voice-live-key'
    assert provider.voice_live_api_version == '2026-04-10'

    # Without the Voice Live environment, Voice Live falls back to the client's Azure OpenAI resource.
    for name in ('AZURE_VOICELIVE_ENDPOINT', 'AZURE_VOICELIVE_API_KEY'):
        monkeypatch.delenv(name)
    provider = AzureProvider(openai_client=client)
    assert provider.voice_live_endpoint == 'https://ga-resource.openai.azure.com'
    assert provider.voice_live_api_key == 'ga-key'


def test_azure_provider_voice_live_only_construction(monkeypatch: pytest.MonkeyPatch):
    """A Voice-Live-only provider constructs from its own arguments or environment alone.

    Regression: the required endpoint/key checks only consulted the Azure OpenAI sources and the
    `AZURE_VOICELIVE_*` environment, so `AzureProvider(voice_live_endpoint=..., voice_live_api_key=...)`
    raised — the documented explicit form was unusable on its own. The api-version check likewise
    re-read the environment instead of the already-resolved Voice Live version, which has a default, so
    a Voice-Live-only environment raised for a version it did not need to be given.
    """
    for name in (
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'OPENAI_API_VERSION',
        'AZURE_VOICELIVE_ENDPOINT',
        'AZURE_VOICELIVE_API_KEY',
        'AZURE_VOICELIVE_API_VERSION',
    ):
        monkeypatch.delenv(name, raising=False)

    # Explicit arguments only, nothing in the environment.
    provider = AzureProvider(voice_live_endpoint='https://vl.services.ai.azure.com', voice_live_api_key='vl-key')
    assert provider.voice_live_endpoint == 'https://vl.services.ai.azure.com'
    assert provider.voice_live_api_key == 'vl-key'
    assert provider.voice_live_api_version == '2026-04-10'

    # Environment only, and without `AZURE_VOICELIVE_API_VERSION` — the default applies.
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://vl.services.ai.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'vl-key')
    provider = AzureProvider()
    assert provider.voice_live_api_version == '2026-04-10'


def test_azure_provider_voice_live_key_is_not_borrowed_by_the_data_plane(monkeypatch: pytest.MonkeyPatch):
    """An Azure OpenAI resource is never authenticated with the Voice Live key.

    Regression: with an Azure OpenAI endpoint configured but its key missing, `AZURE_VOICELIVE_API_KEY`
    was installed into the `AsyncAzureOpenAI` client, so ordinary chat requests went to the Azure OpenAI
    resource holding another resource's credential instead of reporting the incomplete configuration.
    """
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'vl-key')

    with pytest.raises(UserError, match='AZURE_OPENAI_API_KEY'):
        AzureProvider()


def test_azure_provider_voice_live_api_version_is_not_borrowed_by_the_data_plane(
    monkeypatch: pytest.MonkeyPatch,
):
    """`AZURE_VOICELIVE_API_VERSION` must not version the Azure OpenAI chat client.

    Regression: with an Azure OpenAI resource configured but `OPENAI_API_VERSION` unset, the Voice Live
    api-version was handed to `AsyncAzureOpenAI`. The two are unrelated version schemes, so every chat
    request went out with a version the data plane doesn't recognize.
    """
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'ga-key')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-04-10')

    with pytest.raises(UserError, match='OPENAI_API_VERSION'):
        AzureProvider()

    # It still stands in when the whole resource is the Voice Live one, so a Voice-Live-only
    # configuration keeps constructing (the chat client it builds is a formality there).
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT')
    monkeypatch.delenv('AZURE_OPENAI_API_KEY')
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://vl.services.ai.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'vl-key')
    assert AzureProvider().voice_live_api_version == '2026-04-10'


def test_azure_provider_voice_live_credentials_are_coherent(monkeypatch: pytest.MonkeyPatch):
    # With both resource sets configured, the Azure OpenAI (GA) credentials and the Voice Live credentials
    # are each drawn as one coherent set — never mixed (e.g. GA endpoint with the Voice Live key).
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'ga-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://vl.services.ai.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'vl-key')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2026-06-01-preview')

    provider = AzureProvider()
    assert (provider.azure_endpoint, provider.api_key) == ('https://ga.openai.azure.com', 'ga-key')
    assert provider.voice_live_endpoint == 'https://vl.services.ai.azure.com'
    assert provider.voice_live_api_key == 'vl-key'
    assert provider.voice_live_api_version == '2026-06-01-preview'


def test_azure_provider_voice_live_explicit_arguments_beat_env(monkeypatch: pytest.MonkeyPatch):
    # Explicit Voice Live arguments win over the `AZURE_VOICELIVE_*` environment, matching the
    # precedence every other `AzureProvider` field uses. Regression: the environment used to win, so a
    # stale variable silently redirected a session configured in code to a different resource.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'ga-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    monkeypatch.setenv('AZURE_VOICELIVE_ENDPOINT', 'https://stale.services.ai.azure.com')
    monkeypatch.setenv('AZURE_VOICELIVE_API_KEY', 'stale-key')
    monkeypatch.setenv('AZURE_VOICELIVE_API_VERSION', '2020-01-01')

    provider = AzureProvider(
        voice_live_endpoint='https://chosen.services.ai.azure.com/',
        voice_live_api_key='chosen-key',
        voice_live_api_version='2026-06-01-preview',
    )
    assert provider.voice_live_endpoint == 'https://chosen.services.ai.azure.com'
    assert provider.voice_live_api_key == 'chosen-key'
    assert provider.voice_live_api_version == '2026-06-01-preview'
    # The Azure OpenAI (GA) set is untouched by the Voice Live arguments.
    assert (provider.azure_endpoint, provider.api_key) == ('https://ga.openai.azure.com', 'ga-key')


def test_azure_provider_voice_live_explicit_arguments_without_env(monkeypatch: pytest.MonkeyPatch):
    # The other direction: with no `AZURE_VOICELIVE_*` set, the explicit arguments are what configure
    # Voice Live — the capability the docs promise and the only way to set its api-version in code.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'ga-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    monkeypatch.delenv('AZURE_VOICELIVE_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_VOICELIVE_API_KEY', raising=False)
    monkeypatch.delenv('AZURE_VOICELIVE_API_VERSION', raising=False)

    provider = AzureProvider(
        voice_live_endpoint='https://chosen.services.ai.azure.com',
        voice_live_api_key='chosen-key',
        voice_live_api_version='2026-06-01-preview',
    )
    assert provider.voice_live_endpoint == 'https://chosen.services.ai.azure.com'
    assert provider.voice_live_api_key == 'chosen-key'
    assert provider.voice_live_api_version == '2026-06-01-preview'


def test_azure_provider_voice_live_falls_back_to_openai_resource(monkeypatch: pytest.MonkeyPatch):
    # With only the Azure OpenAI resource configured, Voice Live reuses it and defaults the api-version.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://ga.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'ga-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')
    monkeypatch.delenv('AZURE_VOICELIVE_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_VOICELIVE_API_KEY', raising=False)
    monkeypatch.delenv('AZURE_VOICELIVE_API_VERSION', raising=False)

    provider = AzureProvider()
    assert provider.voice_live_endpoint == 'https://ga.openai.azure.com'
    assert provider.voice_live_api_key == 'ga-key'
    assert provider.voice_live_api_version == '2026-04-10'


def test_azure_provider_for_realtime_normalizes_bare_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    provider = AzureProvider.for_realtime(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='test-key-123',
    )

    assert type(provider.client) is AsyncOpenAI
    assert provider.azure_endpoint == 'https://project-id.openai.azure.com/openai/v1'
    assert provider.base_url == 'https://project-id.openai.azure.com/openai/v1/'


def test_azure_provider_for_realtime_preserves_api_version_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')

    provider = AzureProvider.for_realtime(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='test-key-123',
    )

    assert isinstance(provider.client, AsyncAzureOpenAI)
    assert provider.azure_endpoint == 'https://project-id.openai.azure.com'


def test_azure_provider_foundry_serverless_with_openai_model():
    model = OpenAIChatModel(
        model_name='gpt-oss-120b',
        provider=AzureProvider(
            azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
            api_key='test-key-123',
        ),
    )
    assert type(model.client) is AsyncOpenAI


def test_azure_mistral_model_profile_disables_max_completion_tokens():
    """Reported for Azure AI Foundry's Mistral gateway (see #6593): it rejects
    `max_completion_tokens` with a 422 and accepts the legacy `max_tokens` field.

    The profile must set `openai_chat_supports_max_completion_tokens=False` so the
    `max_tokens` setting is sent as the legacy `max_tokens` field instead.
    """
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )

    # Mistral-family models must disable max_completion_tokens.
    mistral_profile = provider.model_profile('mistral-medium-2505')
    assert mistral_profile is not None
    assert mistral_profile.get('openai_chat_supports_max_completion_tokens') is False

    mistralai_profile = provider.model_profile('mistralai-Mixtral-8x22B-Instruct-v0-1')
    assert mistralai_profile is not None
    assert mistralai_profile.get('openai_chat_supports_max_completion_tokens') is False

    ministral_profile = provider.model_profile('ministral-3b')
    assert ministral_profile is not None
    assert ministral_profile.get('openai_chat_supports_max_completion_tokens') is False

    magistral_profile = provider.model_profile('magistral-small-latest')
    assert magistral_profile is not None
    assert magistral_profile.get('openai_chat_supports_max_completion_tokens') is False

    # Non-Mistral models must NOT be affected.
    openai_profile = provider.model_profile('gpt-4o')
    assert openai_profile is not None
    # Default is True (or absent, which falls through to True in the model code).
    assert openai_profile.get('openai_chat_supports_max_completion_tokens', True) is True

    deepseek_profile = provider.model_profile('DeepSeek-R1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('openai_chat_supports_max_completion_tokens', True) is True


async def test_azure_mistral_sends_max_tokens_not_max_completion_tokens(allow_model_requests: None):
    """Reported for Azure AI Foundry's Mistral gateway (see #6593): the model must send
    `max_tokens`, not `max_completion_tokens`.
    """
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    model = OpenAIChatModel('mistral-medium-2505', provider=provider)

    # Verify the profile has the correct flag set.
    profile = model.profile
    assert profile.get('openai_chat_supports_max_completion_tokens') is False


@pytest.mark.parametrize('model_name', ['Ministral-3B', 'magistral-small-latest'])
async def test_azure_mistral_family_sends_max_tokens(allow_model_requests: None, model_name: str):
    """Wire-level regression: a Mistral-family model with no `mistral` prefix (Ministral,
    Magistral) must still route the `max_tokens` setting to the legacy `max_tokens` field
    and omit `max_completion_tokens`. Uses a mock client because VCR matchers ignore the
    request body. Reported for Azure AI Foundry's Mistral gateway (see #6593).
    """
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Resolve the profile exactly as the Azure provider would, then confirm the flag.
    profile = AzureProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_chat_supports_max_completion_tokens') is False

    model = OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model, model_settings=ModelSettings(max_tokens=100))
    await agent.run('Hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['max_tokens'] == 100
    assert 'max_completion_tokens' not in kwargs
