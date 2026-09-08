from __future__ import annotations as _annotations

import subprocess
import sys
from collections.abc import Iterator
from typing import Any, get_args

import pytest

from pydantic_ai import Agent, messages as messages_module, realtime as realtime_module
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.realtime import codec as realtime_codec, infer_realtime_model
from pydantic_ai.realtime.azure import AzureRealtimeModel
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    # Inferring the xAI and Google realtime models eagerly constructs their providers, which import
    # the `xai-sdk` and `google-genai` SDKs, so this dispatch test only runs when both are installed.
    import google.genai  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import xai_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime.azure import (
        LatestAzureRealtimeModelNames,
        LatestAzureRealtimeTranscriptionModelNames,
    )
    from pydantic_ai.realtime.google import LatestGoogleRealtimeModelNames
    from pydantic_ai.realtime.model import KnownRealtimeModelName
    from pydantic_ai.realtime.openai import (
        LatestOpenAIRealtimeModelNames,
        LatestOpenAIRealtimeTranscriptionModelNames,
    )
    from pydantic_ai.realtime.settings import KnownRealtimeTranscriptionModelName
    from pydantic_ai.realtime.xai import LatestXaiRealtimeModelNames, LatestXaiRealtimeTranscriptionModelNames


@pytest.mark.skipif(not imports_successful(), reason='realtime provider packages were not installed')
def test_known_realtime_model_names() -> None:  # pragma: lax no cover
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    generated_names = sorted(
        [f'openai:{name}' for name in get_model_names(LatestOpenAIRealtimeModelNames)]
        + [f'azure:{name}' for name in get_model_names(LatestAzureRealtimeModelNames)]
        + [f'xai:{name}' for name in get_model_names(LatestXaiRealtimeModelNames)]
        + [f'google:{name}' for name in get_model_names(LatestGoogleRealtimeModelNames)]
    )
    assert generated_names == sorted(get_args(KnownRealtimeModelName.__value__))

    generated_transcription_names = sorted(
        ['auto']
        + list(get_model_names(LatestOpenAIRealtimeTranscriptionModelNames))
        + list(get_model_names(LatestXaiRealtimeTranscriptionModelNames))
        + list(get_model_names(LatestAzureRealtimeTranscriptionModelNames))
    )
    assert generated_transcription_names == sorted(get_args(KnownRealtimeTranscriptionModelName.__value__))


def test_star_import_does_not_load_optional_providers() -> None:
    code = """
import sys

class BlockOpenAI:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'openai' or fullname.startswith('openai.'):
            raise ModuleNotFoundError("No module named 'openai'")

sys.meta_path.insert(0, BlockOpenAI())
from pydantic_ai.realtime import *
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_realtime_event_exports_match_public_layers() -> None:
    # The shared message/part events a session yields are not realtime-specific, so they are
    # exported from `pydantic_ai.messages` and the root `pydantic_ai` — never re-exported here.
    # (The `Realtime*Event` control-plane events also live in `pydantic_ai.messages`, for history
    # serialization, but realtime is their home so they *are* exported here.)
    shared_message_events = {
        'SpeechPart',
        'SpeechPartDelta',
        'DeferredToolRequestsEvent',
        'DeferredToolResultsEvent',
        'FunctionToolCallEvent',
        'FunctionToolResultEvent',
        'PartDeltaEvent',
        'PartEndEvent',
        'PartStartEvent',
    }
    assert not shared_message_events & set(realtime_module.__all__)
    assert all(hasattr(messages_module, name) for name in shared_message_events)
    assert 'SessionUsage' not in realtime_module.__all__
    assert 'SessionUsage' in realtime_codec.__all__


@pytest.mark.skipif(not imports_successful(), reason='xai-sdk / google-genai not installed')
def test_infer_realtime_models(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')
    env.set('XAI_API_KEY', 'test')
    env.set('GOOGLE_API_KEY', 'test')
    env.set('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    env.set('AZURE_OPENAI_API_KEY', 'test')

    # Each provider prefix must select its own concrete model class, not just carry the suffix through
    # as `model_name` (which a wrong-class result would also satisfy).
    openai_model = infer_realtime_model('openai:gpt-realtime')
    assert type(openai_model).__name__ == 'OpenAIRealtimeModel'
    assert openai_model.model_name == 'gpt-realtime'

    xai_model = infer_realtime_model('xai:grok-voice-latest')
    assert type(xai_model).__name__ == 'XaiRealtimeModel'
    assert xai_model.model_name == 'grok-voice-latest'

    google_model = infer_realtime_model('google:gemini-2.5-flash-native-audio-latest')
    assert type(google_model).__name__ == 'GoogleRealtimeModel'
    assert google_model.model_name == 'gemini-2.5-flash-native-audio-latest'

    # `google-cloud:` selects Vertex AI directly (no gateway), exactly as in `infer_model`.
    env.set('GOOGLE_CLOUD_PROJECT', 'test-project')
    env.set('GOOGLE_CLOUD_LOCATION', 'us-central1')
    vertex_model = infer_realtime_model('google-cloud:gemini-live-2.5-flash')
    assert type(vertex_model).__name__ == 'GoogleRealtimeModel'
    assert vertex_model.model_name == 'gemini-live-2.5-flash'
    assert getattr(vertex_model, '_provider').client.vertexai

    azure_model = infer_realtime_model('azure:gpt-realtime')
    assert type(azure_model).__name__ == 'AzureRealtimeModel'
    assert azure_model.model_name == 'gpt-realtime'


def test_infer_realtime_model_gateway_openai(env: TestEnv) -> None:
    # `gateway/openai:...` routes the OpenAI realtime protocol through the Pydantic AI Gateway: an
    # `OpenAIRealtimeModel` whose provider derives its base URL and key from `gateway_provider`.
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')

    model = infer_realtime_model('gateway/openai:gpt-realtime')
    # Name-check the class (rather than importing it) to keep this dispatch test light, matching the
    # cases above.
    assert type(model).__name__ == 'OpenAIRealtimeModel'
    assert isinstance(model, OpenAIRealtimeModel)
    assert model.model_name == 'gpt-realtime'
    # The provider carries the gateway base URL, so the realtime WebSocket handshake connects through
    # the gateway rather than directly to OpenAI.
    assert getattr(model, '_provider').base_url == 'https://gateway.pydantic.dev/proxy/openai/'
    assert '/proxy/openai/realtime' in model._realtime_url()  # pyright: ignore[reportPrivateUsage]

    direct_model = OpenAIRealtimeModel('gpt-realtime')
    assert direct_model._realtime_url().split('?', 1)[0] == 'wss://api.openai.com/v1/realtime'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not imports_successful(), reason='xai-sdk / google-genai not installed')
def test_infer_realtime_model_gateway_google(env: TestEnv) -> None:
    # `gateway/google:...` (and its `gateway/google-cloud` alias) route Gemini Live through the gateway's
    # Vertex upstream: a `GoogleRealtimeModel` whose provider derives its base URL and key from
    # `gateway_provider`, with the gateway's bearer auth added to the WebSocket handshake.
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')

    for route in ('gateway/google', 'gateway/google-cloud'):
        model = infer_realtime_model(f'{route}:gemini-live-2.5-flash')
        # Name-check the class (rather than importing it) to keep this dispatch test light.
        assert type(model).__name__ == 'GoogleRealtimeModel'
        assert model.model_name == 'gemini-live-2.5-flash'
        # Both shorthands collapse onto the gateway's Google Cloud (Vertex) route, so the handshake
        # connects through the gateway rather than directly to Vertex.
        assert getattr(model, '_provider').base_url == 'https://gateway.pydantic.dev/proxy/google-vertex'


@pytest.mark.skipif(not imports_successful(), reason='xai-sdk / google-genai not installed')
def test_infer_realtime_model_provider_factory(env: TestEnv) -> None:
    # The realtime counterpart of `infer_model(..., provider_factory)`: the factory is called with the
    # `provider:` prefix exactly as written (gateway routes un-normalized), and every branch — not just
    # `openai` — uses the provider it returns. Nothing is read from the environment, so credentials
    # can come from the application (e.g. per-user keys) rather than `*_API_KEY` variables.
    for name in ('OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'XAI_API_KEY', 'GOOGLE_API_KEY'):
        env.remove(name)
    providers: dict[str, Provider[Any]] = {
        'openai': OpenAIProvider(api_key='factory-key'),
        'azure': AzureProvider.for_realtime(azure_endpoint='https://factory.openai.azure.com', api_key='factory-key'),
        'xai': XaiProvider(api_key='factory-key'),
        'google': GoogleProvider(api_key='factory-key'),
        'gateway/openai': gateway_provider(
            'openai', api_key='factory-key', base_url='https://gateway.pydantic.dev/proxy'
        ),
        'gateway/google': gateway_provider(
            'google-cloud', api_key='factory-key', base_url='https://gateway.pydantic.dev/proxy'
        ),
    }
    seen: list[str] = []

    def provider_factory(provider_name: str) -> Provider[Any]:
        seen.append(provider_name)
        return providers[provider_name]

    for model_id, expected_class in [
        ('openai:gpt-realtime', 'OpenAIRealtimeModel'),
        ('azure:gpt-realtime', 'AzureRealtimeModel'),
        ('xai:grok-voice-latest', 'XaiRealtimeModel'),
        ('google:gemini-live-2.5-flash', 'GoogleRealtimeModel'),
        ('gateway/openai:gpt-realtime', 'OpenAIRealtimeModel'),
        ('gateway/google:gemini-live-2.5-flash', 'GoogleRealtimeModel'),
    ]:
        model = infer_realtime_model(model_id, provider_factory)
        assert type(model).__name__ == expected_class, model_id
        # Identity: the factory's provider is used as-is, not re-inferred from its name.
        assert getattr(model, '_provider') is providers[model_id.partition(':')[0]], model_id
    assert seen == ['openai', 'azure', 'xai', 'google', 'gateway/openai', 'gateway/google']

    # The Azure realtime URL comes from the factory's endpoint, with no `AZURE_OPENAI_ENDPOINT` set.
    azure_model = infer_realtime_model('azure:gpt-realtime', provider_factory)
    assert isinstance(azure_model, AzureRealtimeModel)
    assert azure_model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://factory.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )


def test_infer_realtime_model_preserves_custom_provider_factory_error() -> None:
    def provider_factory(_provider_name: str) -> Provider[Any]:
        raise ValueError('custom provider error')

    with pytest.raises(ValueError, match='custom provider error'):
        infer_realtime_model('openai:gpt-realtime', provider_factory)


def test_infer_realtime_model_provider_factory_unknown_provider() -> None:
    # A provider with no realtime model is still rejected, whatever the factory returns for it.
    seen: list[str] = []

    def provider_factory(provider_name: str) -> Provider[Any]:
        seen.append(provider_name)
        return OpenAIProvider(api_key='factory-key')

    with pytest.raises(UserError, match='Unknown realtime model provider'):
        infer_realtime_model('anthropic:voice', provider_factory)
    with pytest.raises(UserError, match='cannot be routed through the Pydantic AI Gateway'):
        infer_realtime_model('gateway/groq:whisper-voice', provider_factory)
    assert seen == ['anthropic', 'gateway/groq']


def test_azure_rejects_non_azure_provider(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')

    with pytest.raises(UserError, match='requires an `AzureProvider`'):
        AzureRealtimeModel('gpt-realtime', provider='openai')


def test_infer_realtime_model_unknown_provider() -> None:
    with pytest.raises(
        UserError, match='Supported providers are `openai`, `azure`, `xai`, `google`, and `google-cloud`'
    ):
        infer_realtime_model('anthropic:voice')

    with pytest.raises(UserError, match=r'use the `provider:model` format .*; got \'openai\''):
        infer_realtime_model('openai')

    with pytest.raises(UserError, match=r'use the `provider:model` format .*; got \'openai:\''):
        infer_realtime_model('openai:')


@pytest.mark.anyio
async def test_agent_realtime_session_infers_string_model() -> None:
    agent: Agent[None, str] = Agent()
    with pytest.raises(UserError, match='Unknown realtime model'):
        async with agent.realtime('unknown:voice').session():
            pass  # pragma: no cover

    # A gateway route with no realtime support is rejected before any provider is built: Groq is a
    # gateway upstream but has no realtime model, so `gateway/groq` isn't a supported realtime route.
    with pytest.raises(UserError, match='cannot be routed through the Pydantic AI Gateway'):
        infer_realtime_model('gateway/groq:whisper-voice')
