from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.realtime import infer_realtime_model

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    # Inferring the xAI and Google realtime models eagerly constructs their providers, which import
    # the `xai-sdk` and `google-genai` SDKs, so this dispatch test only runs when both are installed.
    import google.genai  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import xai_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]


@pytest.mark.skipif(not imports_successful(), reason='xai-sdk / google-genai not installed')
def test_infer_realtime_models(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')
    env.set('XAI_API_KEY', 'test')
    env.set('GOOGLE_API_KEY', 'test')
    env.set('AZURE_VOICELIVE_ENDPOINT', 'https://resource.services.ai.azure.com')
    env.set('AZURE_VOICELIVE_API_VERSION', '2026-04-10')
    env.set('AZURE_VOICELIVE_API_KEY', 'test')

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

    azure_model = infer_realtime_model('azure-voicelive:gpt-realtime')
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
    assert model.model_name == 'gpt-realtime'
    # The provider carries the gateway base URL, so the realtime WebSocket handshake connects through
    # the gateway rather than directly to OpenAI.
    assert getattr(model, '_provider').base_url == 'https://gateway.pydantic.dev/proxy/openai/'


def test_infer_realtime_model_unknown_provider() -> None:
    with pytest.raises(UserError, match='Supported providers are `openai`, `azure-voicelive`, `xai`, and `google`'):
        infer_realtime_model('anthropic:voice')

    # A non-OpenAI gateway route is rejected: xAI isn't a gateway upstream and Gemini Live isn't the
    # OpenAI protocol, so only `gateway/openai` is proxied for realtime.
    with pytest.raises(UserError, match='gateway/openai'):
        infer_realtime_model('gateway/google:gemini-2.5-flash-native-audio-latest')
