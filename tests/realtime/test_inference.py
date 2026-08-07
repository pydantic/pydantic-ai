from __future__ import annotations as _annotations

import subprocess
import sys

import pytest

from pydantic_ai import Agent, realtime as realtime_module
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import DeferredToolRequestsEvent, DeferredToolResultsEvent
from pydantic_ai.realtime import AzureRealtimeModel, codec as realtime_codec, infer_realtime_model
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

from ..conftest import TestEnv


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
    assert realtime_module.DeferredToolRequestsEvent is DeferredToolRequestsEvent
    assert realtime_module.DeferredToolResultsEvent is DeferredToolResultsEvent
    assert 'SessionUsageEvent' not in realtime_module.__all__
    assert 'SessionUsageEvent' in realtime_codec.__all__


def test_infer_realtime_models(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')
    env.set('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    env.set('AZURE_OPENAI_API_KEY', 'test')

    # Each provider prefix must select its own concrete model class, not just carry the suffix through
    # as `model_name` (which a wrong-class result would also satisfy).
    openai_model = infer_realtime_model('openai:gpt-realtime')
    assert type(openai_model).__name__ == 'OpenAIRealtimeModel'
    assert openai_model.model_name == 'gpt-realtime'

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


def test_azure_rejects_non_azure_provider(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')

    with pytest.raises(UserError, match='requires an `AzureProvider`'):
        AzureRealtimeModel('gpt-realtime', provider='openai')


def test_infer_realtime_model_unknown_provider() -> None:
    with pytest.raises(UserError, match='Supported providers are `openai` and `azure`'):
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
    with pytest.raises(UserError, match='Unknown realtime model provider'):
        infer_realtime_model('gateway/groq:whisper-voice')
