"""Which stateless signed or encrypted round-trip data each model sends back, by canonical system.

These are unit tests of `Model.accepts_stateless_artifacts` and the request each adapter renders from a history
recorded by another provider. They build models offline and render with the adapters' message mapping, because
the claim is about which history data goes out, which a cassette's request matcher doesn't check; the live
round-trips across systems are in `tests/test_canonical_system_live.py`.
"""

from __future__ import annotations as _annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai._canonical_system import accepts_stateless_artifact, stateless_artifact_system
from pydantic_ai.messages import (
    Citation,
    CompactionPart,
    ContentCitationAnchor,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    WebCitationSource,
    _post_compaction_window_for_response,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrock, AsyncAnthropicFoundry, AsyncAnthropicVertex
    from google.genai import Client
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider
    from pydantic_ai.providers.xai import XaiProvider

    class ClaudeCodeProvider(AnthropicProvider):
        """A third-party provider that only renames `AnthropicProvider`, like a Claude Code subscription plugin."""

        @property
        def name(self) -> str:
            return 'claude-code'

    class GatewayOpenAIProvider(OpenAIProvider):
        """The documented custom-gateway subclass of `OpenAIProvider`, which talks to another service."""

        @property
        def name(self) -> str:
            return 'my-gateway'


pytestmark = pytest.mark.skipif(not imports_successful(), reason='model provider SDKs not installed')

SIGNATURE = base64.b64encode(b'signature').decode()
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/'
VERTEX_URL = 'https://us-central1-aiplatform.googleapis.com/'


# Providers and models, built offline.


def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key='test-key')


def azure_client_openai_provider() -> OpenAIProvider:
    return OpenAIProvider(
        openai_client=AsyncAzureOpenAI(
            api_key='test-key', azure_endpoint='https://resource.openai.azure.com', api_version='2025-04-01-preview'
        )
    )


def codex_provider() -> OpenAICodexProvider:
    return OpenAICodexProvider(OpenAICodexCredentials(access_token='a', refresh_token='r', account_id='acct'))


def bedrock_provider() -> BedrockProvider:
    return BedrockProvider(api_key='test-key', region_name='us-east-1')


def gemini_api_model() -> GoogleModel:
    return GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test-key'))


def vertex_client_google_model() -> GoogleModel:
    """A Google Cloud client in `GoogleProvider`, so `system` is `'google'` but the transport is Google Cloud."""
    return GoogleModel(
        'gemini-3-flash-preview', provider=GoogleProvider(client=Client(vertexai=True, api_key='test-key'))
    )


def google_cloud_model() -> GoogleModel:
    return GoogleModel('gemini-3-flash-preview', provider=GoogleCloudProvider(api_key='test-key'))


def bedrock_claude_model() -> BedrockConverseModel:
    return BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider())


def bedrock_nova_model() -> BedrockConverseModel:
    return BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider())


RECEIVERS: dict[str, Callable[[], Model]] = {
    'openai': lambda: OpenAIResponsesModel('gpt-5.5', provider=openai_provider()),
    'openai-codex': lambda: OpenAIResponsesModel('gpt-5.5', provider=codex_provider()),
    'openai+azure-client': lambda: OpenAIResponsesModel('gpt-5.5', provider=azure_client_openai_provider()),
    'azure': lambda: OpenAIResponsesModel(
        'gpt-5.5',
        provider=AzureProvider(
            azure_endpoint='https://resource.openai.azure.com', api_version='2025-04-01-preview', api_key='test-key'
        ),
    ),
    'my-gateway': lambda: OpenAIResponsesModel('gpt-5.5', provider=GatewayOpenAIProvider(api_key='test-key')),
    'anthropic': lambda: AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key')),
    'claude-code': lambda: AnthropicModel('claude-sonnet-4-5', provider=ClaudeCodeProvider(api_key='test-key')),
    'bedrock-claude': bedrock_claude_model,
    'bedrock-nova': bedrock_nova_model,
    'gemini-api': gemini_api_model,
    'google+vertex-client': vertex_client_google_model,
    'google-cloud': google_cloud_model,
    'xai': lambda: XaiModel('grok-4', provider=XaiProvider(api_key='test-key')),
    'compatible-default': lambda: OpenAIChatModel(
        'qwen/qwen3', provider=OpenAICompatibleProvider('my-service', base_url='https://example.com/v1')
    ),
    'compatible-accepts-openai': lambda: OpenAIChatModel(
        'qwen/qwen3',
        provider=OpenAICompatibleProvider('my-proxy', base_url='https://example.com/v1', compatible_systems=['openai']),
    ),
}


_SAME_AS_RESPONSE = 'same-as-response'


def thinking_response(
    provider_name: str | None,
    *,
    part_provider_name: str | None = _SAME_AS_RESPONSE,
    canonical_system: str | None = None,
    provider_url: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        parts=[
            ThinkingPart(
                content='thought',
                signature=SIGNATURE,
                provider_name=provider_name if part_provider_name == _SAME_AS_RESPONSE else part_provider_name,
            )
        ],
        provider_name=provider_name,
        canonical_system=canonical_system,
        provider_url=provider_url,
    )


@dataclass
class AcceptCase:
    id: str
    receiver: str
    response: ModelResponse
    accepted: bool


ACCEPT_CASES = [
    # Own data, recorded before responses carried `canonical_system`.
    AcceptCase('openai own', 'openai', thinking_response('openai'), True),
    AcceptCase('anthropic own', 'anthropic', thinking_response('anthropic'), True),
    AcceptCase('bedrock own on claude', 'bedrock-claude', thinking_response('bedrock'), True),
    AcceptCase('bedrock own on nova', 'bedrock-nova', thinking_response('bedrock'), True),
    AcceptCase('xai own', 'xai', thinking_response('xai'), True),
    AcceptCase('custom own', 'compatible-default', thinking_response('my-service'), True),
    # A part without a `provider_name` is never accepted, whatever its response says.
    AcceptCase('part without provider_name', 'openai', thinking_response('openai', part_provider_name=None), False),
    # A part from another provider than its response is judged by its own name.
    AcceptCase(
        'part provider differs from response',
        'anthropic',
        thinking_response('openai', part_provider_name='anthropic', canonical_system='openai'),
        True,
    ),
    # Legacy names.
    AcceptCase('legacy codex into openai', 'openai', thinking_response('openai-codex'), True),
    AcceptCase('legacy grok into xai', 'xai', thinking_response('grok'), True),
    AcceptCase('legacy google-gla into gemini api', 'gemini-api', thinking_response('google-gla'), True),
    AcceptCase('legacy google-vertex into gemini api', 'gemini-api', thinking_response('google-vertex'), False),
    AcceptCase('legacy google-vertex into google cloud', 'google-cloud', thinking_response('google-vertex'), True),
    # Stamped responses.
    AcceptCase('codex into openai', 'openai', thinking_response('openai-codex', canonical_system='openai'), True),
    AcceptCase('openai into codex', 'openai-codex', thinking_response('openai'), True),
    AcceptCase(
        'azure-client openai into openai',
        'openai',
        thinking_response('openai', canonical_system='azure'),
        False,
    ),
    AcceptCase('azure into openai', 'openai', thinking_response('azure'), False),
    AcceptCase('openai into azure', 'azure', thinking_response('openai'), False),
    AcceptCase('azure into azure-client openai', 'openai+azure-client', thinking_response('azure'), True),
    # The same name with no stamp is accepted as before: it may be this provider's own older history. New
    # `OpenAIProvider` history is always stamped, which tells the two apart.
    AcceptCase('unstamped openai into azure-client openai', 'openai+azure-client', thinking_response('openai'), True),
    AcceptCase(
        'stamped openai into azure-client openai',
        'openai+azure-client',
        thinking_response('openai', canonical_system='openai'),
        False,
    ),
    AcceptCase(
        'stamped azure-client openai into azure-client openai',
        'openai+azure-client',
        thinking_response('openai', canonical_system='azure'),
        True,
    ),
    AcceptCase('openai into renamed openai subclass', 'my-gateway', thinking_response('openai'), False),
    AcceptCase('renamed openai subclass into openai', 'openai', thinking_response('my-gateway'), False),
    AcceptCase('anthropic into claude-code', 'claude-code', thinking_response('anthropic'), True),
    AcceptCase(
        'claude-code into anthropic',
        'anthropic',
        thinking_response('claude-code', canonical_system='anthropic'),
        True,
    ),
    AcceptCase('older claude-code into anthropic', 'anthropic', thinking_response('claude-code'), False),
    AcceptCase('older claude-code into claude-code', 'claude-code', thinking_response('claude-code'), True),
    AcceptCase('anthropic into bedrock claude', 'bedrock-claude', thinking_response('anthropic'), True),
    AcceptCase('anthropic into bedrock nova', 'bedrock-nova', thinking_response('anthropic'), False),
    AcceptCase(
        'bedrock claude into anthropic',
        'anthropic',
        thinking_response('bedrock', canonical_system='anthropic'),
        True,
    ),
    AcceptCase('older bedrock claude into anthropic', 'anthropic', thinking_response('bedrock'), False),
    # Gemini API and Google Cloud share the `google` name.
    AcceptCase(
        'gemini api into vertex client',
        'google+vertex-client',
        thinking_response('google', provider_url=GEMINI_API_URL),
        False,
    ),
    AcceptCase(
        'vertex client into gemini api',
        'gemini-api',
        thinking_response('google', canonical_system='google-cloud', provider_url=VERTEX_URL),
        False,
    ),
    AcceptCase(
        'older vertex client into gemini api',
        'gemini-api',
        thinking_response('google', provider_url=VERTEX_URL),
        False,
    ),
    AcceptCase(
        'older vertex client into google cloud',
        'google-cloud',
        thinking_response('google', provider_url=VERTEX_URL),
        True,
    ),
    AcceptCase(
        'older vertex client in another region into vertex client',
        'google+vertex-client',
        thinking_response('google', provider_url='https://europe-west4-aiplatform.googleapis.com/v1'),
        True,
    ),
    AcceptCase(
        'older vertex data residency endpoint into google cloud',
        'google-cloud',
        thinking_response('google', provider_url='https://aiplatform.eu.rep.googleapis.com/'),
        True,
    ),
    AcceptCase(
        'older global vertex endpoint into gemini api',
        'gemini-api',
        thinking_response('google', provider_url='https://aiplatform.googleapis.com/'),
        False,
    ),
    AcceptCase(
        'older google-cloud on a gemini api client into google cloud',
        'google-cloud',
        thinking_response('google-cloud', provider_url=GEMINI_API_URL),
        False,
    ),
    # A proxy URL, or none, is no evidence either way, so the same name is accepted as before.
    AcceptCase(
        'older google via proxy into vertex client',
        'google+vertex-client',
        thinking_response('google', provider_url='https://proxy.example.com/google'),
        True,
    ),
    AcceptCase(
        'older google without url into vertex client', 'google+vertex-client', thinking_response('google'), True
    ),
    AcceptCase(
        'older google via proxy into google cloud',
        'google-cloud',
        thinking_response('google', provider_url='https://proxy.example.com/google'),
        False,
    ),
    # `compatible_systems` is one-way.
    AcceptCase('openai into compatible proxy', 'compatible-accepts-openai', thinking_response('openai'), True),
    AcceptCase('compatible proxy into openai', 'openai', thinking_response('my-proxy'), False),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in ACCEPT_CASES])
def test_accepts_stateless_artifacts(case: AcceptCase):
    model = RECEIVERS[case.receiver]()
    assert model.accepts_stateless_artifacts(case.response, case.response.parts[0]) is case.accepted


def test_response_level_data_is_judged_by_the_response_provider():
    """Without a part, the response's own `provider_name` decides, as for Gemini signatures on unnamed parts."""
    model = gemini_api_model()
    assert model.accepts_stateless_artifacts(ModelResponse(parts=[], provider_name='google'))
    assert not model.accepts_stateless_artifacts(ModelResponse(parts=[], provider_name='openai'))
    assert not model.accepts_stateless_artifacts(ModelResponse(parts=[], provider_name=None))


def test_stateless_artifact_system():
    assert stateless_artifact_system(thinking_response('openai-codex'), 'openai-codex') == ('openai', True)
    assert stateless_artifact_system(thinking_response('google', provider_url=VERTEX_URL), 'google') == (
        'google-cloud',
        False,
    )
    assert stateless_artifact_system(thinking_response('openai', canonical_system='azure'), 'openai') == (
        'azure',
        False,
    )
    assert not accepts_stateless_artifact(
        thinking_response('openai'), None, system='openai', canonical_system='openai', compatible_systems=frozenset()
    )


@pytest.mark.parametrize(
    ('make_provider', 'name', 'canonical_system'),
    [
        pytest.param(openai_provider, 'openai', 'openai', id='openai'),
        pytest.param(azure_client_openai_provider, 'openai', 'azure', id='openai-azure-client'),
        pytest.param(lambda: GatewayOpenAIProvider(api_key='k'), 'my-gateway', 'my-gateway', id='openai-subclass'),
        pytest.param(codex_provider, 'openai-codex', 'openai', id='openai-codex'),
        pytest.param(
            lambda: OpenAIChatGPTProvider(openai_client=AsyncOpenAI(api_key='k')),
            'openai-chatgpt',
            'openai-chatgpt',
            id='openai-chatgpt',
        ),
        pytest.param(
            lambda: AzureProvider(
                azure_endpoint='https://resource.openai.azure.com', api_version='2025-04-01-preview', api_key='k'
            ),
            'azure',
            'azure',
            id='azure',
        ),
        pytest.param(
            lambda: BedrockMantleProvider(api_key='k', region_name='us-east-1'),
            'bedrock-mantle',
            'bedrock-mantle',
            id='bedrock-mantle',
        ),
        pytest.param(lambda: AnthropicProvider(api_key='k'), 'anthropic', 'anthropic', id='anthropic'),
        pytest.param(lambda: ClaudeCodeProvider(api_key='k'), 'claude-code', 'anthropic', id='anthropic-subclass'),
        pytest.param(
            lambda: AnthropicProvider(anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', api_key='k')),
            'anthropic',
            'anthropic',
            id='anthropic-bedrock-client',
        ),
        pytest.param(
            lambda: AnthropicProvider(
                anthropic_client=AsyncAnthropicVertex(region='us-east5', project_id='p', access_token='t')
            ),
            'anthropic',
            'anthropic',
            id='anthropic-vertex-client',
        ),
        pytest.param(bedrock_provider, 'bedrock', 'bedrock', id='bedrock'),
        pytest.param(lambda: GoogleProvider(api_key='k'), 'google', 'google', id='google'),
        pytest.param(
            lambda: GoogleProvider(client=Client(vertexai=True, api_key='k')),
            'google',
            'google-cloud',
            id='google-vertex-client',
        ),
        pytest.param(lambda: GoogleCloudProvider(api_key='k'), 'google-cloud', 'google-cloud', id='google-cloud'),
        pytest.param(
            lambda: GoogleCloudProvider(client=Client(vertexai=False, api_key='k')),
            'google-cloud',
            'google',
            id='google-cloud-gemini-api-client',
        ),
        pytest.param(lambda: XaiProvider(api_key='k'), 'xai', 'xai', id='xai'),
        pytest.param(lambda: OpenRouterProvider(api_key='k'), 'openrouter', 'openrouter', id='openrouter'),
        pytest.param(
            lambda: SnowflakeProvider(account='acct', token='k'),
            'snowflake',
            'snowflake',
            id='snowflake',
        ),
    ],
)
def test_provider_canonical_system(make_provider: Callable[[], Any], name: str, canonical_system: str):
    provider = make_provider()
    assert provider.name == name
    assert provider.canonical_system == canonical_system
    assert provider.compatible_systems == frozenset()


def test_bedrock_converse_canonical_system_follows_the_model():
    assert bedrock_claude_model().canonical_system == 'anthropic'
    assert bedrock_nova_model().canonical_system == 'bedrock'
    # A model the name doesn't identify, such as an inference profile ARN, can opt in through its profile.
    arn = 'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc'
    assert BedrockConverseModel(arn, provider=bedrock_provider()).canonical_system == 'bedrock'
    assert (
        BedrockConverseModel(
            arn, provider=bedrock_provider(), profile=BedrockModelProfile(bedrock_thinking_variant='anthropic')
        ).canonical_system
        == 'anthropic'
    )


def test_wrapper_and_fallback_models_delegate():
    gemini_api = gemini_api_model()
    vertex = vertex_client_google_model()
    claude_code = RECEIVERS['claude-code']()

    instrumented = InstrumentedModel(claude_code)
    assert instrumented.canonical_system == 'anthropic'
    assert instrumented.compatible_systems == frozenset()
    assert instrumented.accepts_stateless_artifacts(
        thinking_response('anthropic'), thinking_response('anthropic').parts[0]
    )

    fallback = FallbackModel(vertex, gemini_api)
    assert fallback.canonical_system == 'google-cloud'
    assert fallback.compatible_systems == frozenset()
    gemini_api_response = thinking_response('google', provider_url=GEMINI_API_URL)
    assert not fallback.accepts_stateless_artifacts(gemini_api_response, gemini_api_response.parts[0])
    # The window a response's calls are judged in comes from the model that produced it.
    assert fallback._model_for_response(gemini_api_response) is gemini_api  # pyright: ignore[reportPrivateUsage]
    vertex_response = thinking_response('google', canonical_system='google-cloud')
    assert fallback._model_for_response(vertex_response) is vertex  # pyright: ignore[reportPrivateUsage]
    assert fallback._model_for_response(thinking_response('openai')) is vertex  # pyright: ignore[reportPrivateUsage]
    assert InstrumentedModel(fallback)._model_for_response(gemini_api_response) is gemini_api  # pyright: ignore[reportPrivateUsage]

    # Of two models the response could be from, the one that trims before compaction parts.
    chat = OpenAIChatModel('gpt-5.5', provider=openai_provider())
    responses = OpenAIResponsesModel('gpt-5.5', provider=openai_provider())
    openai_response = thinking_response('openai')
    assert FallbackModel(chat, responses)._model_for_response(openai_response) is responses  # pyright: ignore[reportPrivateUsage]
    assert FallbackModel(chat, vertex)._model_for_response(openai_response) is chat  # pyright: ignore[reportPrivateUsage]


def test_model_without_provider_uses_its_system():
    model = TestModel()
    assert model.provider is None
    assert model.canonical_system == 'test'
    assert model.compatible_systems == frozenset()
    assert model._canonical_system_stamp is None  # pyright: ignore[reportPrivateUsage]


def test_canonical_system_stamp_round_trips():
    """A model stamps its responses when its canonical system differs from its name, and the stamp is stored.

    Models on a provider whose canonical system depends on its client always do: either Google provider name
    covers both transports, and `OpenAIProvider` covers OpenAI and Azure.
    """
    assert RECEIVERS['openai']()._canonical_system_stamp == 'openai'  # pyright: ignore[reportPrivateUsage]
    assert RECEIVERS['openai+azure-client']()._canonical_system_stamp == 'azure'  # pyright: ignore[reportPrivateUsage]
    assert RECEIVERS['azure']()._canonical_system_stamp is None  # pyright: ignore[reportPrivateUsage]
    assert RECEIVERS['anthropic']()._canonical_system_stamp is None  # pyright: ignore[reportPrivateUsage]
    assert gemini_api_model()._canonical_system_stamp == 'google'  # pyright: ignore[reportPrivateUsage]
    assert google_cloud_model()._canonical_system_stamp == 'google-cloud'  # pyright: ignore[reportPrivateUsage]
    assert RECEIVERS['openai-codex']()._canonical_system_stamp == 'openai'  # pyright: ignore[reportPrivateUsage]
    assert RECEIVERS['claude-code']()._canonical_system_stamp == 'anthropic'  # pyright: ignore[reportPrivateUsage]
    assert vertex_client_google_model()._canonical_system_stamp == 'google-cloud'  # pyright: ignore[reportPrivateUsage]
    response = thinking_response('openai-codex', canonical_system='openai')
    dumped = ModelMessagesTypeAdapter.dump_python([response], mode='json')
    assert dumped[0]['canonical_system'] == 'openai'
    assert ModelMessagesTypeAdapter.validate_python(dumped) == [response]


# Compaction: the trim, the boundary and the execution-availability window follow the same predicate.


def compaction_history(provider_name: str, *, canonical_system: str | None = None) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('before the compaction')]),
        ModelResponse(parts=[TextPart('earlier answer')], provider_name=provider_name),
        ModelRequest(parts=[UserPromptPart('compact')]),
        ModelResponse(
            parts=[
                CompactionPart(
                    content=None,
                    id='cmp_1',
                    provider_name=provider_name,
                    provider_details={'encrypted_content': 'encrypted'},
                ),
                TextPart('summary answer'),
            ],
            provider_name=provider_name,
            canonical_system=canonical_system,
        ),
        ModelRequest(parts=[UserPromptPart('after the compaction')]),
    ]


@pytest.mark.parametrize(
    ('history_provider', 'canonical_system', 'trims'),
    [
        pytest.param('openai-codex', 'openai', True, id='codex'),
        pytest.param('openai-codex', None, True, id='older-codex'),
        pytest.param('azure', None, False, id='azure'),
        pytest.param('openai', 'azure', False, id='azure-client'),
    ],
)
async def test_compaction_trim_follows_the_predicate(history_provider: str, canonical_system: str | None, trims: bool):
    model = OpenAIResponsesModel('gpt-5.5', provider=openai_provider())
    history = compaction_history(history_provider, canonical_system=canonical_system)
    _, items = await model._map_messages(history, {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    compaction_items = [item for item in items if item.get('type') == 'compaction']
    first_user = items[0].get('content')
    if trims:
        assert compaction_items == [{'id': 'cmp_1', 'encrypted_content': 'encrypted', 'type': 'compaction'}]
        assert first_user != 'before the compaction'
    else:
        assert compaction_items == []
        assert first_user == 'before the compaction'


def test_post_compaction_window_follows_the_serving_model():
    history = compaction_history('openai-codex', canonical_system='openai')
    serving = ModelResponse(parts=[ToolCallPart('tool', {})], provider_name='openai')
    messages = [*history, serving]
    openai_window = _post_compaction_window_for_response(
        messages, serving, OpenAIResponsesModel('gpt-5.5', provider=openai_provider()).accepts_stateless_artifacts
    )
    assert openai_window[0].parts == history[3].parts
    azure_window = _post_compaction_window_for_response(
        messages, serving, RECEIVERS['azure']().accepts_stateless_artifacts
    )
    assert azure_window == messages
    # Without the producer's predicate, only the serving response's own provider's compaction is a boundary.
    assert _post_compaction_window_for_response(messages, serving) == messages
    codex_serving = ModelResponse(parts=[ToolCallPart('tool', {})], provider_name='openai-codex')
    codex_messages = [*history, codex_serving]
    assert _post_compaction_window_for_response(codex_messages, codex_serving)[0].parts == history[3].parts


@pytest.mark.parametrize(
    ('model_factory', 'response', 'trims'),
    [
        pytest.param(
            RECEIVERS['openai'],
            ModelResponse(parts=[], provider_name='openai', canonical_system='openai'),
            True,
            id='openai-responses-own-response',
        ),
        pytest.param(
            RECEIVERS['openai'],
            ModelResponse(parts=[], provider_name='openai'),
            False,
            id='openai-responses-unstamped-response',
        ),
        pytest.param(
            RECEIVERS['openai-codex'],
            ModelResponse(parts=[], provider_name='openai-codex', canonical_system='openai'),
            True,
            id='codex-own-response',
        ),
        # A response recorded before responses carried `canonical_system`, or another model's, can't be judged by
        # this model's predicate.
        pytest.param(
            RECEIVERS['openai-codex'],
            ModelResponse(parts=[], provider_name='openai-codex'),
            False,
            id='codex-unstamped-response',
        ),
        pytest.param(
            RECEIVERS['anthropic'],
            ModelResponse(parts=[], provider_name='openai'),
            False,
            id='another-models-response',
        ),
        # A model that never trims history before a compaction part, like Bedrock with a Claude model, honors
        # no boundary, even one from a system it accepts.
        pytest.param(
            bedrock_claude_model,
            ModelResponse(parts=[], provider_name='bedrock', canonical_system='anthropic'),
            False,
            id='bedrock-claude',
        ),
    ],
)
def test_compaction_window_predicate(model_factory: Callable[[], Model], response: ModelResponse, trims: bool):
    model = model_factory()
    predicate = InstrumentedModel(model)._compaction_window_predicate(response)  # pyright: ignore[reportPrivateUsage]
    if trims:
        assert predicate is not None
        assert predicate == model.accepts_stateless_artifacts
    else:
        assert predicate is None


# What each adapter sends from a history recorded by another provider.


def _json(value: Any) -> Any:
    """`value` with protobuf messages and bytes made comparable."""
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import Message

    if isinstance(value, Message):
        return MessageToDict(value)
    if isinstance(value, list):
        return [_json(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    return value


def reasoning_history(provider_name: str, *, canonical_system: str | None = None) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='thought', id='rs_1', signature=SIGNATURE, provider_name=provider_name),
                TextPart('answer', id='msg_1', provider_name=provider_name),
            ],
            provider_name=provider_name,
            canonical_system=canonical_system,
        ),
        ModelRequest(parts=[UserPromptPart('next')]),
    ]


async def test_openai_responses_replays_codex_encrypted_reasoning():
    model = OpenAIResponsesModel('gpt-5.5', provider=openai_provider())
    _, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        reasoning_history('openai-codex', canonical_system='openai'), {}, ModelRequestParameters()
    )
    assert items == snapshot(
        [
            {'role': 'user', 'content': 'hi'},
            {
                'id': 'rs_1',
                'summary': [{'text': 'thought', 'type': 'summary_text'}],
                'encrypted_content': 'c2lnbmF0dXJl',
                'type': 'reasoning',
            },
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'next'},
        ]
    )


async def test_openai_responses_renders_azure_reasoning_as_text():
    model = OpenAIResponsesModel('gpt-5.5', provider=openai_provider())
    _, items = await model._map_messages(reasoning_history('azure'), {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert items == snapshot(
        [
            {'role': 'user', 'content': 'hi'},
            {
                'role': 'assistant',
                'content': """\
<think>
thought
</think>\
""",
            },
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'next'},
        ]
    )


async def test_openai_responses_keeps_compatible_native_tool_search():
    """Native tool search from a provider with the same canonical system keeps its native shape, without its ids."""
    model = OpenAIResponsesModel('gpt-5.5', provider=openai_provider())
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('find a tool')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['weather']},
                    tool_call_id='tsc_1',
                    id='ts_1',
                    provider_name='openai-codex',
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': []},
                    tool_call_id='tsc_1',
                    provider_name='openai-codex',
                    provider_details={'id': 'tso_1', 'status': 'completed'},
                ),
            ],
            provider_name='openai-codex',
            canonical_system='openai',
        ),
        ModelRequest(parts=[UserPromptPart('next')]),
    ]
    params = ModelRequestParameters()
    _, items = await model._map_messages(model.prepare_messages(history, params), {}, params)  # pyright: ignore[reportPrivateUsage]
    assert [item.get('type') for item in items] == snapshot([None, 'tool_search_call', 'tool_search_output', None])


async def test_openai_chat_replays_extra_content_to_a_compatible_proxy():
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    'get',
                    {},
                    tool_call_id='call_1',
                    provider_name='openai',
                    provider_details={'extra_content': {'google': {'thought_signature': SIGNATURE}}},
                )
            ],
            provider_name='openai',
        ),
        ModelRequest(parts=[ToolReturnPart('get', 'ok', tool_call_id='call_1')]),
    ]
    params = ModelRequestParameters()
    accepting = RECEIVERS['compatible-accepts-openai']()
    assert isinstance(accepting, OpenAIChatModel)
    rejecting = RECEIVERS['compatible-default']()
    assert isinstance(rejecting, OpenAIChatModel)
    accepted = await accepting._map_messages(history, params)  # pyright: ignore[reportPrivateUsage]
    rejected = await rejecting._map_messages(history, params)  # pyright: ignore[reportPrivateUsage]
    assert accepted[1] == snapshot(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'get', 'arguments': '{}'},
                    'extra_content': {'google': {'thought_signature': 'c2lnbmF0dXJl'}},
                }
            ],
        }
    )
    assert rejected[1] == snapshot(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'get', 'arguments': '{}'}}],
        }
    )


async def test_openrouter_replays_reasoning_details_to_a_renamed_subclass():
    class MyOpenRouterProvider(OpenRouterProvider):
        @property
        def name(self) -> str:
            return 'my-openrouter'

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='',
                    signature=SIGNATURE,
                    id='reasoning-1',
                    provider_name='openrouter',
                    provider_details={'type': 'reasoning.encrypted', 'format': 'anthropic-claude-v1', 'index': 0},
                ),
                TextPart('answer'),
            ],
            provider_name='openrouter',
        ),
        ModelRequest(parts=[UserPromptPart('next')]),
    ]
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=MyOpenRouterProvider(api_key='k'))
    messages = await model._map_messages(history, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert messages[1] == snapshot(
        {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_details': [
                {
                    'id': 'reasoning-1',
                    'format': 'anthropic-claude-v1',
                    'index': 0,
                    'type': 'reasoning.encrypted',
                    'data': 'c2lnbmF0dXJl',
                }
            ],
        }
    )


def block_types(messages: Any) -> list[tuple[str, list[str]]]:
    """Each Anthropic message's role and the types of its content blocks."""
    return [
        (message['role'], [block['type'] for block in message['content']])
        for message in cast(list[dict[str, Any]], messages)
    ]


def anthropic_history(provider_name: str, *, canonical_system: str | None = None) -> list[ModelMessage]:
    redacted_id = 'redacted_content' if provider_name == 'bedrock' else 'redacted_thinking'
    return [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='thought', signature=SIGNATURE, provider_name=provider_name),
                ThinkingPart(content='', id=redacted_id, signature='redacted-data', provider_name=provider_name),
                NativeToolCallPart(
                    tool_name='web_search', args={'query': 'q'}, tool_call_id='srvtoolu_1', provider_name=provider_name
                ),
                NativeToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'type': 'web_search_result',
                            'url': 'https://example.com',
                            'title': 't',
                            'encrypted_content': 'e',
                        }
                    ],
                    tool_call_id='srvtoolu_1',
                    provider_name=provider_name,
                ),
                NativeToolCallPart(
                    tool_name=CodeExecutionTool.kind,
                    args={'code': 'print(1)'},
                    tool_call_id='srvtoolu_2',
                    provider_name=provider_name,
                ),
                NativeToolReturnPart(
                    tool_name=CodeExecutionTool.kind,
                    content={
                        'type': 'code_execution_result',
                        'stdout': '1',
                        'stderr': '',
                        'return_code': 0,
                        'content': [],
                    },
                    tool_call_id='srvtoolu_2',
                    provider_name=provider_name,
                ),
                TextPart('answer'),
            ],
            provider_name=provider_name,
            canonical_system=canonical_system,
        ),
        ModelRequest(parts=[UserPromptPart('next')]),
    ]


@dataclass
class ReplayCase:
    id: str
    receiver: Callable[[], Model]
    history_provider: str
    canonical_system: str | None
    expected: Any


ANTHROPIC_REPLAY_CASES = [
    ReplayCase(
        'anthropic-into-claude-code',
        RECEIVERS['claude-code'],
        'anthropic',
        None,
        # The code execution call and result stay behind: they refer to the other account's container.
        snapshot(
            [
                ('user', ['text']),
                ('assistant', ['thinking', 'redacted_thinking', 'server_tool_use', 'web_search_tool_result', 'text']),
                ('user', ['text']),
            ]
        ),
    ),
    ReplayCase(
        'bedrock-claude-into-anthropic',
        RECEIVERS['anthropic'],
        'bedrock',
        'anthropic',
        # Bedrock's `redacted_content` is sent as `redacted_thinking`, with the same data.
        snapshot(
            [
                ('user', ['text']),
                ('assistant', ['thinking', 'redacted_thinking', 'server_tool_use', 'web_search_tool_result', 'text']),
                ('user', ['text']),
            ]
        ),
    ),
    ReplayCase(
        'openai-into-anthropic',
        RECEIVERS['anthropic'],
        'openai',
        None,
        snapshot([('user', ['text']), ('user', ['text']), ('assistant', ['text']), ('user', ['text'])]),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in ANTHROPIC_REPLAY_CASES])
async def test_anthropic_replays_by_canonical_system(case: ReplayCase):
    model = case.receiver()
    assert isinstance(model, AnthropicModel)
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        anthropic_history(case.history_provider, canonical_system=case.canonical_system), ModelRequestParameters(), {}
    )
    assert block_types(messages) == case.expected


BEDROCK_REPLAY_CASES = [
    ReplayCase(
        'anthropic-into-claude',
        bedrock_claude_model,
        'anthropic',
        None,
        snapshot(
            [
                {'role': 'user', 'content': [{'text': 'hi'}]},
                {
                    'role': 'assistant',
                    'content': [
                        {'reasoningContent': {'reasoningText': {'text': 'thought', 'signature': 'c2lnbmF0dXJl'}}},
                        {'reasoningContent': {'redactedContent': b'redacted-data'}},
                        {'text': 'answer'},
                    ],
                },
                {'role': 'user', 'content': [{'text': 'next'}]},
            ]
        ),
    ),
    ReplayCase(
        'claude-code-into-claude',
        bedrock_claude_model,
        'claude-code',
        'anthropic',
        snapshot(
            [
                {'role': 'user', 'content': [{'text': 'hi'}]},
                {
                    'role': 'assistant',
                    'content': [
                        {'reasoningContent': {'reasoningText': {'text': 'thought', 'signature': 'c2lnbmF0dXJl'}}},
                        {'reasoningContent': {'redactedContent': b'redacted-data'}},
                        {'text': 'answer'},
                    ],
                },
                {'role': 'user', 'content': [{'text': 'next'}]},
            ]
        ),
    ),
    ReplayCase(
        'anthropic-into-nova',
        bedrock_nova_model,
        'anthropic',
        None,
        snapshot(
            [
                {'role': 'user', 'content': [{'text': 'hi'}]},
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'text': """\
<think>
thought
</think>\
"""
                        },
                        {
                            'text': """\
<think>

</think>\
"""
                        },
                        {'text': 'answer'},
                    ],
                },
                {'role': 'user', 'content': [{'text': 'next'}]},
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in BEDROCK_REPLAY_CASES])
async def test_bedrock_replays_claude_signatures_by_canonical_system(case: ReplayCase):
    model = case.receiver()
    assert isinstance(model, BedrockConverseModel)
    _, messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        anthropic_history(case.history_provider, canonical_system=case.canonical_system), ModelRequestParameters(), {}
    )
    assert _json(messages) == case.expected


async def test_anthropic_replays_web_search_citations_by_canonical_system():
    """Citations point into web search results by an encrypted index, so they go wherever the results do."""
    citation = {
        'type': 'web_search_result_location',
        'cited_text': 'web excerpt',
        'url': 'https://example.com',
        'title': 'Example',
        'encrypted_index': 'encrypted',
    }
    response = ModelResponse(
        parts=[
            TextPart(
                'An answer.',
                provider_name='anthropic',
                provider_details={'citations': [citation]},
                citations=[
                    Citation(
                        sources=[WebCitationSource(url='https://example.com')],
                        anchor=ContentCitationAnchor(start=0, end=10),
                    )
                ],
            )
        ],
        provider_name='anthropic',
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Search the web.')]),
        response,
        ModelRequest(parts=[UserPromptPart('Continue.')]),
    ]
    model = RECEIVERS['claude-code']()
    assert isinstance(model, AnthropicModel)
    _, messages = await model._map_message(history, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage]
    [replayed] = cast(list[dict[str, Any]], messages[1]['content'])
    assert replayed.get('citations') == [citation]


@pytest.mark.parametrize(
    ('model_factory', 'response', 'expected_signature'),
    [
        pytest.param(
            vertex_client_google_model,
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'get',
                        {},
                        tool_call_id='1',
                        provider_name='google',
                        provider_details={'thought_signature': SIGNATURE},
                    )
                ],
                provider_name='google',
                provider_url=GEMINI_API_URL,
            ),
            b'skip_thought_signature_validator',
            id='gemini-api-into-vertex',
        ),
        # A Gemini API response through a proxy: its URL says nothing, but the stamp `GoogleModel` always sets does.
        pytest.param(
            vertex_client_google_model,
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'get',
                        {},
                        tool_call_id='1',
                        provider_name='google',
                        provider_details={'thought_signature': SIGNATURE},
                    )
                ],
                provider_name='google',
                canonical_system='google',
                provider_url='https://proxy.example.com/google',
            ),
            b'skip_thought_signature_validator',
            id='proxied-gemini-api-into-vertex',
        ),
        pytest.param(
            gemini_api_model,
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'get',
                        {},
                        tool_call_id='1',
                        provider_name='google',
                        provider_details={'thought_signature': SIGNATURE},
                    )
                ],
                provider_name='google',
                canonical_system='google',
                provider_url='https://proxy.example.com/google',
            ),
            b'signature',
            id='proxied-gemini-api-into-gemini-api',
        ),
        pytest.param(
            gemini_api_model,
            ModelResponse(
                parts=[
                    ThinkingPart(content='thought', signature=SIGNATURE, provider_name='google'),
                    ToolCallPart('get', {}, tool_call_id='1', provider_name='google'),
                ],
                provider_name='google',
                canonical_system='google-cloud',
                provider_url=VERTEX_URL,
            ),
            b'skip_thought_signature_validator',
            id='vertex-into-gemini-api',
        ),
        pytest.param(
            google_cloud_model,
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'get',
                        {},
                        tool_call_id='1',
                        provider_name='google',
                        provider_details={'thought_signature': SIGNATURE},
                    )
                ],
                provider_name='google',
                canonical_system='google-cloud',
            ),
            b'signature',
            id='vertex-client-into-google-cloud',
        ),
    ],
)
async def test_google_replays_thought_signatures_to_their_own_transport(
    model_factory: Callable[[], GoogleModel], response: ModelResponse, expected_signature: bytes
):
    """The #10905 case: the Gemini API and Google Cloud reject each other's signatures, so the dummy goes instead."""
    model = model_factory()
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        response,
        ModelRequest(parts=[ToolReturnPart('get', 'ok', tool_call_id='1')]),
    ]
    params = ModelRequestParameters()
    _, contents = await model._map_messages(model.prepare_messages(history, params), params)  # pyright: ignore[reportPrivateUsage]
    function_call = next(
        part
        for content in cast(list[dict[str, Any]], contents)
        if content.get('role') == 'model'
        for part in content['parts']
        if 'function_call' in part
    )
    assert function_call['thought_signature'] == expected_signature


async def test_xai_replays_by_canonical_system():
    model = XaiModel('grok-4', provider=XaiProvider(api_key='k'))
    xai_messages = await model._map_messages(reasoning_history('grok'), ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    openai_messages = await model._map_messages(reasoning_history('openai'), ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert _json(xai_messages[1]) == snapshot(
        {
            'content': [{'text': ''}],
            'role': 'ROLE_ASSISTANT',
            'reasoningContent': 'thought',
            'encryptedContent': 'c2lnbmF0dXJl',
        }
    )
    assert _json(openai_messages[1]) == snapshot(
        {
            'content': [
                {
                    'text': """\
<think>
thought
</think>\
"""
                }
            ],
            'role': 'ROLE_ASSISTANT',
        }
    )

    # A server-side tool result whose call isn't in the history has nothing to attach to, so it's left out.
    orphan_return: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(
            parts=[
                NativeToolReturnPart(
                    tool_name='web_search', content='results', tool_call_id='missing', provider_name='xai'
                ),
                TextPart('answer'),
            ],
            provider_name='xai',
        ),
    ]
    orphan_messages = await model._map_messages(orphan_return, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert [_json(message) for message in orphan_messages[1:]] == snapshot(
        [{'content': [{'text': 'answer'}], 'role': 'ROLE_ASSISTANT'}]
    )


# Anthropic file and container IDs on clients without the Files API.


@dataclass
class AnthropicClientCase:
    id: str
    client: Callable[[], Any] | None
    accepts_ids: bool
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


ANTHROPIC_CLIENT_CASES = [
    AnthropicClientCase('anthropic', None, True),
    AnthropicClientCase('bedrock', lambda: AsyncAnthropicBedrock(aws_region='us-east-1', api_key='k'), False),
    AnthropicClientCase(
        'vertex', lambda: AsyncAnthropicVertex(region='us-east5', project_id='p', access_token='t'), False
    ),
    AnthropicClientCase('foundry', lambda: AsyncAnthropicFoundry(resource='r', api_key='k'), False),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in ANTHROPIC_CLIENT_CASES])
async def test_anthropic_file_and_container_ids_by_client(case: AnthropicClientCase):
    provider = (
        AnthropicProvider(api_key='k') if case.client is None else AnthropicProvider(anthropic_client=case.client())
    )
    model = AnthropicModel('claude-sonnet-4-5', provider=provider)
    assert model.profile.get('anthropic_accepts_file_and_container_ids', True) is case.accepts_ids

    file = UploadedFile(file_id='file_1', provider_name='anthropic', media_type='application/pdf')
    tool = CodeExecutionTool(files=[file])
    params = ModelRequestParameters(native_tools=[tool])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('run it')]),
        ModelResponse(parts=[TextPart('ok')], provider_name='anthropic', provider_details={'container_id': 'cntr_1'}),
        ModelRequest(parts=[UserPromptPart('again')]),
    ]
    container, from_history = model._get_container(history, {})  # pyright: ignore[reportPrivateUsage]
    assert (container, from_history) == (('cntr_1', True) if case.accepts_ids else (None, False))
    assert model._container_upload_file_ids(tool) == (['file_1'] if case.accepts_ids else [])  # pyright: ignore[reportPrivateUsage]

    prompt_with_file: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(['read this', file])])]
    if case.accepts_ids:
        _, messages = await model._map_message(prompt_with_file, params, {})  # pyright: ignore[reportPrivateUsage]
        assert block_types(messages)[0] == ('user', ['text', 'document', 'container_upload'])
        assert model._messages_use_anthropic_uploaded_file(prompt_with_file)  # pyright: ignore[reportPrivateUsage]
    else:
        assert not model._messages_use_anthropic_uploaded_file(prompt_with_file)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(UserError, match='is an Anthropic Files API file'):
            await model._map_message(prompt_with_file, params, {})  # pyright: ignore[reportPrivateUsage]

    # Nor is a paused turn's container.
    paused: list[ModelMessage] = [
        *history[:2],
        ModelResponse(
            parts=[TextPart('working')],
            provider_name='anthropic',
            provider_details={'container_id': 'cntr_2'},
            state='suspended',
        ),
    ]
    assert model._get_container(paused, {}) == (('cntr_2', False) if case.accepts_ids else (None, False))  # pyright: ignore[reportPrivateUsage]


# `OpenAICompatibleProvider`.


def test_openai_compatible_provider_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key-that-must-not-leak')
    provider = OpenAICompatibleProvider('my-service', base_url='https://example.com/v1')
    assert provider.name == 'my-service'
    assert provider.canonical_system == 'my-service'
    assert provider.compatible_systems == frozenset()
    assert provider.base_url == 'https://example.com/v1/'
    assert provider.client.api_key == 'api-key-not-set'
    # The generic profile, without the flags only OpenAI's own API supports.
    profile = provider.model_profile('gpt-5.5')
    assert profile is not None
    assert profile.get('openai_chat_supports_allowed_tools') is None
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    # A `vendor/model` name gets the vendor's profile.
    assert (provider.model_profile('anthropic/claude-sonnet-4-5') or {}).get('json_schema_transformer') is (
        OpenAIJsonSchemaTransformer
    )
    assert (provider.model_profile('anthropic/claude-sonnet-4-5') or {}).get('supports_thinking') is True
    # Static, like every provider's, so it can be called on the class.
    assert OpenAICompatibleProvider.model_profile('gpt-5.5') == profile


def test_openai_compatible_provider_options():
    client = AsyncOpenAI(api_key='k', base_url='https://example.com/v1')
    provider = OpenAICompatibleProvider(
        'my-proxy',
        openai_client=client,
        profile={'supports_tools': False},
        canonical_system='openai',
        compatible_systems=['xai'],
    )
    assert provider.client is client
    assert provider.canonical_system == 'openai'
    assert provider.compatible_systems == frozenset({'xai'})
    # `profile=` applies to the profile models get, over the static default.
    assert OpenAIChatModel('my-model', provider=provider).profile.get('supports_tools') is False
    assert (provider.model_profile('my-model') or {}).get('supports_tools') is not False

    replaced = OpenAICompatibleProvider(
        'p', base_url='https://example.com/v1', api_key='k', profile=lambda name: {'supports_json_object_output': True}
    )
    assert OpenAIChatModel('my-model', provider=replaced).profile.get('supports_json_object_output') is True
    assert replaced.client.api_key == 'k'


def test_openai_compatible_provider_errors():
    with pytest.raises(UserError, match='needs a `base_url` or an `openai_client`'):
        OpenAICompatibleProvider('p')  # pyright: ignore[reportCallIssue]
    with pytest.raises(UserError, match='cannot be used with `openai_client`'):
        OpenAICompatibleProvider('p', openai_client=AsyncOpenAI(api_key='k'), base_url='https://example.com')  # pyright: ignore[reportCallIssue]
