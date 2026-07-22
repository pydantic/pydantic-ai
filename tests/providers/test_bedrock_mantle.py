from __future__ import annotations

from typing import Literal

import httpx
import pytest
from inline_snapshot import snapshot
from typing_extensions import assert_never

from pydantic_ai import UserError
from pydantic_ai.models import infer_model, infer_model_profile
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.gateway import gateway_provider

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncBedrockOpenAI

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.bedrock_mantle import BedrockMantleChatModel, BedrockMantleResponsesModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')]

# These tests inspect local provider configuration and routing without making HTTP requests, so VCR cannot cover them.


@pytest.fixture(autouse=True)
def bedrock_credentials(env: TestEnv) -> None:
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'test-api-key')
    env.set('AWS_DEFAULT_REGION', 'us-east-1')


def test_bedrock_mantle_uses_bedrock_mantle_provider() -> None:
    assert infer_provider_class('bedrock-mantle') is BedrockMantleProvider

    provider = BedrockMantleProvider()
    assert provider.name == 'bedrock-mantle'
    assert provider.base_url == 'https://bedrock-mantle.us-east-1.api.aws/openai/v1/'


def test_bedrock_mantle_endpoint_families() -> None:
    provider = BedrockMantleProvider()

    openai_responses = provider._client_for_interface('openai-responses')  # pyright: ignore[reportPrivateUsage]
    responses = provider._client_for_interface('responses')  # pyright: ignore[reportPrivateUsage]
    chat = provider._client_for_interface('chat')  # pyright: ignore[reportPrivateUsage]

    assert isinstance(openai_responses, AsyncBedrockOpenAI)
    # `openai-responses` (GPT-5.x) is served at `/openai/v1` and is the provider's default client.
    assert openai_responses is provider.client
    assert str(openai_responses.base_url) == 'https://bedrock-mantle.us-east-1.api.aws/openai/v1/'
    # GPT-OSS Responses and Chat share the one `/v1` client.
    assert responses is chat
    assert str(responses.base_url) == 'https://bedrock-mantle.us-east-1.api.aws/v1/'
    # The two clients are distinct instances but share transport (and auth) via `with_options`.
    assert openai_responses is not responses
    assert openai_responses._client is responses._client  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_custom_base_url() -> None:
    # A custom `base_url` is normalized to its origin, so both endpoint families still route correctly.
    provider = BedrockMantleProvider(base_url='https://example.com/bedrock/v1')

    assert (
        str(provider._client_for_interface('openai-responses').base_url)  # pyright: ignore[reportPrivateUsage]
        == 'https://example.com/bedrock/openai/v1/'
    )
    assert str(provider._client_for_interface('responses').base_url) == 'https://example.com/bedrock/v1/'  # pyright: ignore[reportPrivateUsage]
    assert str(provider._client_for_interface('chat').base_url) == 'https://example.com/bedrock/v1/'  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_injected_client() -> None:
    client = AsyncBedrockOpenAI(api_key='test-api-key', aws_region='us-west-2')
    provider = BedrockMantleProvider(openai_client=client)

    # Both endpoint families are derived from the injected client's origin (sharing its transport + auth),
    # so routing works even when the user supplies their own client.
    assert (
        str(provider._client_for_interface('openai-responses').base_url)  # pyright: ignore[reportPrivateUsage]
        == 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/'
    )
    assert (
        str(provider._client_for_interface('responses').base_url)  # pyright: ignore[reportPrivateUsage]
        == 'https://bedrock-mantle.us-west-2.api.aws/v1/'
    )
    assert provider._client_for_interface('openai-responses')._client is client._client  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_model_uses_interface_client() -> None:
    # Each model class routes to the provider client for its interface, so requests hit the right endpoint.
    provider = BedrockMantleProvider()
    responses_model = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider)
    gpt_oss_model = BedrockMantleResponsesModel('openai.gpt-oss-120b', provider=provider)
    chat_model = BedrockMantleChatModel('openai.gpt-oss-safeguard-20b', provider=provider)

    assert responses_model.client is provider._client_for_interface('openai-responses')  # pyright: ignore[reportPrivateUsage]
    assert gpt_oss_model.client is provider._client_for_interface('responses')  # pyright: ignore[reportPrivateUsage]
    assert chat_model.client is provider._client_for_interface('chat')  # pyright: ignore[reportPrivateUsage]
    assert responses_model.client is not gpt_oss_model.client
    assert gpt_oss_model.client is chat_model.client


def test_bedrock_mantle_accepts_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = BedrockMantleProvider(http_client=http_client)

    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_requires_region_or_base_url(env: TestEnv) -> None:
    env.remove('AWS_DEFAULT_REGION')
    env.remove('AWS_REGION')
    with pytest.raises(UserError, match='region'):
        BedrockMantleProvider()


async def test_bedrock_mantle_provider_reopens_http_client() -> None:
    provider = BedrockMantleProvider()
    model = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider)
    first_http_client = model.client._client  # pyright: ignore[reportPrivateUsage]

    async with model:
        pass
    assert first_http_client.is_closed

    async with model:
        assert model.client._client is not first_http_client  # pyright: ignore[reportPrivateUsage]
        assert not model.client._client.is_closed  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('model_id', 'model_interface', 'base_url'),
    [
        (
            'bedrock-mantle:openai.gpt-5.6-luna',
            'responses',
            'https://bedrock-mantle.us-east-1.api.aws/openai/v1/',
        ),
        (
            'bedrock-mantle:openai.gpt-oss-120b',
            'responses',
            'https://bedrock-mantle.us-east-1.api.aws/v1/',
        ),
        (
            'bedrock-mantle:openai.gpt-oss-safeguard-20b',
            'chat',
            'https://bedrock-mantle.us-east-1.api.aws/v1/',
        ),
        (
            'bedrock:openai.gpt-oss-120b',
            'converse',
            'https://bedrock-runtime.us-east-1.amazonaws.com',
        ),
    ],
)
def test_bedrock_mantle_infer_model(
    model_id: str,
    model_interface: Literal['responses', 'chat', 'converse'],
    base_url: str,
) -> None:
    # Providers are inferred from the bearer token + region set by the `bedrock_credentials` fixture.
    model = infer_model(model_id)
    if model_interface == 'responses':
        assert isinstance(model, BedrockMantleResponsesModel)
    elif model_interface == 'chat':
        assert isinstance(model, BedrockMantleChatModel)
    elif model_interface == 'converse':
        assert isinstance(model, BedrockConverseModel)
    else:
        assert_never(model_interface)
    assert (model.model_name, model.base_url) == (model_id.partition(':')[2], base_url)


def test_bedrock_mantle_requires_bedrock_mantle_provider() -> None:
    openai_provider = OpenAIProvider(api_key='test-api-key')
    with pytest.raises(UserError, match='require a `BedrockMantleProvider`'):
        infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: openai_provider)


def test_bedrock_mantle_rejects_non_openai_model() -> None:
    with pytest.raises(UserError, match='not an OpenAI model'):
        infer_model('bedrock-mantle:anthropic.claude-sonnet-5', lambda _: BedrockMantleProvider())


def test_bedrock_mantle_model_rejects_wrong_endpoint_family() -> None:
    # Constructing the wrong model class for a model's endpoint family would misroute the request, so
    # it's rejected at construction with a pointer to the right class.
    with pytest.raises(UserError, match='Chat Completions API'):
        BedrockMantleResponsesModel('openai.gpt-oss-safeguard-20b')
    with pytest.raises(UserError, match='Responses API'):
        BedrockMantleChatModel('openai.gpt-5.6-luna')


def test_bedrock_converse_rejects_proprietary_openai() -> None:
    # Proprietary GPT models are not served by the Converse API: the profile flags them
    # (`bedrock_supported_on_converse=False`) and `BedrockConverseModel` raises at construction with a
    # pointer to `BedrockMantleProvider`. Family-based (not GPT-OSS), so it survives future GPT generations.
    for model_name in ('openai.gpt-5.6-luna', 'openai.gpt-6', 'openai.gpt-8-turbo'):
        assert BedrockProvider.model_profile(model_name) == snapshot({'bedrock_supported_on_converse': False})
        with pytest.raises(UserError, match='BedrockMantleProvider'):
            infer_model(f'bedrock:{model_name}')
    # The open-weight GPT-OSS family remains available on Converse.
    assert isinstance(infer_model('bedrock:openai.gpt-oss-120b'), BedrockConverseModel)
    assert isinstance(infer_model('bedrock:openai.gpt-oss-safeguard-20b'), BedrockConverseModel)


def test_gateway_bedrock_remains_on_converse() -> None:
    provider = gateway_provider('bedrock', api_key='test-api-key', base_url='https://gateway.pydantic.dev/proxy')
    model = infer_model('gateway/bedrock:openai.gpt-oss-120b', lambda _: provider)

    assert isinstance(model, BedrockConverseModel)
    assert model.base_url == 'https://gateway.pydantic.dev/proxy/bedrock'


def test_bedrock_mantle_profiles() -> None:
    # #6517: the vendor `openai.` prefix is stripped, so the OpenAI profile is resolved correctly and
    # GPT-5.6 keeps its real capabilities (phase / reasoning / image output).
    assert infer_model_profile('bedrock-mantle:openai.gpt-5.6-luna') == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': False,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'thinking_always_enabled': False,
            'openai_system_prompt_role': None,
            'openai_chat_supports_web_search': False,
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_reasoning_enabled_by_default': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_responses_supports_reasoning_mode': True,
            'openai_supports_phase': True,
            'openai_supports_prompt_cache_breakpoints': True,
            'bedrock_mantle_interface': 'openai-responses',
            'openai_responses_tool_call_ids_are_response_scoped': True,
            'supported_native_tools': frozenset(),
        }
    )
    # Every GPT-5.x model on Mantle's `/openai/v1` Responses endpoint resets tool-call IDs across
    # separate responses (verified live on 5.5 and 5.6), so response-scoping keys on the interface,
    # not the model version.
    assert (
        infer_model_profile('bedrock-mantle:openai.gpt-5.4').get('openai_responses_tool_call_ids_are_response_scoped')
        is True
    )
    assert (
        infer_model_profile('bedrock-mantle:openai.gpt-5.5').get('openai_responses_tool_call_ids_are_response_scoped')
        is True
    )
    # GPT-OSS on `/v1/responses` keeps globally-unique IDs, so it is not response-scoped.
    assert (
        infer_model_profile('bedrock-mantle:openai.gpt-oss-120b').get(
            'openai_responses_tool_call_ids_are_response_scoped', False
        )
        is False
    )
    assert infer_model_profile('bedrock-mantle:openai.gpt-oss-120b').get('bedrock_mantle_interface') == 'responses'
    assert infer_model_profile('bedrock-mantle:openai.gpt-oss-safeguard-20b').get('bedrock_mantle_interface') == 'chat'
    # Non-OpenAI and unknown models fall back to the default profile (best-effort).
    assert infer_model_profile('bedrock-mantle:anthropic.claude-sonnet-5') == DEFAULT_PROFILE
    assert infer_model_profile('bedrock-mantle:amazon.nova-2-lite-v1:0') == DEFAULT_PROFILE
