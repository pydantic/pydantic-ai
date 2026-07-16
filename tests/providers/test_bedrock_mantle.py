from __future__ import annotations

from typing import Literal

import httpx
import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture
from typing_extensions import assert_never

from pydantic_ai import UserError
from pydantic_ai.models import infer_model, infer_model_profile
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers._bedrock_model_names import bedrock_model_interface
from pydantic_ai.providers.gateway import gateway_provider

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrockMantle
    from openai import AsyncBedrockOpenAI

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.bedrock_mantle import (
        BedrockMantleChatModel,
        BedrockMantleMessagesModel,
        BedrockMantleResponsesModel,
    )
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')]


@pytest.fixture(autouse=True)
def bedrock_credentials(env: TestEnv) -> None:
    env.set('AWS_ACCESS_KEY_ID', 'test-access-key')
    env.set('AWS_SECRET_ACCESS_KEY', 'test-secret-key')
    env.set('AWS_DEFAULT_REGION', 'us-east-1')


def test_bedrock_mantle_uses_bedrock_provider() -> None:
    assert infer_provider_class('bedrock-mantle') is BedrockProvider

    provider = BedrockProvider()
    assert provider.name == 'bedrock'
    assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'
    assert provider.mantle_base_url == 'https://bedrock-mantle.us-east-1.api.aws'


def test_bedrock_mantle_clients() -> None:
    provider = BedrockProvider()

    special_openai = provider.mantle_openai_client('openai.gpt-5.6-luna')
    standard_openai = provider.mantle_openai_client('openai.gpt-oss-120b')
    anthropic = provider.mantle_anthropic_client()

    assert isinstance(special_openai, AsyncBedrockOpenAI)
    assert isinstance(standard_openai, AsyncBedrockOpenAI)
    assert isinstance(anthropic, AsyncAnthropicBedrockMantle)
    assert str(special_openai.base_url) == 'https://bedrock-mantle.us-east-1.api.aws/openai/v1/'
    assert str(standard_openai.base_url) == 'https://bedrock-mantle.us-east-1.api.aws/v1/'
    assert str(anthropic.base_url) == 'https://bedrock-mantle.us-east-1.api.aws/anthropic/'
    assert special_openai._client is standard_openai._client is anthropic._client  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_custom_base_url() -> None:
    provider = BedrockProvider(mantle_base_url='https://example.com/bedrock')

    assert str(provider.mantle_openai_client('openai.gpt-5.6-luna').base_url) == (
        'https://example.com/bedrock/openai/v1/'
    )
    assert str(provider.mantle_openai_client('openai.gpt-oss-120b').base_url) == ('https://example.com/bedrock/v1/')
    assert str(provider.mantle_anthropic_client().base_url) == 'https://example.com/bedrock/anthropic/'


def test_bedrock_mantle_injected_clients() -> None:
    http_client = httpx.AsyncClient()
    openai_client = AsyncBedrockOpenAI(
        api_key='test-api-key',
        aws_region='us-west-2',
        http_client=http_client,
    )
    anthropic_client = AsyncAnthropicBedrockMantle(
        api_key='test-api-key',
        aws_region='us-west-2',
        http_client=http_client,
    )
    provider = BedrockProvider(
        mantle_openai_client=openai_client,
        mantle_anthropic_client=anthropic_client,
    )

    assert provider.mantle_openai_client('openai.gpt-5.6-luna') is openai_client
    assert provider.mantle_anthropic_client() is anthropic_client
    assert provider.mantle_base_url == 'https://bedrock-mantle.us-east-1.api.aws'


def test_bedrock_mantle_infers_region_from_bedrock_client(env: TestEnv) -> None:
    bedrock_client = BedrockProvider(region_name='us-west-2').client
    env.remove('AWS_DEFAULT_REGION')

    provider = BedrockProvider(bedrock_client=bedrock_client)

    assert provider.mantle_base_url == 'https://bedrock-mantle.us-west-2.api.aws'


def test_bedrock_mantle_infers_base_url_from_injected_clients(env: TestEnv, mocker: MockerFixture) -> None:
    env.remove('AWS_DEFAULT_REGION')
    bedrock_client = mocker.Mock()
    bedrock_client.meta.endpoint_url = 'https://bedrock-runtime.example.com'
    bedrock_client.meta.region_name = None
    openai_client = AsyncBedrockOpenAI(
        api_key='test-api-key',
        aws_region='us-east-1',
        base_url='https://openai.example.com/openai/v1',
    )
    anthropic_client = AsyncAnthropicBedrockMantle(
        api_key='test-api-key',
        aws_region='us-east-1',
        base_url='https://anthropic.example.com/anthropic',
    )

    openai_provider = BedrockProvider(bedrock_client=bedrock_client, mantle_openai_client=openai_client)
    anthropic_provider = BedrockProvider(bedrock_client=bedrock_client, mantle_anthropic_client=anthropic_client)
    unconfigured_provider = BedrockProvider(bedrock_client=bedrock_client)

    assert openai_provider.mantle_base_url == 'https://openai.example.com'
    assert anthropic_provider.mantle_base_url == 'https://anthropic.example.com'
    with pytest.raises(UserError, match='pass `mantle_base_url`'):
        unconfigured_provider.mantle_base_url


async def test_bedrock_mantle_provider_reopens_http_client() -> None:
    provider = BedrockProvider()
    model = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider)
    first_http_client = model.client._client  # pyright: ignore[reportPrivateUsage]

    async with model:
        pass
    assert first_http_client.is_closed

    async with model:
        assert model.client._client is not first_http_client  # pyright: ignore[reportPrivateUsage]
        assert not model.client._client.is_closed  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('model_name', 'explicit_mantle', 'expected'),
    [
        ('openai.gpt-5.6-luna', False, 'mantle-openai-responses'),
        ('openai.gpt-5.10', False, 'mantle-openai-responses'),
        ('openai.gpt-oss-120b', False, 'converse'),
        ('openai.gpt-oss-120b', True, 'mantle-openai-responses'),
        ('openai.gpt-oss-safeguard-20b', True, 'mantle-openai-chat'),
        ('openai.future-model', True, 'mantle-openai-chat'),
        ('anthropic.claude-sonnet-5', True, 'mantle-anthropic-messages'),
    ],
)
def test_bedrock_model_interface(model_name: str, explicit_mantle: bool, expected: str) -> None:
    assert bedrock_model_interface(model_name, explicit_mantle=explicit_mantle) == expected


def test_bedrock_mantle_unsupported_model() -> None:
    with pytest.raises(UserError, match='Use the `bedrock:` prefix'):
        infer_model('bedrock-mantle:amazon.nova-2-lite-v1:0')


def test_bedrock_requires_bedrock_provider() -> None:
    openai_provider = OpenAIProvider(api_key='test-api-key')
    with pytest.raises(UserError, match='require a `BedrockProvider`'):
        infer_model('bedrock:openai.gpt-5.6-luna', lambda _: openai_provider)


def test_bedrock_mantle_provider_rejects_wrong_interfaces() -> None:
    provider = BedrockProvider()

    with pytest.raises(UserError, match='does not use a Bedrock Mantle OpenAI interface'):
        provider.mantle_openai_client('anthropic.claude-sonnet-5')
    with pytest.raises(UserError, match='is not an OpenAI model'):
        provider.mantle_model_profile('anthropic.claude-sonnet-5', 'mantle-openai-responses')
    with pytest.raises(UserError, match='is not an Anthropic model'):
        provider.mantle_model_profile('openai.gpt-oss-120b', 'mantle-anthropic-messages')
    with pytest.raises(UserError, match='is not a Bedrock Mantle interface'):
        provider.mantle_model_profile('openai.gpt-oss-120b', 'converse')


@pytest.mark.parametrize(
    ('model_id', 'model_interface', 'system', 'base_url'),
    [
        (
            'bedrock:openai.gpt-5.6-luna',
            'responses',
            'bedrock-mantle',
            'https://bedrock-mantle.us-east-1.api.aws/openai/v1/',
        ),
        (
            'bedrock:openai.gpt-oss-120b',
            'converse',
            'bedrock',
            'https://bedrock-runtime.us-east-1.amazonaws.com',
        ),
        (
            'bedrock-mantle:openai.gpt-oss-120b',
            'responses',
            'bedrock-mantle',
            'https://bedrock-mantle.us-east-1.api.aws/v1/',
        ),
        (
            'bedrock-mantle:openai.gpt-oss-safeguard-20b',
            'chat',
            'bedrock-mantle',
            'https://bedrock-mantle.us-east-1.api.aws/v1/',
        ),
        (
            'bedrock-mantle:anthropic.claude-sonnet-5',
            'messages',
            'bedrock-mantle',
            'https://bedrock-mantle.us-east-1.api.aws/anthropic/',
        ),
    ],
)
def test_bedrock_mantle_infer_model(
    model_id: str,
    model_interface: Literal['responses', 'converse', 'chat', 'messages'],
    system: str,
    base_url: str,
) -> None:
    model = infer_model(model_id)
    if model_interface == 'responses':
        assert isinstance(model, BedrockMantleResponsesModel)
    elif model_interface == 'converse':
        assert isinstance(model, BedrockConverseModel)
    elif model_interface == 'chat':
        assert isinstance(model, BedrockMantleChatModel)
    elif model_interface == 'messages':
        assert isinstance(model, BedrockMantleMessagesModel)
    else:
        assert_never(model_interface)
    assert (model.model_name, model.system, model.base_url) == (model_id.partition(':')[2], system, base_url)


def test_bedrock_mantle_chat_model_supports_gpt_oss() -> None:
    model = BedrockMantleChatModel('openai.gpt-oss-120b')
    assert model.base_url == 'https://bedrock-mantle.us-east-1.api.aws/v1/'


def test_gateway_bedrock_remains_on_converse() -> None:
    provider = gateway_provider('bedrock', api_key='test-api-key', base_url='https://gateway.pydantic.dev/proxy')
    model = infer_model('gateway/bedrock:openai.gpt-oss-120b', lambda _: provider)

    assert isinstance(model, BedrockConverseModel)
    assert model.base_url == 'https://gateway.pydantic.dev/proxy/bedrock'


def test_bedrock_mantle_profiles() -> None:
    assert infer_model_profile('bedrock-mantle:openai.gpt-5.6-luna') == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
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
            'supported_native_tools': frozenset(),
            'openai_responses_tool_call_ids_are_response_scoped': True,
        }
    )
    assert (
        infer_model_profile('bedrock-mantle:openai.gpt-oss-120b').get(
            'openai_responses_tool_call_ids_are_response_scoped', False
        )
        is False
    )
    anthropic_profile = infer_model_profile('bedrock-mantle:anthropic.claude-sonnet-5')
    assert anthropic_profile.get('supports_json_schema_output', False) is False
    assert anthropic_profile.get('supported_native_tools') == frozenset()
    assert infer_model_profile('bedrock:openai.gpt-5.6-luna') == infer_model_profile(
        'bedrock-mantle:openai.gpt-5.6-luna'
    )
    assert infer_model_profile('bedrock-mantle:amazon.nova-2-lite-v1:0') == DEFAULT_PROFILE
