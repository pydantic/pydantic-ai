import re

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.sglang import SGLangProvider

    from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


class CityLocation(BaseModel):
    city: str


def test_sglang_provider() -> None:
    provider = SGLangProvider(base_url='http://localhost:30000/v1/')
    assert provider.name == 'sglang'
    assert provider.base_url == 'http://localhost:30000/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)


def test_sglang_provider_need_base_url(env: TestEnv) -> None:
    env.remove('SGLANG_BASE_URL')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SGLANG_BASE_URL` environment variable or pass it via `SGLangProvider(base_url=...)`'
            ' to use the SGLang provider.'
        ),
    ):
        SGLangProvider()


def test_sglang_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SGLANG_BASE_URL', 'https://custom.sglang.com/v1/')
    provider = SGLangProvider()
    assert provider.base_url == 'https://custom.sglang.com/v1/'


def test_sglang_provider_api_key_placeholder(env: TestEnv) -> None:
    env.remove('SGLANG_API_KEY')
    provider = SGLangProvider(base_url='http://localhost:30000/v1/')
    assert provider.client.api_key == 'api-key-not-set'


def test_sglang_provider_with_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SGLANG_BASE_URL', 'http://localhost:30000/v1/')
    monkeypatch.setenv('SGLANG_API_KEY', 'env-key')
    provider = SGLangProvider()
    assert provider.client.api_key == 'env-key'


def test_sglang_provider_explicit_api_key() -> None:
    provider = SGLangProvider(base_url='http://localhost:30000/v1/', api_key='explicit-key')
    assert provider.client.api_key == 'explicit-key'


def test_sglang_provider_explicit_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SGLANG_BASE_URL', 'https://env.sglang.com/v1/')
    monkeypatch.setenv('SGLANG_API_KEY', 'env-key')
    provider = SGLangProvider(base_url='https://explicit.sglang.com/v1/', api_key='explicit-key')
    assert provider.base_url == 'https://explicit.sglang.com/v1/'
    assert provider.client.api_key == 'explicit-key'


def test_sglang_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:30000/v1/', api_key='test')
    provider = SGLangProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_sglang_provider_openai_client_is_exclusive() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:30000/v1/', api_key='test')
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `base_url`'):
        SGLangProvider(openai_client=openai_client, base_url='http://localhost:30000/v1/')  # type: ignore[call-overload]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `http_client`'):
        SGLangProvider(openai_client=openai_client, http_client=httpx.AsyncClient())  # type: ignore[call-overload]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `api_key`'):
        SGLangProvider(openai_client=openai_client, api_key='test')  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ('model_name', 'schema_transformer'),
    [
        ('meta-llama/Llama-3-8B-Instruct', InlineDefsJsonSchemaTransformer),
        ('google/gemma-3-4b-it', GoogleJsonSchemaTransformer),
        ('Qwen/Qwen3-32B', InlineDefsJsonSchemaTransformer),
        ('Qwen/QwQ-32B', InlineDefsJsonSchemaTransformer),
        ('deepseek-ai/DeepSeek-R1', OpenAIJsonSchemaTransformer),
        ('mistralai/Magistral-Small-2509', OpenAIJsonSchemaTransformer),
        ('CohereLabs/command-a-reasoning-08-2025', OpenAIJsonSchemaTransformer),
        ('openai/gpt-oss-20b', OpenAIJsonSchemaTransformer),
        ('zai-org/GLM-4.7', OpenAIJsonSchemaTransformer),
        ('unknown-model', OpenAIJsonSchemaTransformer),
    ],
)
def test_sglang_provider_model_profile(model_name: str, schema_transformer: type[JsonSchemaTransformer]) -> None:
    profile = SGLangProvider.model_profile(model_name)

    assert profile is not None
    assert profile.get('json_schema_transformer') is schema_transformer


def test_sglang_provider_profile_overrides() -> None:
    provider = SGLangProvider(base_url='http://localhost:30000/v1/')
    for model in (
        'llama-3-8b',
        'qwen3',
        'qwen-3-coder',
        'mistral-small',
        'gemma-3',
        'command-r',
        'gpt-oss-20b',
        'unknown-model',
    ):
        profile = provider.model_profile(model)
        assert profile is not None
        assert profile.get('openai_chat_supports_multiple_system_messages', True) is False
        assert profile.get('openai_chat_supports_document_input', True) is False
        assert profile.get('supports_tool_return_schema', True) is False
        assert profile.get('native_output_requires_schema_in_instructions', False) is True


async def test_sglang_provider_merges_leading_system_messages(allow_model_requests: None) -> None:
    """Mocked because it pins the request shape that strict server-side chat templates reject (issue #5812).

    `instructions` plus `PromptedOutput` must produce a single leading system message carrying both.
    """
    response = completion_message(ChatCompletionMessage(content='{"city": "Paris"}', role='assistant'))
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=SGLangProvider(openai_client=mock_client))
    agent = Agent(model, instructions='Answer accurately.', output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the capital of France?')

    assert result.output == CityLocation(city='Paris')
    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user']
    system_content = messages[0]['content']
    assert system_content.startswith('Answer accurately.')
    assert '"city"' in system_content


async def test_sglang_provider_native_output_injects_schema(allow_model_requests: None) -> None:
    """Mocked because it pins the request shape for `NativeOutput` on SGLang.

    Grammar-constrained decoding is pure token masking, so the schema must also reach the model through
    the instructions (issue #3490), alongside the `json_schema` response format.
    """
    response = completion_message(ChatCompletionMessage(content='{"city": "Paris"}', role='assistant'))
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=SGLangProvider(openai_client=mock_client))
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the capital of France?')

    assert result.output == CityLocation(city='Paris')
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['response_format'] == snapshot(
        {
            'type': 'json_schema',
            'json_schema': {
                'name': 'CityLocation',
                'schema': {
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'title': 'CityLocation',
                    'type': 'object',
                },
                'strict': True,
            },
        }
    )
    assert kwargs['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': "\nAlways respond with a JSON object that's compatible with this schema:\n\n"
                '{"properties": {"city": {"type": "string"}}, "required": ["city"], '
                '"title": "CityLocation", "type": "object"}\n\n'
                "Don't include any text or Markdown fencing before or after.\n",
            },
            {'role': 'user', 'content': 'What is the capital of France?'},
        ]
    )
