import re

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ThinkingPart
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
    from pydantic_ai.providers.vllm import VLLMProvider

    from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


class CityLocation(BaseModel):
    city: str


def test_vllm_provider() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    assert provider.name == 'vllm'
    assert provider.base_url == 'http://localhost:8000/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)


def test_vllm_provider_need_base_url(env: TestEnv) -> None:
    env.remove('VLLM_BASE_URL')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `VLLM_BASE_URL` environment variable or pass it via `VLLMProvider(base_url=...)`'
            ' to use the vLLM provider.'
        ),
    ):
        VLLMProvider()


def test_vllm_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_BASE_URL', 'https://custom.vllm.com/v1/')
    provider = VLLMProvider()
    assert provider.base_url == 'https://custom.vllm.com/v1/'


def test_vllm_provider_api_key_placeholder(env: TestEnv) -> None:
    env.remove('VLLM_API_KEY')
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    assert provider.client.api_key == 'api-key-not-set'


def test_vllm_provider_with_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_BASE_URL', 'http://localhost:8000/v1/')
    monkeypatch.setenv('VLLM_API_KEY', 'env-key')
    provider = VLLMProvider()
    assert provider.client.api_key == 'env-key'


def test_vllm_provider_explicit_api_key() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/', api_key='explicit-key')
    assert provider.client.api_key == 'explicit-key'


def test_vllm_provider_explicit_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_BASE_URL', 'https://env.vllm.com/v1/')
    monkeypatch.setenv('VLLM_API_KEY', 'env-key')
    provider = VLLMProvider(base_url='https://explicit.vllm.com/v1/', api_key='explicit-key')
    assert provider.base_url == 'https://explicit.vllm.com/v1/'
    assert provider.client.api_key == 'explicit-key'


def test_vllm_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = VLLMProvider(http_client=http_client, base_url='http://localhost:8000/v1/')
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


def test_vllm_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:8000/v1/', api_key='test')
    provider = VLLMProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_vllm_provider_openai_client_is_exclusive() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:8000/v1/', api_key='test')
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `base_url`'):
        VLLMProvider(openai_client=openai_client, base_url='http://localhost:8000/v1/')  # type: ignore[call-overload]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `http_client`'):
        VLLMProvider(openai_client=openai_client, http_client=httpx.AsyncClient())  # type: ignore[call-overload]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `api_key`'):
        VLLMProvider(openai_client=openai_client, api_key='test')  # type: ignore[call-overload]


async def test_vllm_provider_recreates_closed_owned_client() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    owned = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
    assert owned is not None
    await owned.aclose()
    async with provider:
        new_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert new_client is not owned
        assert not new_client.is_closed


@pytest.mark.parametrize(
    ('model_name', 'schema_transformer', 'supports_thinking'),
    [
        ('meta-llama/Llama-3-8B-Instruct', InlineDefsJsonSchemaTransformer, False),
        ('google/gemma-3-4b-it', GoogleJsonSchemaTransformer, False),
        ('Qwen/Qwen3-32B', InlineDefsJsonSchemaTransformer, False),
        ('Qwen/QwQ-32B', InlineDefsJsonSchemaTransformer, False),
        ('deepseek-ai/DeepSeek-R1', OpenAIJsonSchemaTransformer, True),
        ('mistralai/Magistral-Small-2509', OpenAIJsonSchemaTransformer, True),
        ('CohereLabs/command-a-reasoning-08-2025', OpenAIJsonSchemaTransformer, True),
        ('openai/gpt-oss-20b', OpenAIJsonSchemaTransformer, False),
        ('unknown-model', OpenAIJsonSchemaTransformer, False),
    ],
)
def test_vllm_provider_model_profile(
    model_name: str, schema_transformer: type[JsonSchemaTransformer], supports_thinking: bool
) -> None:
    profile = VLLMProvider.model_profile(model_name)

    assert profile is not None
    assert profile.get('json_schema_transformer') is schema_transformer
    assert profile.get('supports_thinking', False) is supports_thinking


def test_vllm_provider_profile_overrides() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
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


def test_vllm_provider_family_tool_flags_survive() -> None:
    """vLLM ignores `tool_choice='required'` for gpt-oss (vllm#44216), so the family opt-outs must survive the merge."""
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')

    harmony_profile = provider.model_profile('gpt-oss-20b')
    assert harmony_profile is not None
    assert harmony_profile.get('openai_supports_tool_choice_required', True) is False

    qwen_coder_profile = provider.model_profile('Qwen/Qwen3-Coder-480B-A35B-Instruct')
    assert qwen_coder_profile is not None
    assert qwen_coder_profile.get('openai_supports_tool_choice_required', True) is False
    assert qwen_coder_profile.get('openai_supports_strict_tool_definition', True) is False


async def test_vllm_provider_merges_leading_system_messages(allow_model_requests: None) -> None:
    """Mocked because it pins the request shape that strict vLLM chat templates rejected in issue #5812.

    `instructions` plus `PromptedOutput` must produce a single leading system message carrying both.
    """
    response = completion_message(ChatCompletionMessage(content='{"city": "Paris"}', role='assistant'))
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
    agent = Agent(model, instructions='Answer accurately.', output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the capital of France?')

    assert result.output == CityLocation(city='Paris')
    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user']
    system_content = messages[0]['content']
    assert system_content.startswith('Answer accurately.')
    assert '"city"' in system_content


async def test_vllm_provider_native_output_injects_schema(allow_model_requests: None) -> None:
    """Mocked because it pins the request shape for `NativeOutput` on vLLM.

    Guided decoding is pure token masking, so the schema must also reach the model through the
    instructions (issue #3490), alongside the `json_schema` response format.
    """
    response = completion_message(ChatCompletionMessage(content='{"city": "Paris"}', role='assistant'))
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the capital of France?')

    assert result.output == CityLocation(city='Paris')
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['response_format']['type'] == 'json_schema'
    messages = kwargs['messages']
    assert [message['role'] for message in messages] == ['system', 'user']
    assert '"city"' in messages[0]['content']


async def test_vllm_provider_parses_reasoning_content_fallback(allow_model_requests: None) -> None:
    """Mocked because it pins the wire shape of pre-rename vLLM servers, which a live cassette can't produce.

    The profile prefers `reasoning`, but older vLLM returns `reasoning_content`; both must parse.
    """
    response = completion_message(
        ChatCompletionMessage.model_construct(content='Paris', reasoning_content='Consider France.', role='assistant')
    )
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')

    assert result.output == 'Paris'
    thinking_parts = [part for part in result.response.parts if isinstance(part, ThinkingPart)]
    assert [(part.id, part.content) for part in thinking_parts] == [('reasoning_content', 'Consider France.')]


async def test_vllm_provider_no_duplicate_thinking_parts(allow_model_requests: None) -> None:
    """Mocked because it pins a wire shape a live cassette can't reliably produce.

    vLLM 0.11.2+ returns identical `reasoning` and `reasoning_content` fields for backwards compatibility;
    only one `ThinkingPart` must come out. See https://github.com/vllm-project/vllm/issues/27755.
    """
    response = completion_message(
        ChatCompletionMessage.model_construct(
            content='Paris', reasoning='Consider France.', reasoning_content='Consider France.', role='assistant'
        )
    )
    mock_client = MockOpenAI.create_mock(response)
    model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')

    assert result.output == 'Paris'
    thinking_parts = [part for part in result.response.parts if isinstance(part, ThinkingPart)]
    assert [(part.id, part.content) for part in thinking_parts] == [('reasoning', 'Consider France.')]
