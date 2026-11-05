import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from unittest.mock import patch

import pytest
from genai_prices.data_snapshot import get_snapshot

from pydantic_ai import Agent, UserError
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    DEFAULT_PROFILE,
    AbstractModel,
    Model,
    ModelRequestParameters,
    infer_model,
    infer_model_profile,
    parse_model_id,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import legacy_retry_prompt_part, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers import openai

if not imports_successful():
    pytest.skip('model packages were not installed', allow_module_level=True)  # pragma: lax no cover


# TODO(Marcelo): We need to add Vertex AI to the test cases.

TEST_CASES = [
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/openai:gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIResponsesModel,
        id='gateway/openai:gpt-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/chat:gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIChatModel,
        id='gateway/chat:gpt-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/responses:gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIResponsesModel,
        id='gateway/responses:gpt-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
        id='gateway/groq:llama-3.3-70b-versatile',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/google:gemini-1.5-flash',
        'gemini-1.5-flash',
        'google-cloud',
        'google',
        GoogleModel,
        id='gateway/google:gemini-1.5-flash',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/anthropic:claude-opus-4-7',
        'claude-opus-4-7',
        'anthropic',
        'anthropic',
        AnthropicModel,
        id='gateway/anthropic:claude-opus-4-7',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_gatewayapikey'},
        'gateway/converse:amazon.nova-micro-v1:0',
        'amazon.nova-micro-v1:0',
        'bedrock',
        'bedrock',
        BedrockConverseModel,
        id='gateway/converse:amazon.nova-micro-v1:0',
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'openai:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'openai',
        'openai',
        OpenAIResponsesModel,
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'openai-chat:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'openai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {
            'AZURE_OPENAI_API_KEY': 'azure-openai-api-key',
            'AZURE_OPENAI_ENDPOINT': 'azure-openai-endpoint',
            'OPENAI_API_VERSION': '2024-12-01-preview',
        },
        'azure:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'azure',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {
            'AZURE_OPENAI_API_KEY': 'azure-openai-api-key',
            'AZURE_OPENAI_ENDPOINT': 'azure-openai-endpoint',
            'OPENAI_API_VERSION': '2024-12-01-preview',
        },
        'azure-responses:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'azure',
        'openai',
        OpenAIResponsesModel,
    ),
    pytest.param(
        {'GEMINI_API_KEY': 'gemini-api-key'},
        'google:gemini-1.5-flash',
        'gemini-1.5-flash',
        'google',
        'google',
        GoogleModel,
    ),
    pytest.param(
        {'ANTHROPIC_API_KEY': 'anthropic-api-key'},
        'anthropic:claude-haiku-4-5',
        'claude-haiku-4-5',
        'anthropic',
        'anthropic',
        AnthropicModel,
    ),
    pytest.param(
        {'GROQ_API_KEY': 'groq-api-key'},
        'groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
    ),
    pytest.param(
        {'MISTRAL_API_KEY': 'mistral-api-key'},
        'mistral:mistral-small-latest',
        'mistral-small-latest',
        'mistral',
        'mistral',
        MistralModel,
    ),
    pytest.param(
        {'CO_API_KEY': 'co-api-key'},
        'cohere:command',
        'command',
        'cohere',
        'cohere',
        CohereModel,
    ),
    pytest.param(
        {
            'AWS_ACCESS_KEY_ID': 'test-access-key',
            'AWS_DEFAULT_REGION': 'aws-default-region',
            'AWS_SECRET_ACCESS_KEY': 'test-secret-key',
        },
        'bedrock:bedrock-claude-haiku-4-5',
        'bedrock-claude-haiku-4-5',
        'bedrock',
        'bedrock',
        BedrockConverseModel,
    ),
    pytest.param(
        {'GITHUB_API_KEY': 'github-api-key'},
        'github:xai/grok-3-mini',
        'xai/grok-3-mini',
        'github',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'GITHUB_COPILOT_API_KEY': 'github-copilot-api-key'},
        'github-copilot:claude-haiku-4.5',
        'claude-haiku-4.5',
        'github-copilot',
        'github_copilot',
        GitHubCopilotModel,
    ),
    pytest.param(
        {'META_API_KEY': 'meta-api-key'},
        'meta:muse-spark-1.3',
        'muse-spark-1.3',
        'meta',
        'openai',
        OpenAIResponsesModel,
    ),
    pytest.param(
        {'MOONSHOTAI_API_KEY': 'moonshotai-api-key'},
        'moonshotai:kimi-k2-0711-preview',
        'kimi-k2-0711-preview',
        'moonshotai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'openai-responses:gpt-4o',
        'gpt-4o',
        'openai',
        'openai',
        OpenAIResponsesModel,
    ),
    pytest.param(
        {'OPENROUTER_API_KEY': 'openrouter-api-key'},
        'openrouter:anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.5-sonnet',
        'openrouter',
        'openrouter',
        OpenRouterModel,
    ),
    pytest.param(
        {'VLLM_BASE_URL': 'http://localhost:8000/v1/'},
        'vllm:Qwen/Qwen3-32B',
        'Qwen/Qwen3-32B',
        'vllm',
        'openai',
        OpenAIChatModel,
    ),
]


@pytest.mark.parametrize(
    'mock_env_vars, model_name, expected_model_name, expected_system, module_name, model_class', TEST_CASES
)
def test_infer_model(
    mock_env_vars: dict[str, str],
    model_name: str,
    expected_model_name: str,
    expected_system: str,
    module_name: str,
    model_class: type[Model],
):
    with patch.dict(os.environ, mock_env_vars):
        model_module = import_module(f'pydantic_ai.models.{module_name}')
        expected_model = getattr(model_module, model_class.__name__)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
            m = infer_model(model_name)

        assert isinstance(m, expected_model)
        assert m.model_name == expected_model_name
        assert m.system == expected_system

        assert m.model_id == f'{expected_system}:{expected_model_name}'

        m2 = infer_model(m)
        assert m2 is m


def test_infer_model_with_provider():
    provider_class = openai.OpenAIProvider(api_key='1234', base_url='http://test')
    m = infer_model('openai-chat:gpt-5', lambda x: provider_class)

    assert isinstance(m, OpenAIChatModel)
    assert m._provider is provider_class  # pyright: ignore[reportPrivateUsage]
    assert m._provider.base_url == 'http://test'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('model_name', 'message'),
    [
        pytest.param('foobar', 'Unknown model: foobar', id='unqualified'),
        pytest.param(
            'claude:sonnet-5',
            "Unknown model: claude:sonnet-5. Did you mean 'anthropic:claude-sonnet-5'?",
            id='close-match',
        ),
        pytest.param(
            'claude:potato-5',
            "Unknown model: claude:potato-5. Did you mean 'anthropic:claude-sonnet-5'?",
            id='loose-match',
        ),
        pytest.param(
            'anthropicc:claude-sonnet-5',
            "Unknown model: anthropicc:claude-sonnet-5. Did you mean 'anthropic:claude-sonnet-5'?",
            id='provider-typo',
        ),
        pytest.param(
            'unknown:claude-sonnet-5',
            "Unknown model: unknown:claude-sonnet-5. Did you mean 'anthropic:claude-sonnet-5'?",
            id='known-model-name',
        ),
        pytest.param(
            'anthropic-claude-sonnet-5',
            "Unknown model: anthropic-claude-sonnet-5. Did you mean 'anthropic:claude-sonnet-5'?",
            id='missing-colon',
        ),
        pytest.param(
            'openia:gpt-5.2',
            "Unknown model: openia:gpt-5.2. Did you mean 'openai:gpt-5.2'?",
            id='prefer-direct-provider',
        ),
        pytest.param('unknown:potato', 'Unknown model: unknown:potato', id='no-match'),
    ],
)
def test_infer_str_unknown(model_name: str, message: str):
    with pytest.raises(UserError, match=f'^{re.escape(message)}$'):
        infer_model(model_name)


def test_agent_suggests_known_model_name():
    with pytest.raises(UserError, match="Did you mean 'anthropic:claude-sonnet-5'"):
        Agent('claude:sonnet-5')


def test_infer_model_allows_unknown_name_for_known_provider():
    provider = openai.OpenAIProvider(api_key='1234', base_url='http://test')
    model = infer_model('openai-chat:potato-5', lambda _: provider)

    assert model.model_name == 'potato-5'


def test_infer_model_preserves_custom_provider_factory_error():
    def provider_factory(_provider_name: str):
        raise ValueError('custom provider error')

    with pytest.raises(ValueError, match='custom provider error'):
        infer_model('openai:gpt-5', provider_factory)


def test_infer_model_preserves_provider_initialization_error():
    with (
        patch.object(openai.OpenAIProvider, '__init__', side_effect=ValueError('provider initialization error')),
        pytest.raises(ValueError, match='provider initialization error'),
    ):
        infer_model('openai:gpt-5')


@pytest.mark.parametrize(
    ('model_id', 'expected'),
    [
        pytest.param('openai:gpt-5', ('openai', 'gpt-5'), id='provider:model'),
        pytest.param('anthropic:claude-3', ('anthropic', 'claude-3'), id='anthropic:model'),
        pytest.param('gpt-4', (None, 'gpt-4'), id='no-prefix'),
        pytest.param('unknown-model', (None, 'unknown-model'), id='unknown'),
        pytest.param('custom:model:with:colons', ('custom', 'model:with:colons'), id='multiple-colons'),
        pytest.param('gateway/openai:gpt-5', ('gateway/openai', 'gpt-5'), id='gateway-prefix'),
    ],
)
def test_parse_model_id(model_id: str, expected: tuple[str | None, str]):
    assert parse_model_id(model_id) == expected


@pytest.mark.parametrize(
    ('model_id', 'is_default'),
    [
        pytest.param('openai:gpt-5', False, id='openai'),
        pytest.param('anthropic:claude-sonnet-4-5', False, id='anthropic'),
        pytest.param('gateway/openai:gpt-5', False, id='gateway-openai'),
        pytest.param('gateway/google-cloud:gemini-2.5-pro', False, id='gateway-google-cloud'),
        pytest.param('unknown-provider:some-model', True, id='unknown-provider'),
        pytest.param('unknown-model', True, id='unknown-no-prefix'),
        pytest.param('nebius:model-without-slash', False, id='provider-unknown-model'),
        pytest.param('google:gemini-2.0-flash', False, id='google-shorthand'),
        pytest.param('openrouter:model-without-slash', True, id='openrouter-no-slash'),
        # Together (OpenAI-compatible) returns the OpenAI default profile for a slashless name
        # rather than crashing — like `nebius` above — so it's not `DEFAULT_PROFILE`.
        pytest.param('together:model-without-slash', False, id='together-no-slash'),
    ],
)
def test_infer_model_profile(model_id: str, is_default: bool):
    profile = infer_model_profile(model_id)
    if is_default:
        assert profile is DEFAULT_PROFILE
    else:
        assert profile is not DEFAULT_PROFILE


@pytest.mark.parametrize(
    ('model_id', 'provider_path', 'model_name'),
    [
        pytest.param('openai:gpt-5', 'pydantic_ai.providers.openai.OpenAIProvider', 'gpt-5', id='openai'),
        pytest.param(
            'anthropic:claude-sonnet-4-5',
            'pydantic_ai.providers.anthropic.AnthropicProvider',
            'claude-sonnet-4-5',
            id='anthropic',
        ),
        pytest.param(
            'google:gemini-2.0-flash',
            'pydantic_ai.providers.google.GoogleProvider',
            'gemini-2.0-flash',
            id='google',
        ),
    ],
)
def test_infer_model_profile_matches_provider(model_id: str, provider_path: str, model_name: str):
    """Verify provider profile values are preserved when model metadata is added."""
    module_path, class_name = provider_path.rsplit('.', 1)
    module = import_module(module_path)
    provider_class = getattr(module, class_name)

    profile = infer_model_profile(model_id)
    provider_profile = provider_class.model_profile(model_name)
    assert provider_profile is not None
    assert {key: value for key, value in profile.items() if key != 'context_window'} == provider_profile


@pytest.mark.parametrize(
    ('gateway_model_id', 'direct_model_id'),
    [
        ('gateway/openai:gpt-5', 'openai:gpt-5'),
        ('gateway/chat:gpt-5', 'openai:gpt-5'),
        ('gateway/responses:gpt-5', 'openai:gpt-5'),
        ('gateway/anthropic:claude-sonnet-4-5', 'anthropic:claude-sonnet-4-5'),
        (
            'gateway/converse:anthropic.claude-sonnet-4-5-20250929-v1:0',
            'bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0',
        ),
        ('gateway/google:gemini-2.5-flash', 'google-cloud:gemini-2.5-flash'),
    ],
)
def test_infer_model_profile_context_window_through_gateway(gateway_model_id: str, direct_model_id: str):
    """A gateway model ID gets the same `context_window` as the provider it routes to, aliases included."""
    direct = infer_model_profile(direct_model_id).get('context_window')
    assert direct is not None
    assert infer_model_profile(gateway_model_id).get('context_window') == direct


def test_infer_model_profile_respects_explicit_unknown_context_window():
    """An explicit provider value suppresses the genai-prices lookup for raw model IDs."""
    with patch(
        'pydantic_ai.providers.openai.OpenAIProvider.model_profile',
        return_value=ModelProfile(context_window=None),
    ):
        assert infer_model_profile('openai:gpt-5').get('context_window') is None


def test_infer_model_profile_fills_default_profile_with_context_window():
    """Metadata inferred for a model without a provider profile preserves the default profile."""
    with (
        patch('pydantic_ai.providers.openai.OpenAIProvider.model_profile', return_value=None),
        patch('pydantic_ai.models.lookup_context_window', return_value=123),
    ):
        profile = infer_model_profile('openai:gpt-5')

    assert profile == {**DEFAULT_PROFILE, 'context_window': 123}


def test_prepare_request_rejects_unsupported_text_output():
    params = ModelRequestParameters()

    with pytest.raises(UserError, match='Text output is not supported by this model'):
        TestModel(profile={'supports_text_output': False}).prepare_request(None, params)

    _, prepared = TestModel().prepare_request(None, params)
    assert prepared.allow_text_output is True


def test_custom_provider_instance_method_model_profile():
    """Verify that a custom provider using the old instance-method model_profile pattern still works for non-Temporal usage.

    Before the @staticmethod change, Provider.model_profile was an instance method.
    Custom providers that still define it as `def model_profile(self, model_name)` should
    continue to work when called on an instance (e.g. `provider.model_profile(model_name)`).
    """
    from pydantic_ai.profiles import ModelProfile
    from pydantic_ai.providers import Provider

    class LegacyCustomProvider(Provider[None]):
        """A custom provider using the old instance-method pattern."""

        @property
        def name(self) -> str:
            return 'legacy-custom'

        @property
        def base_url(self) -> str:
            return 'https://example.com'

        @property
        def client(self) -> None:
            return None

        # Old-style instance method (not @staticmethod or @classmethod)
        def model_profile(self, model_name: str) -> ModelProfile | None:  # type: ignore[override]
            return ModelProfile()

    provider = LegacyCustomProvider()
    assert provider.name == 'legacy-custom'
    assert provider.base_url == 'https://example.com'
    assert provider.client is None
    # Instance call should still work
    profile = provider.model_profile('some-model')
    assert isinstance(profile, dict)


def _request_parts(messages: list[ModelMessage]) -> list[list[tuple[str, object]]]:
    """Flatten each `ModelRequest`'s parts to `(type, content)` tuples for compact assertions."""
    return [
        [(type(part).__name__, getattr(part, 'content', None)) for part in message.parts]
        for message in messages
        if isinstance(message, ModelRequest)
    ]


@pytest.mark.parametrize(
    'supports_inline,messages,expected',
    [
        pytest.param(
            False,
            [
                ModelRequest(parts=[UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
                ModelRequest(parts=[SystemPromptPart(content='Be terse.'), UserPromptPart(content='ok?')]),
            ],
            [
                [('UserPromptPart', 'hi')],
                [('UserPromptPart', '<system>Be terse.</system>'), ('UserPromptPart', 'ok?')],
            ],
            id='wraps-non-leading-system-prompt',
        ),
        pytest.param(
            True,
            [
                ModelRequest(parts=[UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
                ModelRequest(parts=[SystemPromptPart(content='Be terse.'), UserPromptPart(content='ok?')]),
            ],
            [
                [('UserPromptPart', 'hi')],
                [('SystemPromptPart', 'Be terse.'), ('UserPromptPart', 'ok?')],
            ],
            id='no-op-when-inline-supported',
        ),
        pytest.param(
            False,
            [
                ModelRequest(parts=[UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
                ModelRequest(
                    parts=[
                        SystemPromptPart(content='A'),
                        SystemPromptPart(content='B'),
                        UserPromptPart(content='c'),
                    ]
                ),
            ],
            [
                [('UserPromptPart', 'hi')],
                [
                    ('UserPromptPart', '<system>A</system>'),
                    ('UserPromptPart', '<system>B</system>'),
                    ('UserPromptPart', 'c'),
                ],
            ],
            id='wraps-multiple-non-leading-system-prompts',
        ),
        pytest.param(
            False,
            [
                ModelRequest(parts=[SystemPromptPart(content='You are helpful.'), UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
            ],
            [[('SystemPromptPart', 'You are helpful.'), ('UserPromptPart', 'hi')]],
            id='keeps-leading-system-prompt',
        ),
        pytest.param(
            False,
            [
                ModelRequest(parts=[SystemPromptPart(content='You are helpful.'), UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
                ModelRequest(parts=[UserPromptPart(content='follow up')]),
            ],
            [
                [('SystemPromptPart', 'You are helpful.'), ('UserPromptPart', 'hi')],
                [('UserPromptPart', 'follow up')],
            ],
            id='no-non-leading-system-prompt-to-wrap',
        ),
        pytest.param(
            False,
            [ModelRequest(parts=[SystemPromptPart(content='hi'), UserPromptPart(content='hello')])],
            [[('SystemPromptPart', 'hi'), ('UserPromptPart', 'hello')]],
            id='single-leading-request',
        ),
        pytest.param(
            False,
            [
                ModelResponse(parts=[TextPart(content='earlier reply')]),
                ModelRequest(parts=[SystemPromptPart(content='Server prompt'), UserPromptPart(content='Follow up')]),
            ],
            [[('SystemPromptPart', 'Server prompt'), ('UserPromptPart', 'Follow up')]],
            id='first-request-is-leading-after-orphan-response',
        ),
        pytest.param(False, [], [], id='no-request'),
    ],
)
def test_prepare_messages_system_prompt_wrapping(
    supports_inline: bool, messages: list[ModelMessage], expected: list[list[tuple[str, object]]]
):
    model = TestModel(profile=ModelProfile(supports_inline_system_prompts=supports_inline))
    assert _request_parts(model.prepare_messages(messages)) == expected


def test_prepare_messages_orders_tool_results_by_call():
    """Results answering the preceding response's calls go out in call order; every other part keeps its slot."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        ModelResponse(
            parts=[
                ToolCallPart('a', {}, tool_call_id='ca'),
                ToolCallPart('b', {}, tool_call_id='cb'),
                ToolCallPart('c', {}, tool_call_id='cc'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('c', 'C', tool_call_id='cc'),
                RetryFeedbackPart('output invalid', retry=RetryInfo(cause='model_retry')),
                ToolReturnPart('b', 'bad args', tool_call_id='cb', outcome='retried'),
                ToolReturnPart('z', 'Z', tool_call_id='not-a-call'),
                ToolReturnPart('a', 'A', tool_call_id='ca'),
                UserPromptPart(content='file from c'),
            ]
        ),
    ]

    prepared = TestModel().prepare_messages(messages)

    assert _request_parts(prepared)[1] == [
        ('ToolReturnPart', 'A'),
        (
            'UserPromptPart',
            "Your previous response failed the application's validation (this is automated feedback, not a message "
            'from the user):\n<retry_feedback>\noutput invalid\n</retry_feedback>',
        ),
        ('ToolReturnPart', 'bad args'),
        ('ToolReturnPart', 'Z'),
        ('ToolReturnPart', 'C'),
        ('UserPromptPart', 'file from c'),
    ]
    # Only the outgoing copy is reordered; the history keeps the order it was stored in.
    assert _request_parts(messages)[1][0] == ('ToolReturnPart', 'C')


def test_prepare_messages_keeps_history_identity_when_tool_results_are_in_call_order():
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('a', {}, tool_call_id='ca'), ToolCallPart('b', {}, tool_call_id='cb')]),
        ModelRequest(parts=[ToolReturnPart('a', 'A', tool_call_id='ca'), ToolReturnPart('b', 'B', tool_call_id='cb')]),
    ]

    assert TestModel().prepare_messages(messages) is messages


def _tool_call_ids(messages: list[ModelMessage]) -> list[tuple[str, str, str | None]]:
    """Each call and return in history order, as `(type, tool_call_id, outcome)`; a retry is a return too."""
    return [
        (type(part).__name__, part.tool_call_id, part.outcome if isinstance(part, ToolReturnPart) else None)
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart)
    ]


def test_prepare_messages_fits_tool_call_ids_to_profile_length():
    """IDs that aren't exactly `tool_call_id_length` ASCII alphanumerics are replaced, consistently per ID.

    A unit test because the edge cases (an 8- or 10-character ID, a 9-character one with `_`, a non-ASCII
    letter) are about which IDs count as fitting, which the Mistral VCR tests only sample. The hashes are
    the same as on the wire there: `call_abc...` maps to `2pESUE4jw` in both.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        ModelResponse(
            parts=[
                ToolCallPart('a', {}, tool_call_id='aB3dE5gH9'),
                ToolCallPart('b', {}, tool_call_id='aB3dE5gH'),
                ToolCallPart('c', {}, tool_call_id='ab_cd_efg'),
                ToolCallPart('d', {}, tool_call_id='aB3dE5gH9X'),
                ToolCallPart('e', {}, tool_call_id='aB3dE5gHé'),
                ToolCallPart('f', {}, tool_call_id='call_abc123def456ghi789jkl012'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('a', 'A', tool_call_id='aB3dE5gH9'),
                # A legacy retry is translated into the retried return it always meant before IDs are fitted.
                legacy_retry_prompt_part('bad args', tool_name='b', tool_call_id='aB3dE5gH'),
                ToolReturnPart('c', 'C', tool_call_id='ab_cd_efg', outcome='retried'),
                ToolReturnPart('d', 'D', tool_call_id='aB3dE5gH9X'),
                ToolReturnPart('e', 'E', tool_call_id='aB3dE5gHé'),
                ToolReturnPart('f', 'F', tool_call_id='call_abc123def456ghi789jkl012'),
            ]
        ),
    ]
    # Dumped rather than deep-copied: copying the deprecated part would construct it again, and warn.
    stored = ModelMessagesTypeAdapter.dump_json(messages)
    model = TestModel(profile=ModelProfile(tool_call_id_length=9))

    prepared = model.prepare_messages(messages)

    assert _tool_call_ids(prepared) == snapshot(
        [
            ('ToolCallPart', 'aB3dE5gH9', None),
            ('ToolCallPart', 'BsnFb66F9', None),
            ('ToolCallPart', '5ZoXBIVHV', None),
            ('ToolCallPart', 'BgySIGZn7', None),
            ('ToolCallPart', 'pFwy2hHZJ', None),
            ('ToolCallPart', '2pESUE4jw', None),
            ('ToolReturnPart', 'aB3dE5gH9', 'success'),
            ('ToolReturnPart', 'BsnFb66F9', 'retried'),
            ('ToolReturnPart', '5ZoXBIVHV', 'retried'),
            ('ToolReturnPart', 'BgySIGZn7', 'success'),
            ('ToolReturnPart', 'pFwy2hHZJ', 'success'),
            ('ToolReturnPart', '2pESUE4jw', 'success'),
        ]
    )
    # The same history prepares to the same IDs every time, and only the outgoing copy changes.
    assert _tool_call_ids(model.prepare_messages(messages)) == _tool_call_ids(prepared)
    assert ModelMessagesTypeAdapter.dump_json(messages) == stored


def test_prepare_messages_keeps_history_identity_when_tool_call_ids_fit():
    """With every ID already fitting, nothing is copied (the `_make_request` identity check relies on it)."""
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('a', {}, tool_call_id='aB3dE5gH9')]),
        ModelRequest(parts=[ToolReturnPart('a', 'A', tool_call_id='aB3dE5gH9')]),
    ]

    assert TestModel(profile=ModelProfile(tool_call_id_length=9)).prepare_messages(messages) is messages


@pytest.mark.parametrize('foreign_first', [True, False], ids=['foreign-first', 'fitting-first'])
def test_prepare_messages_resolves_tool_call_id_collisions_in_history_order(foreign_first: bool):
    """A replacement that's already taken is rehashed, and the ID seen first in history keeps its value.

    A unit test because a real collision needs two IDs to hash alike. A fitting ID that equals another
    ID's hash collides just the same: `call_x` hashes to `nVlA23dIQ`. Whichever comes first keeps that
    value, and the other is rehashed from itself plus a counter, so a growing history never renumbers an
    earlier call.
    """
    foreign = [
        ModelResponse(parts=[ToolCallPart('a', {}, tool_call_id='call_x')]),
        ModelRequest(parts=[ToolReturnPart('a', 'A', tool_call_id='call_x')]),
    ]
    fitting = [
        ModelResponse(parts=[ToolCallPart('b', {}, tool_call_id='nVlA23dIQ')]),
        ModelRequest(parts=[ToolReturnPart('b', 'B', tool_call_id='nVlA23dIQ')]),
    ]
    messages: list[ModelMessage] = [*foreign, *fitting] if foreign_first else [*fitting, *foreign]

    prepared = TestModel(profile=ModelProfile(tool_call_id_length=9)).prepare_messages(messages)

    # One `(tool, ID)` pair per tool: each call still pairs with its own return.
    pairs = {
        (part.tool_name, part.tool_call_id)
        for message in prepared
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart)
    }
    expected = {('a', 'nVlA23dIQ'), ('b', 'Q30Sez398')} if foreign_first else {('a', 'heCJTQ2G1'), ('b', 'nVlA23dIQ')}
    assert pairs == expected


@pytest.mark.parametrize(
    'length,message',
    [
        (0, '`tool_call_id_length` must be at least 1, got 0.'),
        (1, 'The message history has more distinct tool call IDs than fit in 1 alphanumeric characters'),
    ],
)
def test_prepare_messages_rejects_tool_call_id_length_without_room(length: int, message: str):
    """A length that can't hold every distinct ID raises instead of rehashing forever.

    A unit test because no real profile sets such a length; it guards a user-supplied profile.
    """
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('a', {}, tool_call_id=f'call_{index}') for index in range(63)])
    ]

    with pytest.raises(UserError, match=re.escape(message)):
        TestModel(profile=ModelProfile(tool_call_id_length=length)).prepare_messages(messages)


async def test_model_default_async_context_returns_model() -> None:
    model = TestModel()
    assert await AbstractModel.__aenter__(model) is model


def test_profile_context_window_from_genai_prices():
    """`Model.profile` fills `context_window` from genai-prices when no profile layer sets it."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        context_window = infer_model('openai:gpt-5').profile.get('context_window')

    assert context_window is not None
    # Compare against a direct genai-prices query so the test doesn't pin a data value.
    _, model_info = get_snapshot().find_provider_model(
        'gpt-5', provider=None, provider_id='openai', provider_api_url=None
    )
    assert context_window == model_info.context_window


def test_context_window_reads_profile():
    """`Model.context_window` is the profile's `context_window`, `None` when the profile doesn't know it."""
    assert TestModel(profile=ModelProfile(context_window=1234)).context_window == 1234
    assert TestModel().context_window is None


def test_profile_context_window_unknown_model():
    """A model genai-prices doesn't know keeps `context_window` as `None`."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        assert infer_model('openai:potato-gpt').profile.get('context_window') is None


def test_profile_context_window_explicit_override():
    """An explicit `profile=` value wins over the genai-prices lookup."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        model = OpenAIChatModel('gpt-5', profile=ModelProfile(context_window=1234))
    assert model.profile.get('context_window') == 1234


def test_profile_context_window_partial_override():
    """A partial user profile keeps inferred fields alongside its explicit overrides."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        model = OpenAIChatModel('gpt-5', profile=ModelProfile(supports_tools=False))
    assert model.profile.get('context_window') is not None
    assert model.profile.get('supports_tools') is False


def test_profile_context_window_explicit_unknown():
    """An explicit `None` remains authoritative instead of being replaced from genai-prices."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        model = OpenAIChatModel('gpt-5', profile=ModelProfile(context_window=None))
    assert model.profile.get('context_window') is None


def test_profile_context_window_callable_override():
    """A full-control profile callable receives the inferred default and can clear it."""
    inferred_context_windows: list[int | None] = []

    def profile(default: ModelProfile) -> ModelProfile:
        inferred_context_windows.append(default.get('context_window'))
        return ModelProfile(context_window=None)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'x'}):
        model = OpenAIChatModel('gpt-5', profile=profile)
    assert model.profile.get('context_window') is None
    assert inferred_context_windows[0] is not None


def test_wrapper_model_deepcopy():
    """`deepcopy` builds the copy without `__init__`, so `wrapped` is unset when `__getattr__` runs.

    `copy` probes the new instance for `__setstate__`; forwarding that to an unset `wrapped` recursed
    until `RecursionError`. No request is involved, so this is not a VCR test.
    """
    model = WrapperModel(TestModel(custom_output_text='wrapped'))

    copied = deepcopy(model)

    assert copied is not model
    assert copied.wrapped is not model.wrapped
    assert copied.model_name == 'test'
    assert copied.custom_output_text == 'wrapped'


@dataclass(init=False)
class _DataclassTestModel(TestModel):
    """A user's dataclass subclass, which would generate its own `__eq__` and `__repr__` from its fields."""


class _PlainModel(Model):
    """A subclass that isn't a dataclass, so it has no fields to compare."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return 'plain'

    @property
    def system(self) -> str:
        raise NotImplementedError


@dataclass(eq=False)
class _IdentityDataclassModel(_PlainModel):
    """A `@dataclass(eq=False)` subclass, which keeps identity hashing."""

    tag: str = 'model'


class _OwnMethodsModel(_PlainModel):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OwnMethodsModel)

    def __repr__(self) -> str:
        return 'own repr'


def test_models_that_differ_only_in_settings_or_profile_are_not_equal():
    """`settings` and `profile` live on the `Model` base, outside the fields a dataclass subclass compares.

    Not a VCR test: equality makes no request.
    """
    assert TestModel(settings={'temperature': 0.0}) != TestModel(settings={'temperature': 1.0})
    assert TestModel(profile=ModelProfile(supports_tools=True)) != TestModel(profile=ModelProfile(supports_tools=False))
    assert WrapperModel(TestModel(settings={'temperature': 0.0})) != WrapperModel(
        TestModel(settings={'temperature': 1.0})
    )
    assert _DataclassTestModel(settings={'temperature': 0.0}) != _DataclassTestModel(settings={'temperature': 1.0})
    assert TestModel(settings={'temperature': 0.0}) == TestModel(settings={'temperature': 0.0})
    assert TestModel() != object()


def test_a_model_without_dataclass_fields_compares_by_identity():
    """Not a VCR test: equality and hashing make no request."""
    model = _PlainModel()

    assert model == model
    assert model != _PlainModel()
    assert {model: 'kept'}[model] == 'kept'

    identity_dataclass = _IdentityDataclassModel()
    assert identity_dataclass == identity_dataclass
    assert identity_dataclass != _IdentityDataclassModel(), 'equal models must hash equal, and these hash by identity'


def test_a_model_subclass_keeps_its_own_eq_and_repr():
    """Not a VCR test: equality and `repr` make no request."""
    assert _OwnMethodsModel() == _OwnMethodsModel()
    assert repr(_OwnMethodsModel()) == 'own repr'


def test_model_repr_shows_its_name_settings_and_profile():
    """A provider adapter keeps its name and provider out of its dataclass fields' repr.

    Not a VCR test: `repr` makes no request.
    """
    model = OpenAIChatModel('gpt-5', provider=openai.OpenAIProvider(api_key='x'), settings={'temperature': 1.0})

    assert repr(model) == "OpenAIChatModel(model_name='gpt-5', settings={'temperature': 1.0})"
    assert repr(TestModel(seed=1, profile=ModelProfile(supports_tools=False))) == (
        "TestModel(model_name='test', call_tools='all', custom_output_text=None, custom_output_args=None, seed=1, "
        "last_model_request_parameters=None, profile={'supports_tools': False})"
    )
    assert repr(_PlainModel()) == "_PlainModel(model_name='plain')"


@dataclass(init=False)
class _SelfReferencingModel(TestModel):
    """A model whose field reaches back to it, as a routing model holding an agent built on it would."""

    itself: Model | None = None


def test_a_self_referencing_model_repr_does_not_recurse():
    """Not a VCR test: `repr` makes no request."""
    model = _SelfReferencingModel()
    model.itself = model

    assert repr(model).endswith('itself=...)')
