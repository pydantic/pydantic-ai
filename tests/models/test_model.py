import os
import re
import warnings
from importlib import import_module
from unittest.mock import patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryContent, ToolReturn, UserError
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnSource,
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
from pydantic_ai.profiles import ModelProfile

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers import openai
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

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
    """Verify infer_model_profile returns the same profile as the provider's model_profile."""
    module_path, class_name = provider_path.rsplit('.', 1)
    module = import_module(module_path)
    provider_class = getattr(module, class_name)

    profile = infer_model_profile(model_id)
    provider_profile = provider_class.model_profile(model_name)
    assert profile == provider_profile


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


def test_prepare_messages_renders_tool_return_source() -> None:
    first_image = BinaryContent(data=b'first', media_type='image/png')
    second_image = BinaryContent(data=b'second', media_type='image/png')
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[user_image := BinaryContent(data=b'user', media_type='image/png')]),
                UserPromptPart(
                    content=[first_image],
                    source=ToolReturnSource(tool_name='get_file', tool_call_id='call_1'),
                ),
                UserPromptPart(
                    content=[second_image],
                    source=ToolReturnSource(tool_name='get_file', tool_call_id='call_2'),
                ),
            ]
        )
    ]
    model = TestModel()

    prepared = model.prepare_messages(messages)

    prepared_request = prepared[0]
    assert isinstance(prepared_request, ModelRequest)
    assert [part.content for part in prepared_request.parts if isinstance(part, UserPromptPart)] == snapshot(
        [
            [user_image],
            ['<pydantic_ai:tool_return tool_name="get_file" tool_call_id="call_1" />', first_image],
            ['<pydantic_ai:tool_return tool_name="get_file" tool_call_id="call_2" />', second_image],
        ]
    )
    assert all(part.source is None for part in prepared_request.parts if isinstance(part, UserPromptPart))
    # Identity, not equality: `_make_request` skips a redundant `_clean_message_history` pass when
    # `prepare_messages` hands back the same list, so `==` would pass an implementation that rebuilds.
    assert model.prepare_messages(prepared) is prepared
    original_request = messages[0]
    assert isinstance(original_request, ModelRequest)
    original_part = original_request.parts[1]
    assert isinstance(original_part, UserPromptPart)
    assert original_part.source == ToolReturnSource(tool_name='get_file', tool_call_id='call_1')


def test_prepare_messages_leaves_text_only_tool_return_content_unmarked() -> None:
    """Text a tool returns is already attributed by the tool result it accompanies.

    Marking it would change the prompt for every `ToolReturn.content` user on every provider,
    including the ones whose tool results carry media natively and never spill.
    """
    source = ToolReturnSource(tool_name='note', tool_call_id='call_1')
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content='plain tool text', source=source),
                UserPromptPart(content=['a list', 'of plain strings'], source=source),
            ]
        )
    ]
    model = TestModel()

    prepared = model.prepare_messages(messages)

    prepared_request = prepared[0]
    assert isinstance(prepared_request, ModelRequest)
    assert [part.content for part in prepared_request.parts if isinstance(part, UserPromptPart)] == snapshot(
        ['plain tool text', ['a list', 'of plain strings']]
    )
    assert all(part.source is None for part in prepared_request.parts if isinstance(part, UserPromptPart))


@pytest.mark.anyio
async def test_tool_return_source_replays_across_provider_mappers() -> None:
    agent = Agent(TestModel(call_tools=['get_image']))

    @agent.tool_plain
    def get_image() -> ToolReturn:
        return ToolReturn(
            return_value='image returned',
            content=[BinaryContent(data=b'tool image', media_type='image/png')],
        )

    result = await agent.run([BinaryContent(data=b'user image', media_type='image/png')])
    history = result.all_messages()
    replayed = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))

    tool_call = next(
        part
        for message in replayed
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )
    tool_prompt = next(
        part
        for message in replayed
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and part.source is not None
    )
    expected_source = ToolReturnSource(tool_name=tool_call.tool_name, tool_call_id=tool_call.tool_call_id)
    assert tool_prompt.source == expected_source

    openai_model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))
    openai_messages = await openai_model._map_messages(  # pyright: ignore[reportPrivateUsage]
        openai_model.prepare_messages(replayed), ModelRequestParameters()
    )
    google_model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
    _, google_contents = await google_model._map_messages(  # pyright: ignore[reportPrivateUsage]
        google_model.prepare_messages(replayed), ModelRequestParameters()
    )

    assert openai_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [{'image_url': {'url': 'data:image/png;base64,dXNlciBpbWFnZQ=='}, 'type': 'image_url'}],
            },
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'pyd_ai_tool_call_id__get_image',
                        'type': 'function',
                        'function': {'name': 'get_image', 'arguments': '{}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'pyd_ai_tool_call_id__get_image', 'content': 'image returned'},
            {
                'role': 'user',
                'content': [
                    {
                        'text': '<pydantic_ai:tool_return tool_name="get_image" tool_call_id="pyd_ai_tool_call_id__get_image" />',
                        'type': 'text',
                    },
                    {'image_url': {'url': 'data:image/png;base64,dG9vbCBpbWFnZQ=='}, 'type': 'image_url'},
                ],
            },
            {'role': 'assistant', 'content': '{"get_image":"image returned"}'},
        ]
    )
    assert google_contents == snapshot(
        [
            {'role': 'user', 'parts': [{'inline_data': {'data': b'user image', 'mime_type': 'image/png'}}]},
            {
                'role': 'model',
                'parts': [
                    {
                        'function_call': {'name': 'get_image', 'args': {}, 'id': 'pyd_ai_tool_call_id__get_image'},
                        'thought_signature': b'skip_thought_signature_validator',
                    }
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'function_response': {
                            'name': 'get_image',
                            'response': {'return_value': 'image returned'},
                            'id': 'pyd_ai_tool_call_id__get_image',
                        }
                    }
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'text': '<pydantic_ai:tool_return tool_name="get_image" tool_call_id="pyd_ai_tool_call_id__get_image" />'
                    },
                    {'inline_data': {'data': b'tool image', 'mime_type': 'image/png'}},
                ],
            },
            {'role': 'model', 'parts': [{'text': '{"get_image":"image returned"}'}]},
        ]
    )
    # Re-checked after mapping: `prepare_messages` copies, it must never mutate stored history.
    assert tool_prompt.source == expected_source


@pytest.mark.anyio
async def test_model_default_async_context_returns_model() -> None:
    model = TestModel()
    assert await AbstractModel.__aenter__(model) is model
