import os
import warnings
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, UserError
from pydantic_ai._json_schema import JsonSchemaTransformer
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    DEFAULT_PROFILE,
    Model,
    ModelRequestParameters,
    ToolDefinition,
    infer_model,
    infer_model_profile,
    parse_model_id,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputObjectDefinition, StructuredOutputMode
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

        # Test that model_id matches the provider:model string that was passed in
        assert m.model_id == f'{expected_system}:{expected_model_name}'

        m2 = infer_model(m)
        assert m2 is m


def test_infer_model_with_provider():
    from pydantic_ai.providers import openai

    provider_class = openai.OpenAIProvider(api_key='1234', base_url='http://test')
    m = infer_model('openai-chat:gpt-5', lambda x: provider_class)

    assert isinstance(m, OpenAIChatModel)
    assert m._provider is provider_class  # pyright: ignore[reportPrivateUsage]
    assert m._provider.base_url == 'http://test'  # pyright: ignore[reportPrivateUsage]


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')


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


class _Answer(BaseModel):
    answer: str


def _double(x: int) -> int:
    return x * 2


def _schema_recording_transformer(walked: list[list[str]]) -> type[JsonSchemaTransformer]:
    """Build a transformer that records the property names of every schema it walks."""

    class _RecordingTransformer(JsonSchemaTransformer):
        def walk(self) -> dict[str, Any]:
            walked.append(sorted(self.schema.get('properties', {})))
            return super().walk()

        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    return _RecordingTransformer


@pytest.mark.parametrize('default_structured_output_mode', ['tool', 'native', 'prompted'])
def test_prepare_request_transforms_only_the_output_channel_the_resolved_mode_uses(
    default_structured_output_mode: StructuredOutputMode,
):
    """The output schema is walked once per request, for the channel the resolved output mode actually sends.

    The agent graph populates both `output_tools` and `output_object` while `output_mode` is still `'auto'`, and
    `prepare_request` drops whichever one the resolved mode doesn't use. Resolving the mode *before*
    `customize_request_parameters` means the JSON schema transformer never walks the schema about to be discarded.

    A unit test because the saved work is invisible on the wire: the discarded channel never reached the provider
    under either ordering, so no cassette can tell them apart.
    """
    walked: list[list[str]] = []
    prepared: list[ModelRequestParameters] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prepared.append(info.model_request_parameters)
        if len(prepared) == 1:
            return ModelResponse(parts=[ToolCallPart('_double', {'x': 21})])
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'answer': 'ok'})])
        return ModelResponse(parts=[TextPart('{"answer": "ok"}')])

    model = FunctionModel(
        respond,
        profile=ModelProfile(
            json_schema_transformer=_schema_recording_transformer(walked),
            default_structured_output_mode=default_structured_output_mode,
            supports_json_schema_output=True,
        ),
    )
    result = Agent(model, output_type=_Answer, tools=[_double]).run_sync('hello')
    assert result.output == _Answer(answer='ok')

    # Per request: the function tool and exactly one output channel — never the output tool *and* the output object.
    assert walked == [['x'], ['answer'], ['x'], ['answer']]

    params = prepared[0]
    assert params.output_mode == default_structured_output_mode
    assert [t.name for t in params.output_tools] == (
        ['final_result'] if default_structured_output_mode == 'tool' else []
    )
    assert (params.output_object is not None) is (default_structured_output_mode != 'tool')


def _empty_output_object() -> OutputObjectDefinition:
    return OutputObjectDefinition(json_schema={'type': 'object', 'properties': {}})


def _switch_to_prompted(params: ModelRequestParameters) -> ModelRequestParameters:
    return replace(params, output_mode='prompted', allow_text_output=True, output_object=_empty_output_object())


def _reset_to_auto(params: ModelRequestParameters) -> ModelRequestParameters:
    return replace(params, output_mode='auto', allow_text_output=True)


@pytest.mark.parametrize(
    'customize,expected_mode,expected_allow_text',
    [
        pytest.param(_switch_to_prompted, 'prompted', True, id='switches-to-prompted'),
        pytest.param(_reset_to_auto, 'tool', False, id='resets-to-auto'),
    ],
)
def test_prepare_request_re_resolves_output_mode_changed_by_customize_request_parameters(
    customize: Callable[[ModelRequestParameters], ModelRequestParameters],
    expected_mode: StructuredOutputMode,
    expected_allow_text: bool,
):
    """A `customize_request_parameters` override that changes the output mode still gets consistent parameters.

    `prepare_request` resolves the output mode before calling `customize_request_parameters`, but deliberately
    resolves again afterwards. That second pass is normally a no-op, and it's what keeps an override that changes
    the mode anyway from leaving `output_mode`, `allow_text_output` and the output channels describing a mode that
    is no longer in effect.

    A unit test because no in-repo model changes the output mode there — this pins the guarantee for the
    third-party overrides that make it possible.
    """

    class _ModeChangingModel(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return customize(model_request_parameters)

    model = _ModeChangingModel(profile=ModelProfile(default_structured_output_mode='tool'))
    _, params = model.prepare_request(
        None,
        ModelRequestParameters(
            output_mode='auto',
            output_tools=[ToolDefinition(name='final_result')],
            output_object=_empty_output_object(),
        ),
    )

    assert params.output_mode == expected_mode
    assert params.allow_text_output is expected_allow_text
    assert params.output_tools == ([ToolDefinition(name='final_result')] if expected_mode == 'tool' else [])
    assert (params.output_object is not None) is (expected_mode != 'tool')
