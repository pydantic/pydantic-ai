"""Every option in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import pytest
from pydantic import TypeAdapter
from typing_extensions import get_type_hints

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import ImageGeneration, NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    AdvisorProviderSettings,
    AdvisorTool,
    FileSearchProviderSettings,
    FileSearchTool,
    ImageGenerationProviderSettings,
    ImageGenerationTool,
    WebFetchProviderSettings,
    WebFetchTool,
    WebSearchProviderSettings,
    WebSearchTool,
)

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .models import test_anthropic as anthropic_mocks
    from .models.mock_openai import (
        MockOpenAI,
        MockOpenAIResponses,
        get_mock_chat_completion_kwargs,
        get_mock_responses_kwargs,
        response_message,
    )
    from .models.test_openrouter import _openrouter_completion  # pyright: ignore[reportPrivateUsage]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic or openai not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def _anthropic_request_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    text = BetaTextBlock(text='ok', type='text')
    client = anthropic_mocks.MockAnthropic.create_mock(
        anthropic_mocks.completion_message([text], BetaUsage(input_tokens=1, output_tokens=1))
    )
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return anthropic_mocks.get_mock_chat_completion_kwargs(client)[0]['tools'][0]


def _openrouter_request_parameters(tool: AbstractNativeTool) -> dict[str, Any]:
    client = MockOpenAI.create_mock(_openrouter_completion('ok'))
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(openai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return get_mock_chat_completion_kwargs(client)[0]['tools'][0]['parameters']


def _openai_request_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    text = ResponseOutputText(text='ok', type='output_text', annotations=[])
    message = ResponseOutputMessage(id='msg', content=[text], role='assistant', status='completed', type='message')
    client = MockOpenAIResponses.create_mock(response_message([message]))
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return get_mock_responses_kwargs(client)[0]['tools'][0]


def _xai_request_search(tool: AbstractNativeTool) -> dict[str, Any]:
    pytest.importorskip('xai_sdk')
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

    from .models.mock_xai import MockXai, create_response, create_usage, get_mock_chat_create_kwargs

    client = MockXai.create_mock(
        [create_response(content='ok', usage=create_usage(prompt_tokens=1, completion_tokens=1))]
    )
    model = XaiModel('grok-4-fast-non-reasoning', provider=XaiProvider(xai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return get_mock_chat_create_kwargs(client)[0]['tools'][0]['collections_search']


QUALITY_DEPRECATED = r'`quality` is deprecated .*provider_settings=\{"openai"'

ANTHROPIC_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (
        WebSearchTool(provider_settings={'anthropic': {'response_inclusion': 'excluded'}}),
        'response_inclusion',
        'excluded',
    ),
    (
        WebFetchTool(provider_settings={'anthropic': {'response_inclusion': 'excluded'}}),
        'response_inclusion',
        'excluded',
    ),
    (WebFetchTool(provider_settings={'anthropic': {'use_cache': False}}), 'use_cache', False),
    (
        AdvisorTool(model='advisor', provider_settings={'anthropic': {'caching': '5m'}}),
        'caching',
        {'type': 'ephemeral', 'ttl': '5m'},
    ),
]
OPENROUTER_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (WebSearchTool(provider_settings={'openrouter': {'engine': 'exa'}}), 'engine', 'exa'),
    (WebSearchTool(provider_settings={'openrouter': {'mode': 'fast'}}), 'mode', 'fast'),
    (WebSearchTool(provider_settings={'openrouter': {'max_results': 3}}), 'max_results', 3),
    (WebSearchTool(provider_settings={'openrouter': {'max_total_results': 9}}), 'max_total_results', 9),
    (WebSearchTool(provider_settings={'openrouter': {'max_characters': 100}}), 'max_characters', 100),
    (
        AdvisorTool(model='advisor', provider_settings={'openrouter': {'forward_transcript': True}}),
        'forward_transcript',
        True,
    ),
]
OPENAI_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (ImageGenerationTool(provider_settings={'openai': {'action': 'generate'}}), 'action', 'generate'),
    (ImageGenerationTool(provider_settings={'openai': {'background': 'opaque'}}), 'background', 'opaque'),
    (ImageGenerationTool(provider_settings={'openai': {'input_fidelity': 'high'}}), 'input_fidelity', 'high'),
    (ImageGenerationTool(provider_settings={'openai': {'moderation': 'low'}}), 'moderation', 'low'),
    (ImageGenerationTool(provider_settings={'openai': {'model': 'gpt-image-2'}}), 'model', 'gpt-image-2'),
    (ImageGenerationTool(provider_settings={'openai': {'partial_images': 2}}), 'partial_images', 2),
    (ImageGenerationTool(provider_settings={'openai': {'quality': 'high'}}), 'quality', 'high'),
]
XAI_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (
        FileSearchTool(file_store_ids=['store'], provider_settings={'xai': {'instructions': 'search these'}}),
        'instructions',
        'search these',
    ),
    (
        FileSearchTool(file_store_ids=['store'], provider_settings={'xai': {'retrieval_mode': 'keyword'}}),
        'keyword_retrieval',
        {},
    ),
]


@pytest.mark.parametrize(('tool', 'request_key', 'value'), ANTHROPIC_OPTIONS)
def test_anthropic_option_reaches_the_request(tool: AbstractNativeTool, request_key: str, value: Any) -> None:
    assert _anthropic_request_tool(tool)[request_key] == value


@pytest.mark.parametrize(('tool', 'request_key', 'value'), OPENROUTER_OPTIONS)
def test_openrouter_option_reaches_the_request(tool: AbstractNativeTool, request_key: str, value: Any) -> None:
    assert _openrouter_request_parameters(tool)[request_key] == value


@pytest.mark.parametrize(('tool', 'request_key', 'value'), OPENAI_OPTIONS)
def test_openai_option_reaches_the_request(tool: AbstractNativeTool, request_key: str, value: Any) -> None:
    assert _openai_request_tool(tool)[request_key] == value


@pytest.mark.parametrize(('tool', 'request_key', 'value'), XAI_OPTIONS)
def test_xai_option_reaches_the_request(tool: AbstractNativeTool, request_key: str, value: Any) -> None:
    assert _xai_request_search(tool)[request_key] == value


def test_every_declared_option_is_covered() -> None:
    """Adding an option without covering it above fails here, rather than silently doing nothing for users."""
    settings_types: dict[type[AbstractNativeTool], Any] = {
        WebSearchTool: WebSearchProviderSettings,
        WebFetchTool: WebFetchProviderSettings,
        AdvisorTool: AdvisorProviderSettings,
        ImageGenerationTool: ImageGenerationProviderSettings,
        FileSearchTool: FileSearchProviderSettings,
    }
    declared = {
        (tool_type, system, option)
        for tool_type, settings_type in settings_types.items()
        for system, options in get_type_hints(settings_type).items()
        for option in get_type_hints(options)
    }
    covered = {
        (type(tool), system, option)
        for tool, _, _ in [*ANTHROPIC_OPTIONS, *OPENROUTER_OPTIONS, *OPENAI_OPTIONS, *XAI_OPTIONS]
        for system, options in cast(Any, tool).provider_settings.items()
        for option in options
    }
    assert declared == covered


@pytest.mark.parametrize(
    ('request_tool', 'tool_type', 'required', 'system', 'option', 'value'),
    [
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'action', 'generate'),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'background', 'opaque'),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'input_fidelity', 'high'),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'moderation', 'low'),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'model', 'gpt-image-2'),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'partial_images', 2),
        (_openai_request_tool, ImageGenerationTool, {}, 'openai', 'quality', 'high'),
        (_anthropic_request_tool, AdvisorTool, {'model': 'advisor'}, 'anthropic', 'caching', '5m'),
        (_xai_request_search, FileSearchTool, {'file_store_ids': ['store']}, 'xai', 'instructions', 'search these'),
        (_xai_request_search, FileSearchTool, {'file_store_ids': ['store']}, 'xai', 'retrieval_mode', 'keyword'),
    ],
)
def test_moved_field_warns_and_sends_the_same_request_as_nested(
    request_tool: Any, tool_type: Any, required: dict[str, Any], system: str, option: str, value: Any
) -> None:
    with pytest.warns(
        PydanticAIDeprecationWarning, match=rf'`{option}` is deprecated .*provider_settings=\{{"{system}"'
    ) as warnings:
        legacy = tool_type(**required, **{option: value})
    nested = tool_type(**required, provider_settings={system: {option: value}})

    assert warnings[0].filename == __file__
    assert request_tool(legacy) == request_tool(nested)


@pytest.mark.parametrize(
    ('tool_type', 'kwargs'),
    [
        (ImageGenerationTool, {'quality': 'high', 'provider_settings': {'openai': {'quality': 'low'}}}),
        (AdvisorTool, {'model': 'advisor', 'caching': '5m', 'provider_settings': {'anthropic': {'caching': '1h'}}}),
        (
            FileSearchTool,
            {
                'file_store_ids': ['store'],
                'retrieval_mode': 'keyword',
                'provider_settings': {'xai': {'retrieval_mode': 'semantic'}},
            },
        ),
    ],
)
def test_moved_field_conflicting_with_nested_value_is_rejected(tool_type: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(UserError, match='conflicts with `provider_settings'):
        tool_type(**kwargs)


def test_moved_field_matching_nested_value_only_warns() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED):
        tool = ImageGenerationTool(quality='high', provider_settings={'openai': {'quality': 'high'}})

    assert _openai_request_tool(tool)['quality'] == 'high'


def test_rebuilding_a_capability_tool_from_data_does_not_warn() -> None:
    """Durable execution validates tools between activities, and a capability user has nothing to change."""
    tool = ImageGeneration(quality='high').get_native_tools()[0]
    assert isinstance(tool, ImageGenerationTool)
    adapter = TypeAdapter(ModelRequestParameters)
    parameters = ModelRequestParameters(native_tools=[tool])

    assert adapter.validate_python(adapter.dump_python(parameters)).native_tools == [tool]
    assert adapter.validate_json(adapter.dump_json(parameters)).native_tools == [tool]


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param({'kind': 'image_generation', 'quality': 'high'}, id='flat'),
        pytest.param({'tool': {'kind': 'image_generation', 'quality': 'high'}}, id='explicit'),
    ],
)
def test_moved_field_in_a_spec_warns(spec: dict[str, Any]) -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED):
        capability = NativeTool.from_spec(**spec)

    assert isinstance(capability.tool, ImageGenerationTool)
    assert capability.tool.quality == 'high'


def test_moved_field_on_a_tool_passed_to_a_spec_warns_once() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED) as warnings:
        NativeTool.from_spec(tool=ImageGenerationTool(quality='high'))

    assert len(warnings) == 1


def test_each_model_reads_only_its_own_settings() -> None:
    tool = WebSearchTool(
        provider_settings={'anthropic': {'response_inclusion': 'excluded'}, 'openrouter': {'engine': 'exa'}}
    )

    anthropic_tool = _anthropic_request_tool(tool)
    assert anthropic_tool['response_inclusion'] == 'excluded'
    assert 'engine' not in anthropic_tool
    openrouter_parameters = _openrouter_request_parameters(tool)
    assert openrouter_parameters['engine'] == 'exa'
    assert 'response_inclusion' not in openrouter_parameters


@pytest.mark.parametrize('option', ['engine', 'mode'])
def test_value_openrouter_adds_later_is_forwarded(option: str) -> None:
    tool = WebSearchTool(provider_settings={'openrouter': {option: 'added-later'}})  # pyright: ignore[reportArgumentType]

    assert _openrouter_request_parameters(tool)[option] == 'added-later'


UNKNOWN_SETTINGS: Any = {
    'madeup': {},
    'anthropic': {'typo': 1},
    'openrouter': {'typo': 1, 'allowed_domains': ['example.com']},
    'openai': {'typo': 1},
    'xai': {'typo': 1},
}


@pytest.mark.parametrize(
    ('read_request', 'make_tool'),
    [
        (_anthropic_request_tool, WebSearchTool),
        (_anthropic_request_tool, WebFetchTool),
        (_openrouter_request_parameters, WebSearchTool),
        (_openrouter_request_parameters, partial(AdvisorTool, model='advisor')),
        (_openai_request_tool, ImageGenerationTool),
        (_xai_request_search, partial(FileSearchTool, file_store_ids=['store'])),
    ],
)
def test_unknown_settings_are_ignored(read_request: Any, make_tool: Any) -> None:
    """Like model settings, an unknown option is never sent, so it can't override a field like `allowed_domains`."""
    assert read_request(make_tool(provider_settings=UNKNOWN_SETTINGS)) == read_request(make_tool())
