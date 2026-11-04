"""Every option declared in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from typing_extensions import get_type_hints

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import ImageGeneration, NativeTool
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
    from google.protobuf.json_format import MessageToDict

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import (
        OpenAIChatModelSettings,
        _map_openai_image_generation_tool,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.xai import _get_native_tools  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic, openai or xai not installed')


def _anthropic_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test'))
    tools, *_ = model._add_native_tools(  # pyright: ignore[reportPrivateUsage]
        [], ModelRequestParameters(native_tools=[tool]), AnthropicModelSettings()
    )
    return cast(dict[str, Any], tools[0])


def _openrouter_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test'))
    tools, _ = model._get_tool_choice(  # pyright: ignore[reportPrivateUsage]
        OpenAIChatModelSettings(), ModelRequestParameters(native_tools=[tool])
    )
    return cast(dict[str, Any], tools[0])['parameters']


def _anthropic_advisor(tool: AbstractNativeTool) -> dict[str, Any]:
    return {'caching': _anthropic_tool(tool)['caching']['ttl']}


def _openai_image(tool: AbstractNativeTool) -> dict[str, Any]:
    assert isinstance(tool, ImageGenerationTool)
    return dict(_map_openai_image_generation_tool(tool))


def _xai_file_search(tool: AbstractNativeTool) -> dict[str, Any]:
    tools = _get_native_tools(ModelRequestParameters(native_tools=[tool]))
    search = MessageToDict(tools[0], preserving_proto_field_name=True)['collections_search']
    modes = [key.removesuffix('_retrieval') for key in search if key.endswith('_retrieval')]
    return {'instructions': search.get('instructions'), 'retrieval_mode': modes[0] if modes else None}


@dataclass(frozen=True)
class Case:
    tool: Callable[..., AbstractNativeTool]
    system: str
    option: str
    value: Any
    request: Callable[[AbstractNativeTool], dict[str, Any]]


def _advisor(**kwargs: Any) -> AbstractNativeTool:
    return AdvisorTool(model='advisor-model', **kwargs)


def _file_search(**kwargs: Any) -> AbstractNativeTool:
    return FileSearchTool(file_store_ids=['store'], **kwargs)


MOVED_CASES = [
    Case(ImageGenerationTool, 'openai', 'action', 'generate', _openai_image),
    Case(ImageGenerationTool, 'openai', 'background', 'opaque', _openai_image),
    Case(ImageGenerationTool, 'openai', 'input_fidelity', 'high', _openai_image),
    Case(ImageGenerationTool, 'openai', 'moderation', 'low', _openai_image),
    Case(ImageGenerationTool, 'openai', 'model', 'gpt-image-2', _openai_image),
    Case(ImageGenerationTool, 'openai', 'partial_images', 2, _openai_image),
    Case(ImageGenerationTool, 'openai', 'quality', 'high', _openai_image),
    Case(_advisor, 'anthropic', 'caching', '5m', _anthropic_advisor),
    Case(_file_search, 'xai', 'instructions', 'search these', _xai_file_search),
    Case(_file_search, 'xai', 'retrieval_mode', 'keyword', _xai_file_search),
]

CASES = [
    *MOVED_CASES,
    Case(WebSearchTool, 'anthropic', 'response_inclusion', 'excluded', _anthropic_tool),
    Case(WebSearchTool, 'openrouter', 'engine', 'exa', _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'mode', 'fast', _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_results', 3, _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_total_results', 9, _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_characters', 100, _openrouter_tool),
    Case(WebFetchTool, 'anthropic', 'response_inclusion', 'excluded', _anthropic_tool),
    Case(WebFetchTool, 'anthropic', 'use_cache', False, _anthropic_tool),
    Case(_advisor, 'openrouter', 'forward_transcript', True, _openrouter_tool),
]

SETTINGS_TYPES = {
    WebSearchTool: WebSearchProviderSettings,
    WebFetchTool: WebFetchProviderSettings,
    _advisor: AdvisorProviderSettings,
    ImageGenerationTool: ImageGenerationProviderSettings,
    _file_search: FileSearchProviderSettings,
}


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.system}-{c.option}')
def test_option_reaches_the_request(case: Case) -> None:
    tool = case.tool(provider_settings={case.system: {case.option: case.value}})
    assert case.request(tool)[case.option] == case.value


@pytest.mark.parametrize('case', MOVED_CASES, ids=lambda c: f'{c.system}-{c.option}')
def test_moved_field_warns_and_sends_the_same_request_as_nested(case: Case) -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=rf'`{case.option}` is deprecated'):
        legacy = case.tool(**{case.option: case.value})
    nested = case.tool(provider_settings={case.system: {case.option: case.value}})

    assert case.request(legacy) == case.request(nested)
    assert case.request(legacy)[case.option] == case.value


@pytest.mark.parametrize(
    ('case', 'nested_value'),
    [
        pytest.param(MOVED_CASES[0], 'edit', id='openai'),
        pytest.param(MOVED_CASES[7], '1h', id='anthropic'),
        pytest.param(MOVED_CASES[9], 'semantic', id='xai'),
    ],
)
def test_nested_value_wins_over_moved_field(case: Case, nested_value: str) -> None:
    with pytest.warns(PydanticAIDeprecationWarning):
        tool = case.tool(**{case.option: case.value}, provider_settings={case.system: {case.option: nested_value}})

    assert case.request(tool)[case.option] == nested_value


def test_rebuilding_a_capability_tool_from_data_does_not_warn() -> None:
    """Durable execution validates tools between activities, and a capability user has nothing to change."""
    tool = ImageGeneration(quality='high').get_native_tools()[0]
    assert isinstance(tool, ImageGenerationTool)
    adapter = TypeAdapter(ModelRequestParameters)
    parameters = ModelRequestParameters(native_tools=[tool])

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
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
    with pytest.warns(PydanticAIDeprecationWarning, match='`quality` is deprecated'):
        capability = NativeTool.from_spec(**spec)

    assert isinstance(capability.tool, ImageGenerationTool)
    assert capability.tool.quality == 'high'


def test_external_web_access_stays_top_level_without_a_warning() -> None:
    """Every model has to honour `False` or fail, so this field deliberately isn't provider-specific."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        tool = WebSearchTool(external_web_access=False)

    assert 'external_web_access' in tool.restricting_settings


def test_every_declared_option_has_a_case() -> None:
    """Adding an option without mapping it fails here, rather than silently doing nothing for users."""
    declared = {
        (tool, system, option)
        for tool, settings_type in SETTINGS_TYPES.items()
        for system, system_settings in get_type_hints(settings_type).items()
        for option in get_type_hints(system_settings)
    }
    assert declared == {(case.tool, case.system, case.option) for case in CASES}


@pytest.mark.parametrize('tool', SETTINGS_TYPES)
@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'madeup': {}}, id='unknown-system'),
        pytest.param({'openrouter': {'typo': 1}}, id='unknown-option'),
    ],
)
def test_unknown_keys_are_rejected(tool: Callable[..., AbstractNativeTool], settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        tool(provider_settings=settings)


def test_another_systems_option_is_rejected() -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        WebSearchTool(provider_settings={'openrouter': {'response_inclusion': 'full'}})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'openrouter': {'engine': 'nonsense'}}, id='bad-literal'),
        pytest.param({'openrouter': {'max_results': '3'}}, id='wrong-type-not-coerced'),
        pytest.param({'openrouter': {'engine': None}}, id='none-is-not-unset'),
    ],
)
def test_invalid_values_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WebSearchTool(provider_settings=settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.system}-{c.option}')
def test_settings_survive_a_serialization_round_trip(case: Case) -> None:
    """Durable execution sends native tools through this round trip between activities."""
    tool = case.tool(provider_settings={case.system: {case.option: case.value}})
    adapter = TypeAdapter(ModelRequestParameters)

    payload = adapter.dump_json(ModelRequestParameters(native_tools=[tool]))

    assert adapter.validate_json(payload).native_tools == [tool]


def test_payload_with_an_unknown_option_is_rejected() -> None:
    """A payload written by a newer version fails loudly here, not by silently dropping the option."""
    payload = {'native_tools': [{'kind': 'web_search', 'provider_settings': {'openrouter': {'future': 1}}}]}

    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        TypeAdapter(ModelRequestParameters).validate_python(payload)
