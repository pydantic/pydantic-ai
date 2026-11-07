"""Pin which native tool settings each model adapter sends, and what happens to the others.

A setting such as `WebSearchTool.allowed_domains` is usually there for compliance or cost reasons, so a model
that would send the tool without it treats the tool as unsupported for the request: the local fallback runs
instead, or the request is rejected. These are not VCR tests: the rejection and the swap are pre-request
guards that never reach the provider, and a cassette would keep matching a request body that silently lost a
setting.

Options in a tool's `provider_settings` are covered at the end: each one has to reach its own model's request, and
unknown ones are ignored, like model settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import pytest
from pydantic import TypeAdapter
from typing_extensions import get_type_hints

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import ImageGeneration, NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters, resolve_request_tools
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    AdvisorProviderSettings,
    AdvisorTool,
    CodeExecutionTool,
    FileSearchProviderSettings,
    FileSearchTool,
    ImageGenerationProviderSettings,
    ImageGenerationTool,
    MemoryTool,
    WebFetchProviderSettings,
    WebFetchTool,
    WebSearchProviderSettings,
    WebSearchTool,
)
from pydantic_ai.tools import ToolDefinition

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as openai_available:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .mock_openai import (
        MockOpenAI,
        MockOpenAIResponses,
        completion_message as openai_completion,
        get_mock_chat_completion_kwargs as openai_request_kwargs,
        get_mock_responses_kwargs,
        response_message,
    )

with try_import() as anthropic_available:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import (
        MockAnthropic,
        completion_message as anthropic_completion,
        get_mock_chat_completion_kwargs as anthropic_request_kwargs,
    )

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

    from .mock_xai import MockXai, create_response, create_usage, get_mock_chat_create_kwargs


ALL_WEB_SEARCH_RESTRICTIONS = WebSearchTool(
    allowed_domains=['pydantic.dev'], blocked_domains=['example.com'], max_uses=1, external_web_access=False
)
ALL_WEB_FETCH_RESTRICTIONS = WebFetchTool(allowed_domains=['pydantic.dev'], blocked_domains=['example.com'], max_uses=1)


@dataclass
class Case:
    model: Callable[[], Model]
    native_tool: AbstractNativeTool
    unsupported: frozenset[str]


CASES = [
    pytest.param(
        Case(
            lambda: GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'allowed_domains', 'blocked_domains', 'max_uses', 'external_web_access'}),
        ),
        id='google-web-search',
        marks=pytest.mark.skipif(not google_available(), reason='google not installed'),
    ),
    pytest.param(
        Case(
            lambda: GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key')),
            ALL_WEB_FETCH_RESTRICTIONS,
            frozenset({'allowed_domains', 'blocked_domains', 'max_uses'}),
        ),
        id='google-web-fetch',
        marks=pytest.mark.skipif(not google_available(), reason='google not installed'),
    ),
    pytest.param(
        Case(
            lambda: AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'external_web_access'}),
        ),
        id='anthropic-web-search',
        marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
    ),
    pytest.param(
        Case(
            lambda: AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key')),
            ALL_WEB_FETCH_RESTRICTIONS,
            frozenset(),
        ),
        id='anthropic-web-fetch',
        marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
    ),
    pytest.param(
        Case(
            lambda: AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key')),
            CodeExecutionTool(),
            frozenset(),
        ),
        id='anthropic-code-execution',
        marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
    ),
    pytest.param(
        Case(
            lambda: OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'max_uses'}),
        ),
        id='openai-responses-web-search',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
    pytest.param(
        Case(
            lambda: OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key')),
            CodeExecutionTool(),
            frozenset(),
        ),
        id='openai-responses-code-execution',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
    pytest.param(
        Case(
            lambda: OpenAIChatModel('gpt-4o-search-preview', provider=OpenAIProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'allowed_domains', 'blocked_domains', 'max_uses', 'external_web_access'}),
        ),
        id='openai-chat-web-search',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
    pytest.param(
        Case(
            lambda: OpenRouterModel('openai/gpt-4.1', provider=OpenRouterProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'external_web_access'}),
        ),
        id='openrouter-web-search',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
    pytest.param(
        Case(
            lambda: OpenRouterModel('openai/gpt-4.1', provider=OpenRouterProvider(api_key='test-key')),
            AdvisorTool(model='claude-opus-4-8', provider_settings={'anthropic': {'max_uses': 1}}),
            frozenset(),
        ),
        id='openrouter-advisor',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
    pytest.param(
        Case(
            lambda: GroqModel('groq/compound', provider=GroqProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'max_uses', 'external_web_access'}),
        ),
        id='groq-web-search',
        marks=pytest.mark.skipif(not groq_available(), reason='groq not installed'),
    ),
    pytest.param(
        Case(
            lambda: XaiModel('grok-4', provider=XaiProvider(api_key='test-key')),
            ALL_WEB_SEARCH_RESTRICTIONS,
            frozenset({'max_uses', 'external_web_access'}),
        ),
        id='xai-web-search',
        marks=pytest.mark.skipif(not xai_available(), reason='xai not installed'),
    ),
    pytest.param(
        Case(
            lambda: XaiModel('grok-4', provider=XaiProvider(api_key='test-key')),
            CodeExecutionTool(),
            frozenset(),
        ),
        id='xai-code-execution',
        marks=pytest.mark.skipif(not xai_available(), reason='xai not installed'),
    ),
]


@pytest.mark.parametrize('case', CASES)
def test_unsupported_native_tool_settings(case: Case):
    assert case.model().unsupported_native_tool_settings(case.native_tool) == case.unsupported


def test_resolve_request_tools_without_unsupported_settings_applies_every_setting():
    """A caller that doesn't say which settings its model can't apply gets the type-level check only."""
    web_search = WebSearchTool(allowed_domains=['pydantic.dev'])
    params = resolve_request_tools(ModelRequestParameters(native_tools=[web_search]), frozenset({WebSearchTool}))
    assert params.native_tools == [web_search]


def test_restricting_settings():
    assert WebSearchTool().restricting_settings == frozenset()
    assert WebSearchTool(external_web_access=True).restricting_settings == frozenset()
    assert ALL_WEB_SEARCH_RESTRICTIONS.restricting_settings == snapshot(
        frozenset({'allowed_domains', 'blocked_domains', 'max_uses', 'external_web_access'})
    )
    assert (
        WebFetchTool(provider_settings={'anthropic': {'max_content_tokens': 1000}}).restricting_settings == frozenset()
    )
    assert ALL_WEB_FETCH_RESTRICTIONS.restricting_settings == snapshot(
        frozenset({'allowed_domains', 'blocked_domains', 'max_uses'})
    )
    assert CodeExecutionTool().restricting_settings == frozenset()


@pytest.mark.skipif(not google_available(), reason='google not installed')
class TestGoogle:
    def model(self, model_name: str = 'gemini-3-flash-preview') -> GoogleModel:
        return GoogleModel(model_name, provider=GoogleProvider(api_key='test-key'))

    def test_web_search_without_restrictions_is_sent(self):
        _, params = self.model().prepare_request(None, ModelRequestParameters(native_tools=[WebSearchTool()]))
        assert params.native_tools == [WebSearchTool()]

    def test_restricted_web_search_without_fallback_raises(self):
        params = ModelRequestParameters(native_tools=[WebSearchTool(allowed_domains=['pydantic.dev'])])
        with pytest.raises(UserError) as exc_info:
            self.model().prepare_request(None, params)
        assert str(exc_info.value) == snapshot(
            'Native tool setting(s) `WebSearchTool.allowed_domains` not supported by this model. '
            'Sending the tool without them would drop the restriction they set. Remove the settings, use a model '
            'that supports them, or provide a local fallback that applies them itself (e.g. '
            "`WebFetch(allowed_domains=[...], local=True)`). `WebSearch`'s own constraint fields disable its `local` "
            'fallback, so for a search fallback put them on the native tool: '
            '`WebSearch(native=WebSearchTool(allowed_domains=[...]), local=...)`.'
        )

    def test_restricted_web_fetch_without_fallback_raises(self):
        params = ModelRequestParameters(native_tools=[WebFetchTool(max_uses=3, blocked_domains=['example.com'])])
        with pytest.raises(UserError) as exc_info:
            self.model().prepare_request(None, params)
        assert str(exc_info.value) == snapshot(
            'Native tool setting(s) `WebFetchTool.blocked_domains`, `WebFetchTool.max_uses` not supported by this '
            'model. '
            'Sending the tool without them would drop the restriction they set. Remove the settings, use a model '
            'that supports them, or provide a local fallback that applies them itself (e.g. '
            "`WebFetch(allowed_domains=[...], local=True)`). `WebSearch`'s own constraint fields disable its `local` "
            'fallback, so for a search fallback put them on the native tool: '
            '`WebSearch(native=WebSearchTool(allowed_domains=[...]), local=...)`.'
        )

    def test_restricted_optional_web_search_is_dropped(self):
        params = ModelRequestParameters(native_tools=[WebSearchTool(blocked_domains=['example.com'], optional=True)])
        _, prepared = self.model().prepare_request(None, params)
        assert prepared.native_tools == []

    def test_restricted_web_search_uses_local_fallback(self):
        fallback = ToolDefinition(name='web_search', unless_native='web_search')
        params = ModelRequestParameters(
            native_tools=[WebSearchTool(allowed_domains=['pydantic.dev'])], function_tools=[fallback]
        )
        _, prepared = self.model().prepare_request(None, params)
        assert prepared.native_tools == []
        assert [tool.name for tool in prepared.function_tools] == ['web_search']

    def test_unrestricted_web_search_drops_local_fallback(self):
        fallback = ToolDefinition(name='web_search', unless_native='web_search')
        params = ModelRequestParameters(native_tools=[WebSearchTool()], function_tools=[fallback])
        _, prepared = self.model().prepare_request(None, params)
        assert prepared.native_tools == [WebSearchTool()]
        assert prepared.function_tools == []

    def test_restricted_web_search_does_not_force_prompted_output(self):
        """A native tool swapped for its fallback doesn't reach the wire, so it doesn't block output tools."""
        fallback = ToolDefinition(name='web_search', unless_native='web_search')
        output_tool = ToolDefinition(name='final_result', kind='output')
        params = ModelRequestParameters(
            native_tools=[WebSearchTool(allowed_domains=['pydantic.dev'])],
            function_tools=[fallback],
            output_tools=[output_tool],
            output_mode='tool',
        )
        _, prepared = self.model('gemini-2.5-flash').prepare_request(None, params)
        assert prepared.output_mode == 'tool'
        assert prepared.native_tools == []

    def test_unsupported_type_is_reported_before_settings(self):
        """With a tool of an unsupported type and one with unsupported settings, the type error names the former."""
        params = ModelRequestParameters(
            native_tools=[
                WebSearchTool(allowed_domains=['pydantic.dev']),
                WebFetchTool(),
                MemoryTool(),
            ]
        )
        with pytest.raises(UserError, match=r"Native tool\(s\) \['MemoryTool'\] not supported"):
            self.model().prepare_request(None, params)


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
def test_openrouter_web_search_without_external_web_access_raises():
    model = OpenRouterModel('openai/gpt-4.1', provider=OpenRouterProvider(api_key='test-key'))
    params = ModelRequestParameters(native_tools=[WebSearchTool(external_web_access=False)])
    with pytest.raises(UserError, match=r'`WebSearchTool.external_web_access` not supported'):
        model.prepare_request(None, params)


# `provider_settings`: each option reaches its own model's request, unknown ones are ignored, and the fields that
# moved into it warn.


def _anthropic_request_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    text = BetaTextBlock(text='ok', type='text')
    client = MockAnthropic.create_mock(anthropic_completion([text], BetaUsage(input_tokens=1, output_tokens=1)))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return anthropic_request_kwargs(client)[0]['tools'][0]


def _openrouter_request_parameters(tool: AbstractNativeTool) -> dict[str, Any]:
    completion = openai_completion(ChatCompletionMessage(content='ok', role='assistant'))
    completion.provider = 'test'  # pyright: ignore[reportAttributeAccessIssue]  # OpenRouter responses always carry it
    client = MockOpenAI.create_mock(completion)
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(openai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return openai_request_kwargs(client)[0]['tools'][0]['parameters']


def _openai_request_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    text = ResponseOutputText(text='ok', type='output_text', annotations=[])
    message = ResponseOutputMessage(id='msg', content=[text], role='assistant', status='completed', type='message')
    client = MockOpenAIResponses.create_mock(response_message([message]))
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return get_mock_responses_kwargs(client)[0]['tools'][0]


def _xai_request_search(tool: AbstractNativeTool) -> dict[str, Any]:
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
    (WebFetchTool(provider_settings={'anthropic': {'enable_citations': True}}), 'citations', {'enabled': True}),
    (WebFetchTool(provider_settings={'anthropic': {'max_content_tokens': 1000}}), 'max_content_tokens', 1000),
    (
        AdvisorTool(model='advisor', provider_settings={'anthropic': {'caching': '5m'}}),
        'caching',
        {'type': 'ephemeral', 'ttl': '5m'},
    ),
    (AdvisorTool(model='advisor', provider_settings={'anthropic': {'max_uses': 2}}), 'max_uses', 2),
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


UNKNOWN_SETTINGS: Any = {
    'madeup': {},
    'anthropic': {'typo': 1},
    'openrouter': {'typo': 1, 'allowed_domains': ['example.com']},
    'openai': {'typo': 1},
    'xai': {'typo': 1},
}


_needs_anthropic = pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
_needs_xai = pytest.mark.skipif(not xai_available(), reason='xai not installed')


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.usefixtures('allow_model_requests')
class TestProviderSettings:
    @_needs_anthropic
    @pytest.mark.parametrize(('tool', 'request_key', 'value'), ANTHROPIC_OPTIONS)
    def test_anthropic_option_reaches_the_request(self, tool: AbstractNativeTool, request_key: str, value: Any) -> None:
        assert _anthropic_request_tool(tool)[request_key] == value

    @pytest.mark.parametrize(('tool', 'request_key', 'value'), OPENROUTER_OPTIONS)
    def test_openrouter_option_reaches_the_request(
        self, tool: AbstractNativeTool, request_key: str, value: Any
    ) -> None:
        assert _openrouter_request_parameters(tool)[request_key] == value

    @pytest.mark.parametrize(('tool', 'request_key', 'value'), OPENAI_OPTIONS)
    def test_openai_option_reaches_the_request(self, tool: AbstractNativeTool, request_key: str, value: Any) -> None:
        assert _openai_request_tool(tool)[request_key] == value

    @_needs_xai
    @pytest.mark.parametrize(('tool', 'request_key', 'value'), XAI_OPTIONS)
    def test_xai_option_reaches_the_request(self, tool: AbstractNativeTool, request_key: str, value: Any) -> None:
        assert _xai_request_search(tool)[request_key] == value

    def test_every_declared_option_is_covered(self) -> None:
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
            pytest.param(
                _anthropic_request_tool,
                AdvisorTool,
                {'model': 'advisor'},
                'anthropic',
                'caching',
                '5m',
                marks=_needs_anthropic,
            ),
            pytest.param(
                _anthropic_request_tool,
                AdvisorTool,
                {'model': 'advisor'},
                'anthropic',
                'max_uses',
                2,
                marks=_needs_anthropic,
            ),
            pytest.param(
                _anthropic_request_tool, WebFetchTool, {}, 'anthropic', 'enable_citations', True, marks=_needs_anthropic
            ),
            pytest.param(
                _anthropic_request_tool,
                WebFetchTool,
                {},
                'anthropic',
                'max_content_tokens',
                1000,
                marks=_needs_anthropic,
            ),
            pytest.param(
                _xai_request_search,
                FileSearchTool,
                {'file_store_ids': ['store']},
                'xai',
                'instructions',
                'search these',
                marks=_needs_xai,
            ),
            pytest.param(
                _xai_request_search,
                FileSearchTool,
                {'file_store_ids': ['store']},
                'xai',
                'retrieval_mode',
                'keyword',
                marks=_needs_xai,
            ),
        ],
    )
    def test_moved_field_warns_and_sends_the_same_request_as_nested(
        self, request_tool: Any, tool_type: Any, required: dict[str, Any], system: str, option: str, value: Any
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
            (AdvisorTool, {'model': 'advisor', 'max_uses': 1, 'provider_settings': {'anthropic': {'max_uses': 2}}}),
            (
                WebFetchTool,
                {'max_content_tokens': 1000, 'provider_settings': {'anthropic': {'max_content_tokens': 2000}}},
            ),
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
    def test_moved_field_conflicting_with_nested_value_is_rejected(
        self, tool_type: Any, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(UserError, match='conflicts with `provider_settings'):
            tool_type(**kwargs)

    def test_moved_field_matching_nested_value_only_warns(self) -> None:
        with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED):
            tool = ImageGenerationTool(quality='high', provider_settings={'openai': {'quality': 'high'}})

        assert _openai_request_tool(tool)['quality'] == 'high'

    def test_rebuilding_a_capability_tool_from_data_does_not_warn(self) -> None:
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
    def test_moved_field_in_a_spec_warns(self, spec: dict[str, Any]) -> None:
        with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED):
            capability = NativeTool.from_spec(**spec)

        assert isinstance(capability.tool, ImageGenerationTool)
        assert capability.tool.quality == 'high'

    def test_moved_field_on_a_tool_passed_to_a_spec_warns_once(self) -> None:
        with pytest.warns(PydanticAIDeprecationWarning, match=QUALITY_DEPRECATED) as warnings:
            NativeTool.from_spec(tool=ImageGenerationTool(quality='high'))

        assert len(warnings) == 1

    @_needs_anthropic
    def test_each_model_reads_only_its_own_settings(self) -> None:
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
    def test_value_openrouter_adds_later_is_forwarded(self, option: str) -> None:
        tool = WebSearchTool(provider_settings={'openrouter': {option: 'added-later'}})  # pyright: ignore[reportArgumentType]

        assert _openrouter_request_parameters(tool)[option] == 'added-later'

    @pytest.mark.parametrize(
        ('read_request', 'make_tool'),
        [
            pytest.param(_anthropic_request_tool, WebSearchTool, marks=_needs_anthropic),
            pytest.param(_anthropic_request_tool, WebFetchTool, marks=_needs_anthropic),
            (_openrouter_request_parameters, WebSearchTool),
            (_openrouter_request_parameters, partial(AdvisorTool, model='advisor')),
            (_openai_request_tool, ImageGenerationTool),
            pytest.param(_xai_request_search, partial(FileSearchTool, file_store_ids=['store']), marks=_needs_xai),
        ],
    )
    def test_unknown_settings_are_ignored(self, read_request: Any, make_tool: Any) -> None:
        """Like model settings, an unknown option is never sent, so it can't override a field like `allowed_domains`."""
        assert read_request(make_tool(provider_settings=UNKNOWN_SETTINGS)) == read_request(make_tool())
