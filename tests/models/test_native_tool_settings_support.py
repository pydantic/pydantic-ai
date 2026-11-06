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
from typing import Any

import pytest
from typing_extensions import get_type_hints

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters, resolve_request_tools
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    AdvisorProviderSettings,
    AdvisorTool,
    CodeExecutionTool,
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

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .mock_openai import (
        MockOpenAI,
        completion_message as openai_completion,
        get_mock_chat_completion_kwargs as openai_request_kwargs,
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
            AdvisorTool(model='claude-opus-4-8', max_uses=1),
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
    assert WebFetchTool(max_content_tokens=1000).restricting_settings == frozenset()
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


# `provider_settings`: each option reaches its own model's request, and unknown ones are ignored.


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


UNKNOWN_SETTINGS: Any = {
    'madeup': {},
    'anthropic': {'typo': 1},
    'openrouter': {'typo': 1, 'allowed_domains': ['example.com']},
}


_needs_anthropic = pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.usefixtures('allow_model_requests')
class TestProviderSettings:
    @_needs_anthropic
    @pytest.mark.parametrize(('tool', 'option', 'value'), ANTHROPIC_OPTIONS)
    def test_anthropic_option_reaches_the_request(self, tool: AbstractNativeTool, option: str, value: Any) -> None:
        assert _anthropic_request_tool(tool)[option] == value

    @pytest.mark.parametrize(('tool', 'option', 'value'), OPENROUTER_OPTIONS)
    def test_openrouter_option_reaches_the_request(self, tool: AbstractNativeTool, option: str, value: Any) -> None:
        assert _openrouter_request_parameters(tool)[option] == value

    def test_every_declared_option_is_covered(self) -> None:
        """Adding an option without covering it above fails here, rather than silently doing nothing for users."""
        settings_types: dict[type[AbstractNativeTool], Any] = {
            WebSearchTool: WebSearchProviderSettings,
            WebFetchTool: WebFetchProviderSettings,
            AdvisorTool: AdvisorProviderSettings,
        }
        declared = {
            (tool_type, provider, option)
            for tool_type, settings_type in settings_types.items()
            for provider, options in get_type_hints(settings_type).items()
            for option in get_type_hints(options)
        }
        covered = {(type(tool), 'anthropic', option) for tool, option, _ in ANTHROPIC_OPTIONS} | {
            (type(tool), 'openrouter', option) for tool, option, _ in OPENROUTER_OPTIONS
        }
        assert declared == covered

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
        ],
    )
    def test_unknown_settings_are_ignored(self, read_request: Any, make_tool: Any) -> None:
        """Like model settings, an unknown option is never sent, so it can't override a field like `allowed_domains`."""
        assert read_request(make_tool(provider_settings=UNKNOWN_SETTINGS)) == read_request(make_tool())
