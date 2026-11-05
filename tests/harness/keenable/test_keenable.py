"""Tests for the KeenableSearch capability and KeenableSearchToolset."""

from __future__ import annotations

import functools
import json
from collections.abc import AsyncIterator, Callable

import anyio
import httpx
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict, Unpack

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_harness.keenable import (
    HttpKeenableClient,
    KeenableSearch,
    KeenableSearchToolset,
    KeenableSource,
)


class _Budgets(TypedDict, total=False):
    num_results: int
    max_snippet_chars: int
    max_page_chars: int


# A realistic Keenable result: the API populates `snippet` with the page text
# and leaves the deprecated `description` empty.
def _result(url: str, title: str | None = 'A title', snippet: str = 'Page text') -> dict[str, object]:
    return {'url': url, 'title': title, 'description': '', 'snippet': snippet}


class FakeClient:
    """Stands in for the HTTP client; records calls and returns canned payloads."""

    def __init__(
        self,
        *,
        results: list[dict[str, object]] | None = None,
        page: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.page = page if page is not None else {}
        self.error = error
        self.queries: list[str] = []
        self.max_results: list[int] = []
        self.urls: list[str] = []

    async def search(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        self.queries.append(query)
        self.max_results.append(max_results)
        if self.error is not None:
            raise self.error
        return self.results

    async def fetch(self, url: str) -> dict[str, object]:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.page


def _toolset(client: FakeClient, **budgets: Unpack[_Budgets]) -> KeenableSearchToolset[None]:
    return KeenableSearch[None](client=client, **budgets).get_toolset()


def _text(returned: ToolReturn[str]) -> str:
    # `ToolReturn.return_value` is typed as the full multi-modal content union,
    # so narrow it once here rather than at every assertion.
    value = returned.return_value
    assert isinstance(value, str)
    return value


def _capture_requests(monkeypatch: pytest.MonkeyPatch, respond: Callable[[httpx.Request], httpx.Response]):
    """Route every `httpx.AsyncClient` the code builds through `respond`, recording the requests."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    monkeypatch.setattr(
        httpx, 'AsyncClient', functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
    )
    return seen


def _research(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Search, read one page, then answer: the loop the default instructions ask for."""
    step = sum(isinstance(message, ModelResponse) for message in messages)
    if step == 0:
        return ModelResponse(parts=[ToolCallPart('web_search', {'query': 'Pydantic AI agent framework'})])
    if step == 1:
        return ModelResponse(parts=[ToolCallPart('get_page', {'url': 'https://example.com/'})])
    return ModelResponse(parts=[TextPart('Done.')])


class TestKeenableSearch:
    @pytest.mark.vcr(additional_matchers=['json_body'])
    async def test_agent_searches_and_reads_a_page(self, monkeypatch: pytest.MonkeyPatch):
        # Recorded against the keyless public endpoints, so the cassette holds
        # no credential. The body matcher keeps `max_results` on the wire.
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        agent = Agent(FunctionModel(_research), capabilities=[KeenableSearch(num_results=2, max_snippet_chars=80)])

        result = await agent.run('Research Pydantic AI.')

        returns = {
            part.tool_name: part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert (returns['web_search'].content, returns['web_search'].metadata) == snapshot(
            (
                """\
1. GitHub - NVIDIA-AI-Blueprints/pydantic-ai: AI Agent Framework, the Pydantic way
   https://github.com/NVIDIA-AI-Blueprints/pydantic-ai
   ### *Pydantic AI is a Python agent framework designed to help you quickly, confi
2. GitHub - ZacharyHampton/pydantic-ai: AI Agent Framework, the Pydantic way
   https://github.com/ZacharyHampton/pydantic-ai
   ### GenAI Agent Framework, the Pydantic way 1. **Built by the Pydantic Team**: [\
""",
                {
                    'sources': [
                        {
                            'url': 'https://github.com/NVIDIA-AI-Blueprints/pydantic-ai',
                            'title': 'GitHub - NVIDIA-AI-Blueprints/pydantic-ai: AI Agent Framework, the Pydantic way',
                        },
                        {
                            'url': 'https://github.com/ZacharyHampton/pydantic-ai',
                            'title': 'GitHub - ZacharyHampton/pydantic-ai: AI Agent Framework, the Pydantic way',
                        },
                    ]
                },
            )
        )
        assert (returns['get_page'].content, returns['get_page'].metadata) == snapshot(
            (
                'This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.',
                {'sources': [{'url': 'https://example.com/', 'title': 'Example Domain'}]},
            )
        )

    def test_default_instructions_mention_both_tools(self):
        instructions = KeenableSearch[None]().get_instructions()

        assert isinstance(instructions, str)
        assert '`web_search`' in instructions
        assert '`get_page`' in instructions
        assert 'untrusted' in instructions

    @pytest.mark.parametrize(('native_web_search', 'sent_tools'), [(False, ['web_search']), (True, [])])
    async def test_web_search_tool_is_a_core_web_search_fallback(self, native_web_search: bool, sent_tools: list[str]):
        client = FakeClient(results=[_result('https://example.com/one', 'One')])
        keenable = KeenableSearch[object](num_results=3, client=client)
        sent: list[list[str]] = []

        def search_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append([tool.name for tool in info.function_tools])
            if len(messages) == 1 and not native_web_search:
                return ModelResponse(parts=[ToolCallPart('web_search', {'query': 'cats'})])
            return ModelResponse(parts=[TextPart('done')])

        supported = frozenset({WebSearchTool}) if native_web_search else frozenset[type[WebSearchTool]]()
        model = FunctionModel(search_once, profile=ModelProfile(supported_native_tools=supported))
        result = await Agent(model, capabilities=[WebSearch[object](local=keenable.web_search_tool())]).run('Search.')

        assert sent[0] == sent_tools
        returns = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if native_web_search:
            assert returns == []
        else:
            assert [part.metadata for part in returns] == [
                {'sources': [{'url': 'https://example.com/one', 'title': 'One'}]}
            ]
            assert client.max_results == [3]

    def test_guidance_overrides_and_disables_instructions(self):
        # The same contract as `ExaSearch` and `YouSearch`, so the three swap freely.
        assert KeenableSearch[None](guidance='Just search.').get_instructions() == 'Just search.'
        assert KeenableSearch[None](guidance='').get_instructions() is None

    @pytest.mark.parametrize(
        ('budgets', 'message'),
        [
            ({'num_results': 0}, '`num_results` must be a positive integer'),
            ({'num_results': 51}, '`num_results` must be at most 50, got 51'),
            ({'max_snippet_chars': 0}, '`max_snippet_chars` must be a positive integer'),
            ({'max_page_chars': 0}, '`max_page_chars` must be a positive integer'),
            # The annotation is advisory; a config-driven caller can pass a float,
            # a string, or a bool, which would otherwise only fail later as a
            # slice bound, or in the bool's case, not fail at all.
            ({'num_results': 1.5}, '`num_results` must be a positive integer, got 1.5'),
            ({'max_page_chars': '10'}, "`max_page_chars` must be a positive integer, got '10'"),
            ({'max_page_chars': True}, '`max_page_chars` must be a positive integer, got True'),
            ({'num_results': True}, '`num_results` must be a positive integer, got True'),
        ],
    )
    def test_budgets_are_validated(self, budgets: _Budgets, message: str):
        with pytest.raises(ValueError, match=message):
            KeenableSearch[None](**budgets)

    def test_from_spec_sets_a_distinct_id(self):
        assert KeenableSearch[None].from_spec().id == 'keenable_search'
        assert KeenableSearch[None].from_spec(id='docs_search').id == 'docs_search'

    def test_from_spec_sets_the_base_capability_fields(self):
        """`id`, `description` and `defer_loading` live on `AbstractCapability`, so a spec must name them too."""
        definitions = AgentSpec.model_json_schema_with_capabilities([KeenableSearch])['$defs']
        assert {'id', 'description', 'defer_loading'} <= definitions['spec_params_KeenableSearch']['properties'].keys()
        capability = KeenableSearch[None].from_spec(id='second', description='The second one.', defer_loading=True)
        assert (capability.id, capability.description, capability.defer_loading) == ('second', 'The second one.', True)

    def test_from_spec_builds_the_default_client(self):
        capability = KeenableSearch[None].from_spec(num_results=2, max_snippet_chars=10, max_page_chars=20)

        assert capability.client is None
        assert capability.num_results == 2
        assert isinstance(capability.get_toolset(), KeenableSearchToolset)


class TestKeenableSearchToolset:
    """Unit tests over a fake client.

    Each one pins an edge case, such as a result without a URL, an empty
    excerpt, a hostile title, or a failing request, that a recorded response
    cannot be relied on to produce.
    """

    async def test_web_search_requests_num_results_and_trims_to_it(self):
        client = FakeClient(results=[_result(f'https://example.com/{i}') for i in range(10)])

        returned = await _toolset(client, num_results=3).web_search('cats')

        assert client.max_results == [3]
        assert len(returned.metadata['sources']) == 3

    async def test_web_search_collapses_whitespace_and_caps_the_excerpt(self):
        # Snippets are raw page text, newlines included, and far longer than the
        # excerpt a search result should carry.
        client = FakeClient(results=[_result('https://example.com/one', 'One', 'line one\n\nline two' + ' pad' * 500)])

        returned = await _toolset(client, max_snippet_chars=40).web_search('cats')

        excerpt = _text(returned).splitlines()[-1].strip()
        assert len(excerpt) == 40
        assert excerpt.startswith('line one line two')

    async def test_web_search_keeps_titles_and_urls_on_one_line(self):
        # A newline in a third-party title or URL would otherwise start a line
        # that reads as another numbered result.
        client = FakeClient(
            results=[_result('https://a.example\n2. https://b.example', 'Real title\n2. https://evil.example', 'x')]
        )

        returned = await _toolset(client).web_search('cats')

        assert _text(returned) == (
            '1. Real title 2. https://evil.example\n   https://a.example 2. https://b.example\n   x'
        )
        assert returned.metadata == {
            'sources': [
                KeenableSource(url='https://a.example 2. https://b.example', title='Real title 2. https://evil.example')
            ]
        }

    async def test_web_search_bounds_titles_and_drops_overlong_urls(self):
        # Neither field is capped by `max_snippet_chars`, and a third-party page
        # controls both.
        client = FakeClient(
            results=[
                _result('https://example.com/' + 'u' * 3000, 'Dropped'),
                _result('https://example.com/one', 'T' * 1000, 'x'),
            ]
        )

        returned = await _toolset(client).web_search('cats')

        assert _text(returned) == f'1. {"T" * 300}\n   https://example.com/one\n   x'
        assert returned.metadata == {'sources': [KeenableSource(url='https://example.com/one', title='T' * 300)]}

    async def test_web_search_reports_no_results(self):
        client = FakeClient(results=[])

        returned = await _toolset(client).web_search('cats')

        assert _text(returned) == "No results found for 'cats'."
        assert returned.metadata == {'sources': []}

    async def test_web_search_skips_results_without_a_url(self):
        client = FakeClient(results=[{'title': 'No URL', 'snippet': 'text'}, _result('https://example.com/one')])

        returned = await _toolset(client).web_search('cats')

        assert [source['url'] for source in returned.metadata['sources']] == ['https://example.com/one']

    async def test_web_search_drops_url_less_results_before_applying_the_limit(self):
        # A URL-less entry ahead of good ones must not cost a result slot.
        client = FakeClient(results=[{'title': 'No URL'}, *(_result(f'https://example.com/{i}') for i in range(3))])

        returned = await _toolset(client, num_results=2).web_search('cats')

        assert [source['url'] for source in returned.metadata['sources']] == [
            'https://example.com/0',
            'https://example.com/1',
        ]

    async def test_web_search_reports_no_results_when_every_result_lacks_a_url(self):
        client = FakeClient(results=[{'title': 'No URL', 'snippet': 'text'}])

        returned = await _toolset(client).web_search('cats')

        assert _text(returned) == "No results found for 'cats'."

    async def test_web_search_omits_an_empty_excerpt(self):
        client = FakeClient(results=[{'url': 'https://example.com/one', 'title': 'One'}])

        returned = await _toolset(client).web_search('cats')

        assert _text(returned) == '1. One\n   https://example.com/one'

    async def test_web_search_falls_back_to_the_url_when_a_result_has_no_title(self):
        client = FakeClient(results=[_result('https://example.com/one', None)])

        returned = await _toolset(client).web_search('cats')

        assert _text(returned).startswith('1. https://example.com/one')
        assert returned.metadata['sources'][0]['title'] is None

    async def test_get_page_truncates_and_marks_long_pages(self):
        client = FakeClient(page={'content': 'x' * 100})

        returned = await _toolset(client, max_page_chars=60).get_page('https://example.com/one')

        # The marker is part of the budget, not extra.
        assert _text(returned) == f'{"x" * 17}\n[... page text truncated at 60 characters]'
        assert len(_text(returned)) == 60
        assert returned.metadata['sources'][0] == KeenableSource(url='https://example.com/one', title=None)

    async def test_get_page_uses_an_ellipsis_when_the_budget_cannot_hold_the_marker(self):
        client = FakeClient(page={'content': 'x' * 100})

        returned = await _toolset(client, max_page_chars=5).get_page('https://example.com/one')

        assert _text(returned) == f'{"x" * 4}…'

    async def test_get_page_retries_on_empty_content(self):
        client = FakeClient(page={'content': ''})

        with pytest.raises(ModelRetry, match='No readable content'):
            await _toolset(client).get_page('https://example.com/one')

    @pytest.mark.parametrize('status', [404, 429, 500])
    async def test_transient_http_errors_become_retries(self, status: int):
        request = httpx.Request('POST', 'https://api.keenable.ai/v1/search/public')
        error = httpx.HTTPStatusError('boom', request=request, response=httpx.Response(status, request=request))
        client = FakeClient(error=error)

        with pytest.raises(ModelRetry, match='Keenable request failed'):
            await _toolset(client).web_search('cats')

    async def test_status_retry_prompt_omits_the_request_url(self):
        request = httpx.Request('GET', 'https://user:secret@proxy.example/v1/fetch?url=https://example.com')
        error = httpx.HTTPStatusError('boom', request=request, response=httpx.Response(429, request=request))
        client = FakeClient(error=error)

        with pytest.raises(ModelRetry) as excinfo:
            await _toolset(client).get_page('https://example.com/one')

        assert str(excinfo.value) == 'Keenable request failed with HTTP 429 Too Many Requests.'

    @pytest.mark.parametrize('status', [401, 402, 403])
    async def test_auth_errors_propagate(self, status: int):
        request = httpx.Request('POST', 'https://api.keenable.ai/v1/search')
        error = httpx.HTTPStatusError('nope', request=request, response=httpx.Response(status, request=request))
        client = FakeClient(error=error)

        with pytest.raises(httpx.HTTPStatusError):
            await _toolset(client).web_search('cats')

    async def test_network_errors_become_retries(self):
        client = FakeClient(error=httpx.ConnectError('unreachable'))

        with pytest.raises(ModelRetry, match='Keenable request failed'):
            await _toolset(client).get_page('https://example.com/one')

    @pytest.mark.parametrize(
        ('budgets', 'message'),
        [
            ({'num_results': -1}, '`num_results` must be a positive integer'),
            ({'num_results': 51}, '`num_results` must be at most 50'),
            ({'max_snippet_chars': 0}, '`max_snippet_chars` must be a positive integer'),
            ({'max_page_chars': -10}, '`max_page_chars` must be a positive integer'),
        ],
    )
    def test_the_toolset_validates_budgets_when_built_directly(self, budgets: _Budgets, message: str):
        # The toolset is exported, so it cannot rely on `KeenableSearch` having
        # validated first: a negative budget slices from the end and would return
        # the wrong text rather than fail.
        with pytest.raises(ValueError, match=message):
            KeenableSearchToolset[None](
                client=FakeClient(),
                num_results=budgets.get('num_results', 5),
                max_snippet_chars=budgets.get('max_snippet_chars', 500),
                max_page_chars=budgets.get('max_page_chars', 10_000),
            )


class TestHttpKeenableClient:
    """Wire tests against a mock transport.

    They pin what the client sends and how it shapes odd responses: a cassette
    does not catch a header or field the code stops sending, and the live API
    cannot be made to return a malformed, oversized, or stalled body.
    """

    async def test_default_client_is_keyless(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        seen = _capture_requests(monkeypatch, lambda _request: httpx.Response(200, json={'results': []}))

        await HttpKeenableClient().search('cats', max_results=5)
        await HttpKeenableClient().fetch('https://example.com/one')

        assert [request.url.path for request in seen] == ['/v1/search/public', '/v1/fetch/public']
        assert all('x-api-key' not in request.headers for request in seen)
        assert all(request.headers['x-keenable-title'] == 'Pydantic AI Harness' for request in seen)

    @pytest.mark.parametrize('source', ['environment', 'argument'])
    async def test_api_key_selects_the_keyed_endpoint(self, monkeypatch: pytest.MonkeyPatch, source: str):
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        if source == 'environment':
            monkeypatch.setenv('KEENABLE_API_KEY', 'keen_live_x')
            client = HttpKeenableClient()
        else:
            monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
            client = HttpKeenableClient(api_key='keen_live_x')
        seen = _capture_requests(monkeypatch, lambda _request: httpx.Response(200, json={'results': []}))

        await client.search('cats', max_results=5)
        await client.fetch('https://example.com/one')

        assert [request.url.path for request in seen] == ['/v1/search', '/v1/fetch']
        assert all(request.headers['x-api-key'] == 'keen_live_x' for request in seen)
        # The key travels only as a header: never in the URL, never in the body.
        assert all('keen_live_x' not in str(request.url) and b'keen_live_x' not in request.content for request in seen)

    async def test_an_injected_http_client_carries_every_request_and_stays_open(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        seen: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={'results': [_result('https://example.com/one')], 'content': '# A page'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            client = HttpKeenableClient(http_client=http_client)

            assert await client.search('cats', max_results=5) == [_result('https://example.com/one')]
            assert (await client.fetch('https://example.com/one'))['content'] == '# A page'
            assert not http_client.is_closed

        assert [request.url.path for request in seen] == ['/v1/search/public', '/v1/fetch/public']

    async def test_the_api_key_never_follows_a_redirect(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        seen: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(302, headers={'Location': 'http://elsewhere.example/collect'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as http_client:
            client = HttpKeenableClient(api_key='keen_live_x', http_client=http_client)

            with pytest.raises(httpx.HTTPStatusError):
                await client.search('cats', max_results=5)

        assert [str(request.url) for request in seen] == ['https://api.keenable.ai/v1/search']

    async def test_base_url_is_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.setenv('KEENABLE_API_URL', 'http://localhost:8080/')
        seen = _capture_requests(monkeypatch, lambda _request: httpx.Response(200, json={'results': []}))

        await HttpKeenableClient().search('cats', max_results=5)

        assert [str(request.url) for request in seen] == ['http://localhost:8080/v1/search/public']

    async def test_http_client_parses_search_and_fetch(self, monkeypatch: pytest.MonkeyPatch):
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith('/v1/search'):
                return httpx.Response(200, json={'results': [_result('https://example.com/one'), 'not-a-dict']})
            return httpx.Response(200, json={'content': '# A page'})

        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        seen = _capture_requests(monkeypatch, respond)
        client = HttpKeenableClient()

        assert await client.search('cats', max_results=7) == [_result('https://example.com/one')]
        assert await client.fetch('https://example.com/one') == {'content': '# A page'}

        search, fetch = seen
        assert str(search.url) == 'https://api.keenable.ai/v1/search/public'
        assert json.loads(search.content) == {'query': 'cats', 'max_results': 7}
        assert str(fetch.url) == 'https://api.keenable.ai/v1/fetch/public?url=https%3A%2F%2Fexample.com%2Fone'

    @pytest.mark.parametrize('payload', [{'results': 'nope'}, [], {}])
    async def test_http_client_tolerates_unexpected_payloads(self, monkeypatch: pytest.MonkeyPatch, payload: object):
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        _capture_requests(monkeypatch, lambda _request: httpx.Response(200, json=payload))
        client = HttpKeenableClient()

        assert await client.search('cats', max_results=5) == []
        assert await client.fetch('https://example.com/one') == (payload if isinstance(payload, dict) else {})

    @pytest.mark.parametrize('body', [b'', b'<html>gateway timeout</html>', b'{"results": "\xff"}'])
    async def test_malformed_2xx_bodies_become_retries(self, monkeypatch: pytest.MonkeyPatch, body: bytes):
        # A 2xx that is not JSON is neither an `httpx.HTTPError` nor a deliberate
        # exception; without this it would kill the run as a bare decode error.
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        _capture_requests(monkeypatch, lambda _request: httpx.Response(200, content=body))
        toolset = KeenableSearchToolset[None](client=None, num_results=5, max_snippet_chars=500, max_page_chars=10_000)

        with pytest.raises(ModelRetry, match='malformed response'):
            await toolset.web_search('cats')
        with pytest.raises(ModelRetry, match='malformed response'):
            await toolset.get_page('https://example.com/one')

    async def test_oversized_bodies_become_retries(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(HttpKeenableClient, '_MAX_RESPONSE_BYTES', 32)
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        _capture_requests(monkeypatch, lambda _request: httpx.Response(200, content=b'{"results": []}' + b' ' * 64))
        toolset = KeenableSearchToolset[None](client=None, num_results=5, max_snippet_chars=500, max_page_chars=10_000)

        with pytest.raises(ModelRetry, match='response body exceeds 32 bytes'):
            await toolset.web_search('cats')

    async def test_a_slow_body_hits_the_request_deadline(self, monkeypatch: pytest.MonkeyPatch):
        # Each chunk arrives well within the per-read timeout, so only the
        # deadline on the whole request stops it.
        async def drip() -> AsyncIterator[bytes]:
            while True:
                yield b' '
                await anyio.sleep(0.001)

        monkeypatch.setattr(HttpKeenableClient, '_TIMEOUT_SECONDS', 0.05)
        monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
        monkeypatch.delenv('KEENABLE_API_URL', raising=False)
        _capture_requests(monkeypatch, lambda _request: httpx.Response(200, content=drip()))
        toolset = KeenableSearchToolset[None](client=None, num_results=5, max_snippet_chars=500, max_page_chars=10_000)

        with pytest.raises(ModelRetry, match='timed out'):
            await toolset.web_search('cats')

    @pytest.mark.parametrize('base_url', ['http://localhost:8080', 'https://keenable.internal'])
    def test_base_url_accepts_https_and_loopback_http(self, base_url: str):
        assert HttpKeenableClient(api_key='', base_url=base_url) is not None

    @pytest.mark.parametrize(
        'base_url',
        [
            'http://example.com',
            'https://',
            'not-a-url',
            # The endpoint path is appended to the base URL, so a query or fragment
            # would land in front of it and the request would miss the endpoint.
            'https://api.keenable.ai?token=x',
            'https://api.keenable.ai#frag',
            # An empty query or fragment parses as none at all, but the path would
            # still land behind the `?` or `#`.
            'https://api.keenable.ai?',
            'https://api.keenable.ai#',
        ],
    )
    def test_base_url_rejects_plaintext_hostless_and_query_bearing_urls(self, base_url: str):
        with pytest.raises(UserError, match=r'^`base_url` must be an https:// URL, or http:// on loopback'):
            HttpKeenableClient(api_key='', base_url=base_url)

    def test_base_url_errors_name_the_environment_variable_it_came_from(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('KEENABLE_API_URL', 'http://example.com')

        with pytest.raises(UserError, match=r'^`KEENABLE_API_URL` must be an https:// URL'):
            HttpKeenableClient(api_key='')

    def test_base_url_rejects_credentials_without_echoing_them(self):
        with pytest.raises(UserError, match=r'^`base_url` must not contain credentials') as excinfo:
            HttpKeenableClient(api_key='', base_url='https://user:secret@proxy.example')

        assert 'secret' not in str(excinfo.value)
