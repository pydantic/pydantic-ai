"""Tests for the You.com capabilities (`YouSearch`, `YouResearch`) and their toolsets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

# `tests/youdotcom` shadows the installed `youdotcom` for a bare `from youdotcom import ...`,
# so import the SDK through its submodules (which the shadow package does not define).
import youdotcom.models as models
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel
from youdotcom.errors import YouError

from pydantic_ai_harness.youdotcom import (
    ExtractionModeName,
    FinanceEffortName,
    ResearchEffortName,
    YouResearch,
    YouResearchToolset,
    YouSearch,
    YouSearchToolset,
)
from pydantic_ai_harness.youdotcom._toolset import default_client


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _text(output: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = output.return_value
    assert isinstance(body, str)
    return body


def _you_error(status: int) -> YouError:
    """A real `YouError` whose `status_code` comes from a synthetic response."""
    request = httpx.Request('POST', 'https://api.you.com/v1/search')
    return YouError('You.com error', httpx.Response(status_code=status, request=request, text='boom'))


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    """A raw `httpx.HTTPStatusError` carrying `status`, as a transport layer might raise it."""
    request = httpx.Request('POST', 'https://api.you.com/v1/search')
    response = httpx.Response(status_code=status, request=request, text='boom')
    return httpx.HTTPStatusError('status error', request=request, response=response)


class _NoStatusYouError(YouError):
    """A `YouError` with no `status_code` attribute, to exercise the non-int branch."""

    def __init__(self) -> None:
        Exception.__init__(self, 'no status')
        self.message = 'no status'


# --- response builders ------------------------------------------------------


def _web(
    url: str | None = 'https://a.dev',
    *,
    title: str | None = 'A',
    highlights: list[str] | None = None,
    markdown: str | None = None,
    snippets: list[str] | None = None,
    description: str | None = None,
    page_age: datetime | None = None,
) -> models.WebResult:
    contents = None
    if highlights is not None or markdown is not None:
        contents = models.Contents(highlights=highlights, markdown=markdown)
    return models.WebResult(
        url=url, title=title, snippets=snippets, description=description, page_age=page_age, contents=contents
    )


def _search(
    *web: models.WebResult, search_uuid: str | None = None, latency: float | None = None
) -> models.SearchResponse:
    metadata = models.SearchMetadata(search_uuid=search_uuid, latency=latency) if search_uuid or latency else None
    return models.SearchResponse(results=models.Results(web=list(web)), metadata=metadata)


def _contents(
    *,
    url: str | None = 'https://a.dev',
    title: str | None = 'A',
    markdown: str | None = 'body',
    site_name: str | None = None,
) -> list[models.ContentsResponse]:
    metadata = models.ContentsMetadata(site_name=site_name) if site_name is not None else None
    return [models.ContentsResponse(url=url, title=title, markdown=markdown, metadata=metadata)]


def _answer(
    answer: str = 'the answer',
    *,
    web: Sequence[tuple[str, str]] = (),
    citations: Sequence[str] = (),
) -> models.AnswerResponse:
    results = models.AnswerResults(web=[models.AnswerSearchResult(url=u, title=t) for u, t in web]) if web else None
    cites = [models.AnswerCitation(source=s) for s in citations] if citations else None
    return models.AnswerResponse(answer=answer, citations=cites, results=results)


def _research(
    content: str | dict[str, object] = 'deep answer',
    *,
    sources: Sequence[tuple[str, str | None]] = (('https://s.dev', 'S'),),
    warnings: list[str] | None = None,
    object_output: bool = False,
) -> models.ResearchResponse:
    return models.ResearchResponse(
        output=models.Output(
            content=content,
            content_type=models.ContentType.OBJECT if object_output else models.ContentType.TEXT,
            sources=[models.Source(url=u, title=t) for u, t in sources],
        ),
        warnings=warnings,
    )


def _finance(
    content: str = 'finance answer', *, sources: Sequence[tuple[str, str | None]] = (('https://f.dev', 'F'),)
) -> models.FinanceResearchResponse:
    return models.FinanceResearchResponse(
        output=models.FinanceResearchOutput(
            content=content,
            content_type=models.FinanceResearchContentType.TEXT,
            sources=[models.FinanceResearchSource(url=u, title=t) for u, t in sources],
        )
    )


@dataclass
class _FakeYouClient:
    """In-memory `YouClient` double: canned responses, recorded call arguments."""

    search_response: models.SearchResponse = field(default_factory=models.SearchResponse)
    contents_response: list[models.ContentsResponse] = field(default_factory=list[models.ContentsResponse])
    answer_response: models.AnswerResponse = field(default_factory=lambda: _answer())
    research_result: models.ResearchResponse | models.TaskResponse = field(default_factory=lambda: _research())
    finance_response: models.FinanceResearchResponse = field(default_factory=lambda: _finance())
    error: Exception | None = None
    search_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    contents_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    answer_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    research_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    finance_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def _guard(self) -> None:
        """Raise the configured error, if any, before returning a canned response."""
        if self.error is not None:
            raise self.error

    async def search_async(
        self,
        *,
        query: str,
        count: int | None = None,
        freshness: str | None = None,
        country: str | None = None,
        extraction: models.Extraction | Mapping[str, object] | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.SearchResponse:
        self._guard()
        self.search_calls.append(
            {
                'query': query,
                'count': count,
                'freshness': freshness,
                'country': country,
                'extraction': extraction,
                'include_domains': include_domains,
                'exclude_domains': exclude_domains,
                'boost_domains': boost_domains,
            }
        )
        return self.search_response

    async def contents_async(
        self,
        *,
        urls: Sequence[str] | None = None,
        formats: Sequence[models.ContentsFormats] | None = None,
    ) -> list[models.ContentsResponse]:
        self._guard()
        self.contents_calls.append({'urls': urls, 'formats': formats})
        return self.contents_response

    async def answer_async(
        self,
        *,
        query: str,
        freshness: str | None = None,
        country: str | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.AnswerResponse:
        self._guard()
        self.answer_calls.append(
            {
                'query': query,
                'freshness': freshness,
                'country': country,
                'include_domains': include_domains,
                'exclude_domains': exclude_domains,
                'boost_domains': boost_domains,
            }
        )
        return self.answer_response

    async def research_async(
        self,
        *,
        input: str,
        research_effort: models.ResearchEffort | None = None,
        background: bool | None = None,
        source_control: models.SourceControl | Mapping[str, object] | None = None,
        output_schema: Mapping[str, object] | None = None,
    ) -> models.ResearchResult:
        self._guard()
        self.research_calls.append(
            {
                'input': input,
                'research_effort': research_effort,
                'background': background,
                'source_control': source_control,
                'output_schema': output_schema,
            }
        )
        return self.research_result

    async def finance_research_async(
        self,
        *,
        input: str,
        research_effort: models.FinanceResearchEffort | None = None,
    ) -> models.FinanceResearchResponse:
        self._guard()
        self.finance_calls.append({'input': input, 'research_effort': research_effort})
        return self.finance_response


def _search_toolset(
    client: _FakeYouClient,
    *,
    num_results: int = 10,
    extraction_mode: ExtractionModeName = 'highlights',
    max_text_chars: int = 10_000,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    boost_domains: list[str] | None = None,
    freshness: str | None = None,
    country: str | None = None,
) -> YouSearchToolset[None]:
    return YouSearch[None](
        client=client,
        num_results=num_results,
        extraction_mode=extraction_mode,
        max_text_chars=max_text_chars,
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
        boost_domains=boost_domains or [],
        freshness=freshness,
        country=country,
    ).get_toolset()


def _research_toolset(
    client: _FakeYouClient,
    *,
    research_effort: ResearchEffortName = 'standard',
    finance_effort: FinanceEffortName = 'deep',
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    boost_domains: list[str] | None = None,
    freshness: str | None = None,
    country: str | None = None,
    output_schema: Mapping[str, object] | None = None,
) -> YouResearchToolset[None]:
    return YouResearch[None](
        client=client,
        research_effort=research_effort,
        finance_effort=finance_effort,
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
        boost_domains=boost_domains or [],
        freshness=freshness,
        country=country,
        output_schema=output_schema,
    ).get_toolset()


class TestWebSearch:
    async def test_formats_results_with_excerpts_and_metadata(self) -> None:
        client = _FakeYouClient(
            search_response=_search(
                _web('https://a.dev', title='A', highlights=['alpha', 'beta'], page_age=datetime(2026, 7, 1)),
                _web('https://b.dev', title=None),
                search_uuid='uuid-1',
                latency=0.42,
            )
        )
        output = await _search_toolset(client).web_search('rust frameworks')
        assert _text(output) == (
            "Found 2 results for 'rust frameworks':\n\n"
            'Title: A\nURL: https://a.dev\nPublished: 2026-07-01\n\n- alpha\n- beta'
            '\n\n---\n\n'
            'Title: (untitled)\nURL: https://b.dev'
        )
        assert output.metadata == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': None}],
            'search_uuid': 'uuid-1',
            'latency': 0.42,
        }

    async def test_requests_highlights_count_and_domains(self) -> None:
        client = _FakeYouClient(search_response=_search(_web()))
        await _search_toolset(client, num_results=3, include_domains=['a.dev']).web_search('q')
        call = client.search_calls[0]
        assert call['count'] == 3
        assert call['include_domains'] == ['a.dev']
        extraction = call['extraction']
        assert isinstance(extraction, models.Extraction)
        assert extraction.extraction_mode == models.ExtractionMode.HIGHLIGHTS

    async def test_full_page_extraction_uses_markdown_body(self) -> None:
        client = _FakeYouClient(search_response=_search(_web('https://a.dev', title='A', markdown='x' * 500)))
        toolset = _search_toolset(client, extraction_mode='full_page', max_text_chars=100)
        output = await toolset.web_search('q')
        extraction = client.search_calls[0]['extraction']
        assert isinstance(extraction, models.Extraction)
        assert extraction.extraction_mode == models.ExtractionMode.FULL_PAGE
        assert extraction.full_page is not None
        assert extraction.full_page.extraction_formats == [models.ExtractionFormat.MARKDOWN]
        marker = '\n[... page text truncated at 100 characters]'
        page_text = _text(output).split('\n\n')[-1]
        # The marker counts against the cap, so the whole truncated body stays within max_text_chars.
        assert page_text.startswith('x') and page_text.endswith(marker) and len(page_text) == 100

    async def test_full_page_prefers_markdown_over_highlights(self) -> None:
        client = _FakeYouClient(
            search_response=_search(_web('https://a.dev', title='A', highlights=['alpha'], markdown='full text'))
        )
        output = await _search_toolset(client, extraction_mode='full_page').web_search('q')
        assert _text(output) == "Found 1 result for 'q':\n\nTitle: A\nURL: https://a.dev\n\nfull text"

    async def test_highlights_mode_prefers_highlights_over_markdown(self) -> None:
        client = _FakeYouClient(
            search_response=_search(_web('https://a.dev', title='A', highlights=['alpha'], markdown='full text'))
        )
        output = await _search_toolset(client, extraction_mode='highlights').web_search('q')
        assert _text(output) == "Found 1 result for 'q':\n\nTitle: A\nURL: https://a.dev\n\n- alpha"

    async def test_snippets_then_description_fallback(self) -> None:
        client = _FakeYouClient(search_response=_search(_web('https://a.dev', title='A', snippets=['snip'])))
        assert '- snip' in _text(await _search_toolset(client).web_search('q'))
        client = _FakeYouClient(search_response=_search(_web('https://a.dev', title='A', description='desc')))
        assert 'desc' in _text(await _search_toolset(client).web_search('q'))

    async def test_no_results_and_metadata_absent(self) -> None:
        # A bare SearchResponse has `results is None`, exercising the no-results branch.
        client = _FakeYouClient(search_response=models.SearchResponse())
        output = await _search_toolset(client).web_search('nothing')
        assert _text(output) == "No results found for 'nothing'."
        assert output.metadata == {'sources': [], 'search_uuid': None, 'latency': None}

    async def test_empty_web_list_reports_no_results(self) -> None:
        client = _FakeYouClient(search_response=_search())
        output = await _search_toolset(client).web_search('nothing')
        assert _text(output) == "No results found for 'nothing'."

    async def test_contents_present_but_empty_falls_back_to_snippets(self) -> None:
        result = models.WebResult(
            url='https://a.dev', title='A', contents=models.Contents(highlights=[], markdown=None), snippets=['snip']
        )
        client = _FakeYouClient(search_response=_search(result))
        assert '- snip' in _text(await _search_toolset(client).web_search('q'))

    async def test_skips_urlless_results_and_caps_count(self) -> None:
        client = _FakeYouClient(
            search_response=_search(
                _web(None, title='no url'),
                _web('https://a.dev', title='A'),
                _web('https://b.dev', title='B'),
                _web('https://c.dev', title='C'),
            )
        )
        output = await _search_toolset(client, num_results=2).web_search('q')
        assert output.metadata['sources'] == [
            {'url': 'https://a.dev', 'title': 'A'},
            {'url': 'https://b.dev', 'title': 'B'},
        ]


class TestGetPage:
    async def test_returns_markdown_and_requests_formats(self) -> None:
        client = _FakeYouClient(contents_response=_contents(markdown='hello page'))
        output = await _search_toolset(client).get_page('https://a.dev')
        assert client.contents_calls == [
            {'urls': ['https://a.dev'], 'formats': [models.ContentsFormats.MARKDOWN, models.ContentsFormats.METADATA]}
        ]
        assert _text(output) == 'Title: A\nURL: https://a.dev\n\nhello page'
        assert output.metadata == {'sources': [{'url': 'https://a.dev', 'title': 'A'}]}

    async def test_title_falls_back_to_site_name_and_url_to_arg(self) -> None:
        client = _FakeYouClient(contents_response=_contents(url=None, title=None, markdown='body', site_name='Example'))
        output = await _search_toolset(client).get_page('https://req.dev')
        assert _text(output) == 'Title: Example\nURL: https://req.dev\n\nbody'

    async def test_untitled_when_metadata_absent(self) -> None:
        client = _FakeYouClient(contents_response=_contents(url='https://a.dev', title=None, markdown='body'))
        output = await _search_toolset(client).get_page('https://a.dev')
        assert _text(output) == 'Title: (untitled)\nURL: https://a.dev\n\nbody'

    async def test_untitled_when_site_name_missing(self) -> None:
        response = models.ContentsResponse(
            url='https://a.dev', title=None, markdown='body', metadata=models.ContentsMetadata(site_name=None)
        )
        client = _FakeYouClient(contents_response=[response])
        output = await _search_toolset(client).get_page('https://a.dev')
        assert _text(output) == 'Title: (untitled)\nURL: https://a.dev\n\nbody'

    async def test_truncates_over_cap(self) -> None:
        client = _FakeYouClient(contents_response=_contents(markdown='y' * 500))
        output = await _search_toolset(client, max_text_chars=100).get_page('https://a.dev')
        marker = '\n[... page text truncated at 100 characters]'
        page_text = _text(output).split('\n\n')[-1]
        assert page_text.startswith('y') and page_text.endswith(marker) and len(page_text) == 100

    async def test_truncation_marker_only_when_cap_below_marker(self) -> None:
        # A cap smaller than the marker leaves no room for head text; the marker alone is returned.
        client = _FakeYouClient(contents_response=_contents(markdown='y' * 200))
        output = await _search_toolset(client, max_text_chars=5).get_page('https://a.dev')
        assert _text(output) == 'Title: A\nURL: https://a.dev\n\n\n[... page text truncated at 5 characters]'

    async def test_no_content_raises_model_retry(self) -> None:
        with pytest.raises(ModelRetry, match='No content could be retrieved'):
            await _search_toolset(_FakeYouClient(contents_response=[])).get_page('https://gone.dev')

    async def test_null_markdown_raises_model_retry(self) -> None:
        client = _FakeYouClient(contents_response=_contents(markdown=None))
        with pytest.raises(ModelRetry, match='No content could be retrieved'):
            await _search_toolset(client).get_page('https://a.dev')


class TestRecoverableErrors:
    async def test_transport_failure_becomes_model_retry(self) -> None:
        client = _FakeYouClient(error=httpx.ConnectError('connection refused'))
        with pytest.raises(ModelRetry, match='You.com request failed: connection refused'):
            await _search_toolset(client).web_search('q')

    async def test_rate_limit_becomes_model_retry(self) -> None:
        client = _FakeYouClient(error=_you_error(429))
        with pytest.raises(ModelRetry, match='You.com request failed'):
            await _search_toolset(client).web_search('q')

    @pytest.mark.parametrize('status', [401, 402, 403])
    async def test_auth_and_billing_propagate(self, status: int) -> None:
        client = _FakeYouClient(error=_you_error(status))
        with pytest.raises(YouError):
            await _search_toolset(client).web_search('q')

    async def test_you_error_without_status_becomes_model_retry(self) -> None:
        client = _FakeYouClient(error=_NoStatusYouError())
        with pytest.raises(ModelRetry, match='You.com request failed'):
            await _search_toolset(client).web_search('q')

    @pytest.mark.parametrize('status', [401, 402, 403])
    async def test_http_status_error_auth_propagates(self, status: int) -> None:
        client = _FakeYouClient(error=_http_status_error(status))
        with pytest.raises(httpx.HTTPStatusError):
            await _search_toolset(client).web_search('q')

    async def test_http_status_error_server_error_becomes_model_retry(self) -> None:
        client = _FakeYouClient(error=_http_status_error(500))
        with pytest.raises(ModelRetry, match='You.com request failed'):
            await _search_toolset(client).web_search('q')


class TestAnswer:
    async def test_answer_with_web_sources(self) -> None:
        client = _FakeYouClient(answer_response=_answer('An answer.', web=[('https://a.dev', 'A')]))
        output = await _research_toolset(client).answer('why?')
        assert _text(output) == 'An answer.\n\nSources:\n- A: https://a.dev'
        assert output.metadata == {'sources': [{'url': 'https://a.dev', 'title': 'A'}]}
        assert client.answer_calls[0]['query'] == 'why?'

    async def test_answer_falls_back_to_citation_urls(self) -> None:
        # `results` is present but its web list is empty, so sources come from citations.
        response = models.AnswerResponse(
            answer='A.', citations=[models.AnswerCitation(source='https://c.dev')], results=models.AnswerResults(web=[])
        )
        client = _FakeYouClient(answer_response=response)
        output = await _research_toolset(client).answer('q')
        assert _text(output) == 'A.\n\nSources:\n- (untitled): https://c.dev'

    async def test_answer_without_sources(self) -> None:
        client = _FakeYouClient(answer_response=_answer('Bare answer.'))
        output = await _research_toolset(client).answer('q')
        assert _text(output) == 'Bare answer.'
        assert output.metadata == {'sources': []}

    async def test_empty_answer_raises_model_retry(self) -> None:
        client = _FakeYouClient(answer_response=_answer(''))
        with pytest.raises(ModelRetry, match='Answer returned no content'):
            await _research_toolset(client).answer('q')


class TestResearch:
    async def test_text_answer_with_sources_and_warnings(self) -> None:
        client = _FakeYouClient(research_result=_research('Deep.', warnings=['heads up']))
        output = await _research_toolset(client, freshness='week', country='us').research('question')
        assert _text(output) == 'Warnings:\n- heads up\n\nDeep.\n\nSources:\n- S: https://s.dev'
        call = client.research_calls[0]
        assert call['background'] is False
        assert call['research_effort'] == models.ResearchEffort.STANDARD
        source_control = call['source_control']
        assert isinstance(source_control, models.SourceControl)
        assert source_control.freshness == 'week'
        assert source_control.country == 'us'

    async def test_structured_output_is_rendered_as_json(self) -> None:
        client = _FakeYouClient(research_result=_research({'finding': 'x'}, object_output=True, sources=[]))
        output = await _research_toolset(client, research_effort='deep', output_schema={'type': 'object'}).research('q')
        assert _text(output) == json.dumps({'finding': 'x'})
        assert client.research_calls[0]['output_schema'] == {'type': 'object'}

    async def test_no_source_control_when_unconfigured(self) -> None:
        client = _FakeYouClient(research_result=_research('Deep.'))
        await _research_toolset(client).research('q')
        assert client.research_calls[0]['source_control'] is None

    async def test_background_task_response_raises_model_retry(self) -> None:
        task = models.TaskResponse(
            task_id='t1',
            type='research',
            status=models.TaskResponseStatus.QUEUED,
            stream_url='https://x',
            created_at=datetime(2026, 1, 1),
        )
        client = _FakeYouClient(research_result=task)
        with pytest.raises(ModelRetry, match='did not return a synthesized answer'):
            await _research_toolset(client).research('q')

    async def test_empty_content_raises_model_retry(self) -> None:
        client = _FakeYouClient(research_result=_research('', sources=[]))
        with pytest.raises(ModelRetry, match='Research returned no answer'):
            await _research_toolset(client).research('q')


class TestFinanceResearch:
    async def test_finance_answer_with_sources(self) -> None:
        client = _FakeYouClient(finance_response=_finance('Revenue up.'))
        output = await _research_toolset(client, finance_effort='exhaustive').finance_research('ACME')
        assert _text(output) == 'Revenue up.\n\nSources:\n- F: https://f.dev'
        assert client.finance_calls[0]['research_effort'] == models.FinanceResearchEffort.EXHAUSTIVE

    async def test_empty_finance_answer_raises_model_retry(self) -> None:
        client = _FakeYouClient(finance_response=_finance(''))
        with pytest.raises(ModelRetry, match='Finance research returned no answer'):
            await _research_toolset(client).finance_research('q')


class TestYouSearchCapability:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('YDC_API_KEY', raising=False)
        monkeypatch.delenv('YOU_API_KEY_AUTH', raising=False)
        with pytest.raises(UserError, match='YDC_API_KEY'):
            YouSearch[None]().get_toolset()

    def test_default_client_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('YDC_API_KEY', 'test-key')
        toolset = YouSearch[None]().get_toolset()
        assert isinstance(toolset, YouSearchToolset)

    def test_default_client_accepts_legacy_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('YDC_API_KEY', raising=False)
        monkeypatch.setenv('YOU_API_KEY_AUTH', 'legacy-key')
        toolset = YouSearch[None]().get_toolset()
        assert isinstance(toolset, YouSearchToolset)

    def test_default_client_sets_attribution_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_client() passes app_name/app_version/app_title/app_url to You()."""
        monkeypatch.setenv('YDC_API_KEY', 'test-key')
        with patch('pydantic_ai_harness.youdotcom._toolset.You') as you_cls:
            default_client(timeout_ms=60_000)
        kwargs = you_cls.call_args.kwargs
        assert kwargs['app_name'] == 'pydantic-ai-harness'
        assert kwargs['app_title'] == 'Pydantic AI Harness'
        assert kwargs['app_url'] == 'https://github.com/pydantic/pydantic-ai-harness'
        assert isinstance(kwargs['app_version'], str) and kwargs['app_version']
        assert kwargs['api_key_auth'] is None
        assert kwargs['timeout_ms'] == 60_000

    @pytest.mark.parametrize('num_results', [0, 21])
    def test_num_results_out_of_bounds_rejected(self, num_results: int) -> None:
        with pytest.raises(ValueError, match='num_results must be between 1 and 20'):
            YouSearch[None](num_results=num_results)

    def test_max_text_chars_floor(self) -> None:
        with pytest.raises(ValueError, match='max_text_chars must be at least 1'):
            YouSearch[None](max_text_chars=0)

    @pytest.mark.parametrize(('exclude', 'boost'), [(['b.dev'], []), ([], ['b.dev'])])
    def test_include_domains_conflicts(self, exclude: list[str], boost: list[str]) -> None:
        with pytest.raises(ValueError, match='include_domains cannot be combined'):
            YouSearch[None](include_domains=['a.dev'], exclude_domains=exclude, boost_domains=boost)

    def test_invalid_extraction_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="extraction_mode must be 'highlights' or 'full_page'"):
            YouSearch[None](extraction_mode='full-page')  # pyright: ignore[reportArgumentType]

    def test_invalid_freshness_rejected(self) -> None:
        with pytest.raises(ValueError, match='freshness must be one of'):
            YouSearch[None](freshness='fortnight')

    @pytest.mark.parametrize(
        'freshness',
        [
            '2026-99-99to2026-02-31',  # impossible month/day
            '2026-02-01to2026-01-01',  # reversed range
            '2026-01-01to2026-02-01\n',  # trailing newline
            '２０２６-０１-０１to２０２６-０２-０１',  # full-width (non-ASCII) digits
            '2026-01-01to2026-02-01to2026-03-01',  # trailing characters
        ],
    )
    def test_malformed_freshness_range_rejected(self, freshness: str) -> None:
        with pytest.raises(ValueError, match='freshness must be one of'):
            YouSearch[None](freshness=freshness)

    def test_valid_freshness_range_and_keyword(self) -> None:
        assert YouSearch[None](freshness='2026-01-01to2026-02-01').freshness == '2026-01-01to2026-02-01'
        assert YouSearch[None](freshness='2026-01-01to2026-01-01').freshness == '2026-01-01to2026-01-01'
        assert YouSearch[None](freshness='month').freshness == 'month'

    def test_instructions_default_custom_and_empty(self) -> None:
        default = YouSearch[None]().get_instructions()
        assert isinstance(default, str) and 'web_search' in default and 'get_page' in default
        assert YouSearch[None](guidance='Use the You.com tools.').get_instructions() == 'Use the You.com tools.'
        assert YouSearch[None](guidance='').get_instructions() is None


class TestYouResearchCapability:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('YDC_API_KEY', raising=False)
        monkeypatch.delenv('YOU_API_KEY_AUTH', raising=False)
        with pytest.raises(UserError, match='YDC_API_KEY'):
            YouResearch[None]().get_toolset()

    def test_default_client_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('YDC_API_KEY', 'test-key')
        toolset = YouResearch[None]().get_toolset()
        assert isinstance(toolset, YouResearchToolset)

    def test_invalid_research_effort_rejected(self) -> None:
        with pytest.raises(ValueError, match='research_effort must be one of'):
            YouResearch[None](research_effort='frontier')  # pyright: ignore[reportArgumentType]

    def test_invalid_finance_effort_rejected(self) -> None:
        with pytest.raises(ValueError, match='finance_effort must be one of'):
            YouResearch[None](finance_effort='lite')  # pyright: ignore[reportArgumentType]

    def test_include_domains_conflict(self) -> None:
        with pytest.raises(ValueError, match='include_domains cannot be combined'):
            YouResearch[None](include_domains=['a.dev'], exclude_domains=['b.dev'])

    def test_output_schema_rejected_with_lite(self) -> None:
        with pytest.raises(ValueError, match="output_schema is not supported with research_effort='lite'"):
            YouResearch[None](research_effort='lite', output_schema={'type': 'object'})

    def test_invalid_freshness_rejected(self) -> None:
        with pytest.raises(ValueError, match='freshness must be one of'):
            YouResearch[None](freshness='sometime')

    def test_instructions_default_custom_and_empty(self) -> None:
        default = YouResearch[None]().get_instructions()
        assert isinstance(default, str) and 'answer' in default and 'finance_research' in default
        assert YouResearch[None](guidance='Research with You.com.').get_instructions() == 'Research with You.com.'
        assert YouResearch[None](guidance='').get_instructions() is None


class TestAgentRun:
    async def test_agent_uses_all_tools_and_instructions(self) -> None:
        client = _FakeYouClient(
            search_response=_search(_web('https://a.dev', title='A', highlights=['alpha'])),
            contents_response=_contents(url='https://b.dev', title='B', markdown='beta'),
            answer_response=_answer('Answer.', web=[('https://c.dev', 'C')]),
            research_result=_research('Research.', sources=[('https://d.dev', 'D')]),
            finance_response=_finance('Finance.', sources=[('https://e.dev', 'E')]),
        )
        agent = Agent(TestModel(), capabilities=[YouSearch(client=client), YouResearch(client=client)])

        result = await agent.run('Go.')
        messages = result.all_messages()
        first = messages[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None
        assert 'web_search' in first.instructions
        assert 'finance_research' in first.instructions

        calls = {
            part.tool_name: part.args_as_dict()
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        returns = {
            part.tool_name: part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        metadata = {
            part.tool_name: part.metadata
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert set(returns) == {'web_search', 'get_page', 'answer', 'research', 'finance_research'}
        query = calls['web_search']['query']
        assert returns['web_search'] == f'Found 1 result for {query!r}:\n\nTitle: A\nURL: https://a.dev\n\n- alpha'
        assert returns['get_page'] == 'Title: B\nURL: https://b.dev\n\nbeta'
        assert returns['answer'] == 'Answer.\n\nSources:\n- C: https://c.dev'
        assert returns['research'] == 'Research.\n\nSources:\n- D: https://d.dev'
        assert returns['finance_research'] == 'Finance.\n\nSources:\n- E: https://e.dev'
        assert metadata['web_search'] == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}],
            'search_uuid': None,
            'latency': None,
        }
        assert metadata['get_page'] == {'sources': [{'url': 'https://b.dev', 'title': 'B'}]}
        assert metadata['answer'] == {'sources': [{'url': 'https://c.dev', 'title': 'C'}]}


class TestAgentSpec:
    def test_spec_schema_includes_capabilities(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([YouSearch, YouResearch])
        dumped = json.dumps(schema)
        assert 'YouSearch' in dumped and 'YouResearch' in dumped

    def test_search_from_spec_builds_capability(self) -> None:
        capability = YouSearch[None].from_spec(num_results=3, extraction_mode='full_page', include_domains=['a.dev'])
        assert capability.num_results == 3
        assert capability.extraction_mode == 'full_page'
        assert capability.include_domains == ['a.dev']
        assert capability.client is None

    def test_research_from_spec_builds_capability(self) -> None:
        capability = YouResearch[None].from_spec(research_effort='deep', boost_domains=['a.dev'])
        assert capability.research_effort == 'deep'
        assert capability.boost_domains == ['a.dev']
        assert capability.client is None

    def test_agent_loads_from_spec_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('YDC_API_KEY', 'test-key')
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - YouSearch:\n      num_results: 3\n')
        agent = Agent.from_file(spec, custom_capability_types=[YouSearch, YouResearch])
        assert isinstance(agent, Agent)
