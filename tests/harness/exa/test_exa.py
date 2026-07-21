"""Tests for the ExaSearch capability and ExaSearchToolset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import pytest
from exa_py.api import (
    ContentsOptions,
    DeepOutputSchema,
    DeepSearchOutput,
    DeepSearchOutputGrounding,
    DeepSearchOutputGroundingCitation,
    Result,
    SearchResponse,
    SearchType,
    TextContentsOptions,
)
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.exa import ExaSearch, ExaSearchToolset


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _text(output: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = output.return_value
    assert isinstance(body, str)
    return body


def _result(
    url: str = 'https://example.dev/page',
    *,
    title: str | None = 'Example page',
    text: str | None = None,
    highlights: list[str] | None = None,
    author: str | None = None,
    published_date: str | None = None,
) -> Result:
    """A real `exa_py` result with the fields the toolset reads."""
    return Result(
        url=url,
        id=url,
        title=title,
        text=text,
        highlights=highlights,
        author=author,
        published_date=published_date,
    )


def _response(*results: Result) -> SearchResponse[Result]:
    """A real `exa_py` response wrapping the given results."""
    return SearchResponse(results=list(results), resolved_search_type=None, auto_date=None)


def _deep_response(
    content: str | dict[str, object],
    *,
    citations: Sequence[tuple[str, str]] = (),
    results: Sequence[Result] = (),
) -> SearchResponse[Result]:
    """A real `exa_py` deep-search response with a synthesized output."""
    grounding = (
        [
            DeepSearchOutputGrounding(
                field='answer',
                citations=[DeepSearchOutputGroundingCitation(url=url, title=title) for url, title in citations],
                confidence='high',
            )
        ]
        if citations
        else []
    )
    return SearchResponse(
        results=list(results),
        resolved_search_type=None,
        auto_date=None,
        output=DeepSearchOutput(content=content, grounding=grounding),
    )


@dataclass
class _FakeExaClient:
    """In-memory `ExaClient` double: canned responses, recorded call arguments."""

    search_response: SearchResponse[Result] = field(default_factory=_response)
    contents_response: SearchResponse[Result] = field(default_factory=_response)
    deep_response: SearchResponse[Result] = field(default_factory=_response)
    error: Exception | None = None
    search_calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    get_contents_calls: list[tuple[str, TextContentsOptions]] = field(
        default_factory=list[tuple[str, TextContentsOptions]]
    )

    async def search(
        self,
        query: str,
        *,
        contents: ContentsOptions | Literal[False],
        num_results: int | None = None,
        type: SearchType | None = None,
        output_schema: DeepOutputSchema | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse[Result]:
        if self.error is not None:
            raise self.error
        self.search_calls.append(
            {
                'query': query,
                'contents': contents,
                'num_results': num_results,
                'type': type,
                'output_schema': output_schema,
                'include_domains': include_domains,
                'exclude_domains': exclude_domains,
            }
        )
        return self.deep_response if type is not None else self.search_response

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        if self.error is not None:
            raise self.error
        self.get_contents_calls.append((urls, text))
        return self.contents_response


def _toolset(
    client: _FakeExaClient,
    *,
    num_results: int = 5,
    max_text_chars: int = 10_000,
    text_summary: bool | str = False,
    include_deep_search: bool = False,
    include_domains: Sequence[str] = (),
    exclude_domains: Sequence[str] = (),
) -> ExaSearchToolset[None]:
    return ExaSearch[None](
        num_results=num_results,
        max_text_chars=max_text_chars,
        text_summary=text_summary,
        include_deep_search=include_deep_search,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        client=client,
    ).get_toolset()


class TestWebSearch:
    async def test_formats_results_with_excerpts_and_metadata(self) -> None:
        client = _FakeExaClient(
            search_response=_response(
                _result(
                    'https://a.dev',
                    title='A',
                    highlights=['alpha excerpt', 'another excerpt'],
                    author='Ada',
                    published_date='2026-07-01',
                ),
                _result('https://b.dev', title=None),
            )
        )
        output = await _toolset(client).web_search('rust web frameworks')
        assert _text(output) == (
            "Found 2 results for 'rust web frameworks':\n\n"
            'Title: A\n'
            'URL: https://a.dev\n'
            'Published: 2026-07-01\n'
            'Author: Ada\n'
            '\n'
            '- alpha excerpt\n'
            '- another excerpt'
            '\n\n---\n\n'
            'Title: (untitled)\n'
            'URL: https://b.dev'
        )
        assert output.metadata == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': None}]
        }

    async def test_requests_highlights_num_results_and_domains(self) -> None:
        client = _FakeExaClient(search_response=_response(_result()))
        toolset = _toolset(client, num_results=3, include_domains=['a.dev', 'b.dev'])
        await toolset.web_search('q')
        assert client.search_calls == [
            {
                'query': 'q',
                'contents': {'highlights': True},
                'num_results': 3,
                'type': None,
                'output_schema': None,
                'include_domains': ['a.dev', 'b.dev'],
                'exclude_domains': None,
            }
        ]

    async def test_exclude_domains_plumbed(self) -> None:
        client = _FakeExaClient(search_response=_response(_result()))
        await _toolset(client, exclude_domains=['spam.dev']).web_search('q')
        assert client.search_calls[0]['include_domains'] is None
        assert client.search_calls[0]['exclude_domains'] == ['spam.dev']

    async def test_no_results(self) -> None:
        client = _FakeExaClient()
        output = await _toolset(client).web_search('nothing to see')
        assert _text(output) == "No results found for 'nothing to see'."
        assert output.metadata == {'sources': []}

    async def test_num_results_enforced_on_oversized_response(self) -> None:
        client = _FakeExaClient(
            search_response=_response(
                _result('https://a.dev', title='A'),
                _result('https://b.dev', title='B'),
                _result('https://c.dev', title='C'),
            )
        )
        output = await _toolset(client, num_results=2).web_search('q')
        assert _text(output) == (
            "Found 2 results for 'q':\n\nTitle: A\nURL: https://a.dev\n\n---\n\nTitle: B\nURL: https://b.dev"
        )
        assert output.metadata == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': 'B'}]
        }


class TestGetPage:
    async def test_returns_page_text_with_headroom_request(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result('https://a.dev', title='A', text='alpha text')))
        output = await _toolset(client, max_text_chars=100).get_page('https://a.dev')
        assert client.get_contents_calls == [('https://a.dev', {'max_characters': 101})]
        assert _text(output) == 'Title: A\nURL: https://a.dev\n\nalpha text'
        assert output.metadata == {'sources': [{'url': 'https://a.dev', 'title': 'A'}]}

    async def test_headroom_request_stops_at_api_ceiling(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result(text='alpha')))
        await _toolset(client, max_text_chars=10_000).get_page('https://a.dev')
        assert client.get_contents_calls == [('https://a.dev', {'max_characters': 10_000})]

    async def test_text_at_cap_is_not_truncated(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result(title='T', text='x' * 10)))
        output = await _toolset(client, max_text_chars=10).get_page('https://example.dev/page')
        assert _text(output) == f'Title: T\nURL: https://example.dev/page\n\n{"x" * 10}'

    async def test_text_over_cap_is_truncated_keeping_the_head(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result(title='T', text='x' * 10 + 'T')))
        output = await _toolset(client, max_text_chars=10).get_page('https://example.dev/page')
        assert _text(output) == (
            f'Title: T\nURL: https://example.dev/page\n\n{"x" * 10}\n[... page text truncated at 10 characters]'
        )

    async def test_no_results_raises_model_retry(self) -> None:
        client = _FakeExaClient()
        with pytest.raises(ModelRetry, match='No content could be retrieved'):
            await _toolset(client).get_page('https://gone.dev')

    async def test_result_without_text_raises_model_retry(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result('https://a.dev', title='A', text=None)))
        with pytest.raises(ModelRetry, match='No content could be retrieved'):
            await _toolset(client).get_page('https://a.dev')


class TestRecoverableErrors:
    async def test_non_2xx_becomes_model_retry(self) -> None:
        client = _FakeExaClient(error=ValueError('Request failed with status code 429: rate limited'))
        with pytest.raises(ModelRetry, match='Exa request failed: .*429'):
            await _toolset(client).web_search('q')

    async def test_auth_failure_propagates(self) -> None:
        client = _FakeExaClient(error=ValueError('Request failed with status code 401: invalid API key'))
        with pytest.raises(ValueError, match='status code 401'):
            await _toolset(client).web_search('q')

    async def test_network_failure_becomes_model_retry(self) -> None:
        client = _FakeExaClient(error=httpx.ConnectError('connection refused'))
        with pytest.raises(ModelRetry, match='Exa request failed: connection refused'):
            await _toolset(client).get_page('https://a.dev')


class TestDeepSearch:
    async def test_returns_answer_with_deduplicated_sources(self) -> None:
        client = _FakeExaClient(
            deep_response=_deep_response(
                'Deep answer.',
                citations=[('https://a.dev', 'A'), ('https://b.dev', ''), ('https://a.dev', 'A')],
            )
        )
        output = await _toolset(client, include_deep_search=True).deep_search('why is the sky blue?')
        assert client.search_calls == [
            {
                'query': 'why is the sky blue?',
                'contents': False,
                'num_results': None,
                'type': 'deep',
                'output_schema': {'type': 'text'},
                'include_domains': None,
                'exclude_domains': None,
            }
        ]
        assert _text(output) == 'Deep answer.\n\nSources:\n- A: https://a.dev\n- (untitled): https://b.dev'
        assert output.metadata == {
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': ''}]
        }

    async def test_domains_plumbed(self) -> None:
        client = _FakeExaClient(deep_response=_deep_response('Answer.'))
        toolset = _toolset(client, include_deep_search=True, include_domains=['a.dev'])
        await toolset.deep_search('q')
        assert client.search_calls[0]['include_domains'] == ['a.dev']

    async def test_structured_content_is_rendered_as_json(self) -> None:
        client = _FakeExaClient(deep_response=_deep_response({'finding': 'x'}, citations=[('https://a.dev', 'A')]))
        output = await _toolset(client, include_deep_search=True).deep_search('q')
        assert _text(output) == '{"finding": "x"}\n\nSources:\n- A: https://a.dev'

    async def test_without_grounding_falls_back_to_result_sources(self) -> None:
        client = _FakeExaClient(deep_response=_deep_response('Answer.', results=[_result('https://a.dev', title='A')]))
        output = await _toolset(client, include_deep_search=True).deep_search('q')
        assert _text(output) == 'Answer.\n\nSources:\n- A: https://a.dev'
        assert output.metadata == {'sources': [{'url': 'https://a.dev', 'title': 'A'}]}

    async def test_without_any_sources_returns_answer_only(self) -> None:
        client = _FakeExaClient(deep_response=_deep_response('Answer.'))
        output = await _toolset(client, include_deep_search=True).deep_search('q')
        assert _text(output) == 'Answer.'
        assert output.metadata == {'sources': []}

    async def test_answer_is_not_capped_by_max_text_chars(self) -> None:
        long_answer = 'a' * 50
        client = _FakeExaClient(deep_response=_deep_response(long_answer))
        output = await _toolset(client, include_deep_search=True, max_text_chars=10).deep_search('q')
        assert _text(output) == long_answer

    async def test_missing_output_raises_model_retry(self) -> None:
        client = _FakeExaClient(deep_response=_response())
        with pytest.raises(ModelRetry, match='Deep search returned no answer'):
            await _toolset(client, include_deep_search=True).deep_search('q')

    async def test_empty_content_raises_model_retry(self) -> None:
        client = _FakeExaClient(deep_response=_deep_response(''))
        with pytest.raises(ModelRetry, match='Deep search returned no answer'):
            await _toolset(client, include_deep_search=True).deep_search('q')

    def test_tool_absent_by_default_and_present_when_enabled(self) -> None:
        client = _FakeExaClient()
        assert list(_toolset(client).tools) == ['web_search', 'get_page']
        assert list(_toolset(client, include_deep_search=True).tools) == ['web_search', 'get_page', 'deep_search']


class TestTextSummary:
    async def test_summary_prepended_to_results(self) -> None:
        client = _FakeExaClient(
            search_response=_deep_response('Rust is fast.', results=[_result('https://a.dev', title='A')])
        )
        output = await _toolset(client, text_summary=True).web_search('is rust fast?')
        assert client.search_calls[0]['output_schema'] == {'type': 'text'}
        assert _text(output) == (
            "Summary: Rust is fast.\n\nFound 1 result for 'is rust fast?':\n\nTitle: A\nURL: https://a.dev"
        )

    async def test_summary_format_hint_sent_as_description(self) -> None:
        client = _FakeExaClient(search_response=_response(_result()))
        await _toolset(client, text_summary='One sentence with the year.').web_search('q')
        assert client.search_calls[0]['output_schema'] == {
            'type': 'text',
            'description': 'One sentence with the year.',
        }

    async def test_off_by_default(self) -> None:
        client = _FakeExaClient(search_response=_response(_result()))
        await _toolset(client).web_search('q')
        assert client.search_calls[0]['output_schema'] is None

    async def test_missing_output_returns_results_only(self) -> None:
        client = _FakeExaClient(search_response=_response(_result('https://a.dev', title='A')))
        output = await _toolset(client, text_summary=True).web_search('q')
        assert _text(output) == "Found 1 result for 'q':\n\nTitle: A\nURL: https://a.dev"

    async def test_non_text_output_returns_results_only(self) -> None:
        client = _FakeExaClient(search_response=_deep_response({'not': 'text'}, results=[_result(title='A')]))
        output = await _toolset(client, text_summary=True).web_search('q')
        assert _text(output).startswith("Found 1 result for 'q':")

    async def test_no_results_short_circuits_before_summary(self) -> None:
        client = _FakeExaClient(search_response=_deep_response('Summary without results.'))
        output = await _toolset(client, text_summary=True).web_search('q')
        assert _text(output) == "No results found for 'q'."


class TestExaSearch:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('EXA_API_KEY', raising=False)
        with pytest.raises(UserError, match='EXA_API_KEY'):
            ExaSearch[None]().get_toolset()

    def test_default_client_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('EXA_API_KEY', 'test-key')
        toolset = ExaSearch[None]().get_toolset()
        assert isinstance(toolset, ExaSearchToolset)

    @pytest.mark.parametrize('num_results', [0, 101])
    def test_num_results_out_of_bounds_rejected(self, num_results: int) -> None:
        with pytest.raises(ValueError, match=f'num_results must be between 1 and 100, got {num_results}'):
            ExaSearch[None](num_results=num_results)

    @pytest.mark.parametrize('max_text_chars', [0, 10_001])
    def test_max_text_chars_out_of_bounds_rejected(self, max_text_chars: int) -> None:
        with pytest.raises(ValueError, match=f'max_text_chars must be between 1 and 10000, got {max_text_chars}'):
            ExaSearch[None](max_text_chars=max_text_chars)

    def test_include_and_exclude_domains_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match='include_domains or exclude_domains, not both'):
            ExaSearch[None](include_domains=['a.dev'], exclude_domains=['b.dev'])

    def test_instructions_reference_the_tools(self) -> None:
        instructions = ExaSearch[None]().get_instructions()
        assert isinstance(instructions, str)
        assert 'web_search' in instructions
        assert 'get_page' in instructions
        assert 'cite' in instructions
        assert 'deep_search' not in instructions

    def test_instructions_cover_deep_search_when_enabled(self) -> None:
        base = ExaSearch[None]().get_instructions()
        instructions = ExaSearch[None](include_deep_search=True).get_instructions()
        assert isinstance(base, str) and isinstance(instructions, str)
        assert instructions.startswith(base)
        assert 'escalate to `deep_search`' in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        capability = ExaSearch[None](include_deep_search=True, guidance='Research with the Exa tools.')
        assert capability.get_instructions() == 'Research with the Exa tools.'

    def test_empty_guidance_disables_instructions(self) -> None:
        assert ExaSearch[None](guidance='').get_instructions() is None

    async def test_agent_run_uses_all_tools_and_instructions(self) -> None:
        client = _FakeExaClient(
            search_response=_response(_result('https://a.dev', title='A', highlights=['alpha'])),
            contents_response=_response(_result('https://b.dev', title='B', text='beta')),
            deep_response=_deep_response('Deep answer.', citations=[('https://c.dev', 'C')]),
        )
        agent = Agent(TestModel(), capabilities=[ExaSearch(include_deep_search=True, client=client)])

        result = await agent.run('Research something.')

        messages = result.all_messages()
        first = messages[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None
        assert 'web_search' in first.instructions
        assert 'deep_search' in first.instructions

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
        assert set(returns) == {'web_search', 'get_page', 'deep_search'}
        query = calls['web_search']['query']
        assert returns['web_search'] == f'Found 1 result for {query!r}:\n\nTitle: A\nURL: https://a.dev\n\n- alpha'
        assert returns['get_page'] == 'Title: B\nURL: https://b.dev\n\nbeta'
        assert returns['deep_search'] == 'Deep answer.\n\nSources:\n- C: https://c.dev'
        assert metadata['web_search'] == {'sources': [{'url': 'https://a.dev', 'title': 'A'}]}
        assert metadata['get_page'] == {'sources': [{'url': 'https://b.dev', 'title': 'B'}]}
        assert metadata['deep_search'] == {'sources': [{'url': 'https://c.dev', 'title': 'C'}]}


class TestAgentSpec:
    def test_spec_schema_includes_exa_search(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ExaSearch])
        assert 'ExaSearch' in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        capability = ExaSearch[None].from_spec(
            num_results=3,
            text_summary=True,
            include_deep_search=True,
            include_domains=['a.dev'],
        )
        assert capability.num_results == 3
        assert capability.text_summary is True
        assert capability.include_deep_search is True
        assert capability.include_domains == ['a.dev']
        assert capability.client is None

    def test_agent_loads_from_spec_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('EXA_API_KEY', 'test-key')
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - ExaSearch:\n      num_results: 3\n')
        agent = Agent.from_file(spec, custom_capability_types=[ExaSearch])
        assert isinstance(agent, Agent)
