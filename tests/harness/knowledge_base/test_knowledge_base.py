"""Tests for the KnowledgeBase capability, its toolset, and the in-memory backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pytest

from pydantic_ai import Agent, DeferredToolRequests, ToolDefinition
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
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
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import FileSearchTool
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import (
    Document,
    FilterNode,
    InMemoryKnowledgeBase,
    KnowledgeBaseToolset,
    MetadataFilter,
    Passage,
    SearchMode,
    allows_read,
    matches,
    normalize_filter,
)

_DOCUMENTS = [
    Document(
        'returns',
        'Returns: items can be returned within 30 days.\n\nRefunds go to the original payment method.',
        source='https://shop.example/returns',
        metadata={'tenant_id': 'acme', 'section': 'policies'},
    ),
    Document(
        'shipping',
        'Orders ship within two business days.',
        source='https://shop.example/shipping',
        metadata={'tenant_id': 'acme', 'section': 'faq'},
    ),
    Document(
        'other-returns',
        'Returns are not accepted.',
        source='https://other.example/returns',
        metadata={'tenant_id': 'globex', 'section': 'policies'},
    ),
]


def _text(output: ToolReturn[str]) -> str:
    body = output.return_value
    assert isinstance(body, str)
    return body


def _ctx(deps: object = None) -> RunContext[object]:
    return RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), tool_call_id='call-1', run_id='run-1')


@dataclass
class _SearchOnly:
    """A backend that only searches: canned passages, recorded calls."""

    passages: Sequence[Passage] = ()
    error: Exception | None = None
    calls: list[tuple[str, int, FilterNode | None]] = field(default_factory=list[tuple[str, int, 'FilterNode | None']])
    options: list[tuple[SearchMode | None, float | None]] = field(
        default_factory=list[tuple['SearchMode | None', 'float | None']]
    )

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        if self.error is not None:
            raise self.error
        self.calls.append((query, limit, filter))
        self.options.append((mode, score_threshold))
        return self.passages


@dataclass
class _NativeCapable(_SearchOnly):
    """A search-only backend the run's provider could also search natively."""

    def file_search_tool(self, ctx: RunContext[object], *, limit: int) -> FileSearchTool | None:
        return FileSearchTool(file_store_ids=['store-1'], max_num_results=limit)


def _toolset(backend: object, **options: object) -> KnowledgeBaseToolset[object]:
    return KnowledgeBase[object](backend, **options).get_toolset()  # pyright: ignore[reportArgumentType]


class TestInMemoryKnowledgeBase:
    async def test_ranks_chunks_and_keeps_document_identity(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        passages = await backend.search('returned within days', limit=5, filter=None)
        assert [passage.document_id for passage in passages] == ['returns', 'shipping']
        assert passages[0] == Passage(
            text='Returns: items can be returned within 30 days.\n\nRefunds go to the original payment method.',
            source='https://shop.example/returns',
            score=passages[0].score,
            document_id='returns',
            metadata={'tenant_id': 'acme', 'section': 'policies'},
        )
        assert passages[0].score is not None and passages[0].score > (passages[1].score or 0)

    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            ({'tenant_id': 'acme'}, ['returns']),
            ({'tenant_id': 'globex'}, ['other-returns']),
            ({'section': ['faq', 'policies'], 'tenant_id': 'acme'}, ['returns']),
            ({'section': ['faq']}, []),
            ({'missing': 'x'}, []),
            ({}, ['other-returns', 'returns']),
        ],
    )
    async def test_filter_keeps_matching_documents(self, filter: MetadataFilter, expected: list[str]) -> None:
        passages = await InMemoryKnowledgeBase(_DOCUMENTS).search('returns', limit=5, filter=normalize_filter(filter))
        assert [passage.document_id for passage in passages] == expected

    async def test_limit_and_no_match(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        assert len(await backend.search('returns', limit=1, filter=None)) == 1
        assert await backend.search('zeppelin', limit=5, filter=None) == []
        assert await backend.search('!!!', limit=5, filter=None) == []
        assert await InMemoryKnowledgeBase().search('returns', limit=5, filter=None) == []

    async def test_long_documents_are_split_on_paragraphs(self) -> None:
        text = 'alpha first.\n\nbeta second.\n\n\n\ngamma third paragraph is longer than the limit.'
        backend = InMemoryKnowledgeBase([Document('d', text)], chunk_chars=30)
        passages = await backend.search('alpha beta gamma', limit=5, filter=None)
        assert sorted(passage.text for passage in passages) == [
            'alpha first.\n\nbeta second.',
            'gamma third paragraph is longer than the limit.',
        ]

    async def test_add_replaces_by_id_and_read_returns_the_document(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        backend.add([Document('shipping', 'Orders ship the same day.')])
        assert await backend.read('shipping', max_chars=100, filter=None) == Document(
            'shipping', 'Orders ship the same day.'
        )
        assert await backend.read('nope', max_chars=100, filter=None) is None

    async def test_read_applies_the_filter(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        acme = normalize_filter({'tenant_id': 'acme'})
        assert await backend.read('returns', max_chars=100, filter=acme) is not None
        assert await backend.read('other-returns', max_chars=100, filter=acme) is None

    async def test_blank_documents_and_paragraphs_add_no_chunks(self) -> None:
        backend = InMemoryKnowledgeBase([Document('blank', ''), Document('spaced', '\n\n  \n\nalpha')])
        passages = await backend.search('alpha', limit=5, filter=None)
        assert [(passage.document_id, passage.text) for passage in passages] == [('spaced', 'alpha')]

    async def test_keyword_is_the_only_mode(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        assert len(await backend.search('returns', limit=5, filter=None, mode='keyword')) == 2
        with pytest.raises(UserError, match="only supports keyword search, not 'semantic'"):
            await backend.search('returns', limit=5, filter=None, mode='semantic')

    async def test_score_threshold_drops_weak_chunks(self) -> None:
        backend = InMemoryKnowledgeBase(_DOCUMENTS)
        scores = [passage.score or 0 for passage in await backend.search('returns', limit=5, filter=None)]
        kept = await backend.search('returns', limit=5, filter=None, score_threshold=max(scores))
        assert [passage.score for passage in kept] == [max(scores)]

    def test_chunk_chars_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='chunk_chars must be positive'):
            InMemoryKnowledgeBase(chunk_chars=0)


class TestSearchKnowledgeBase:
    async def test_renders_passages_with_sources_documents_and_metadata(self) -> None:
        toolset = _toolset(InMemoryKnowledgeBase(_DOCUMENTS), number_of_results=1)
        output = await toolset.search_knowledge_base(_ctx(), 'returned within days')
        assert _text(output) == (
            "Found 1 result for 'returned within days':\n\n"
            f'Source: https://shop.example/returns (score {output.metadata["results"][0]["score"]:.2f})\n'
            'Document: returns\n'
            'Returns: items can be returned within 30 days.\n\nRefunds go to the original payment method.'
        )
        assert output.metadata['results'] == [
            {
                'source': 'https://shop.example/returns',
                'score': output.metadata['results'][0]['score'],
                'document_id': 'returns',
                'metadata': {'tenant_id': 'acme', 'section': 'policies'},
            }
        ]

    async def test_search_only_backend_shows_no_document_ids(self) -> None:
        backend = _SearchOnly(
            passages=[
                Passage('alpha', source='s3://bucket/a.md\nSource: s3://evil', score=0.5, document_id='a'),
                Passage('beta'),
            ]
        )
        output = await _toolset(backend).search_knowledge_base(_ctx(), 'q')
        assert _text(output) == (
            "Found 2 results for 'q':\n\n"
            'Source: s3://bucket/a.md Source: s3://evil (score 0.50)\nalpha'
            '\n\n---\n\n'
            'Source: (unknown)\nbeta'
        )
        assert backend.calls == [('q', 5, None)]

    async def test_show_sources_off_keeps_provenance_in_metadata_only(self) -> None:
        toolset = _toolset(InMemoryKnowledgeBase(_DOCUMENTS), number_of_results=1, show_sources=False)
        output = await toolset.search_knowledge_base(_ctx(), 'shipping business days')
        assert _text(output) == (
            "Found 1 result for 'shipping business days':\n\nDocument: shipping\nOrders ship within two business days."
        )
        assert output.metadata['results'][0]['source'] == 'https://shop.example/shipping'

    async def test_no_results(self) -> None:
        output = await _toolset(_SearchOnly()).search_knowledge_base(_ctx(), 'nothing')
        assert _text(output) == "No results found for 'nothing'."
        assert output.metadata == {'results': []}

    async def test_number_of_results_enforced_on_oversized_backend_response(self) -> None:
        backend = _SearchOnly(passages=[Passage('a'), Passage('b'), Passage('c')])
        output = await _toolset(backend, number_of_results=2).search_knowledge_base(_ctx(), 'q')
        assert len(output.metadata['results']) == 2
        assert backend.calls[0][1] == 2

    async def test_static_filter_is_passed_to_the_backend(self) -> None:
        backend = _SearchOnly()
        await _toolset(backend, filter={'tenant_id': 'acme'}).search_knowledge_base(_ctx(), 'q')
        assert backend.calls == [('q', 5, {'field': 'tenant_id', 'operator': '==', 'value': 'acme'})]

    async def test_filter_callable_is_evaluated_per_run(self) -> None:
        backend = _SearchOnly()

        def tenant(ctx: RunContext[object]) -> MetadataFilter:
            return {'tenant_id': str(ctx.deps)}

        await _toolset(backend, filter=tenant).search_knowledge_base(_ctx('globex'), 'q')
        assert backend.calls[0][2] == {'field': 'tenant_id', 'operator': '==', 'value': 'globex'}

    async def test_async_filter_callable_is_awaited(self) -> None:
        backend = _SearchOnly()

        async def tenant(ctx: RunContext[object]) -> MetadataFilter:
            return {'tenant_id': 'acme'}

        await _toolset(backend, filter=tenant).search_knowledge_base(_ctx(), 'q')
        assert backend.calls[0][2] == {'field': 'tenant_id', 'operator': '==', 'value': 'acme'}

    async def test_search_mode_and_threshold_are_passed_to_the_backend(self) -> None:
        backend = _SearchOnly()
        await _toolset(backend, search_mode='hybrid', score_threshold=0.4).search_knowledge_base(_ctx(), 'q')
        assert backend.options == [('hybrid', 0.4)]

    async def test_tree_filter_is_passed_through_checked(self) -> None:
        backend = _SearchOnly()
        tree: MetadataFilter = {
            'operator': 'OR',
            'conditions': [
                {'field': 'year', 'operator': '>=', 'value': 2024},
                {'field': 'source', 'operator': 'not in', 'value': ('drafts',)},
            ],
        }
        await _toolset(backend, filter=tree).search_knowledge_base(_ctx(), 'q')
        assert backend.calls[0][2] == {
            'operator': 'OR',
            'conditions': [
                {'field': 'year', 'operator': '>=', 'value': 2024},
                {'field': 'source', 'operator': 'not in', 'value': ['drafts']},
            ],
        }

    async def test_backend_retry_is_passed_through(self) -> None:
        backend = _SearchOnly(error=ModelRetry('throttled'))
        with pytest.raises(ModelRetry, match='throttled'):
            await _toolset(backend).search_knowledge_base(_ctx(), 'q')

    async def test_tool_description_mentions_what_the_knowledge_base_holds(self) -> None:
        plain = await _toolset(_SearchOnly()).get_tools(_ctx())
        described = await _toolset(_SearchOnly(), description='product manuals').get_tools(_ctx())
        assert list(plain) == ['search_knowledge_base']
        assert plain['search_knowledge_base'].tool_def.description == (
            'Search the knowledge base for passages relevant to a query.'
        )
        assert described['search_knowledge_base'].tool_def.description == (
            'Search the knowledge base for passages relevant to a query. The knowledge base holds product manuals.'
        )


class TestToolsetOnItsOwn:
    async def test_searches_and_reads_without_durable_operations(self) -> None:
        toolset = KnowledgeBaseToolset[object](InMemoryKnowledgeBase(_DOCUMENTS), number_of_results=1)
        found = await toolset.search_knowledge_base(_ctx(), 'shipping business days')
        assert [record['document_id'] for record in found.metadata['results']] == ['shipping']
        read = await toolset.read_document(_ctx(), 'shipping')
        assert _text(read).endswith('Orders ship within two business days.')


class TestReadDocument:
    async def test_tool_is_only_offered_by_readable_backends(self) -> None:
        assert list(await _toolset(InMemoryKnowledgeBase()).get_tools(_ctx())) == [
            'search_knowledge_base',
            'read_document',
        ]
        assert list(await _toolset(_SearchOnly()).get_tools(_ctx())) == ['search_knowledge_base']

    async def test_returns_the_document_with_its_source(self) -> None:
        output = await _toolset(InMemoryKnowledgeBase(_DOCUMENTS)).read_document(_ctx(), 'shipping')
        assert _text(output) == (
            'Document: shipping\nSource: https://shop.example/shipping\n\nOrders ship within two business days.'
        )
        assert output.metadata == {
            'document': {
                'document_id': 'shipping',
                'source': 'https://shop.example/shipping',
                'metadata': {'tenant_id': 'acme', 'section': 'faq'},
                'offset': 0,
                'next_offset': None,
            }
        }

    async def test_long_document_is_cut_with_a_marker(self) -> None:
        backend = InMemoryKnowledgeBase([Document('long', 'x' * 30)])
        output = await _toolset(backend, max_document_chars=10, show_sources=False).read_document(_ctx(), 'long')
        assert (
            _text(output)
            == f'Document: long\n\n{"x" * 10}\n[... the document continues; call read_document with offset=10]'
        )
        assert output.metadata['document']['next_offset'] == 10

    async def test_later_parts_are_read_by_offset(self) -> None:
        backend = InMemoryKnowledgeBase([Document('long', 'a' * 10 + 'b' * 10 + 'c' * 5)])
        toolset = _toolset(backend, max_document_chars=10, show_sources=False)
        second = await toolset.read_document(_ctx(), 'long', offset=10)
        assert _text(second) == (
            f'Document: long\nFrom character: 10\n\n{"b" * 10}\n[... the document continues; call read_document with offset=20]'
        )
        last = await toolset.read_document(_ctx(), 'long', offset=20)
        assert _text(last) == f'Document: long\nFrom character: 20\n\n{"c" * 5}'
        assert (last.metadata['document']['offset'], last.metadata['document']['next_offset']) == (20, None)

    @pytest.mark.parametrize(('offset', 'message'), [(-1, 'offset must be 0 or more'), (30, 'past the end')])
    async def test_bad_offsets_ask_the_model_to_retry(self, offset: int, message: str) -> None:
        toolset = _toolset(InMemoryKnowledgeBase([Document('short', 'x' * 25)]))
        with pytest.raises(ModelRetry, match=message):
            await toolset.read_document(_ctx(), 'short', offset=offset)

    async def test_document_outside_the_run_filter_reads_as_missing(self) -> None:
        """A tenant-scoped agent can't open another tenant's document by guessing its id."""

        def tenant(ctx: RunContext[object]) -> MetadataFilter:
            return {'tenant_id': str(ctx.deps)}

        toolset = _toolset(InMemoryKnowledgeBase(_DOCUMENTS), filter=tenant)
        assert _text(await toolset.read_document(_ctx('acme'), 'returns')).startswith('Document: returns')
        with pytest.raises(ModelRetry, match="No document with id 'other-returns'"):
            await toolset.read_document(_ctx('acme'), 'other-returns')

    async def test_unknown_document_asks_the_model_to_retry(self) -> None:
        with pytest.raises(ModelRetry, match="No document with id 'nope'"):
            await _toolset(InMemoryKnowledgeBase()).read_document(_ctx(), 'nope')


class TestKnowledgeBase:
    @pytest.mark.parametrize('number_of_results', [0, 101])
    def test_number_of_results_out_of_bounds_rejected(self, number_of_results: int) -> None:
        with pytest.raises(ValueError, match='number_of_results must be between 1 and 100'):
            KnowledgeBase[None](_SearchOnly(), number_of_results=number_of_results)

    def test_max_document_chars_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='max_document_chars must be positive'):
            KnowledgeBase[None](_SearchOnly(), max_document_chars=0)

    def test_native_needs_a_native_capable_backend(self) -> None:
        with pytest.raises(ValueError, match='native needs a backend'):
            KnowledgeBase[None](_SearchOnly(), native=True)

    @pytest.mark.parametrize(
        'options',
        [{'filter': {'tenant_id': 'acme'}}, {'search_mode': 'semantic'}, {'score_threshold': 0.5}],
    )
    def test_native_uses_the_providers_settings(self, options: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="native search uses the provider's own settings"):
            KnowledgeBase[None](_NativeCapable(), native=True, **options)  # pyright: ignore[reportArgumentType]

    def test_static_filter_is_checked_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Unknown comparison operator '~'"):
            KnowledgeBase[None](_SearchOnly(), filter={'field': 'year', 'operator': '~', 'value': 1})

    def test_instructions_follow_the_backend_and_description(self) -> None:
        searching = KnowledgeBase[None](_SearchOnly()).get_instructions()
        reading = KnowledgeBase[None](InMemoryKnowledgeBase(), description='manuals').get_instructions()
        assert isinstance(searching, str) and isinstance(reading, str)
        assert 'search_knowledge_base' in searching and 'read_document' not in searching
        assert 'a knowledge base of manuals' in reading and 'read_document' in reading

    def test_custom_and_empty_guidance(self) -> None:
        assert KnowledgeBase[None](_SearchOnly(), guidance='Search first.').get_instructions() == 'Search first.'
        assert KnowledgeBase[None](_SearchOnly(), guidance='').get_instructions() is None

    def test_catalog_description_reads_as_an_instruction(self) -> None:
        assert KnowledgeBase[None](_SearchOnly()).get_description() == (
            'Search a knowledge base for passages before answering questions it could cover.'
        )
        assert KnowledgeBase[None](_SearchOnly(), description='manuals').get_description() == (
            'Search a knowledge base of manuals for passages before answering questions it could cover.'
        )

    def test_two_that_agree_are_one_and_two_that_differ_raise(self) -> None:
        backend = InMemoryKnowledgeBase()
        first = KnowledgeBase[None](backend, number_of_results=3)
        same = KnowledgeBase[None](backend, number_of_results=3)
        other = KnowledgeBase[None](backend, number_of_results=3, filter={'tenant_id': 'acme'})
        assert type(first).combine([first, same]) is first
        with pytest.raises(UserError, match="disagree on 'filter'"):
            type(first).combine([first, other])

    async def test_agent_run_searches_and_reads_with_the_run_filter(self) -> None:
        seen: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append([tool.name for tool in info.function_tools])
            if len(seen) == 1:
                return ModelResponse(parts=[ToolCallPart('search_knowledge_base', {'query': 'returns'})])
            if len(seen) == 2:
                return ModelResponse(parts=[ToolCallPart('read_document', {'document_id': 'returns'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=str,
            capabilities=[KnowledgeBase(InMemoryKnowledgeBase(_DOCUMENTS), filter=lambda ctx: {'tenant_id': ctx.deps})],
        )
        result = await agent.run('What is the return policy?', deps='acme')

        first = result.all_messages()[0]
        assert isinstance(first, ModelRequest) and first.instructions is not None
        assert 'read_document' in first.instructions
        assert seen[0] == ['search_knowledge_base', 'read_document']
        returns = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [part.tool_name for part in returns] == ['search_knowledge_base', 'read_document']
        assert [record['document_id'] for record in returns[0].metadata['results']] == ['returns']
        assert str(returns[1].content).startswith('Document: returns')

    async def test_retry_inside_the_agent_run_is_fed_back_to_the_model(self) -> None:
        """The durable operation carries the retry out as data, and the tool raises it again."""
        agent = Agent(TestModel(), capabilities=[KnowledgeBase(_SearchOnly(error=ModelRetry('throttled')))])
        with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'search_knowledge_base' exceeded max retries"):
            await agent.run('Search.')

    async def test_deferred_loading_hides_the_tools_until_loaded(self) -> None:
        backend = _SearchOnly(passages=[Passage('alpha')])
        seen: list[tuple[list[str], str | None]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            first = messages[0]
            assert isinstance(first, ModelRequest)
            seen.append(([tool.name for tool in info.function_tools], first.instructions))
            if len(seen) == 1:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'knowledge_base'})])
            if len(seen) == 2:
                return ModelResponse(parts=[ToolCallPart('search_knowledge_base', {'query': 'q'})])
            return ModelResponse(parts=[TextPart('done')])

        capability = KnowledgeBase[None](backend, description='manuals', defer_loading=True)
        await Agent(FunctionModel(respond), deps_type=type(None), capabilities=[capability]).run('Search.')

        first_tools, first_instructions = seen[0]
        assert 'search_knowledge_base' not in first_tools and 'load_capability' in first_tools
        assert first_instructions is not None
        assert 'Search a knowledge base of manuals for passages' in first_instructions
        assert 'search_knowledge_base' in seen[1][0]
        assert len(backend.calls) == 1

    async def test_search_can_require_approval(self) -> None:
        backend = _SearchOnly(passages=[Passage('alpha')])

        async def require_approval(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [replace(tool_def, kind='unapproved') for tool_def in tool_defs]

        agent = Agent(
            TestModel(),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[KnowledgeBase[None](backend), PrepareTools[None](require_approval)],
        )
        result = await agent.run('Search.')
        assert backend.calls == []
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['search_knowledge_base']


class TestNativeFileSearch:
    async def test_native_tool_replaces_the_local_search_where_supported(self) -> None:
        sent: list[tuple[list[str], list[str]]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(
                (
                    [tool.name for tool in info.function_tools],
                    [tool.kind for tool in info.model_request_parameters.native_tools],
                )
            )
            return ModelResponse(parts=[TextPart('done')])

        from pydantic_ai.profiles import ModelProfile

        model = FunctionModel(respond, profile=ModelProfile(supported_native_tools=frozenset({FileSearchTool})))
        await Agent(model, capabilities=[KnowledgeBase(_NativeCapable(), native=True)]).run('Search.')
        assert sent == [([], ['file_search'])]

    async def test_local_search_is_the_fallback_where_unsupported(self) -> None:
        sent: list[tuple[list[str], list[str]]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(
                (
                    [tool.name for tool in info.function_tools],
                    [tool.kind for tool in info.model_request_parameters.native_tools],
                )
            )
            return ModelResponse(parts=[TextPart('done')])

        from pydantic_ai.profiles import ModelProfile

        model = FunctionModel(respond, profile=ModelProfile(supported_native_tools=frozenset()))
        await Agent(model, capabilities=[KnowledgeBase(_NativeCapable(), native=True)]).run('Search.')
        assert sent == [(['search_knowledge_base'], [])]


class TestAgentSpec:
    def test_unknown_package_attribute_raises(self) -> None:
        import pydantic_ai_harness.knowledge_base as package

        with pytest.raises(AttributeError, match='has no attribute'):
            getattr(package, 'Nope')

    def test_spec_schema_includes_the_capability(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([KnowledgeBase])
        assert 'KnowledgeBase' in json.dumps(schema)


class TestFilters:
    _METADATA: dict[str, object] = {'tenant_id': 'acme', 'year': 2025, 'tags': 'manual'}

    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            (None, True),
            ({}, True),
            ({'tenant_id': 'acme'}, True),
            ({'tenant_id': 'globex'}, False),
            ({'tenant_id': ['acme', 'globex'], 'year': 2025}, True),
            ({'field': 'year', 'operator': '>', 'value': 2024}, True),
            ({'field': 'year', 'operator': '>=', 'value': 2026}, False),
            ({'field': 'year', 'operator': '<', 'value': 2026}, True),
            ({'field': 'year', 'operator': '<=', 'value': 2024}, False),
            ({'field': 'year', 'operator': '!=', 'value': 2025}, False),
            ({'field': 'year', 'operator': 'not in', 'value': [2024]}, True),
            ({'field': 'tenant_id', 'operator': '>', 'value': 3}, False),
            ({'field': 'missing', 'operator': '==', 'value': 'x'}, False),
            ({'field': 'missing', 'operator': '>', 'value': 1}, False),
            ({'field': 'missing', 'operator': '!=', 'value': 'x'}, True),
            ({'field': 'missing', 'operator': 'not in', 'value': ['x']}, True),
            (
                {
                    'operator': 'OR',
                    'conditions': [
                        {'field': 'tenant_id', 'operator': '==', 'value': 'globex'},
                        {'field': 'year', 'operator': '==', 'value': 2025},
                    ],
                },
                True,
            ),
            (
                {
                    'operator': 'AND',
                    'conditions': [
                        {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                        {'operator': 'NOT', 'conditions': [{'field': 'tags', 'operator': '==', 'value': 'manual'}]},
                    ],
                },
                False,
            ),
        ],
    )
    def test_matches(self, filter: MetadataFilter | None, expected: bool) -> None:
        assert matches(self._METADATA, filter) is expected

    def test_shorthand_becomes_a_tree(self) -> None:
        assert normalize_filter({'tenant_id': 'acme', 'source': ('a', 'b')}) == {
            'operator': 'AND',
            'conditions': [
                {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                {'field': 'source', 'operator': 'in', 'value': ['a', 'b']},
            ],
        }

    @pytest.mark.parametrize(
        ('filter', 'message'),
        [
            ({'operator': 'XOR', 'conditions': [{'field': 'a', 'operator': '==', 'value': 1}]}, 'Unknown logical'),
            ({'operator': 'AND', 'conditions': []}, 'non-empty list'),
            ({'operator': 'AND', 'conditions': 'abc'}, 'non-empty list'),
            ({'operator': 'AND', 'conditions': [{'a': 1}]}, 'Not a filter condition'),
            ({'operator': 'AND', 'conditions': ['a']}, 'Not a filter condition'),
            ({'field': 3, 'operator': '==', 'value': 1}, 'needs a field name'),
            ({'field': 'a', 'operator': '~', 'value': 1}, 'Unknown comparison'),
            ({'field': 'a', 'operator': 'in', 'value': 'x'}, 'needs a list'),
            ({'field': 'a', 'operator': 'in', 'value': 3}, 'needs a list'),
            ({'field': 'a', 'operator': '==', 'value': [1]}, 'needs a single value'),
            ({'a': {'nested': 1}}, 'needs a list'),
        ],
    )
    def test_malformed_filters_are_rejected(self, filter: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            normalize_filter(filter)  # pyright: ignore[reportArgumentType]


class TestRequireFilter:
    def test_needs_a_filter(self) -> None:
        with pytest.raises(ValueError, match='require_filter needs a filter'):
            KnowledgeBase[None](_SearchOnly(), require_filter=True)

    async def test_a_filter_that_resolves_to_nothing_refuses_to_search_and_read(self) -> None:
        def tenant(ctx: RunContext[object]) -> MetadataFilter | None:
            return None if ctx.deps is None else {'tenant_id': str(ctx.deps)}

        backend = _SearchOnly()
        searching = _toolset(backend, filter=tenant, require_filter=True)
        with pytest.raises(UserError, match='require_filter is set'):
            await searching.search_knowledge_base(_ctx(None), 'q')
        assert backend.calls == []
        await searching.search_knowledge_base(_ctx('acme'), 'q')
        assert backend.calls[0][2] == {'field': 'tenant_id', 'operator': '==', 'value': 'acme'}

        reading = _toolset(InMemoryKnowledgeBase(_DOCUMENTS), filter=tenant, require_filter=True)
        with pytest.raises(UserError, match='require_filter is set'):
            await reading.read_document(_ctx(None), 'returns')


class TestCanRead:
    async def test_a_backend_can_turn_reading_off(self) -> None:
        class _Switchable(InMemoryKnowledgeBase):
            can_read = False

        assert list(await _toolset(_Switchable(_DOCUMENTS)).get_tools(_ctx())) == ['search_knowledge_base']
        instructions = KnowledgeBase[None](_Switchable()).get_instructions()
        assert isinstance(instructions, str) and 'read_document' not in instructions


class TestAllowsRead:
    @pytest.mark.parametrize(
        ('metadata', 'filter', 'expected'),
        [
            ({}, None, True),
            ({'tenant_id': 'acme'}, {'tenant_id': 'acme'}, True),
            ({'tenant_id': 'acme'}, {'tenant_id': 'globex'}, False),
            ({}, {'field': 'tenant_id', 'operator': '!=', 'value': 'globex'}, False),
            ({}, {'field': 'tenant_id', 'operator': 'not in', 'value': ['globex']}, False),
            ({'tenant_id': 'acme'}, {'field': 'tenant_id', 'operator': '!=', 'value': 'globex'}, True),
            (
                {'tenant_id': 'acme'},
                {
                    'operator': 'AND',
                    'conditions': [
                        {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                        {'field': 'year', 'operator': '!=', 'value': 2020},
                    ],
                },
                False,
            ),
        ],
    )
    def test_every_named_field_must_be_present(
        self, metadata: dict[str, object], filter: MetadataFilter | None, expected: bool
    ) -> None:
        assert allows_read(metadata, normalize_filter(filter)) is expected

    async def test_in_memory_read_fails_closed(self) -> None:
        backend = InMemoryKnowledgeBase([Document('untagged', 'text')])
        not_globex = normalize_filter({'field': 'tenant_id', 'operator': '!=', 'value': 'globex'})
        assert len(await backend.search('text', limit=5, filter=not_globex)) == 1
        assert await backend.read('untagged', max_chars=100, filter=not_globex) is None
