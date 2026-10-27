"""Tests for the `Knowledge` capability.

Searches run through `Agent(..., capabilities=[Knowledge(...)])` with a `FunctionModel` that calls
`search_knowledge`, over files in a `LocalWorkspace`. Embeddings come from a keyword-count model, so
similarity is predictable and every embedding request is recorded.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest
from inline_snapshot import snapshot
from opentelemetry.trace import get_tracer

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import LocalWorkspace, PrefixTools
from pydantic_ai.embeddings import Embedder, EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceError
from pydantic_ai_harness import Knowledge
from pydantic_ai_harness.knowledge import (
    DEFAULT_FILE_SUFFIXES,
    EmbeddedChunk,
    InMemoryVectorStore,
    KnowledgeDocument,
    KnowledgeMatch,
    KnowledgeSyncResult,
    RecursiveChunker,
)

try:
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    logfire_installed = False
else:
    logfire_installed = True

VOCABULARY = ('refund', 'shipping', 'password', 'invoice', 'warranty')


class KeywordEmbeddingModel(EmbeddingModel):
    """Embeds text as the count of each `VOCABULARY` word in it, and records every request."""

    def __init__(self, name: str = 'keywords', *, fail_on_request: int | None = None) -> None:
        super().__init__()
        self._name = name
        self._fail_on_request = fail_on_request
        self.requests: list[tuple[EmbedInputType, list[str]]] = []

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def system(self) -> str:
        return 'test'

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        texts, _ = self.prepare_embed(inputs, settings)
        self.requests.append((input_type, texts))
        if len(self.requests) == self._fail_on_request:
            raise RuntimeError('embedding service unavailable')
        return EmbeddingResult(
            embeddings=[[float(text.lower().count(word)) for word in VOCABULARY] for text in texts],
            inputs=texts,
            input_type=input_type,
            model_name=self._name,
            provider_name='test',
        )

    @property
    def document_requests(self) -> list[list[str]]:
        return [texts for input_type, texts in self.requests if input_type == 'document']


def _write(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _searching(*queries: str, tool: str = 'search_knowledge') -> FunctionModel:
    """A model that calls `tool` once per query, in parallel, then answers 'done'."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool, {'query': query}) for query in queries])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def _matches(part: ToolReturnPart) -> list[KnowledgeMatch]:
    content = part.content
    assert isinstance(content, list)
    matches: list[KnowledgeMatch] = []
    for match in content:  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(match, KnowledgeMatch)
        matches.append(match)
    return matches


def _ranked(part: ToolReturnPart) -> list[tuple[str, float]]:
    return [(match.document_id, round(match.score, 3)) for match in _matches(part)]


DOCS = {
    'docs/refunds.md': 'Refund policy: a refund is issued within 14 days of the refund request.',
    'docs/shipping.md': 'Shipping takes 3 days. Shipping is free over $50.',
    'docs/account/passwords.md': 'Reset your password from the account page.',
    'docs/notes.bin': 'refund refund refund',
    'docs/.drafts/refund.md': 'refund draft',
    'docs/.hidden.md': 'refund hidden',
}


class TestKnowledge:
    async def test_search_returns_the_most_similar_passages_from_workspace_files(self, tmp_path: Path) -> None:
        _write(tmp_path, DOCS)
        model = KeywordEmbeddingModel()
        agent = Agent(
            _searching('how do I get a refund?'),
            capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=model, max_results=2)],
        )

        result = await agent.run('question')

        [part] = _returns(result.all_messages())
        assert part.tool_name == 'search_knowledge'
        assert _ranked(part) == snapshot([('docs/refunds.md', 1.0), ('docs/account/passwords.md', 0.0)])
        assert _matches(part)[0] == snapshot(
            KnowledgeMatch(
                document_id='docs/refunds.md',
                text='Refund policy: a refund is issued within 14 days of the refund request.',
                score=1.0,
            )
        )
        # Hidden entries and files without a text suffix are not indexed.
        assert model.document_requests == snapshot(
            [
                [
                    'Reset your password from the account page.',
                    'Refund policy: a refund is issued within 14 days of the refund request.',
                    'Shipping takes 3 days. Shipping is free over $50.',
                ]
            ]
        )

    async def test_instructions_name_the_description(self, tmp_path: Path) -> None:
        captured: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured.append(info.instructions)
            return ModelResponse(parts=[TextPart('done')])

        documents = [KnowledgeDocument('faq', 'refund')]
        await Agent(FunctionModel(respond), capabilities=[Knowledge(documents, embedder=KeywordEmbeddingModel())]).run(
            'q'
        )
        await Agent(
            FunctionModel(respond),
            capabilities=[Knowledge(documents, embedder=KeywordEmbeddingModel(), description='the support handbook')],
        ).run('q')

        assert captured == snapshot(
            [
                'Search the knowledge base before answering a question it may cover, rather than answering from memory, and say which document the answer comes from.',
                'Search the support handbook before answering a question it may cover, rather than answering from memory, and say which document the answer comes from.',
            ]
        )

    async def test_sync_embeds_only_new_and_changed_documents_and_removes_deleted_ones(self, tmp_path: Path) -> None:
        _write(tmp_path, {'docs/a.md': 'refund', 'docs/b.md': 'shipping', 'docs/c.md': 'invoice'})
        model = KeywordEmbeddingModel()
        store = InMemoryVectorStore()
        knowledge = Knowledge[None](['docs'], embedder=model, store=store)
        workspace = LocalWorkspaceBackend(tmp_path)

        assert await knowledge.sync(workspace) == snapshot(
            KnowledgeSyncResult(documents=3, embedded=3, removed=0, chunks_embedded=3)
        )
        assert await knowledge.sync(workspace) == snapshot(
            KnowledgeSyncResult(documents=3, embedded=0, removed=0, chunks_embedded=0)
        )

        (tmp_path / 'docs/b.md').write_text('warranty')
        (tmp_path / 'docs/c.md').unlink()
        (tmp_path / 'docs/d.md').write_text('password')
        assert await knowledge.sync(workspace) == snapshot(
            KnowledgeSyncResult(documents=3, embedded=2, removed=1, chunks_embedded=2)
        )
        assert model.document_requests == snapshot([['refund', 'shipping', 'invoice'], ['warranty', 'password']])
        assert sorted(await store.fingerprints()) == ['docs/a.md', 'docs/b.md', 'docs/d.md']

    async def test_searches_in_later_runs_reuse_the_store(self, tmp_path: Path) -> None:
        _write(tmp_path, {'docs/a.md': 'refund'})
        model = KeywordEmbeddingModel()
        agent = Agent(
            _searching('refund'), capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=model)]
        )

        await agent.run('first')
        await agent.run('second')

        assert model.requests == snapshot([('query', ['refund']), ('document', ['refund']), ('query', ['refund'])])

    async def test_a_file_the_agent_changes_mid_run_is_searched_as_changed(self, tmp_path: Path) -> None:
        _write(tmp_path, {'docs/a.md': 'refund'})

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('search_knowledge', {'query': 'warranty'})])
            if len(messages) == 3:
                (tmp_path / 'docs/a.md').write_text('warranty')
                return ModelResponse(parts=[ToolCallPart('search_knowledge', {'query': 'warranty'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=KeywordEmbeddingModel())],
        )
        result = await agent.run('q')

        assert [_ranked(part) for part in _returns(result.all_messages())] == snapshot(
            [[('docs/a.md', 0.0)], [('docs/a.md', 1.0)]]
        )

    async def test_parallel_searches_embed_the_corpus_once(self, tmp_path: Path) -> None:
        _write(tmp_path, {'docs/a.md': 'refund', 'docs/b.md': 'shipping'})
        model = KeywordEmbeddingModel()
        agent = Agent(
            _searching('refund', 'shipping'),
            capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=model, max_results=1)],
        )

        result = await agent.run('q')

        assert [_ranked(part) for part in _returns(result.all_messages())] == snapshot(
            [[('docs/a.md', 1.0)], [('docs/b.md', 1.0)]]
        )
        assert model.document_requests == snapshot([['refund', 'shipping']])

    async def test_concurrent_searches_over_different_workspaces_each_search_their_own(self, tmp_path: Path) -> None:
        _write(tmp_path / 'alpha', {'docs/a.md': 'refund alpha'})
        _write(tmp_path / 'beta', {'docs/a.md': 'refund beta'})

        class HoldAlphaQuery(KeywordEmbeddingModel):
            """Holds the alpha query until the beta workspace's document is embedded, so the searches overlap."""

            def __init__(self) -> None:
                super().__init__()
                self.beta_embedded = anyio.Event()

            async def embed(
                self,
                inputs: str | Sequence[str],
                *,
                input_type: EmbedInputType,
                settings: EmbeddingSettings | None = None,
            ) -> EmbeddingResult:
                if input_type == 'query' and inputs == 'refund alpha':
                    await self.beta_embedded.wait()
                result = await super().embed(inputs, input_type=input_type, settings=settings)
                if input_type == 'document' and 'refund beta' in inputs:
                    self.beta_embedded.set()
                return result

        agent = Agent(capabilities=[Knowledge(['docs'], embedder=HoldAlphaQuery())])
        texts: dict[str, list[str]] = {}

        async def search(name: str) -> None:
            result = await agent.run(
                'q', model=_searching(f'refund {name}'), workspace=LocalWorkspaceBackend(tmp_path / name)
            )
            texts[name] = [match.text for part in _returns(result.all_messages()) for match in _matches(part)]

        async with anyio.create_task_group() as tg:
            tg.start_soon(search, 'alpha')
            tg.start_soon(search, 'beta')

        assert texts == {'alpha': ['refund alpha'], 'beta': ['refund beta']}

    async def test_a_new_embedding_model_embeds_every_document_again(self, tmp_path: Path) -> None:
        store = InMemoryVectorStore()
        documents = [KnowledgeDocument('a', 'refund'), KnowledgeDocument('b', 'shipping')]
        first = KeywordEmbeddingModel('v1')
        second = KeywordEmbeddingModel('v2')

        await Knowledge[None](documents, embedder=first, store=store).sync()
        await Knowledge[None](documents, embedder=Embedder(first), store=store).sync()
        assert await Knowledge[None](documents, embedder=second, store=store).sync() == snapshot(
            KnowledgeSyncResult(documents=2, embedded=2, removed=0, chunks_embedded=2)
        )
        assert first.document_requests == snapshot([['refund', 'shipping']])

    async def test_new_chunking_embeds_every_document_again(self) -> None:
        store = InMemoryVectorStore()
        documents = [KnowledgeDocument('a', 'refund policy. ' * 10)]
        model = KeywordEmbeddingModel()

        await Knowledge[None](documents, embedder=model, store=store).sync()
        resynced = await Knowledge[None](
            documents, embedder=model, store=store, chunker=RecursiveChunker(chunk_size=50, chunk_overlap=0)
        ).sync()

        assert resynced == snapshot(KnowledgeSyncResult(documents=1, embedded=1, removed=0, chunks_embedded=4))

    async def test_embedding_requests_span_documents_in_batches(self) -> None:
        model = KeywordEmbeddingModel()
        documents = [KnowledgeDocument(f'doc-{i}', f'refund {i}') for i in range(130)]

        result = await Knowledge[None](documents, embedder=model).sync()

        assert result == snapshot(KnowledgeSyncResult(documents=130, embedded=130, removed=0, chunks_embedded=130))
        assert [len(texts) for texts in model.document_requests] == snapshot([64, 64, 2])

    async def test_a_failed_embedding_request_keeps_the_documents_already_embedded(self) -> None:
        model = KeywordEmbeddingModel(fail_on_request=2)
        documents = [KnowledgeDocument(f'doc-{i}', f'refund {i}') for i in range(100)]
        store = InMemoryVectorStore()
        knowledge = Knowledge[None](documents, embedder=model, store=store)

        with pytest.raises(RuntimeError, match='embedding service unavailable'):
            await knowledge.sync()

        assert len(await store.fingerprints()) == 64
        assert await knowledge.sync() == snapshot(
            KnowledgeSyncResult(documents=100, embedded=36, removed=0, chunks_embedded=36)
        )

    async def test_a_document_with_no_text_is_stored_without_chunks(self) -> None:
        model = KeywordEmbeddingModel()
        knowledge = Knowledge[None](
            [KnowledgeDocument('empty', '  \n'), KnowledgeDocument('a', 'refund')], embedder=model
        )

        assert await knowledge.sync() == snapshot(
            KnowledgeSyncResult(documents=2, embedded=2, removed=0, chunks_embedded=1)
        )
        assert await knowledge.sync() == snapshot(
            KnowledgeSyncResult(documents=2, embedded=0, removed=0, chunks_embedded=0)
        )
        assert model.document_requests == [['refund']]


class TestKnowledgeSources:
    async def test_a_listed_file_is_read_whatever_its_suffix(self, tmp_path: Path) -> None:
        _write(tmp_path, {'notes.bin': 'refund', 'docs/a.md': 'shipping'})
        store = InMemoryVectorStore()
        knowledge = Knowledge[None](['notes.bin', Path('docs/a.md')], embedder=KeywordEmbeddingModel(), store=store)

        await knowledge.sync(LocalWorkspaceBackend(tmp_path))

        assert sorted(await store.fingerprints()) == ['docs/a.md', 'notes.bin']

    async def test_the_working_directory_as_a_source_names_documents_by_relative_path(self, tmp_path: Path) -> None:
        _write(tmp_path, {'a.md': 'refund', 'sub/b.TXT': 'shipping', 'c.py': 'print()'})
        knowledge = Knowledge[None](['.', 'sub/../a.md'], embedder=KeywordEmbeddingModel())

        with pytest.raises(UserError, match=r"produce the document id 'a\.md' more than once"):
            await knowledge.sync(LocalWorkspaceBackend(tmp_path))

        store = InMemoryVectorStore()
        knowledge = Knowledge[None](['./'], embedder=KeywordEmbeddingModel(), store=store)
        await knowledge.sync(LocalWorkspaceBackend(tmp_path))
        assert sorted(await store.fingerprints()) == ['a.md', 'sub/b.TXT']

    async def test_file_suffixes_select_directory_files(self, tmp_path: Path) -> None:
        _write(tmp_path, {'src/a.py': 'refund', 'src/b.md': 'shipping'})
        store = InMemoryVectorStore()
        knowledge = Knowledge[None](['src'], embedder=KeywordEmbeddingModel(), file_suffixes=['.PY'], store=store)

        await knowledge.sync(LocalWorkspaceBackend(tmp_path))

        assert list(await store.fingerprints()) == ['src/a.py']
        assert '.md' in DEFAULT_FILE_SUFFIXES

    async def test_a_symlinked_directory_loop_is_walked_once(self, tmp_path: Path) -> None:
        _write(tmp_path, {'docs/a.md': 'refund'})
        os.symlink(tmp_path / 'docs', tmp_path / 'docs' / 'loop')
        store = InMemoryVectorStore()
        knowledge = Knowledge[None](['docs'], embedder=KeywordEmbeddingModel(), store=store)

        await knowledge.sync(LocalWorkspaceBackend(tmp_path))

        assert sorted(await store.fingerprints()) == ['docs/a.md']

    async def test_a_symlink_out_of_the_source_and_working_directory_is_skipped(self, tmp_path: Path) -> None:
        work = tmp_path / 'work'
        _write(work, {'docs/a.md': 'refund', 'shared/b.md': 'shipping'})
        (tmp_path / 'secret.md').write_text('password')
        os.symlink(tmp_path / 'secret.md', work / 'docs' / 'leak.md')
        os.symlink(work / 'shared', work / 'docs' / 'shared')
        store = InMemoryVectorStore()

        await Knowledge[None](['docs'], embedder=KeywordEmbeddingModel(), store=store).sync(LocalWorkspaceBackend(work))

        assert sorted(await store.fingerprints()) == ['docs/a.md', 'docs/shared/b.md']

    async def test_a_directory_outside_the_working_directory_is_read(self, tmp_path: Path) -> None:
        _write(tmp_path, {'library/a.md': 'refund', 'work/notes.md': 'shipping'})
        store = InMemoryVectorStore()
        knowledge = Knowledge[None]([tmp_path / 'library'], embedder=KeywordEmbeddingModel(), store=store)

        await knowledge.sync(LocalWorkspaceBackend(tmp_path / 'work'))

        assert list(await store.fingerprints()) == [f'{(tmp_path / "library").as_posix()}/a.md']

    async def test_non_utf8_bytes_are_replaced(self, tmp_path: Path) -> None:
        (tmp_path / 'a.txt').write_bytes(b'refund \xff')
        model = KeywordEmbeddingModel()

        await Knowledge[None](['a.txt'], embedder=model).sync(LocalWorkspaceBackend(tmp_path))

        assert model.document_requests == [['refund \ufffd']]

    async def test_a_missing_path_fails_the_run(self, tmp_path: Path) -> None:
        agent = Agent(
            _searching('refund'),
            capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=KeywordEmbeddingModel())],
        )

        with pytest.raises(UserError, match=r"`Knowledge` source 'docs' does not exist in the workspace\."):
            await agent.run('q')

    async def test_a_custom_source_is_loaded_on_every_sync_with_the_workspace(self, tmp_path: Path) -> None:
        seen: list[Workspace] = []
        texts = ['refund']

        class Tickets:
            async def load(self, workspace: Workspace) -> Sequence[KnowledgeDocument]:
                seen.append(workspace)
                return [KnowledgeDocument('ticket-1', texts[-1])]

        knowledge = Knowledge[None]([Tickets()], embedder=KeywordEmbeddingModel())
        await knowledge.sync(LocalWorkspaceBackend(tmp_path))
        texts.append('warranty')
        assert await knowledge.sync() == snapshot(
            KnowledgeSyncResult(documents=1, embedded=1, removed=0, chunks_embedded=1)
        )

        assert await seen[0].working_dir() == str(tmp_path)
        assert not seen[1].attached

    async def test_a_workspace_error_fails_the_tool_call(self) -> None:
        class Broken:
            async def load(self, workspace: Workspace) -> Sequence[KnowledgeDocument]:
                raise WorkspaceError('backend refused the read')

        agent = Agent(_searching('refund'), capabilities=[Knowledge([Broken()], embedder=KeywordEmbeddingModel())])

        result = await agent.run('q')

        [part] = _returns(result.all_messages())
        assert (part.outcome, part.content) == snapshot(('failed', 'backend refused the read'))


class TestKnowledgeWorkspace:
    async def test_path_sources_without_a_workspace_fail_the_run_at_its_start(self) -> None:
        model = KeywordEmbeddingModel()
        agent = Agent(_searching('refund'), capabilities=[Knowledge(['docs'], embedder=model)])

        with pytest.raises(UserError, match='`Knowledge` needs a workspace'):
            await agent.run('q')
        assert model.requests == []

    async def test_inline_documents_need_no_workspace(self) -> None:
        agent = Agent(
            _searching('refund'),
            capabilities=[Knowledge([KnowledgeDocument('faq', 'refund')], embedder=KeywordEmbeddingModel())],
        )

        result = await agent.run('q')

        assert [_ranked(part) for part in _returns(result.all_messages())] == [[('faq', 1.0)]]

    async def test_own_workspace_is_read_instead_of_the_runs(self, tmp_path: Path) -> None:
        _write(tmp_path / 'app', {'docs/a.md': 'refund'})
        _write(tmp_path / 'run', {'docs/a.md': 'shipping'})
        knowledge = Knowledge(
            ['docs'], embedder=KeywordEmbeddingModel(), workspace=LocalWorkspaceBackend(tmp_path / 'app')
        )

        result = await Agent(_searching('refund'), capabilities=[knowledge]).run('q')
        assert [_ranked(part) for part in _returns(result.all_messages())] == [[('docs/a.md', 1.0)]]
        result = await Agent(_searching('refund'), capabilities=[LocalWorkspace(tmp_path / 'run'), knowledge]).run('q')
        assert [_ranked(part) for part in _returns(result.all_messages())] == [[('docs/a.md', 1.0)]]
        assert await knowledge.sync(LocalWorkspaceBackend(tmp_path / 'run')) == snapshot(
            KnowledgeSyncResult(documents=1, embedded=0, removed=0, chunks_embedded=0)
        )

    async def test_sync_without_a_workspace_for_path_sources_raises(self) -> None:
        with pytest.raises(UserError, match=r'`Knowledge.sync\(\)` needs the workspace'):
            await Knowledge[None](['docs'], embedder=KeywordEmbeddingModel()).sync()

    def test_a_workspace_capability_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match=r'`Knowledge\(workspace=...\)` takes a workspace backend'):
            Knowledge[None](['docs'], embedder=KeywordEmbeddingModel(), workspace=LocalWorkspace(tmp_path))  # pyright: ignore[reportArgumentType]


class TestKnowledgeConfiguration:
    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'sources': []}, 'needs at least one source'),
            ({'max_results': 0}, '`Knowledge.max_results` must be positive'),
            ({'file_suffixes': ['md']}, r"entries start with a dot, such as `.md`: \['md'\]"),
        ],
    )
    def test_invalid_configuration_is_refused(self, kwargs: dict[str, object], message: str) -> None:
        arguments: dict[str, object] = {'sources': ['docs'], 'embedder': KeywordEmbeddingModel(), **kwargs}
        with pytest.raises(UserError, match=message):
            Knowledge[None](**arguments)  # pyright: ignore[reportArgumentType]

    async def test_two_agreeing_capabilities_are_one(self) -> None:
        tools: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tools.append([tool.name for tool in info.function_tools])
            return ModelResponse(parts=[TextPart('done')])

        spec = Knowledge.from_spec(['docs'], embedder='openai:text-embedding-3-small', file_suffixes=['.md'])
        code = Knowledge(('docs',), embedder='openai:text-embedding-3-small', file_suffixes=('.MD',))
        await Agent(FunctionModel(respond), capabilities=[spec, code]).run('q', workspace=LocalWorkspaceBackend('.'))

        assert tools == [['search_knowledge']]

    def test_two_disagreeing_capabilities_raise(self) -> None:
        model = KeywordEmbeddingModel()
        with pytest.raises(UserError, match="Capability id 'knowledge' is used by multiple Knowledge capabilities"):
            Agent('test', capabilities=[Knowledge(['docs'], embedder=model), Knowledge(['faq'], embedder=model)])

    async def test_two_knowledge_bases_need_their_own_ids_and_prefixes(self) -> None:
        handbook = Knowledge([KnowledgeDocument('handbook', 'refund')], embedder=KeywordEmbeddingModel())
        api = Knowledge(
            [KnowledgeDocument('api', 'invoice')],
            embedder=KeywordEmbeddingModel(),
            description='the API docs',
            id='api',
        )
        agent = Agent(
            _searching('invoice', tool='api_search_knowledge'),
            capabilities=[handbook, PrefixTools(api, prefix='api')],
        )

        result = await agent.run('q')

        [part] = _returns(result.all_messages())
        assert (part.tool_name, _ranked(part)) == ('api_search_knowledge', [('api', 1.0)])

    def test_agent_spec(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {
                        'Knowledge': {
                            'sources': ['docs'],
                            'embedder': 'openai:text-embedding-3-small',
                            'description': 'the manual',
                            'max_results': 3,
                            'file_suffixes': ['.md'],
                            'chunk_size': 500,
                            'chunk_overlap': 50,
                        }
                    }
                ],
            },
            custom_capability_types=[Knowledge],
        )
        [knowledge] = [
            capability for capability in agent.root_capability.capabilities if isinstance(capability, Knowledge)
        ]
        assert knowledge == Knowledge(
            ('docs',),
            embedder='openai:text-embedding-3-small',
            chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
            description='the manual',
            max_results=3,
            file_suffixes=frozenset({'.md'}),
        )

    def test_agent_spec_schema_lists_the_serializable_options(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([Knowledge])
        params = schema['$defs']['spec_params_Knowledge']
        assert (sorted(params['properties']), params['required']) == snapshot(
            (
                [
                    'chunk_overlap',
                    'chunk_size',
                    'description',
                    'embedder',
                    'file_suffixes',
                    'id',
                    'max_results',
                    'sources',
                ],
                ['sources', 'embedder'],
            )
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
class TestKnowledgeTelemetry:
    @pytest.mark.usefixtures('instrument_all_agents')
    async def test_each_sync_is_a_span_with_what_it_changed(self, tmp_path: Path, capfire: CaptureLogfire) -> None:
        _write(tmp_path, {'docs/a.md': 'refund', 'docs/b.md': 'shipping'})
        agent = Agent(
            _searching('refund'),
            capabilities=[LocalWorkspace(tmp_path), Knowledge(['docs'], embedder=KeywordEmbeddingModel())],
        )

        await agent.run('q')

        [span] = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'knowledge_sync']
        assert {key: value for key, value in span['attributes'].items() if key.startswith('knowledge.')} == snapshot(
            {
                'knowledge.documents': 2,
                'knowledge.documents_embedded': 2,
                'knowledge.documents_removed': 0,
                'knowledge.chunks_embedded': 2,
            }
        )

    async def test_sync_outside_a_run_spans_on_the_tracer_passed(self, capfire: CaptureLogfire) -> None:
        knowledge = Knowledge[None]([KnowledgeDocument('a', 'refund')], embedder=KeywordEmbeddingModel())

        await knowledge.sync()
        await knowledge.sync(tracer=get_tracer('test'))

        spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'knowledge_sync']
        assert [span['attributes']['knowledge.documents_embedded'] for span in spans] == [0]


class TestRecursiveChunker:
    def test_short_text_is_one_chunk(self) -> None:
        assert RecursiveChunker().chunk('  short text \n') == ['short text']
        assert RecursiveChunker().chunk(' \n ') == []

    def test_paragraphs_are_merged_up_to_the_chunk_size(self) -> None:
        text = 'aaaa\n\nbbbb\n\ncccc\n\ndddd'
        assert RecursiveChunker(chunk_size=12, chunk_overlap=0).chunk(text) == snapshot(
            ['aaaa\n\nbbbb', 'cccc\n\ndddd']
        )

    def test_overlap_repeats_trailing_pieces(self) -> None:
        text = 'one two three four five six'
        assert RecursiveChunker(chunk_size=14, chunk_overlap=6).chunk(text) == snapshot(
            ['one two three', 'three four', 'four five six']
        )

    def test_a_long_piece_is_split_at_finer_separators(self) -> None:
        text = 'intro\n\n' + 'word ' * 6 + '\n\noutro'
        assert RecursiveChunker(chunk_size=12, chunk_overlap=0).chunk(text) == snapshot(
            ['intro', 'word word', 'word word', 'word word', 'outro']
        )

    def test_a_long_first_or_last_piece_is_split_on_its_own(self) -> None:
        text = 'word ' * 6 + '\n\nend'
        assert RecursiveChunker(chunk_size=12, chunk_overlap=0).chunk(text) == snapshot(
            ['word word', 'word word', 'word word', 'end']
        )
        assert RecursiveChunker(chunk_size=12, chunk_overlap=0).chunk('start\n\n' + 'word ' * 6) == snapshot(
            ['start', 'word word', 'word word', 'word word']
        )

    def test_text_without_separators_is_cut(self) -> None:
        assert RecursiveChunker(chunk_size=4, chunk_overlap=1).chunk('abcdefghij') == snapshot(['abcd', 'defg', 'ghij'])
        assert RecursiveChunker(chunk_size=4, chunk_overlap=0).chunk('abcdefghij') == snapshot(['abcd', 'efgh', 'ij'])

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'chunk_size': 0}, '`RecursiveChunker.chunk_size` must be positive'),
            ({'chunk_size': 10, 'chunk_overlap': 10}, 'smaller than `chunk_size`'),
            ({'chunk_overlap': -1}, 'non-negative'),
            ({'separators': ('\n', '')}, 'cannot contain an empty string'),
        ],
    )
    def test_invalid_configuration_is_refused(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(UserError, match=message):
            RecursiveChunker(**kwargs)  # pyright: ignore[reportArgumentType]


class TestInMemoryVectorStore:
    async def test_search_ranks_by_cosine_similarity(self) -> None:
        store = InMemoryVectorStore()
        await store.replace_document(
            'a',
            fingerprint='1',
            chunks=[EmbeddedChunk(text='x', vector=[1.0, 0.0]), EmbeddedChunk(text='zero', vector=[0.0, 0.0])],
        )
        await store.replace_document('b', fingerprint='2', chunks=[EmbeddedChunk(text='y', vector=[1.0, 1.0])])

        matches = await store.search([1.0, 0.0], limit=2)
        assert [(match.document_id, match.text, round(match.score, 3)) for match in matches] == snapshot(
            [('a', 'x', 1.0), ('b', 'y', 0.707)]
        )
        assert [match.score for match in await store.search([0.0, 0.0], limit=1)] == [0.0]

        await store.remove_documents(['a', 'missing'])
        assert await store.fingerprints() == {'b': '2'}

    async def test_a_query_of_another_size_is_refused(self) -> None:
        store = InMemoryVectorStore()
        await store.replace_document('a', fingerprint='1', chunks=[EmbeddedChunk(text='x', vector=[1.0, 0.0])])

        with pytest.raises(UserError, match='holds 2-dimensional vectors, but the query has 3'):
            await store.search([1.0, 0.0, 0.0], limit=1)
