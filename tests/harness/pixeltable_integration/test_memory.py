"""Harness `Memory` backed by `PixeltableMemoryStore`, next to the `Pixeltable` catalog tools."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pixeltable as pxt
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation
from pydantic_ai_harness.pixeltable import Pixeltable, PixeltableMemoryStore, PixeltableToolset

from .support import create_table, insert_rows, tiny_embed

HANDBOOK = 'The release pipeline publishes wheels from a git clone until a GitHub Release exists.'
NOTEBOOK = 'zzz qqq nnn rrrr'
PARAPHRASE = 'How do we publish this package before PyPI?'


@pytest.fixture
def root() -> Iterator[str]:
    name = f'harness_pxt_pat_{uuid.uuid4().hex[:8]}'
    pxt.create_dir(name)
    yield name
    pxt.drop_dir(name, force=True)


def _memory_context(messages: list[ModelMessage]) -> str:
    contexts = [
        content.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for content in part.content
        if isinstance(content, TextContent) and content.content.startswith('<memory>\n')
    ]
    return contexts[-1] if contexts else ''


class TestPixeltableMemory:
    async def test_memory_notebook_write_and_inject(self, root: str) -> None:
        store = PixeltableMemoryStore(table_name=f'{root}.memory')

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if 'uv' not in _memory_context(messages):
                return ModelResponse(
                    parts=[ToolCallPart('write_memory', {'content': '- user prefers uv'}, tool_call_id='w1')]
                )
            return ModelResponse(parts=[TextPart('injected-ok')])

        agent = Agent(FunctionModel(model), capabilities=[Memory(store)])
        # The snapshot is refreshed before every model request, so the write shows up in the same run.
        first = await agent.run('Remember that I prefer uv.')
        assert first.output == 'injected-ok'
        file = await store.read('main/MEMORY.md', max_chars=1_000)
        assert file is not None
        assert file.content == '- user prefers uv\n'
        second = await agent.run('What do I prefer?')
        assert second.output == 'injected-ok'

    async def test_retrieve_is_catalog_not_memory_notebook(self, root: str) -> None:
        # A paraphrase finds the handbook through the embedding index; the lexical notebook
        # search is not a retrieval substitute.
        store = PixeltableMemoryStore(table_name=f'{root}.memory')
        await store.write('main/MEMORY.md', NOTEBOOK, expected_version=None)
        chunks = create_table(f'{root}.chunks', {'text': pxt.String})
        insert_rows(chunks, [{'text': HANDBOOK}])
        chunks.add_embedding_index('text', string_embed=tiny_embed)
        tools = PixeltableToolset(tables=[f'{root}.chunks'], max_rows=5, max_chars=4000)

        hit = tools.similarity_search(f'{root}.chunks', HANDBOOK, 'text', limit=1)
        assert hit['rows'][0]['text'] == HANDBOOK
        lexical = await store.search('', PARAPHRASE, limit=10, max_files=10, max_chars=400, max_file_chars=1_000)
        assert lexical.matches == []

    def test_equality_lookup_support_ticket_shape(self, root: str) -> None:
        tickets = create_table(f'{root}.tickets', {'status': pxt.String, 'note': pxt.String})
        insert_rows(
            tickets,
            [
                {'status': 'open', 'note': 'reset password'},
                {'status': 'closed', 'note': 'old ticket'},
                {'status': 'open', 'note': 'wire delay'},
            ],
        )
        tools = PixeltableToolset(tables=[f'{root}.tickets'], max_rows=5, max_chars=4000)
        result = tools.query_table(f'{root}.tickets', where={'status': 'open'})
        assert {row['note'] for row in result['rows']} == {'reset password', 'wire delay'}

    async def test_receipt_gap_replay_conflicts(self, root: str) -> None:
        # Without its receipt, a replayed create sees the file it made and cannot tell it apart
        # from someone else's, so it conflicts instead of overwriting.
        store = PixeltableMemoryStore(table_name=f'{root}.memory')
        operation = MemoryOperation(id='run-1:call-1', fingerprint='write:notes/main.md:one')
        first = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        assert not first.replayed
        t = store.table
        t.delete(where=t.path == f'__op__/{operation.id}')
        with pytest.raises(MemoryConflictError):
            await store.write('notes/main.md', 'one', expected_version=None, operation=operation)

    def test_memory_spec_backend_cannot_be_pixeltable(self) -> None:
        # A table store needs a constructed instance; the YAML `backend` names built-in stores only.
        with pytest.raises(ValueError, match='backend'):
            Memory.from_spec(backend='pixeltable')  # pyright: ignore[reportArgumentType]


N_FILES = 20_000
N_CHUNKS = 20_000


@pytest.mark.expensive
class TestPixeltableVolume:
    """Local volume checks at 20k rows (about 5 s together)."""

    async def test_memory_prefix_list_and_search(self, root: str) -> None:
        store = PixeltableMemoryStore(table_name=f'{root}.memory')
        rows: list[dict[str, object]] = [
            {
                'path': f'tenant-{"a" if i % 10 else "b"}/n{i:05d}.md',
                'kind': 'file',
                'content': 'alpha note' if i % 10 else 'beta private',
                'version': uuid.uuid4().hex,
                'last_operation_id': None,
                'fingerprint': None,
                'existed': None,
            }
            for i in range(N_FILES)
        ]
        insert_rows(store.table, rows)
        listed = await store.list_paths('tenant-a/', limit=50)
        result = await store.search('tenant-a/', 'alpha', limit=10, max_files=100, max_chars=4_000, max_file_chars=200)
        assert len(listed) == 50
        assert all(path.startswith('tenant-a/') for path in listed)
        assert result.scanned == 100
        assert result.truncated
        assert result.matches
        assert all(match.path.startswith('tenant-a/') for match in result.matches)

    def test_catalog_query_similarity_and_allowlist(self, root: str) -> None:
        chunks = create_table(f'{root}.chunks', {'text': pxt.String, 'status': pxt.String})
        create_table(f'{root}.secret', {'text': pxt.String})
        payload: list[dict[str, object]] = [
            {'text': 'cats sit on mats' if i == 0 else f'row {i} filler', 'status': 'open' if i % 3 == 0 else 'closed'}
            for i in range(N_CHUNKS)
        ]
        insert_rows(chunks, payload)
        chunks.add_embedding_index('text', string_embed=tiny_embed)
        toolset = Pixeltable(tables=[f'{root}.chunks'], max_rows=20).get_toolset()
        assert isinstance(toolset, PixeltableToolset)

        queried = toolset.query_table(f'{root}.chunks', where={'status': 'open'}, limit=50)
        assert queried['truncated']
        assert len(queried['rows']) == 20
        assert all(row['status'] == 'open' for row in queried['rows'])
        similar = toolset.similarity_search(f'{root}.chunks', 'cats sit on mats', 'text', limit=3)
        assert similar['rows'][0]['text'] == 'cats sit on mats'
        with pytest.raises(ModelRetry, match='allowlist'):
            toolset.query_table(f'{root}.secret')
