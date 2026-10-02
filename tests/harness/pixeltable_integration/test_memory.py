"""Harness `Memory` backed by `PixeltableMemoryStore`, next to the `Pixeltable` catalog tools."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pixeltable as pxt
import pytest

from pydantic_ai import Agent
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
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation, PixeltableMemoryStore

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

    async def test_missing_receipt_does_not_reapply_write(self, root: str) -> None:
        store = PixeltableMemoryStore(table_name=f'{root}.memory')
        operation = MemoryOperation(id='run-1:call-1', fingerprint='write:notes/main.md:one')
        first = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        assert not first.replayed
        table = store.table
        table.delete(where=table.path == f'__op__/{operation.id}')
        with pytest.raises(MemoryConflictError):
            await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
