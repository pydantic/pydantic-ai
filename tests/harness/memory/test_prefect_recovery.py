"""Prefect recovery of a `Memory` run whose snapshot compaction drops mid-run.

A Prefect flow retry runs the flow again, and each task whose result is cached returns it instead of running, such as
`Memory`'s snapshot load. The run here crashes in its first tool call after memory changed, and the retry compacts
the snapshot away during its tool loop and puts it back. It must put back the snapshot the first attempt recorded,
without reading the changed store.
"""

from __future__ import annotations

from pathlib import Path

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
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.memory import InMemoryMemoryStore, Memory, MemoryFile
from tests.conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


class _CountingStore(InMemoryMemoryStore):
    """Counts reads, so a retry that read the store again shows."""

    reads = 0

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        self.reads += 1
        return await super().read(path, max_chars=max_chars)


def _snapshots(messages: list[ModelMessage]) -> list[str]:
    """The memory snapshots in the messages a model was sent, each a `<memory>` frame."""
    return [
        ''.join(item.content for item in part.content if isinstance(item, TextContent))
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and isinstance(first := part.content[0], TextContent)
        and first.content.startswith('<memory>')
    ]


async def test_flow_retry_restores_the_recorded_snapshot_without_reading_the_store(
    prefect_test_server: None, tmp_path: Path
) -> None:
    store = _CountingStore({'main/MEMORY.md': '- version A'})
    # The memory snapshots each model request carried, recorded where the model runs.
    snapshots_sent: list[list[str]] = []
    reads_at_crash: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        snapshots_sent.append(_snapshots(messages))
        if len(snapshots_sent) < 5:
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id=f'call-{len(snapshots_sent)}')])
        return ModelResponse(parts=[TextPart('done')])

    async def lookup() -> str:
        """Look something up. The first call changes memory, then crashes the flow."""
        if not reads_at_crash:
            current = await store.read('main/MEMORY.md', max_chars=100)
            assert current is not None
            await store.write('main/MEMORY.md', '- version B', expected_version=current.version)
            reads_at_crash.append(store.reads)
            raise RuntimeError('worker lost')
        return 'found'

    agent = Agent(
        FunctionModel(model),
        name='memory_recovery',
        tools=[lookup],
        capabilities=[
            SlidingWindowCompaction(max_messages=4, keep_messages=2, preserve_first_user_message=False),
            Memory(store=store),
            PrefectDurability(),
        ],
    )

    @flow(retries=1, retry_delay_seconds=0)
    async def remembering() -> str:
        return (await agent.run('what do I like?')).output

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        assert await remembering() == 'done'

    # The retry replays the first request, so the model runs once on the first attempt and four times on the retry.
    # Compaction drops the snapshot during the retry's tool loop; every request still sends exactly the one recorded.
    assert len(snapshots_sent) == 5
    [snapshot] = snapshots_sent[0]
    assert '- version A' in snapshot
    assert all(sent == [snapshot] for sent in snapshots_sent)
    # Recovery restores the recorded snapshot load instead of reading the changed store.
    assert store.reads == reads_at_crash[0]
