"""Conversation content and decorative metadata have independent concurrency boundaries."""

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, SqliteStepStore, StepEvent
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SqliteConversationStore,
    conversation_text,
    ensure_inactive,
)


async def test_roundtrip_search_paging_and_media(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('repair renderer')]),
        ModelResponse(parts=[TextPart('result ' * 12000)]),
        ModelRequest(parts=[UserPromptPart([BinaryContent(data=b'\xff' * 70000, media_type='image/png')])]),
    ]
    first = await store.save(summary=ConversationSummary(workspace='/a', title='Renderer'), messages=messages)
    second = await store.save(summary=ConversationSummary(workspace='/b', title='Other'), messages=[])
    reopened = SqliteConversationStore(database=store.database)
    loaded = await reopened.get(conversation_id=first.id)
    assert ModelMessagesTypeAdapter.dump_json(loaded.messages) == ModelMessagesTypeAdapter.dump_json(messages)
    assert loaded.summary.revision == 1
    assert loaded.summary.message_count == 3
    assert [e.id for e in await reopened.listing(limit=1)] == [second.id]
    assert [e.id for e in await reopened.listing(limit=1, offset=1)] == [first.id]
    assert [e.id for e in await reopened.listing(query='REPAIR')] == [first.id]
    assert await reopened.listing(query='not present') == []
    assert conversation_text(messages).startswith('user: repair renderer')
    with closing(sqlite3.connect(store.database)) as conn:
        assert conn.execute('SELECT count(*) FROM media').fetchone()[0] >= 2


async def test_content_and_naming_compare_and_swap(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    source = ConversationSummary(workspace='/a', title='Fallback')
    first = await store.save(summary=source, messages=[])
    assert await store.name(source=first, title='Generated', tags=('sqlite',), tokens=12)
    # Content writes preserve a concurrent metadata-only update.
    second = await store.save(summary=first, messages=[ModelRequest(parts=[UserPromptPart('new')])])
    assert second.title == 'Generated'
    assert second.naming_tokens == 12
    assert second.naming_version == 1
    assert not await store.name(source=first, title='Stale')
    with pytest.raises(ConversationConflict, match='another process'):
        await store.save(summary=first, messages=[])
    assert await store.name(source=second, title='Mine', manual=True)
    current = (await store.get(conversation_id=first.id)).summary
    assert current.updated_at == second.updated_at
    assert current.revision == second.revision
    assert not await store.name(source=current, title='Overwrite manual')
    assert not await store.name(source=second, title='Older manual', manual=True)
    assert await store.name(source=current, title='Mine again', manual=True)
    await store.delete(source=current)
    assert not await store.name(source=current, title='Resurrect', manual=True)
    with pytest.raises(LookupError, match='No saved session'):
        await store.get(conversation_id=first.id)
    with pytest.raises(ConversationConflict, match='deleted'):
        await store.save(summary=current, messages=[])
    with pytest.raises(ConversationConflict):
        await store.delete(source=current)


async def test_stale_delete_does_not_erase_new_content(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    second = await store.save(summary=first, messages=[])
    with pytest.raises(ConversationConflict):
        await store.delete(source=first)
    assert (await store.get(conversation_id=first.id)).summary == second
    for limit, offset in [(0, 0), (1, -1)]:
        with pytest.raises(ValueError):
            await store.listing(limit=limit, offset=offset)


async def test_corrupt_history_is_not_an_empty_session(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    with closing(sqlite3.connect(store.database)) as conn:
        conn.execute('UPDATE conversations SET messages=?', ('not json',))
        conn.commit()
    with pytest.raises(ValueError):
        await store.get(conversation_id=first.id)


def test_live_owner_is_not_a_recovery_point(monkeypatch: pytest.MonkeyPatch) -> None:
    running = ConversationSummary(workspace='/a', outcome='running', owner_pid=os.getpid())
    with pytest.raises(ConversationConflict, match='busy'):
        ensure_inactive(running)
    ensure_inactive(replace(running, outcome='completed'))
    ensure_inactive(replace(running, owner_pid=None))

    def gone(pid: int, signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, 'kill', gone)
    ensure_inactive(running)

    def denied(pid: int, signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(os, 'kill', denied)
    with pytest.raises(ConversationConflict):
        ensure_inactive(running)


async def test_delete_removes_run_data_but_not_other_conversations(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    second = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    steps = SqliteStepStore(database=store.database)
    for record in (first, second):
        await steps.register_run(RunRecord(run_id=record.id, conversation_id=record.id))
        await steps.append_event(StepEvent(run_id=record.id, kind='run_started', step_index=0))
        await steps.save_snapshot(ContinuableSnapshot(run_id=record.id, step_index=0, messages=[]))
    await store.delete(source=first)
    assert await steps.get_run(run_id=first.id) is None
    assert await steps.latest_snapshot(run_id=first.id) is None
    assert await steps.list_events(run_id=first.id) == []
    assert await steps.latest_snapshot(run_id=second.id) is not None
    assert len(await store.listing()) == 1


async def test_stale_naming_usage_is_still_accounted(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    second = await store.save(summary=first, messages=[])
    assert not await store.name(source=first, title='Too late', tokens=123)
    current = (await store.get(conversation_id=first.id)).summary
    assert current.title == second.title
    assert current.revision == second.revision
    assert current.naming_tokens == 123


async def test_unicode_and_multimodal_search(tmp_path: Path) -> None:

    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    messages = [
        ModelRequest(
            parts=[UserPromptPart(['école', TextContent('Straße'), BinaryContent(data=b'x', media_type='image/png')])]
        )
    ]
    saved = await store.save(
        summary=ConversationSummary(workspace='/a', title='Éducation', tags=('Übung',)), messages=messages
    )
    assert conversation_text(messages) == 'user: école\nuser: Straße'
    assert conversation_text([ModelRequest(parts=[ToolReturnPart('tool', 'secret', tool_call_id='x')])]) == ''
    for query in ('ÉCOLE', 'STRASSE', 'éDUCATION', 'üBUNG'):
        assert [entry.id for entry in await store.listing(query=query)] == [saved.id]


@pytest.mark.parametrize(
    'output,busy',
    [('"python.exe","123","Console","1","2 K"\n', True), ('INFO: no tasks\n', False), ('"other.exe","456"\n', False)],
)
def test_windows_owner_probe(monkeypatch: pytest.MonkeyPatch, output: str, busy: bool) -> None:

    def tasklist(
        args: list[str], *, capture_output: bool, text: bool, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        assert args == [r'C:\Windows\System32\tasklist.exe', '/FI', 'PID eq 123', '/FO', 'CSV', '/NH']
        assert capture_output and text and check and timeout == 10
        return subprocess.CompletedProcess(args, 0, stdout=output)

    monkeypatch.setenv('SystemRoot', r'C:\Windows')
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(subprocess, 'run', tasklist)
    running = ConversationSummary(workspace='/a', outcome='running', owner_pid=123)
    if busy:
        with pytest.raises(ConversationConflict, match='busy'):
            ensure_inactive(running)
    else:
        ensure_inactive(running)


async def test_delete_with_only_run_catalog(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    saved = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])
    with closing(sqlite3.connect(store.database)) as conn:
        conn.execute('CREATE TABLE runs (run_id TEXT, conversation_id TEXT)')
        conn.commit()
    await store.delete(source=saved)
    assert await store.listing() == []


@pytest.mark.parametrize('root', ['', '.', r'C:Windows', r'\Windows'])
def test_windows_probe_rejects_relative_system_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    monkeypatch.setenv('SystemRoot', root)
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(ValueError, match='absolute Windows path'):
        ensure_inactive(ConversationSummary(workspace='/a', outcome='running', owner_pid=123))
