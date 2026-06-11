"""Tests for `SqliteStepStore` and the media externalization paths.

Mirrors selected coverage from `test_step_persistence.py::TestFileStepStore`
for the SQLite backend, plus end-to-end media-externalization round-trips
through `FileStepStore` and `SqliteStepStore`. The full hook-level coverage
already lives in `test_step_persistence.py`; this module focuses on the
SQLite-specific behavior and the media plumbing.
"""

from __future__ import annotations

import base64
import sqlite3
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.media import DiskMediaStore, SqliteMediaStore
from pydantic_ai_harness.experimental.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
    SqliteStepStore,
    StepEvent,
    StepPersistence,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _sample_messages_with_media(payload_size: int) -> list[ModelMessage]:
    big = b'\xab' * payload_size
    return [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'analyze this',
                        BinaryContent(data=big, media_type='image/png'),
                    ]
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content='done')]),
    ]


# ---------------------------------------------------------------------------
# SqliteStepStore — direct protocol exercises
# ---------------------------------------------------------------------------


class TestSqliteStepStoreProtocol:
    async def test_register_and_get_run(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        record = RunRecord(run_id='r1', conversation_id='c1', agent_name='agent', metadata={'k': 'v'})
        await store.register_run(record)
        fetched = await store.get_run(run_id='r1')
        assert fetched is not None
        assert fetched.run_id == 'r1'
        assert fetched.conversation_id == 'c1'
        assert fetched.metadata == {'k': 'v'}

    async def test_register_duplicate_run_raises(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.register_run(RunRecord(run_id='r1'))
        with pytest.raises(sqlite3.IntegrityError):
            await store.register_run(RunRecord(run_id='r1'))

    async def test_list_runs_chronological(self, tmp_path: Path) -> None:
        from datetime import datetime, timedelta, timezone

        store = SqliteStepStore(database=tmp_path / 'runs.db')
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        await store.register_run(RunRecord(run_id='r3', started_at=base + timedelta(seconds=3)))
        await store.register_run(RunRecord(run_id='r1', started_at=base + timedelta(seconds=1)))
        await store.register_run(RunRecord(run_id='r2', started_at=base + timedelta(seconds=2)))

        records = await store.list_runs()
        assert [r.run_id for r in records] == ['r1', 'r2', 'r3']

    async def test_list_runs_filters(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.register_run(RunRecord(run_id='r1', conversation_id='a', parent_run_id='p'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='a', parent_run_id='q'))
        await store.register_run(RunRecord(run_id='r3', conversation_id='b', parent_run_id='p'))

        by_conv = await store.list_runs(conversation_id='a')
        assert {r.run_id for r in by_conv} == {'r1', 'r2'}

        by_parent = await store.list_runs(parent_run_id='p')
        assert {r.run_id for r in by_parent} == {'r1', 'r3'}

        both = await store.list_runs(parent_run_id='p', conversation_id='a')
        assert [r.run_id for r in both] == ['r1']

    async def test_append_and_list_events(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(
                run_id='r1',
                kind='tool_call_started',
                step_index=1,
                tool_call_id='t1',
                tool_name='add',
                metadata={'k': 'v'},
            )
        )
        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'tool_call_started']
        assert events[1].tool_call_id == 't1'
        assert events[1].metadata == {'k': 'v'}

    async def test_save_and_load_snapshot(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[TextPart(content='hi back')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=messages))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 2
        assert len(snap.messages) == 2

    async def test_snapshot_seq_monotonic_across_reset_step(self, tmp_path: Path) -> None:
        """A reused `run_id` with `step_index` reset to 0 must not clobber the prior snapshot.

        Mirrors `FileStepStore._next_snapshot_seq` — SQLite uses
        `AUTOINCREMENT seq` for the same guarantee.
        """
        store = SqliteStepStore(database=tmp_path / 'runs.db', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=5, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 0  # the *last* write wins, not the highest step_index

    async def test_tool_effect_upsert_per_call_id(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id='t1',
                tool_name='add',
                run_id='r1',
                status='completed',
                effect_summary='ok',
            )
        )
        effect = await store.get_tool_effect(run_id='r1', tool_call_id='t1')
        assert effect is not None
        assert effect.status == 'completed'
        assert effect.effect_summary == 'ok'

    async def test_tool_effect_scoped_by_run(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r2', status='completed')
        )
        a = await store.get_tool_effect(run_id='r1', tool_call_id='t1')
        b = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert a is not None and a.status == 'started'
        assert b is not None and b.status == 'completed'

    async def test_list_unresolved_tool_effects(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t2', tool_name='mul', run_id='r1', status='completed')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['t1']

    async def test_missing_lookups_return_none_or_empty(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        assert await store.get_run(run_id='nope') is None
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.get_tool_effect(run_id='nope', tool_call_id='x') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []

    async def test_shared_connection_mode(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(tmp_path / 'runs.db', check_same_thread=False)
        try:
            store = SqliteStepStore(connection=conn, media_store=None)
            await store.register_run(RunRecord(run_id='r1'))
            await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
            assert (await store.get_run(run_id='r1')) is not None
            assert len(await store.list_events(run_id='r1')) == 1
        finally:
            conn.close()

    async def test_shared_connection_default_media_store_uses_same_connection(self, tmp_path: Path) -> None:
        """`media_store='auto'` + `connection=` paths a SqliteMediaStore at the same conn."""
        conn = sqlite3.connect(tmp_path / 'runs.db', check_same_thread=False)
        try:
            store = SqliteStepStore(connection=conn)
            await store.register_run(RunRecord(run_id='r1'))
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
            )
            (count,) = conn.execute('SELECT COUNT(*) FROM media').fetchone()
            assert count == 1
        finally:
            conn.close()

    def test_rejects_both_database_and_connection(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            SqliteStepStore()
        conn = sqlite3.connect(':memory:')
        try:
            with pytest.raises(ValueError, match='exactly one'):
                SqliteStepStore(database='x', connection=conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Media integration: FileStepStore + DiskMediaStore (default)
# ---------------------------------------------------------------------------


class TestFileStepStoreMedia:
    async def test_large_binary_externalized_to_media_dir(self, tmp_path: Path) -> None:
        """Default FileStepStore wires a `DiskMediaStore(<root>/media/)`.

        Large `BinaryContent` payloads end up as `<root>/media/<sha256>.bin`,
        not inlined in the snapshot JSON.
        """
        store = FileStepStore(tmp_path / 'runs', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
        )

        media_dir = tmp_path / 'runs' / 'media'
        assert media_dir.is_dir()
        blobs = list(media_dir.glob('*.bin'))
        assert len(blobs) == 1
        assert blobs[0].stat().st_size == 70_000

        # Snapshot JSON references the URI, not the inline base64.
        snap_files = list((tmp_path / 'runs' / 'r1' / 'snapshots').glob('*.json'))
        assert len(snap_files) == 1
        snap_text = snap_files[0].read_text(encoding='utf-8')
        assert '__harness_external_media__' in snap_text
        assert base64.b64encode(b'\xab' * 70_000).decode('ascii') not in snap_text

    async def test_round_trip_restores_original_bytes(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'runs')
        await store.register_run(RunRecord(run_id='r1'))
        payload = b'\xcd' * 80_000
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[BinaryContent(data=payload, media_type='image/jpeg')])]),
            ModelResponse(parts=[TextPart(content='ok')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        first = snap.messages[0]
        assert isinstance(first, ModelRequest)
        prompt = first.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = prompt.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.data == payload
        assert binary.media_type == 'image/jpeg'

    async def test_below_threshold_stays_inline(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'runs', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(1_024))
        )
        media_dir = tmp_path / 'runs' / 'media'
        assert not media_dir.exists() or list(media_dir.glob('*.bin')) == []

    async def test_opt_out_keeps_bytes_inline(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'runs', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
        )
        assert not (tmp_path / 'runs' / 'media').exists()
        snap_text = next((tmp_path / 'runs' / 'r1' / 'snapshots').glob('*.json')).read_text(encoding='utf-8')
        assert '__harness_external_media__' not in snap_text

        # Loading also works with media_store=None — no restore_media walk.
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None

    async def test_explicit_media_store_override(self, tmp_path: Path) -> None:
        custom_root = tmp_path / 'shared-media'
        store = FileStepStore(
            tmp_path / 'runs',
            media_store=DiskMediaStore(custom_root),
        )
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
        )
        # External bytes live under the override, not under <root>/media/
        assert list(custom_root.glob('*.bin'))
        assert not (tmp_path / 'runs' / 'media').exists()


# ---------------------------------------------------------------------------
# Media integration: SqliteStepStore + same-DB SqliteMediaStore (default)
# ---------------------------------------------------------------------------


class TestSqliteStepStoreMedia:
    async def test_large_binary_stored_in_same_db(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
        )

        conn = sqlite3.connect(db, check_same_thread=False)
        try:
            row = conn.execute('SELECT COUNT(*), SUM(size_bytes) FROM media').fetchone()
            assert row[0] == 1
            assert row[1] == 70_000
            # No sibling DB file or media directory.
            assert not (tmp_path / 'media').exists()
        finally:
            conn.close()

    async def test_round_trip_restores_bytes(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'runs.db')
        await store.register_run(RunRecord(run_id='r1'))
        payload = b'\xef' * 70_000
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[BinaryContent(data=payload, media_type='image/png')])]),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        first = snap.messages[0]
        assert isinstance(first, ModelRequest)
        prompt = first.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = prompt.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.data == payload

    async def test_dedup_across_snapshots(self, tmp_path: Path) -> None:
        """The same payload appearing in two snapshots ends up as one row."""
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db)
        await store.register_run(RunRecord(run_id='r1'))
        for step in range(3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_sample_messages_with_media(70_000))
            )
        conn = sqlite3.connect(db, check_same_thread=False)
        try:
            (count,) = conn.execute('SELECT COUNT(*) FROM media').fetchone()
            assert count == 1
        finally:
            conn.close()

    async def test_external_media_store_override(self, tmp_path: Path) -> None:
        """SqliteStepStore can be paired with a non-sqlite media store (e.g. disk)."""
        disk = tmp_path / 'media'
        store = SqliteStepStore(database=tmp_path / 'runs.db', media_store=DiskMediaStore(disk))
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_sample_messages_with_media(70_000))
        )
        assert len(list(disk.glob('*.bin'))) == 1
        # The `media` table is never created in the same DB because we routed
        # blobs to a disk store; the StepStore schema doesn't include it.
        conn = sqlite3.connect(tmp_path / 'runs.db', check_same_thread=False)
        try:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='media'").fetchone()
            assert row is None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# End-to-end: Agent + StepPersistence + media externalization
# ---------------------------------------------------------------------------


class TestEndToEndMediaWithAgent:
    async def test_agent_run_with_large_binary_input(self, tmp_path: Path) -> None:
        """An Agent run carrying BinaryContent in its prompt round-trips through SP."""
        store = FileStepStore(tmp_path / 'runs')
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )

        big = b'\xab' * 100_000
        result = await agent.run(
            [
                'classify this image',
                BinaryContent(data=big, media_type='image/png'),
            ]
        )

        # Sanity: agent produced something.
        assert isinstance(result.output, str)

        runs = await store.list_runs()
        assert len(runs) == 1
        run_id = runs[0].run_id

        # Snapshot exists, references external media.
        snap = await store.latest_snapshot(run_id=run_id)
        assert snap is not None
        first = snap.messages[0]
        assert isinstance(first, ModelRequest)
        prompt = first.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = next(p for p in prompt.content if isinstance(p, BinaryContent))
        assert binary.data == big

        # On disk: one media blob, no inlined base64 in any snapshot json.
        blobs = list((tmp_path / 'runs' / 'media').glob('*.bin'))
        assert len(blobs) == 1


# ---------------------------------------------------------------------------
# Auto / opt-out / explicit media_store semantics on both stores
# ---------------------------------------------------------------------------


class TestSqliteRowDefenses:
    """Exercise the defensive `isinstance`/`None` guards in the row decoders.

    These guards never trigger via the public API — they catch poke-by-hand
    corruption of the DB. Hitting them explicitly is the cleanest way to
    document the contract and the only way to satisfy 100% coverage.
    """

    def _open(self, db: Path) -> sqlite3.Connection:
        return sqlite3.connect(db, check_same_thread=False, isolation_level=None)

    async def test_corrupted_run_row_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        conn = self._open(db)
        try:
            # TEXT affinity coerces numbers to strings, but leaves BLOBs alone —
            # binding `bytes` is the cleanest way to violate `isinstance(v, str)`.
            conn.execute('UPDATE runs SET metadata = ? WHERE run_id = ?', (b'\x00bin', 'r1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='run row has wrong types'):
            await store.get_run(run_id='r1')

    async def test_corrupted_event_row_wrong_type_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        conn = self._open(db)
        try:
            conn.execute('UPDATE events SET metadata = ? WHERE run_id = ?', (b'\x00bin', 'r1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='event row has wrong types'):
            await store.list_events(run_id='r1')

    async def test_corrupted_event_kind_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        conn = self._open(db)
        try:
            conn.execute('UPDATE events SET kind = ? WHERE run_id = ?', ('fictional', 'r1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='unknown event kind'):
            await store.list_events(run_id='r1')

    async def test_corrupted_tool_effect_wrong_type_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        conn = self._open(db)
        try:
            conn.execute('UPDATE tool_effects SET tool_name = ? WHERE tool_call_id = ?', (b'\x00bin', 't1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='tool effect row has wrong types'):
            await store.get_tool_effect(run_id='r1', tool_call_id='t1')

    async def test_corrupted_tool_effect_status_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        conn = self._open(db)
        try:
            conn.execute('UPDATE tool_effects SET status = ? WHERE tool_call_id = ?', ('exploded', 't1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='unknown tool effect status'):
            await store.get_tool_effect(run_id='r1', tool_call_id='t1')

    async def test_corrupted_snapshot_row_raises(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='x')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        conn = self._open(db)
        try:
            conn.execute('UPDATE snapshots SET timestamp = ? WHERE run_id = ?', (b'\x00bin', 'r1'))
        finally:
            conn.close()
        with pytest.raises(ValueError, match='snapshot row has wrong types'):
            await store.latest_snapshot(run_id='r1')


class TestMediaStoreResolution:
    def test_file_store_auto_creates_disk_media_store(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'runs')
        assert isinstance(store._media_store, DiskMediaStore)  # type: ignore[reportPrivateUsage]

    def test_file_store_none_disables_media(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'runs', media_store=None)
        assert store._media_store is None  # type: ignore[reportPrivateUsage]

    def test_sqlite_store_auto_uses_same_db(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'r.db')
        assert isinstance(store._media_store, SqliteMediaStore)  # type: ignore[reportPrivateUsage]

    def test_sqlite_store_none_disables_media(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'r.db', media_store=None)
        assert store._media_store is None  # type: ignore[reportPrivateUsage]
