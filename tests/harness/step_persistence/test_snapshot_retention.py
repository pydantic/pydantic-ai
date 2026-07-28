"""Tests for the opt-in `max_snapshots_per_run` retention bound.

Covers the shared retain invariant through the public `save_snapshot` path on
all three shipped stores, the per-backend pruning idioms (list rebuild,
`unlink`, indexed `DELETE`), constructor validation, `from_spec` forwarding,
and one end-to-end `Agent` run with the bound set.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    SqliteStepStore,
    StepPersistence,
    StepStore,
    continue_run,
)
from pydantic_ai_harness.step_persistence._types import SnapshotState

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


BACKENDS = ['memory', 'file', 'sqlite']


def _make_store(kind: str, tmp_path: Path, keep: int | None) -> tuple[StepStore, Callable[[str], int]]:
    """Build a bounded store plus a callable that counts a run's stored snapshots."""
    if kind == 'memory':
        mem = InMemoryStepStore(max_snapshots_per_run=keep)
        return mem, lambda run_id: len(mem._snapshots.get(run_id, []))
    if kind == 'file':
        root = tmp_path / 'runs'
        return FileStepStore(root, max_snapshots_per_run=keep), lambda run_id: len(
            list((root / run_id / 'snapshots').glob('*.json'))
        )
    db = tmp_path / 'runs.db'

    def count(run_id: str) -> int:
        conn = sqlite3.connect(db, check_same_thread=False)
        try:
            (total,) = conn.execute('SELECT COUNT(*) FROM snapshots WHERE run_id = ?', (run_id,)).fetchone()
            return total
        finally:
            conn.close()

    return SqliteStepStore(database=db, media_store=None, max_snapshots_per_run=keep), count


async def _save(store: StepStore, run_id: str, step: int, state: SnapshotState = 'complete') -> None:
    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[TextPart(content=f's{step}')]),
    ]
    await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=step, messages=msgs, state=state))


class TestRetentionBound:
    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_none_keeps_every_snapshot(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, None)
        for step in range(5):
            await _save(store, 'r1', step)
        assert count('r1') == 5

    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_count_under_limit_keeps_all(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, 5)
        for step in range(3):
            await _save(store, 'r1', step)
        assert count('r1') == 3

    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_count_over_limit_prunes_to_window(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, 2)
        for step in range(4):
            await _save(store, 'r1', step)
        # All complete: retain set collapses to the newest snapshot's window.
        assert count('r1') == 2
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 3

    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_tail_all_interrupted_retains_complete_below_window(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, 2)
        await _save(store, 'r1', 0, state='complete')
        await _save(store, 'r1', 1, state='complete')
        await _save(store, 'r1', 2, state='interrupted')
        await _save(store, 'r1', 3, state='interrupted')
        # top-2 = {step2, step3} (both interrupted); + newest overall (step3);
        # + newest complete (step1, below the window). step0 is evicted.
        assert count('r1') == 3

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None
        assert default.state == 'complete'
        assert default.step_index == 1

        frontier = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert frontier is not None
        assert frontier.state == 'interrupted'
        assert frontier.step_index == 3

    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_keep_one_with_newest_interrupted_keeps_two(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, 1)
        await _save(store, 'r1', 0, state='complete')
        await _save(store, 'r1', 1, state='interrupted')
        # N=1 correctness floor: the window (step1) is interrupted, so the
        # newest complete (step0) is still retained -- two rows, not one.
        assert count('r1') == 2
        default = await store.latest_snapshot(run_id='r1')
        assert default is not None
        assert default.step_index == 0

    @pytest.mark.parametrize('kind', BACKENDS)
    async def test_all_interrupted_no_complete_retained(self, kind: str, tmp_path: Path) -> None:
        store, count = _make_store(kind, tmp_path, 2)
        for step in range(3):
            await _save(store, 'r1', step, state='interrupted')
        assert count('r1') == 2
        assert await store.latest_snapshot(run_id='r1') is None
        frontier = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert frontier is not None
        assert frontier.step_index == 2

    @pytest.mark.parametrize('kind', BACKENDS)
    def test_rejects_bound_below_one(self, kind: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='max_snapshots_per_run must be >= 1'):
            _make_store(kind, tmp_path, 0)

    @pytest.mark.parametrize('kind', BACKENDS)
    def test_rejects_non_int_bound(self, kind: str, tmp_path: Path) -> None:
        # A positive non-int passes the `< 1` floor but breaks pruning after a
        # write (slice `TypeError` / bad SQL bind), so reject it at construction.
        with pytest.raises(ValueError, match='max_snapshots_per_run must be an int'):
            _make_store(kind, tmp_path, 1.5)  # pyright: ignore[reportArgumentType]


class TestFileStorePruneEdgeCases:
    async def test_non_numeric_snapshot_file_is_ignored(self, tmp_path: Path) -> None:
        """A stray non-`{seq}.json` file in the snapshots dir is skipped, not deleted."""
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        await _save(store, 'r1', 0)
        stray = root / 'r1' / 'snapshots' / 'notanumber.json'
        stray.write_text('{}', encoding='utf-8')
        await _save(store, 'r1', 1)
        await _save(store, 'r1', 2)

        assert stray.exists()
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 2

    async def test_corrupt_sibling_file_does_not_block_bounded_save(self, tmp_path: Path) -> None:
        """A truncated `{seq}.json` sibling is skipped; the bounded save still succeeds and prunes."""
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        await _save(store, 'r1', 0)
        # Corrupt an existing numeric snapshot file: parseable filename, unparsable body.
        (root / 'r1' / 'snapshots' / '0.json').write_text('{ truncated', encoding='utf-8')

        # These would raise out of save_snapshot if prune let the parse error escape.
        await _save(store, 'r1', 1)
        await _save(store, 'r1', 2)

        # The corrupt file is neither retained-counted nor deleted; parseable ones pruned to the bound.
        assert (root / 'r1' / 'snapshots' / '0.json').exists()
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 2

    async def test_malformed_sibling_does_not_evict_valid_complete_snapshot(self, tmp_path: Path) -> None:
        """A JSON-valid but fieldless sibling must not be trusted as the newest complete snapshot.

        `{"state": "complete"}` parses and reads as `complete`, but lacks
        `messages`/`timestamp`/`step_index`. If the prune trusted it, it would
        become the retained newest `complete` and evict the real older complete
        snapshot, after which `latest_snapshot()` would fail on the retained
        malformed file instead of returning the valid one. Both the prune and
        the read path must skip it.
        """
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        await _save(store, 'r1', 0, state='complete')  # valid complete, seq 0
        snap_dir = root / 'r1' / 'snapshots'
        # Higher-seq malformed sibling: valid JSON, `complete`, but no fields.
        (snap_dir / '1.json').write_text('{"state": "complete"}', encoding='utf-8')
        # A valid interrupted snapshot (seq 2) becomes the newest overall, so the
        # malformed file would be the highest-seq `complete` if it were trusted.
        await _save(store, 'r1', 2, state='interrupted')  # triggers the prune

        assert (snap_dir / '0.json').exists()
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.state == 'complete'
        assert latest.step_index == 0

    async def test_prune_skips_candidate_that_vanishes_during_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concurrent prune can unlink a candidate between this prune's glob and its read.

        `_sync_prune_snapshots` enumerates every `{seq}.json`, then reads each
        to learn its state. If an overlapping bounded save's prune unlinks a
        non-retained candidate in that window, the parse must skip it rather
        than let `FileNotFoundError` fail a save whose new snapshot already
        landed.
        """
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        # Only one save before arming: 0.json is still present (prune is a no-op
        # at one file), so the next save's prune enumerates it and reads it.
        await _save(store, 'r1', 0)

        vanishing = root / 'r1' / 'snapshots' / '0.json'
        original_read_text = Path.read_text

        def racing_read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
            if self == vanishing and vanishing.exists():
                vanishing.unlink()
                raise FileNotFoundError(vanishing)
            return original_read_text(self, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, 'read_text', racing_read_text)

        # This save's prune enumerates {0,1} and reads 0.json, which vanishes
        # mid-parse. The parse must skip it rather than raise.
        await _save(store, 'r1', 1)

        assert not vanishing.exists()
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 1

    async def test_prune_tolerates_candidate_unlinked_by_concurrent_prune(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A candidate can be deleted by a concurrent prune between enumeration and this prune's unlink.

        Without `missing_ok`, the second prune's `unlink` on an already-removed
        file raises `FileNotFoundError` after the new snapshot has been written.
        """
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        await _save(store, 'r1', 0)
        # Seed a second non-retained sibling so the next save's prune unlinks two
        # files: the raced victim (0.json) and one survivor of the race (1.json).
        snap_dir = root / 'r1' / 'snapshots'
        (snap_dir / '1.json').write_text(
            '{"state": "complete", "step_index": 1, "timestamp": "2026-01-01T00:00:00+00:00", "messages": []}',
            encoding='utf-8',
        )

        victim = snap_dir / '0.json'
        original_unlink = Path.unlink

        def racing_unlink(self: Path, missing_ok: bool = False) -> None:
            if self == victim and victim.exists():
                original_unlink(victim)  # a concurrent prune removed it first
            return original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, 'unlink', racing_unlink)

        # This save's prune unlinks the non-retained 0.json (already removed by
        # the race) and 1.json. `missing_ok` must swallow the vanished 0.json.
        await _save(store, 'r1', 2)

        assert not victim.exists()
        assert not (snap_dir / '1.json').exists()
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 2

    async def test_concurrent_bounded_saves_stay_bounded(self, tmp_path: Path) -> None:
        """Overlapping saves against one bounded run: none raises and the bound holds.

        Each `save_snapshot` dispatches its write+prune to a worker thread, so
        the prunes run in parallel over a shared snapshots dir. This is the
        prune-vs-prune race in the raw: one prune unlinking a candidate the
        other has enumerated must not fail the second save.
        """
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None, max_snapshots_per_run=1)
        await _save(store, 'r1', 0)

        await asyncio.gather(*(_save(store, 'r1', step) for step in range(1, 8)))

        # Retain set upper bound: newest N + newest overall + newest complete.
        remaining = list((root / 'r1' / 'snapshots').glob('*.json'))
        assert len(remaining) <= 3
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None

    async def test_read_skips_candidate_that_vanishes_mid_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concurrent prune can unlink an enumerated candidate before the reader opens it.

        The default read enumerates the interrupted newest, then reads it to
        check its state. If a prune on another worker thread unlinks that file
        between enumeration and read, `latest_snapshot` must skip it and reach
        the retained `complete` snapshot rather than raising `FileNotFoundError`.
        """
        root = tmp_path / 'runs'
        store = FileStepStore(root, media_store=None)
        await _save(store, 'r1', 0, state='complete')
        await _save(store, 'r1', 1, state='interrupted')

        vanishing = root / 'r1' / 'snapshots' / '1.json'
        original_read_text = Path.read_text

        def racing_read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
            if self == vanishing and vanishing.exists():
                vanishing.unlink()
                raise FileNotFoundError(vanishing)
            return original_read_text(self, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, 'read_text', racing_read_text)

        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == 0


class TestSqliteStorePruneEdgeCases:
    async def test_large_bound_does_not_exceed_sql_variable_limit(self, tmp_path: Path) -> None:
        """Pruning with a large bound must not bind one parameter per retained seq.

        SQLite caps host parameters (999 pre-3.32, 32766 after). A retain set
        enumerated as `seq NOT IN (?, ?, ...)` trips that ceiling; the set-based
        DELETE keeps the parameter count constant. Rows are seeded in bulk and a
        single bounded save then triggers the prune that would have overflowed.
        """
        db = tmp_path / 'runs.db'
        keep = 32_766
        store = SqliteStepStore(database=db, media_store=None, max_snapshots_per_run=keep)
        await _save(store, 'r1', 0)  # creates the schema and the first row

        conn = sqlite3.connect(db, check_same_thread=False, isolation_level=None)
        try:
            conn.executemany(
                'INSERT INTO snapshots (run_id, step_index, timestamp, state, messages) VALUES (?, ?, ?, ?, ?)',
                [('r1', step, '2026-01-01T00:00:00+00:00', 'complete', '[]') for step in range(1, keep + 2)],
            )
        finally:
            conn.close()

        # r1 now holds keep+2 rows; this bounded save runs the DELETE that,
        # under a per-seq-parameter scheme, would raise "too many SQL variables".
        await _save(store, 'r1', keep + 2)

        conn = sqlite3.connect(db, check_same_thread=False)
        try:
            (total,) = conn.execute('SELECT COUNT(*) FROM snapshots WHERE run_id = ?', ('r1',)).fetchone()
        finally:
            conn.close()
        assert total == keep
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        assert latest.step_index == keep + 2


class TestFromSpecForwarding:
    async def test_memory_backend_forwards_bound(self) -> None:
        cap = StepPersistence.from_spec(max_snapshots_per_run=3)
        assert isinstance(cap.store, InMemoryStepStore)
        assert cap.store._max_snapshots_per_run == 3

    async def test_file_backend_forwards_bound(self, tmp_path: Path) -> None:
        cap = StepPersistence.from_spec(backend='file', directory=tmp_path, max_snapshots_per_run=4)
        assert isinstance(cap.store, FileStepStore)
        assert cap.store._max_snapshots_per_run == 4

    async def test_sqlite_backend_forwards_bound(self, tmp_path: Path) -> None:
        cap = StepPersistence.from_spec(backend='sqlite', database=str(tmp_path / 'runs.db'), max_snapshots_per_run=5)
        assert isinstance(cap.store, SqliteStepStore)
        assert cap.store._max_snapshots_per_run == 5


class TestBoundedAgentRun:
    async def test_multi_step_run_stays_bounded(self, tmp_path: Path) -> None:
        """A run with several settled tool-call boundaries prunes to the bound."""

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            calls = sum(
                1
                for m in messages
                if isinstance(m, ModelResponse) and any(isinstance(p, ToolCallPart) for p in m.parts)
            )
            if calls < 3:
                return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id=f'c{calls}')])
            return ModelResponse(parts=[TextPart('done')])

        store = InMemoryStepStore(max_snapshots_per_run=1)
        agent: Agent[None, str] = Agent(
            FunctionModel(model),
            capabilities=[StepPersistence(store=store, agent_name='worker')],
        )

        @agent.tool_plain
        def lookup() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'ok'

        result = await agent.run('go')
        assert result.output == 'done'

        run_id = (await store.list_runs())[-1].run_id
        # Three settled CallToolsNode boundaries produced multiple complete
        # snapshots; the bound collapses them to the newest.
        assert len(store._snapshots[run_id]) == 1
        history = await continue_run(store, run_id=run_id)
        assert history
