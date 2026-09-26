"""Storage backends for step events, snapshots, and tool-effect records."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

import anyio.to_thread
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pydantic_ai_harness.media import (
    DiskMediaStore,
    MediaStore,
    SqliteMediaStore,
    externalize_media,
    restore_media,
)
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
    ToolEffectStatus,
)

_logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so a concurrent reader sees either the old file or the new one.

    Run and snapshot files are read while a run is still writing -- by a resuming
    worker, or by the `conversation_search` corpus reader. A plain `write_text`
    truncates first, so a reader can observe a half-written file. Writing to a
    uniquely-named temporary sibling and `os.replace`-ing it into position makes
    the swap atomic on POSIX and Windows; the unique name keeps two concurrent
    writers from sharing a temporary. The suffix is `.tmp`, not `.json`, so a
    partial file never enters a `*.json` scan.
    """
    tmp = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_DEFAULT_MEDIA_THRESHOLD_BYTES = 64 * 1024
_AutoMedia = Literal['auto']

_VALID_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,200}$')

_EVENT_KIND_TABLE: dict[str, EventKind] = {
    'run_started': 'run_started',
    'run_completed': 'run_completed',
    'run_failed': 'run_failed',
    'model_request_started': 'model_request_started',
    'model_request_completed': 'model_request_completed',
    'model_request_failed': 'model_request_failed',
    'tool_call_started': 'tool_call_started',
    'tool_call_completed': 'tool_call_completed',
    'tool_call_failed': 'tool_call_failed',
}

_TOOL_STATUS_TABLE: dict[str, ToolEffectStatus] = {
    'started': 'started',
    'completed': 'completed',
    'failed': 'failed',
}


def _validate_id(value: str, *, field: str) -> None:
    r"""Reject identifiers that would let a caller escape the store directory.

    `FileStepStore` interpolates `run_id` into a path. Allowing `..`, `/`,
    `\`, empty input, or oversized values would enable path traversal.
    """
    if not _VALID_ID_RE.fullmatch(value) or '..' in value:
        raise ValueError(f'invalid {field}: {value!r}')


def _validate_max_snapshots(value: object) -> None:
    """Reject a retention bound below the correctness floor or of the wrong type.

    `None` keeps every snapshot. `1` is the smallest bound the retain
    invariant still serves both read modes at; `0` or negatives cannot. A
    non-`int` (e.g. `1.5`) passes the `< 1` check but breaks pruning later
    with a slice `TypeError` or a bad SQL bind, so reject it at construction.
    """
    if value is None:
        return
    if not isinstance(value, int):
        raise ValueError(f'max_snapshots_per_run must be an int >= 1 or None, got {value!r}')
    if value < 1:
        raise ValueError(f'max_snapshots_per_run must be >= 1 or None, got {value!r}')


def _snapshot_key_sequence(key: str) -> int | None:
    """Return the monotonic prefix used by `StepPersistence` snapshot keys."""
    prefix, separator, _ = key.partition(':')
    if not separator:
        return None
    try:
        value = int(prefix)
    except ValueError:
        return None
    return value if value >= 0 else None


def _retained_seqs(entries: list[tuple[int, SnapshotState]], keep: int) -> set[int]:
    """Return the `seq` values to keep when bounding a run to `keep` snapshots.

    The retain set is the newest `keep` by `seq`, plus the newest snapshot
    overall and the newest `complete` snapshot. The last two hold the
    invariant that both read modes stay served:
    `latest_snapshot(include_interrupted=True)` needs the newest overall, and
    the default read needs the newest `complete`. They survive the case where
    the newest `keep` snapshots are all `interrupted` and the newest resumable
    `complete` sits below that window. `entries` need not be sorted.
    """
    if not entries:  # pragma: no cover
        return set()
    by_seq = sorted(entries, key=lambda entry: entry[0])
    retained = {seq for seq, _ in by_seq[-keep:]}
    retained.add(by_seq[-1][0])
    complete_seqs = [seq for seq, state in by_seq if state == 'complete']
    if complete_seqs:
        retained.add(complete_seqs[-1])
    return retained


@runtime_checkable
class StepStore(Protocol):
    """Async protocol for step-persistence backends.

    All methods are async so the same protocol covers in-memory stores and
    file/database stores that would otherwise block the event loop.

    `latest_snapshot` returns the newest snapshot whose `state` is
    `complete`; with `include_interrupted=True` it returns the newest
    snapshot regardless of state (see `SnapshotState`).
    """

    async def register_run(self, record: RunRecord) -> None: ...  # pragma: no cover

    async def get_run(self, *, run_id: str) -> RunRecord | None: ...  # pragma: no cover

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        """Return matching `RunRecord`s sorted by `started_at` ascending.

        Both filters are optional and AND-combine when both are set. The
        chronological ordering is part of the protocol -- callers may pick
        the most recent run with `[-1]`.
        """
        ...  # pragma: no cover

    async def append_event(self, event: StepEvent) -> None: ...  # pragma: no cover

    async def list_events(self, *, run_id: str) -> list[StepEvent]: ...  # pragma: no cover

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None: ...  # pragma: no cover

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False
    ) -> ContinuableSnapshot | None: ...  # pragma: no cover

    async def record_tool_effect(self, record: ToolEffectRecord) -> None: ...  # pragma: no cover

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str
    ) -> ToolEffectRecord | None: ...  # pragma: no cover

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]: ...  # pragma: no cover


class InMemoryStepStore:
    """Process-local store, suitable for tests and single-process orchestrators.

    Pass `max_snapshots_per_run` to bound per-run snapshot growth (see
    `save_snapshot`). `None` keeps every snapshot.
    """

    def __init__(self, *, max_snapshots_per_run: int | None = None) -> None:
        _validate_max_snapshots(max_snapshots_per_run)
        self._max_snapshots_per_run = max_snapshots_per_run
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[StepEvent]] = defaultdict(list)
        self._snapshots: dict[str, list[ContinuableSnapshot]] = defaultdict(list)
        self._snapshot_key_high_water: dict[str, tuple[int, set[str]]] = {}
        self._tool_effects: dict[tuple[str, str], ToolEffectRecord] = {}

    async def register_run(self, record: RunRecord) -> None:
        self._runs[record.run_id] = record

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        records = [
            r
            for r in self._runs.values()
            if (parent_run_id is None or r.parent_run_id == parent_run_id)
            and (conversation_id is None or r.conversation_id == conversation_id)
        ]
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        events = self._events[event.run_id]
        if event.idempotency_key is not None:
            for prior in events:
                if prior.idempotency_key == event.idempotency_key:
                    return
        events.append(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return list(self._events.get(run_id, ()))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        snaps = self._snapshots[snapshot.run_id]
        if snapshot.idempotency_key is not None:
            sequence = _snapshot_key_sequence(snapshot.idempotency_key)
            high_water = self._snapshot_key_high_water.get(snapshot.run_id)
            if sequence is not None and high_water is not None:
                prior_sequence, keys_at_high_water = high_water
                if sequence < prior_sequence or (
                    sequence == prior_sequence and snapshot.idempotency_key in keys_at_high_water
                ):
                    return
            if any(prior.idempotency_key == snapshot.idempotency_key for prior in snaps):
                return
            if sequence is not None:
                # Only keys at the highest sequence are retained, bounding replay suppression
                # independently of snapshot retention while rejecting every older operation.
                if high_water is None or sequence > high_water[0]:
                    self._snapshot_key_high_water[snapshot.run_id] = (sequence, {snapshot.idempotency_key})
                else:
                    high_water[1].add(snapshot.idempotency_key)
        snaps.append(snapshot)
        if self._max_snapshots_per_run is None or len(snaps) <= self._max_snapshots_per_run:
            return
        entries: list[tuple[int, SnapshotState]] = [(index, snap.state) for index, snap in enumerate(snaps)]
        retained = _retained_seqs(entries, self._max_snapshots_per_run)
        self._snapshots[snapshot.run_id] = [snap for index, snap in enumerate(snaps) if index in retained]

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        snaps = self._snapshots.get(run_id)
        if not snaps:
            return None
        for snap in reversed(snaps):
            if include_interrupted or snap.state == 'complete':
                return snap
        return None

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate: `interrupted` snapshots are skipped
        unless `include_interrupted=True`. Not part of the `StepStore` protocol:
        the `conversation_search` capability consumes it through its narrower
        `SnapshotStore` protocol.
        """
        snaps = self._snapshots.get(run_id, ())
        return [snap for snap in snaps if include_interrupted or snap.state == 'complete']

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        self._tool_effects[(record.run_id, record.tool_call_id)] = record

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return self._tool_effects.get((run_id, tool_call_id))

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return [r for r in self._tool_effects.values() if r.run_id == run_id and r.status == 'started']


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'expected str|None, got {type(value).__name__}')
    return value


def _snapshot_state(value: object) -> SnapshotState:
    """Parse a persisted snapshot `state`, defaulting missing values to `complete`.

    Rows and files written before the field existed were all gate-checked
    provider-valid, so `complete` is the correct reading for them.
    """
    if value is None or value == 'complete':
        return 'complete'
    if value == 'interrupted':
        return 'interrupted'
    raise ValueError(f'unknown snapshot state: {value!r}')


def _snapshot_fields_ok(data: dict[str, object]) -> bool:
    """Whether a parsed snapshot file has the fields the loader reads.

    Checks presence of `messages`, `timestamp`, and `step_index` -- the fields
    `_sync_load_latest_snapshot` needs to rebuild a `ContinuableSnapshot`. A
    file that is valid JSON but omits them (e.g. `{"state": "complete"}`) is not
    a snapshot; the prune and the read path skip it so it is never retained
    over, nor returned instead of, a valid snapshot -- matching how they skip a
    file that fails to parse. A file that has the fields but with wrong types is
    a distinct corruption the loader still rejects loudly.
    """
    return 'messages' in data and 'timestamp' in data and 'step_index' in data


_STR_STR_DICT_ADAPTER: TypeAdapter[dict[str, str]] = TypeAdapter(dict[str, str])
_OBJECT_DICT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_STRING_ADAPTER: TypeAdapter[str] = TypeAdapter(str)


def _str_str_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    return _STR_STR_DICT_ADAPTER.validate_python(value)


def _load_json_object(text: str) -> dict[str, object]:
    return _OBJECT_DICT_ADAPTER.validate_json(text)


def _event_to_dict(event: StepEvent) -> dict[str, object]:
    return {
        'run_id': event.run_id,
        'kind': event.kind,
        'step_index': event.step_index,
        'timestamp': event.timestamp.isoformat(),
        'conversation_id': event.conversation_id,
        'parent_run_id': event.parent_run_id,
        'agent_name': event.agent_name,
        'tool_call_id': event.tool_call_id,
        'tool_name': event.tool_name,
        'error': event.error,
        'metadata': dict(event.metadata),
        'idempotency_key': event.idempotency_key,
    }


def _event_from_dict(data: dict[str, object]) -> StepEvent:
    run_id = data['run_id']
    kind_raw = data['kind']
    step_raw = data['step_index']
    timestamp_raw = data['timestamp']
    if not (
        isinstance(run_id, str)
        and isinstance(kind_raw, str)
        and isinstance(step_raw, int)
        and isinstance(timestamp_raw, str)
    ):
        raise ValueError('event payload has wrong types')
    kind = _EVENT_KIND_TABLE.get(kind_raw)
    if kind is None:
        raise ValueError(f'unknown event kind: {kind_raw!r}')
    return StepEvent(
        run_id=run_id,
        kind=kind,
        step_index=step_raw,
        timestamp=datetime.fromisoformat(timestamp_raw),
        conversation_id=_opt_str(data.get('conversation_id')),
        parent_run_id=_opt_str(data.get('parent_run_id')),
        agent_name=_opt_str(data.get('agent_name')),
        tool_call_id=_opt_str(data.get('tool_call_id')),
        tool_name=_opt_str(data.get('tool_name')),
        error=_opt_str(data.get('error')),
        metadata=_str_str_dict(data.get('metadata')),
        idempotency_key=_opt_str(data.get('idempotency_key')),
    )


def _run_to_dict(record: RunRecord) -> dict[str, object]:
    return {
        'run_id': record.run_id,
        'conversation_id': record.conversation_id,
        'parent_run_id': record.parent_run_id,
        'agent_name': record.agent_name,
        'metadata': dict(record.metadata),
        'started_at': record.started_at.isoformat(),
        'registration_id': record.registration_id,
    }


def _run_from_dict(data: dict[str, object]) -> RunRecord:
    run_id = data['run_id']
    started_at_raw = data['started_at']
    if not (isinstance(run_id, str) and isinstance(started_at_raw, str)):
        raise ValueError('run record has wrong types')
    return RunRecord(
        run_id=run_id,
        conversation_id=_opt_str(data.get('conversation_id')),
        parent_run_id=_opt_str(data.get('parent_run_id')),
        agent_name=_opt_str(data.get('agent_name')),
        metadata=_str_str_dict(data.get('metadata')),
        started_at=datetime.fromisoformat(started_at_raw),
        registration_id=_opt_str(data.get('registration_id')),
    )


def _tool_effect_to_dict(record: ToolEffectRecord) -> dict[str, object]:
    return {
        'tool_call_id': record.tool_call_id,
        'tool_name': record.tool_name,
        'run_id': record.run_id,
        'status': record.status,
        'started_at': record.started_at.isoformat(),
        'ended_at': record.ended_at.isoformat() if record.ended_at is not None else None,
        'idempotency_key': record.idempotency_key,
        'effect_summary': record.effect_summary,
    }


def _tool_effect_from_dict(data: dict[str, object]) -> ToolEffectRecord:
    tool_call_id = data['tool_call_id']
    tool_name = data['tool_name']
    run_id = data['run_id']
    status_raw = data['status']
    started_at_raw = data['started_at']
    if not (
        isinstance(tool_call_id, str)
        and isinstance(tool_name, str)
        and isinstance(run_id, str)
        and isinstance(status_raw, str)
        and isinstance(started_at_raw, str)
    ):
        raise ValueError('tool effect record has wrong types')
    status = _TOOL_STATUS_TABLE.get(status_raw)
    if status is None:
        raise ValueError(f'unknown tool effect status: {status_raw!r}')
    ended_at_raw = data.get('ended_at')
    ended_at = datetime.fromisoformat(ended_at_raw) if isinstance(ended_at_raw, str) else None
    return ToolEffectRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        run_id=run_id,
        status=status,
        started_at=datetime.fromisoformat(started_at_raw),
        ended_at=ended_at,
        idempotency_key=_opt_str(data.get('idempotency_key')),
        effect_summary=_opt_str(data.get('effect_summary')),
    )


class FileStepStore:
    """Filesystem-backed store using JSONL for events/tool effects and JSON for snapshots.

    Layout under the configured root directory:

    ```text
    <root>/
      <run_id>/
        run.json
        events.jsonl
        tool_effects.jsonl
        snapshots/
          <seq>.json
    ```

    Snapshot filenames are a per-run monotonic counter assigned at write
    time (see `_next_snapshot_seq`), not `step_index` -- `ctx.run_step`
    resets each `Agent.run` so a reused `run_id` would otherwise clash.
    The actual `step_index` is preserved as a field inside the JSON.

    `run_id` is validated against `[A-Za-z0-9_.-]{1,200}` (with `..` rejected)
    to prevent path traversal. Blocking I/O is dispatched to a worker thread
    via `anyio.to_thread`, so capability hooks do not stall the event loop.

    `BinaryContent` payloads and text parts at or above
    `media_threshold_bytes` (default 64 KiB) are externalized through the
    configured `MediaStore` to keep snapshot JSON small. The default backs
    onto `<root>/media/<sha256>.bin`. Pass `media_store=None` to keep
    payloads inline, or pass a custom `MediaStore` to redirect (e.g.
    `S3MediaStore(...)`). Snapshots written before text externalization
    existed still restore: `restore_media` recognizes the older marker shape.

    `max_snapshots_per_run` (default `None`, unbounded) bounds per-run
    snapshot growth: after each write, `{seq}.json` files outside the retain
    set are unlinked (see `_sync_prune_snapshots`).
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        _validate_max_snapshots(max_snapshots_per_run)
        self._max_snapshots_per_run = max_snapshots_per_run
        self._root = Path(directory)
        resolved: MediaStore | None
        if media_store == 'auto':
            resolved = DiskMediaStore(self._root / 'media')
        else:
            resolved = media_store
        self._media_store: MediaStore | None = resolved
        self._media_threshold_bytes = media_threshold_bytes

    def _run_dir(self, run_id: str) -> Path:
        _validate_id(run_id, field='run_id')
        return self._root / run_id

    async def register_run(self, record: RunRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_register_run, record)

    def _sync_register_run(self, record: RunRecord) -> None:
        run_dir = self._run_dir(record.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'snapshots').mkdir(exist_ok=True)
        _atomic_write_text(run_dir / 'run.json', json.dumps(_run_to_dict(record)))

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_run, run_id)

    def _sync_get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_dir(run_id) / 'run.json'
        if not path.exists():
            return None
        return _run_from_dict(_load_json_object(path.read_text(encoding='utf-8')))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_runs, parent_run_id, conversation_id)

    def _sync_list_runs(
        self,
        parent_run_id: str | None,
        conversation_id: str | None,
    ) -> list[RunRecord]:
        if not self._root.exists():
            return []
        records: list[RunRecord] = []
        for sub in self._root.iterdir():
            run_file = sub / 'run.json'
            if not run_file.exists():
                continue
            record = _run_from_dict(_load_json_object(run_file.read_text(encoding='utf-8')))
            if parent_run_id is not None and record.parent_run_id != parent_run_id:
                continue
            if conversation_id is not None and record.conversation_id != conversation_id:
                continue
            records.append(record)
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await anyio.to_thread.run_sync(self._sync_append_event, event)

    def _sync_append_event(self, event: StepEvent) -> None:
        run_dir = self._run_dir(event.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        if event.idempotency_key is not None:
            for prior in self._sync_list_events(event.run_id):
                if prior.idempotency_key == event.idempotency_key:
                    return
        line = json.dumps(_event_to_dict(event))
        with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await anyio.to_thread.run_sync(self._sync_list_events, run_id)

    def _sync_list_events(self, run_id: str) -> list[StepEvent]:
        path = self._run_dir(run_id) / 'events.jsonl'
        if not path.exists():
            return []
        events: list[StepEvent] = []
        for raw in path.read_text(encoding='utf-8').splitlines():
            if raw.strip():
                events.append(_event_from_dict(_load_json_object(raw)))
        return events

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        await anyio.to_thread.run_sync(self._sync_save_snapshot, snapshot, messages_json)

    def _sync_save_snapshot(self, snapshot: ContinuableSnapshot, messages_json: object) -> None:
        run_dir = self._run_dir(snapshot.run_id)
        snap_dir = run_dir / 'snapshots'
        snap_dir.mkdir(parents=True, exist_ok=True)
        key_path = run_dir / 'snapshot-keys.jsonl'
        if snapshot.idempotency_key is not None and key_path.exists():
            recorded_keys = {
                _STRING_ADAPTER.validate_json(raw)
                for raw in key_path.read_text(encoding='utf-8').splitlines()
                if raw.strip()
            }
            if snapshot.idempotency_key in recorded_keys:
                return
        if snapshot.idempotency_key is not None:
            for path in snap_dir.glob('*.json'):
                try:
                    prior = _load_json_object(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, ValueError, ValidationError):
                    continue
                if prior.get('idempotency_key') == snapshot.idempotency_key:
                    return
        payload: dict[str, object] = {
            'run_id': snapshot.run_id,
            'step_index': snapshot.step_index,
            'conversation_id': snapshot.conversation_id,
            'parent_run_id': snapshot.parent_run_id,
            'agent_name': snapshot.agent_name,
            'timestamp': snapshot.timestamp.isoformat(),
            'state': snapshot.state,
            'messages': messages_json,
            'idempotency_key': snapshot.idempotency_key,
        }
        seq = self._next_snapshot_seq(snap_dir)
        _atomic_write_text(snap_dir / f'{seq}.json', json.dumps(payload))
        if snapshot.idempotency_key is not None:
            with key_path.open('a', encoding='utf-8') as fp:
                fp.write(json.dumps(snapshot.idempotency_key) + '\n')
        self._sync_prune_snapshots(snap_dir)

    def _sync_prune_snapshots(self, snap_dir: Path) -> None:
        """Drop snapshot files outside the retain set when bounded.

        No-op when `max_snapshots_per_run` is `None`. Externalized media is
        content-addressed and may be shared across snapshots and runs, so a
        dropped `{seq}.json` never triggers a media delete -- orphaned-blob GC
        is a separate concern (see the capability README non-goals).

        A sibling file that fails to parse, or that parses but lacks the fields
        a snapshot needs (`_snapshot_fields_ok`), is skipped: not retained and
        not deleted. The newly-written snapshot has already landed, so a corrupt
        or structurally-incomplete older file must neither turn a bounded save
        into a failure nor be trusted as the newest `complete` snapshot and
        retained over a valid one. The read path skips the same files.

        A concurrent bounded save pruning on another worker thread can unlink a
        non-retained candidate between this enumeration and either the parse or
        the unlink below. Both tolerate the vanished file (skip / `missing_ok`),
        mirroring how `_sync_load_latest_snapshot` skips a candidate that
        disappears mid-read: the new snapshot has already landed, so a raced
        delete must not fail the save.
        """
        if self._max_snapshots_per_run is None:
            return
        paths_by_seq: dict[int, Path] = {}
        entries: list[tuple[int, SnapshotState]] = []
        for path in snap_dir.glob('*.json'):
            try:
                seq = int(path.stem)
            except ValueError:
                continue
            try:
                data = _load_json_object(path.read_text(encoding='utf-8'))
                state = _snapshot_state(data.get('state'))
            except FileNotFoundError:
                continue
            except (ValueError, ValidationError):
                continue
            if not _snapshot_fields_ok(data):
                continue
            entries.append((seq, state))
            paths_by_seq[seq] = path
        if len(entries) <= self._max_snapshots_per_run:
            return
        retained = _retained_seqs(entries, self._max_snapshots_per_run)
        for seq, path in paths_by_seq.items():
            if seq not in retained:
                path.unlink(missing_ok=True)

    @staticmethod
    def _next_snapshot_seq(snap_dir: Path) -> int:
        """Append-only monotonic counter per run directory.

        Using the snapshot's `step_index` as the filename would collide when
        the same `run_id` is reused across `Agent.run` calls -- `ctx.run_step`
        resets to 0 each call. The physical sequence is independent of
        `step_index`, which lives inside the JSON payload.
        """
        max_seq = -1
        for path in snap_dir.glob('*.json'):
            try:
                seq = int(path.stem)
            except ValueError:
                continue
            if seq > max_seq:
                max_seq = seq
        return max_seq + 1

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        loaded = await anyio.to_thread.run_sync(self._sync_load_latest_snapshot, run_id, include_interrupted)
        if loaded is None:
            return None
        data, messages_json = loaded
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        timestamp_raw = data['timestamp']
        step_raw = data['step_index']
        if not (isinstance(timestamp_raw, str) and isinstance(step_raw, int)):
            raise ValueError('snapshot has wrong types')
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_raw,
            messages=messages,
            conversation_id=_opt_str(data.get('conversation_id')),
            parent_run_id=_opt_str(data.get('parent_run_id')),
            agent_name=_opt_str(data.get('agent_name')),
            timestamp=datetime.fromisoformat(timestamp_raw),
            state=_snapshot_state(data.get('state')),
            idempotency_key=_opt_str(data.get('idempotency_key')),
        )

    def _sync_load_latest_snapshot(
        self, run_id: str, include_interrupted: bool
    ) -> tuple[dict[str, object], object] | None:
        snap_dir = self._run_dir(run_id) / 'snapshots'
        if not snap_dir.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for path in snap_dir.glob('*.json'):
            try:
                candidates.append((int(path.stem), path))
            except ValueError:
                continue
        for _, path in sorted(candidates, key=lambda c: c[0], reverse=True):
            # A bounded save pruning on another worker thread can unlink a
            # non-retained candidate between this enumeration and the read.
            # The newest overall and newest `complete` are always retained, so
            # skipping a vanished candidate still reaches the right snapshot (or
            # a correct `None`).
            try:
                data = _load_json_object(path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                continue
            # Skip a JSON-valid but structurally-incomplete file so it never
            # masks the newest loadable snapshot below it (see the prune).
            if not _snapshot_fields_ok(data):
                continue
            if include_interrupted or _snapshot_state(data.get('state')) == 'complete':
                return data, data['messages']
        return None

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate: `interrupted` snapshots are skipped
        unless `include_interrupted=True`. Snapshots that fail to read or parse
        are skipped and logged, so one damaged file does not hide the rest of
        the run's history. Not part of the `StepStore` protocol: the
        `conversation_search` capability consumes it through its narrower
        `SnapshotStore` protocol.
        """
        texts = await anyio.to_thread.run_sync(self._sync_read_snapshot_texts, run_id)
        snapshots: list[ContinuableSnapshot] = []
        for text in texts:
            try:
                data = _load_json_object(text)
                state = _snapshot_state(data.get('state'))
                if state == 'interrupted' and not include_interrupted:
                    continue
                messages_json = data['messages']
                if self._media_store is not None:
                    messages_json = await restore_media(messages_json, media_store=self._media_store)
                messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
                timestamp_raw = data['timestamp']
                step_raw = data['step_index']
                if not (isinstance(timestamp_raw, str) and isinstance(step_raw, int)):
                    raise ValueError('snapshot has wrong types')
                snapshots.append(
                    ContinuableSnapshot(
                        run_id=run_id,
                        step_index=step_raw,
                        messages=messages,
                        conversation_id=_opt_str(data.get('conversation_id')),
                        parent_run_id=_opt_str(data.get('parent_run_id')),
                        agent_name=_opt_str(data.get('agent_name')),
                        timestamp=datetime.fromisoformat(timestamp_raw),
                        state=state,
                        idempotency_key=_opt_str(data.get('idempotency_key')),
                    )
                )
            except Exception:
                _logger.warning('Skipping unparsable snapshot for run %s', run_id, exc_info=True)
        return snapshots

    def _sync_read_snapshot_texts(self, run_id: str) -> list[str]:
        snap_dir = self._run_dir(run_id) / 'snapshots'
        if not snap_dir.exists():
            return []
        candidates: list[tuple[int, Path]] = []
        for path in snap_dir.glob('*.json'):
            try:
                candidates.append((int(path.stem), path))
            except ValueError:
                continue
        texts: list[str] = []
        for _, path in sorted(candidates, key=lambda c: c[0]):
            try:
                texts.append(path.read_text(encoding='utf-8'))
            except OSError:
                _logger.warning('Skipping unreadable snapshot file %s', path, exc_info=True)
        return texts

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_record_tool_effect, record)

    def _sync_record_tool_effect(self, record: ToolEffectRecord) -> None:
        run_dir = self._run_dir(record.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_tool_effect_to_dict(record))
        with (run_dir / 'tool_effects.jsonl').open('a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_tool_effect, run_id, tool_call_id)

    def _sync_get_tool_effect(self, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        path = self._run_dir(run_id) / 'tool_effects.jsonl'
        if not path.exists():
            return None
        latest: ToolEffectRecord | None = None
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            record = _tool_effect_from_dict(_load_json_object(raw))
            if record.tool_call_id == tool_call_id:
                latest = record
        return latest

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_unresolved_tool_effects, run_id)

    def _sync_list_unresolved_tool_effects(self, run_id: str) -> list[ToolEffectRecord]:
        path = self._run_dir(run_id) / 'tool_effects.jsonl'
        if not path.exists():
            return []
        latest_by_call: dict[str, ToolEffectRecord] = {}
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            record = _tool_effect_from_dict(_load_json_object(raw))
            latest_by_call[record.tool_call_id] = record
        return [r for r in latest_by_call.values() if r.status == 'started']


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    parent_run_id TEXT,
    agent_name TEXT,
    metadata TEXT NOT NULL,
    started_at TEXT NOT NULL,
    registration_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_conv ON runs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    conversation_id TEXT,
    parent_run_id TEXT,
    agent_name TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    error TEXT,
    metadata TEXT NOT NULL,
    idempotency_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);

CREATE TABLE IF NOT EXISTS snapshots (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    conversation_id TEXT,
    parent_run_id TEXT,
    agent_name TEXT,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'complete',
    messages TEXT NOT NULL,
    idempotency_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_run ON snapshots(run_id, seq);

CREATE TABLE IF NOT EXISTS snapshot_idempotency_keys (
    run_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    PRIMARY KEY (run_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS tool_effects (
    run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    idempotency_key TEXT,
    effect_summary TEXT,
    PRIMARY KEY (run_id, tool_call_id)
);
"""


class SqliteStepStore:
    """SQLite-backed store. Single file holds runs, events, snapshots, tool effects + media.

    Pass either `database=` (path; connections opened short-lived per call) or
    `connection=` (caller-owned `sqlite3.Connection`). When `database=` is
    used, the default `media_store` is a `SqliteMediaStore` against the same
    file (sibling `media` table), so a single-file deployment is the default.

    `BinaryContent` payloads and text parts at or above
    `media_threshold_bytes` (default 64 KiB) are externalized through that
    store, leaving a URI reference in the `snapshots.messages` JSON. Pass
    `media_store=None` to keep payloads inline. Snapshots written before text
    externalization existed still restore: `restore_media` recognizes the
    older marker shape.

    A caller-owned `connection=` **must** be created with
    `check_same_thread=False`. Store methods dispatch SQL onto worker threads
    via `anyio.to_thread`, so the stdlib default (`check_same_thread=True`)
    raises `sqlite3.ProgrammingError` on first use. The `database=` path sets
    this internally; `connection=` cannot, so it is the caller's responsibility.

    Concurrency: WAL mode is enabled on connections opened from a `database=`
    path (a caller-owned `connection=` keeps whatever journal mode the caller
    set). Concurrent readers + a single writer are safe; this matches the
    access pattern of one capability per `Agent.run`.

    Schema:

    - `runs(run_id PK, conversation_id, parent_run_id, agent_name, metadata, started_at, registration_id)`
    - `events(seq PK AUTOINCREMENT, run_id, kind, step_index, timestamp, ..., idempotency_key)`
    - `snapshots(seq PK AUTOINCREMENT, run_id, step_index, ..., messages)` --
      the `AUTOINCREMENT` `seq` mirrors `FileStepStore._next_snapshot_seq`
      so `ctx.run_step` resets across `Agent.run` cannot collide.
    - `snapshot_idempotency_keys(run_id, idempotency_key PK)` -- survives pruning.
    - `tool_effects(run_id, tool_call_id PK)` -- upsert per
      `(run_id, tool_call_id)`, matching `InMemoryStepStore` semantics.
    - `media(sha256 PK, media_type, bytes, size_bytes)` -- `INSERT OR IGNORE`
      for content-addressed dedup.

    The `runs.run_id` PK enforces the "explicit `run_id` is single-shot"
    contract -- `register_run` raises `sqlite3.IntegrityError` on reuse,
    which `StepPersistence.before_run` converts to a friendlier
    `ValueError` via its own pre-check.

    `max_snapshots_per_run` (default `None`, unbounded) bounds per-run
    snapshot growth: after each write, one indexed `DELETE` prunes rows
    outside the retain set (see `_sync_prune_snapshots`).
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        _validate_max_snapshots(max_snapshots_per_run)
        self._max_snapshots_per_run = max_snapshots_per_run
        self._database = Path(database) if database is not None else None
        self._connection = connection
        self._schema_ready = False
        resolved: MediaStore | None
        if media_store == 'auto':
            if self._database is not None:
                resolved = SqliteMediaStore(database=self._database)
            else:
                assert connection is not None
                resolved = SqliteMediaStore(connection=connection)
        else:
            resolved = media_store
        self._media_store: MediaStore | None = resolved
        self._media_threshold_bytes = media_threshold_bytes

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        assert self._database is not None
        self._database.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._database, isolation_level=None)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._connection is None:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.executescript(_SQLITE_SCHEMA)
        # Databases created before the `state` column existed: `CREATE TABLE
        # IF NOT EXISTS` keeps their old shape, so add the column here. The
        # default backfills existing rows as `complete`, which is correct --
        # the gated design only ever wrote provider-valid snapshots.
        # Attempt-and-catch rather than a `PRAGMA table_info` pre-check: two
        # store instances migrating the same old file concurrently would both
        # pass the pre-check and the loser's `ALTER` would raise. Only a column
        # that is already there is benign, so confirm that is what happened --
        # a locked or readonly database raises the same `OperationalError`, and
        # swallowing it would mark the schema ready while `state` is still
        # missing, with every later snapshot read failing on `no such column:
        # state` and `_schema_ready` never letting the migration retry.
        try:
            conn.execute("ALTER TABLE snapshots ADD COLUMN state TEXT NOT NULL DEFAULT 'complete'")
        except sqlite3.OperationalError:
            if 'state' not in self._snapshot_columns(conn):
                raise
        for table in ('events', 'snapshots'):
            try:
                conn.execute(f'ALTER TABLE {table} ADD COLUMN idempotency_key TEXT')
            except sqlite3.OperationalError:
                if 'idempotency_key' not in self._table_columns(conn, table):
                    raise  # pragma: no cover
        try:
            conn.execute('ALTER TABLE runs ADD COLUMN registration_id TEXT')
        except sqlite3.OperationalError:
            if 'registration_id' not in self._table_columns(conn, 'runs'):
                raise  # pragma: no cover
        conn.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency '
            'ON events(run_id, idempotency_key) WHERE idempotency_key IS NOT NULL'
        )
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS snapshots_idempotency_ignore
            BEFORE INSERT ON snapshots
            WHEN NEW.idempotency_key IS NOT NULL AND EXISTS (
                SELECT 1 FROM snapshot_idempotency_keys
                WHERE run_id = NEW.run_id AND idempotency_key = NEW.idempotency_key
            )
            BEGIN
                SELECT RAISE(IGNORE);
            END;
            CREATE TRIGGER IF NOT EXISTS snapshots_idempotency_record
            AFTER INSERT ON snapshots
            WHEN NEW.idempotency_key IS NOT NULL
            BEGIN
                INSERT OR IGNORE INTO snapshot_idempotency_keys (run_id, idempotency_key)
                VALUES (NEW.run_id, NEW.idempotency_key);
            END;
            """
        )
        conn.commit()
        self._schema_ready = True

    @staticmethod
    def _snapshot_columns(conn: sqlite3.Connection) -> set[str]:
        return {row[1] for row in conn.execute('PRAGMA table_info(snapshots)')}

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}

    async def register_run(self, record: RunRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_register_run, record)

    def _sync_register_run(self, record: RunRecord) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                'INSERT INTO runs ('
                'run_id, conversation_id, parent_run_id, agent_name, metadata, started_at, registration_id'
                ') VALUES (?, ?, ?, ?, ?, ?, ?)',
                (
                    record.run_id,
                    record.conversation_id,
                    record.parent_run_id,
                    record.agent_name,
                    json.dumps(dict(record.metadata)),
                    record.started_at.isoformat(),
                    record.registration_id,
                ),
            )
        finally:
            self._maybe_close(conn)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_run, run_id)

    def _sync_get_run(self, run_id: str) -> RunRecord | None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(
                'SELECT run_id, conversation_id, parent_run_id, agent_name, metadata, started_at, registration_id '
                'FROM runs WHERE run_id = ?',
                (run_id,),
            ).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            return None
        return _run_from_row(row)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_runs, parent_run_id, conversation_id)

    def _sync_list_runs(
        self,
        parent_run_id: str | None,
        conversation_id: str | None,
    ) -> list[RunRecord]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            clauses: list[str] = []
            params: list[object] = []
            if parent_run_id is not None:
                clauses.append('parent_run_id = ?')
                params.append(parent_run_id)
            if conversation_id is not None:
                clauses.append('conversation_id = ?')
                params.append(conversation_id)
            sql = (
                'SELECT run_id, conversation_id, parent_run_id, agent_name, metadata, started_at, registration_id '
                'FROM runs'
            )
            if clauses:
                sql += ' WHERE ' + ' AND '.join(clauses)
            sql += ' ORDER BY started_at ASC'
            rows = conn.execute(sql, params).fetchall()
        finally:
            self._maybe_close(conn)
        return [_run_from_row(row) for row in rows]

    async def append_event(self, event: StepEvent) -> None:
        await anyio.to_thread.run_sync(self._sync_append_event, event)

    def _sync_append_event(self, event: StepEvent) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                'INSERT INTO events ('
                'run_id, kind, step_index, timestamp, conversation_id, parent_run_id, '
                'agent_name, tool_call_id, tool_name, error, metadata, idempotency_key'
                ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(run_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING',
                (
                    event.run_id,
                    event.kind,
                    event.step_index,
                    event.timestamp.isoformat(),
                    event.conversation_id,
                    event.parent_run_id,
                    event.agent_name,
                    event.tool_call_id,
                    event.tool_name,
                    event.error,
                    json.dumps(dict(event.metadata)),
                    event.idempotency_key,
                ),
            )
        finally:
            self._maybe_close(conn)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await anyio.to_thread.run_sync(self._sync_list_events, run_id)

    def _sync_list_events(self, run_id: str) -> list[StepEvent]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                'SELECT run_id, kind, step_index, timestamp, conversation_id, parent_run_id, '
                'agent_name, tool_call_id, tool_name, error, metadata, idempotency_key '
                'FROM events WHERE run_id = ? ORDER BY seq ASC',
                (run_id,),
            ).fetchall()
        finally:
            self._maybe_close(conn)
        return [_event_from_row(row) for row in rows]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        await anyio.to_thread.run_sync(self._sync_save_snapshot, snapshot, messages_json)

    def _sync_save_snapshot(self, snapshot: ContinuableSnapshot, messages_json: object) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                'INSERT INTO snapshots ('
                'run_id, step_index, conversation_id, parent_run_id, agent_name, timestamp, state, messages, '
                'idempotency_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    snapshot.run_id,
                    snapshot.step_index,
                    snapshot.conversation_id,
                    snapshot.parent_run_id,
                    snapshot.agent_name,
                    snapshot.timestamp.isoformat(),
                    snapshot.state,
                    json.dumps(messages_json),
                    snapshot.idempotency_key,
                ),
            )
            self._sync_prune_snapshots(conn, snapshot.run_id)
        finally:
            self._maybe_close(conn)

    def _sync_prune_snapshots(self, conn: sqlite3.Connection, run_id: str) -> None:
        """Delete this run's snapshot rows outside the retain set when bounded.

        No-op when `max_snapshots_per_run` is `None`. Externalized media in the
        sibling `media` table is content-addressed and may be shared across
        snapshots and runs, so a deleted snapshot row never removes a media row
        -- orphaned-blob GC is a separate concern (see the capability README
        non-goals).

        The retain set is computed inside SQL rather than enumerated as bound
        parameters: the newest `keep` by `seq` (which subsumes the newest
        overall, since `keep >= 1`) plus the newest `complete`, mirroring
        `_retained_seqs`. Binding one parameter per retained `seq` would trip
        SQLite's variable limit once `keep` grows large.
        """
        if self._max_snapshots_per_run is None:
            return
        conn.execute(
            'DELETE FROM snapshots WHERE run_id = ? AND seq NOT IN ('
            'SELECT seq FROM (SELECT seq FROM snapshots WHERE run_id = ? ORDER BY seq DESC LIMIT ?) '
            'UNION '
            "SELECT seq FROM (SELECT seq FROM snapshots WHERE run_id = ? AND state = 'complete' "
            'ORDER BY seq DESC LIMIT 1))',
            (run_id, run_id, self._max_snapshots_per_run, run_id),
        )

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        row = await anyio.to_thread.run_sync(self._sync_load_latest_snapshot, run_id, include_interrupted)
        if row is None:
            return None
        step_index, conv_id, parent_id, agent_name, timestamp_iso, state, messages_json_text, idempotency_key = row
        messages_json: object = json.loads(messages_json_text)
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_index,
            messages=messages,
            conversation_id=conv_id,
            parent_run_id=parent_id,
            agent_name=agent_name,
            timestamp=datetime.fromisoformat(timestamp_iso),
            state=state,
            idempotency_key=idempotency_key,
        )

    def _sync_load_latest_snapshot(
        self, run_id: str, include_interrupted: bool
    ) -> tuple[int, str | None, str | None, str | None, str, SnapshotState, str, str | None] | None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            sql = (
                'SELECT step_index, conversation_id, parent_run_id, agent_name, timestamp, state, messages, '
                'idempotency_key '
                'FROM snapshots WHERE run_id = ?'
            )
            if not include_interrupted:
                sql += " AND state = 'complete'"
            row = conn.execute(sql + ' ORDER BY seq DESC LIMIT 1', (run_id,)).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            return None
        step_index, conv_id, parent_id, agent_name, timestamp_iso, state_raw, messages_json_text, key = row
        if not (isinstance(step_index, int) and isinstance(timestamp_iso, str) and isinstance(messages_json_text, str)):
            raise ValueError('snapshot row has wrong types')
        return (
            step_index,
            _opt_str(conv_id),
            _opt_str(parent_id),
            _opt_str(agent_name),
            timestamp_iso,
            _snapshot_state(state_raw),
            messages_json_text,
            _opt_str(key),
        )

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate: `interrupted` snapshots are skipped
        unless `include_interrupted=True`. Rows that fail to parse are skipped
        and logged, so one damaged row does not hide the rest of the run's
        history. Not part of the `StepStore` protocol: the `conversation_search`
        capability consumes it through its narrower `SnapshotStore` protocol.
        """
        rows = await anyio.to_thread.run_sync(self._sync_load_snapshot_rows, run_id, include_interrupted)
        snapshots: list[ContinuableSnapshot] = []
        for row in rows:
            try:
                step_index, conv_id, parent_id, agent_name, timestamp_iso, state_raw, messages_json_text, key = row
                if not (
                    isinstance(step_index, int)
                    and isinstance(timestamp_iso, str)
                    and isinstance(messages_json_text, str)
                ):
                    raise ValueError('snapshot row has wrong types')
                messages_json: object = json.loads(messages_json_text)
                if self._media_store is not None:
                    messages_json = await restore_media(messages_json, media_store=self._media_store)
                messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
                snapshots.append(
                    ContinuableSnapshot(
                        run_id=run_id,
                        step_index=step_index,
                        messages=messages,
                        conversation_id=_opt_str(conv_id),
                        parent_run_id=_opt_str(parent_id),
                        agent_name=_opt_str(agent_name),
                        timestamp=datetime.fromisoformat(timestamp_iso),
                        state=_snapshot_state(state_raw),
                        idempotency_key=_opt_str(key),
                    )
                )
            except Exception:
                _logger.warning('Skipping unparsable snapshot row for run %s', run_id, exc_info=True)
        return snapshots

    def _sync_load_snapshot_rows(self, run_id: str, include_interrupted: bool) -> list[tuple[object, ...]]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            sql = (
                'SELECT step_index, conversation_id, parent_run_id, agent_name, timestamp, state, messages, '
                'idempotency_key '
                'FROM snapshots WHERE run_id = ?'
            )
            if not include_interrupted:
                sql += " AND state = 'complete'"
            rows = conn.execute(sql + ' ORDER BY seq ASC', (run_id,)).fetchall()
        finally:
            self._maybe_close(conn)
        return rows

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_record_tool_effect, record)

    def _sync_record_tool_effect(self, record: ToolEffectRecord) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                'INSERT INTO tool_effects ('
                'run_id, tool_call_id, tool_name, status, started_at, ended_at, idempotency_key, effect_summary'
                ') VALUES (?, ?, ?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(run_id, tool_call_id) DO UPDATE SET '
                'tool_name=excluded.tool_name, status=excluded.status, started_at=excluded.started_at, '
                'ended_at=excluded.ended_at, idempotency_key=excluded.idempotency_key, '
                'effect_summary=excluded.effect_summary',
                (
                    record.run_id,
                    record.tool_call_id,
                    record.tool_name,
                    record.status,
                    record.started_at.isoformat(),
                    record.ended_at.isoformat() if record.ended_at is not None else None,
                    record.idempotency_key,
                    record.effect_summary,
                ),
            )
        finally:
            self._maybe_close(conn)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_tool_effect, run_id, tool_call_id)

    def _sync_get_tool_effect(self, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(
                'SELECT run_id, tool_call_id, tool_name, status, started_at, ended_at, '
                'idempotency_key, effect_summary FROM tool_effects '
                'WHERE run_id = ? AND tool_call_id = ?',
                (run_id, tool_call_id),
            ).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            return None
        return _tool_effect_from_row(row)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_unresolved_tool_effects, run_id)

    def _sync_list_unresolved_tool_effects(self, run_id: str) -> list[ToolEffectRecord]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                'SELECT run_id, tool_call_id, tool_name, status, started_at, ended_at, '
                'idempotency_key, effect_summary FROM tool_effects '
                "WHERE run_id = ? AND status = 'started'",
                (run_id,),
            ).fetchall()
        finally:
            self._maybe_close(conn)
        return [_tool_effect_from_row(row) for row in rows]


def _run_from_row(row: tuple[object, ...]) -> RunRecord:
    run_id, conv_id, parent_id, agent_name, metadata_text, started_at_iso, registration_id = row
    if not (isinstance(run_id, str) and isinstance(metadata_text, str) and isinstance(started_at_iso, str)):
        raise ValueError('run row has wrong types')
    return RunRecord(
        run_id=run_id,
        conversation_id=_opt_str(conv_id),
        parent_run_id=_opt_str(parent_id),
        agent_name=_opt_str(agent_name),
        metadata=_str_str_dict(json.loads(metadata_text)),
        started_at=datetime.fromisoformat(started_at_iso),
        registration_id=_opt_str(registration_id),
    )


def _event_from_row(row: tuple[object, ...]) -> StepEvent:
    (
        run_id,
        kind_raw,
        step_raw,
        timestamp_raw,
        conv_id,
        parent_id,
        agent_name,
        tool_call_id,
        tool_name,
        error,
        metadata_text,
        idempotency_key,
    ) = row
    if not (
        isinstance(run_id, str)
        and isinstance(kind_raw, str)
        and isinstance(step_raw, int)
        and isinstance(timestamp_raw, str)
        and isinstance(metadata_text, str)
    ):
        raise ValueError('event row has wrong types')
    kind = _EVENT_KIND_TABLE.get(kind_raw)
    if kind is None:
        raise ValueError(f'unknown event kind: {kind_raw!r}')
    return StepEvent(
        run_id=run_id,
        kind=kind,
        step_index=step_raw,
        timestamp=datetime.fromisoformat(timestamp_raw),
        conversation_id=_opt_str(conv_id),
        parent_run_id=_opt_str(parent_id),
        agent_name=_opt_str(agent_name),
        tool_call_id=_opt_str(tool_call_id),
        tool_name=_opt_str(tool_name),
        error=_opt_str(error),
        metadata=_str_str_dict(json.loads(metadata_text)),
        idempotency_key=_opt_str(idempotency_key),
    )


def _tool_effect_from_row(row: tuple[object, ...]) -> ToolEffectRecord:
    (
        run_id,
        tool_call_id,
        tool_name,
        status_raw,
        started_at_raw,
        ended_at_raw,
        idempotency_key,
        effect_summary,
    ) = row
    if not (
        isinstance(run_id, str)
        and isinstance(tool_call_id, str)
        and isinstance(tool_name, str)
        and isinstance(status_raw, str)
        and isinstance(started_at_raw, str)
    ):
        raise ValueError('tool effect row has wrong types')
    status = _TOOL_STATUS_TABLE.get(status_raw)
    if status is None:
        raise ValueError(f'unknown tool effect status: {status_raw!r}')
    return ToolEffectRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        run_id=run_id,
        status=status,
        started_at=datetime.fromisoformat(started_at_raw),
        ended_at=datetime.fromisoformat(ended_at_raw) if isinstance(ended_at_raw, str) else None,
        idempotency_key=_opt_str(idempotency_key),
        effect_summary=_opt_str(effect_summary),
    )
