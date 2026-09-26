"""MongoDB-backed step-persistence store.

Mirrors `SqliteStepStore`: an append-only event log, continuable snapshots,
and a tool-effect ledger, keyed off one `AsyncMongoClient`. Binary and text
parts at or above `media_threshold_bytes` are externalized to a `MediaStore`
(default: a `MongoMediaStore` sharing the same client). This is a per-value
offload, not an aggregate size cap: a snapshot made of many parts each below
the threshold can still exceed MongoDB's 16MB BSON limit and fail on insert.
Lower `media_threshold_bytes` if your snapshots accumulate many mid-sized
parts.

Collections (created lazily on first write):

- `runs` -- `_id = run_id`; `insert_one` atomically enforces the single-shot
  `run_id` contract (duplicates surface as `ValueError`).
- `events` -- one document per event, ordered by a monotonic `seq`.
- `snapshots` -- one document per snapshot; latest-per-run is the highest
  `seq`, matching `SqliteStepStore`'s `AUTOINCREMENT seq` so a reused
  `run_id` whose `step_index` reset to 0 cannot clobber an earlier snapshot.
- `snapshot_idempotency_keys` -- replay suppression retained after pruning.
- `tool_effects` -- upsert per `(run_id, tool_call_id)`.
- `counters` -- one document per sequence name, incremented atomically with
  `$inc` to allocate `seq` values.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
from pydantic_ai_harness.media._mongo import MongoMediaStore  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _AutoMedia,  # pyright: ignore[reportPrivateUsage]
    _event_from_dict,  # pyright: ignore[reportPrivateUsage]
    _event_to_dict,  # pyright: ignore[reportPrivateUsage]
    _opt_str,  # pyright: ignore[reportPrivateUsage]
    _retained_seqs,  # pyright: ignore[reportPrivateUsage]
    _run_from_dict,  # pyright: ignore[reportPrivateUsage]
    _run_to_dict,  # pyright: ignore[reportPrivateUsage]
    _snapshot_state,  # pyright: ignore[reportPrivateUsage]
    _tool_effect_from_dict,  # pyright: ignore[reportPrivateUsage]
    _tool_effect_to_dict,  # pyright: ignore[reportPrivateUsage]
    _validate_max_snapshots,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
)

_MongoDocument = dict[str, object]

_logger = logging.getLogger(__name__)


class MongoStepStore:
    """MongoDB-backed step-persistence store; see the module docstring for layout.

    Pass `client=` (a shared `AsyncMongoClient`) or `db_url=` (the store
    creates and owns its own client). `database=` is always required. With a
    shared client the caller owns its lifecycle; with `db_url=` call
    `aclose()` to release the owned client.

    `media_store` defaults to `'auto'`, which builds a `MongoMediaStore` on
    the same client and database (binary/text payloads land in sibling
    `media` collections). Pass `None` to keep payloads inline in the snapshot
    document, or pass any `MediaStore` (e.g. `S3MediaStore`) to redirect.
    Parts whose byte length is >= `media_threshold_bytes` are externalized.

    `max_snapshots_per_run` (default `None`, unbounded) bounds per-run
    snapshot growth: after each write, one `delete_many` prunes the documents
    outside the retain set (see `_prune_snapshots`).
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_MongoDocument] | None = None,
        db_url: str | None = None,
        database: str | None = None,
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        if (client is None) == (db_url is None):
            raise ValueError('provide exactly one of `client=` or `db_url=`')
        if database is None:
            raise ValueError('`database=` is required')
        _validate_max_snapshots(max_snapshots_per_run)
        self._max_snapshots_per_run = max_snapshots_per_run

        if client is not None:
            self._client = client
            self._own_client = False
        else:
            assert db_url is not None
            self._client = AsyncMongoClient[_MongoDocument](db_url)
            self._own_client = True

        self._db: AsyncDatabase[_MongoDocument] = self._client[database]

        resolved: MediaStore | None
        if media_store == 'auto':
            resolved = MongoMediaStore(client=self._client, database=database)
        else:
            resolved = media_store
        self._media_store = resolved
        self._media_threshold_bytes = media_threshold_bytes
        self._indexes_ready = False

    async def aclose(self) -> None:
        """Close the client if this store created it; no-op for a shared client."""
        if self._own_client:
            await self._client.close()

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._db['runs'].create_index('conversation_id', sparse=True)
        await self._db['runs'].create_index('parent_run_id', sparse=True)
        await self._db['runs'].create_index('started_at')
        await self._db['events'].create_index([('run_id', 1), ('seq', 1)])
        await self._db['events'].create_index(
            [('run_id', 1), ('idempotency_key', 1)],
            unique=True,
            partialFilterExpression={'idempotency_key': {'$type': 'string'}},
        )
        await self._db['snapshots'].create_index([('run_id', 1), ('seq', -1)])
        await self._db['snapshots'].create_index(
            [('run_id', 1), ('idempotency_key', 1)],
            unique=True,
            partialFilterExpression={'idempotency_key': {'$type': 'string'}},
        )
        await self._db['snapshots'].create_index([('run_id', 1), ('state', 1), ('seq', -1)])
        await self._db['tool_effects'].create_index([('run_id', 1), ('tool_call_id', 1)], unique=True)
        await self._db['tool_effects'].create_index([('run_id', 1), ('status', 1)])
        self._indexes_ready = True

    async def _next_seq(self, name: str) -> int:
        doc = await self._db['counters'].find_one_and_update(
            {'_id': name},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        assert doc is not None
        seq = doc['seq']
        assert isinstance(seq, int)
        return seq

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_indexes()
        try:
            await self._db['runs'].insert_one({'_id': record.run_id, **_run_to_dict(record)})
        except DuplicateKeyError as exc:
            raise ValueError(f'run_id {record.run_id!r} is already in the store') from exc

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        doc = await self._db['runs'].find_one({'_id': run_id})
        if doc is None:
            return None
        return _run_from_dict(doc)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        query: _MongoDocument = {}
        if parent_run_id is not None:
            query['parent_run_id'] = parent_run_id
        if conversation_id is not None:
            query['conversation_id'] = conversation_id
        # Sort by the parsed instant, not the stored ISO string: a lexicographic
        # sort would misorder mixed-offset timestamps, and the `StepStore`
        # contract (matching the in-memory/sqlite/file stores) is instant order.
        records = [_run_from_dict(doc) async for doc in self._db['runs'].find(query)]
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_indexes()
        doc = _event_to_dict(event)
        if event.idempotency_key is None:
            del doc['idempotency_key']
            doc['seq'] = await self._next_seq('events')
            await self._db['events'].insert_one(doc)
            return
        await self._db['events'].update_one(
            {'run_id': event.run_id, 'idempotency_key': event.idempotency_key},
            {'$setOnInsert': {**doc, 'seq': await self._next_seq('events')}},
            upsert=True,
        )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        cursor = self._db['events'].find({'run_id': run_id}).sort('seq', 1)
        return [_event_from_dict(doc) async for doc in cursor]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_indexes()
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        doc: _MongoDocument = {
            'run_id': snapshot.run_id,
            'step_index': snapshot.step_index,
            'conversation_id': snapshot.conversation_id,
            'parent_run_id': snapshot.parent_run_id,
            'agent_name': snapshot.agent_name,
            'timestamp': snapshot.timestamp.isoformat(),
            'state': snapshot.state,
            'messages': json.dumps(messages_json),
            'idempotency_key': snapshot.idempotency_key,
        }
        if snapshot.idempotency_key is None:
            del doc['idempotency_key']
            doc['seq'] = await self._next_seq('snapshots')
            await self._db['snapshots'].insert_one(doc)
        else:
            key_id: _MongoDocument = {'run_id': snapshot.run_id, 'key': snapshot.idempotency_key}
            if await self._db['snapshot_idempotency_keys'].find_one({'_id': key_id}) is not None:
                return
            doc['seq'] = await self._next_seq('snapshots')
            try:
                await self._db['snapshots'].insert_one(doc)
            except DuplicateKeyError:
                await self._db['snapshot_idempotency_keys'].update_one(
                    {'_id': key_id}, {'$setOnInsert': {'_id': key_id}}, upsert=True
                )
                return
            # This ledger is independent of retained snapshot documents, so pruning cannot
            # make a previously applied key eligible again. Its size is bounded by keyed saves.
            await self._db['snapshot_idempotency_keys'].update_one(
                {'_id': key_id}, {'$setOnInsert': {'_id': key_id}}, upsert=True
            )
        await self._prune_snapshots(snapshot.run_id)

    async def _prune_snapshots(self, run_id: str) -> None:
        """Delete this run's snapshot documents outside the retain set when bounded.

        No-op when `max_snapshots_per_run` is `None`. Externalized media in the
        sibling `media` collections is content-addressed and may be shared
        across snapshots and runs, so a deleted snapshot document never removes
        a blob -- orphaned-blob GC is a separate concern (see the capability
        README non-goals).

        The retain set comes from `_retained_seqs`, so the newest overall and
        the newest `complete` survive even when the newest `keep` documents are
        all `interrupted`.
        """
        if self._max_snapshots_per_run is None:
            return
        entries: list[tuple[int, SnapshotState]] = []
        async for doc in self._db['snapshots'].find({'run_id': run_id}, {'seq': 1, 'state': 1}):
            seq = doc['seq']
            assert isinstance(seq, int)
            entries.append((seq, _snapshot_state(doc.get('state'))))
        retained = _retained_seqs(entries, self._max_snapshots_per_run)
        await self._db['snapshots'].delete_many({'run_id': run_id, 'seq': {'$nin': sorted(retained)}})

    async def _snapshot_from_doc(self, run_id: str, doc: _MongoDocument) -> ContinuableSnapshot:
        messages_text = doc['messages']
        step_index = doc['step_index']
        timestamp_raw = doc['timestamp']
        if not (isinstance(messages_text, str) and isinstance(step_index, int) and isinstance(timestamp_raw, str)):
            raise ValueError('snapshot document has wrong types')

        messages_json: object = json.loads(messages_text)
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_index,
            messages=messages,
            conversation_id=_opt_str(doc.get('conversation_id')),
            parent_run_id=_opt_str(doc.get('parent_run_id')),
            agent_name=_opt_str(doc.get('agent_name')),
            timestamp=datetime.fromisoformat(timestamp_raw),
            state=_snapshot_state(doc.get('state')),
            idempotency_key=_opt_str(doc.get('idempotency_key')),
        )

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        query: _MongoDocument = {'run_id': run_id}
        if not include_interrupted:
            query['state'] = 'complete'
        doc = await self._db['snapshots'].find_one(query, sort=[('seq', -1)])
        if doc is None:
            return None
        return await self._snapshot_from_doc(run_id, doc)

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate: `interrupted` snapshots are skipped
        unless `include_interrupted=True`. Documents that fail to parse are
        skipped and logged, so one damaged document does not hide the rest of
        the run's history. Not part of the `StepStore` protocol: the
        `conversation_search` capability consumes it through its narrower
        `SnapshotStore` protocol.
        """
        query: _MongoDocument = {'run_id': run_id}
        if not include_interrupted:
            query['state'] = 'complete'
        snapshots: list[ContinuableSnapshot] = []
        async for doc in self._db['snapshots'].find(query).sort('seq', 1):
            try:
                snapshots.append(await self._snapshot_from_doc(run_id, doc))
            except Exception:
                _logger.warning('Skipping unparsable snapshot document for run %s', run_id, exc_info=True)
        return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_indexes()
        await self._db['tool_effects'].update_one(
            {'run_id': record.run_id, 'tool_call_id': record.tool_call_id},
            {'$set': _tool_effect_to_dict(record)},
            upsert=True,
        )

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        doc = await self._db['tool_effects'].find_one({'run_id': run_id, 'tool_call_id': tool_call_id})
        if doc is None:
            return None
        return _tool_effect_from_dict(doc)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        cursor = self._db['tool_effects'].find({'run_id': run_id, 'status': 'started'})
        return [_tool_effect_from_dict(doc) async for doc in cursor]
