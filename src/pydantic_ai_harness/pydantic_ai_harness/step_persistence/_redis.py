"""Redis-backed step-persistence store over a caller-owned client.

Like `RedisSpendStore` and `RedisPlanStore`, this store never imports `redis`:
it depends only on the `RedisClient` protocol below, so installing the harness
pulls in no Redis driver. Pass your own `redis.asyncio.Redis` client.

Keys under the configured `prefix`. Every key that belongs to one run carries a
`{run_id}` hash tag, so all of them land on one Redis Cluster slot and one Lua
script can write them together:

```text
<prefix>:run:{<run_id>}                 string  RunRecord JSON
<prefix>:events:{<run_id>}              list    one StepEvent JSON per append
<prefix>:event_keys:{<run_id>}          set     idempotency keys of appended events
<prefix>:snapshots:{<run_id>}           zset    member `<seq>:<state>`, score `seq`
<prefix>:snapshots:seq:{<run_id>}       string  counter allocating `seq`
<prefix>:snapshot_payloads:{<run_id>}   hash    `seq` -> snapshot JSON, history as message keys
<prefix>:snapshot_keys:{<run_id>}       set     idempotency keys of saved snapshots
<prefix>:messages:{<run_id>}            hash    SHA-256 -> message JSON
<prefix>:tool_effects:{<run_id>}        hash    `tool_call_id` -> ToolEffectRecord JSON
<prefix>:runs:all                       set     every run id
<prefix>:runs:conversation:<cid>        set     run ids in one conversation
<prefix>:runs:parent:<pid>              set     run ids spawned by one run
```

Snapshots are keyed by a per-run monotonic `seq`, not `step_index`:
`ctx.run_step` resets each `Agent.run`, so a reused `run_id` would clobber an
earlier snapshot. The index member carries the state (`12:complete`), so
choosing what to read and what to prune costs one `ZRANGE` and no payload reads.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
from pydantic_ai_harness.step_persistence._messages import keyed_messages, referenced_history_json
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _OBJECT_LIST_ADAPTER,  # pyright: ignore[reportPrivateUsage]
    _event_from_dict,  # pyright: ignore[reportPrivateUsage]
    _event_to_dict,  # pyright: ignore[reportPrivateUsage]
    _load_json_object,  # pyright: ignore[reportPrivateUsage]
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

_logger = logging.getLogger(__name__)

_DEFAULT_PREFIX = 'pydantic-ai-harness:step'

_WRITE_SCRIPT = """
local op = ARGV[1]
local ttl = tonumber(ARGV[2])
if op == 'event' then
  if ARGV[3] == '0' or redis.call('SISMEMBER', KEYS[3], ARGV[4]) == 0 then
    redis.call('RPUSH', KEYS[2], ARGV[5])
    if ARGV[3] == '1' then
      redis.call('SADD', KEYS[3], ARGV[4])
    end
  end
elseif op == 'snapshot' then
  if ARGV[3] == '0' or redis.call('SISMEMBER', KEYS[7], ARGV[4]) == 0 then
    for i = 8, #ARGV, 2 do
      redis.call('HSETNX', KEYS[8], ARGV[i], ARGV[i + 1])
    end
    local seq = redis.call('INCR', KEYS[5])
    redis.call('HSET', KEYS[6], seq, ARGV[6])
    redis.call('ZADD', KEYS[4], seq, seq .. ':' .. ARGV[5])
    if ARGV[3] == '1' then
      redis.call('SADD', KEYS[7], ARGV[4])
    end
    local keep = tonumber(ARGV[7])
    if keep > 0 then
      local members = redis.call('ZRANGE', KEYS[4], 0, -1)
      local newest_complete = 0
      for i = #members, 1, -1 do
        if string.sub(members[i], -9) == ':complete' then
          newest_complete = i
          break
        end
      end
      for i = 1, #members - keep do
        if i ~= newest_complete then
          redis.call('HDEL', KEYS[6], string.match(members[i], '^[^:]*'))
          redis.call('ZREM', KEYS[4], members[i])
        end
      end
    end
  end
else
  redis.call('HSET', KEYS[9], ARGV[3], ARGV[4])
end
if ttl > 0 then
  for i = 1, #KEYS do
    redis.call('EXPIRE', KEYS[i], ttl)
  end
end
return 1
"""
"""Applies one write to a run, together with the replay check and the pruning it implies.

KEYS are the run's keys in `RedisStepStore._run_keys` order. ARGV starts with the
operation and the TTL in seconds (zero for none), followed by the operation's own values:

- `event`: whether a key was given (`1`/`0`), the key, the event JSON.
- `snapshot`: whether a key was given, the key, the state, the payload JSON, the
  `max_snapshots_per_run` bound (zero for none), then SHA-256 and message JSON pairs.
- `tool_effect`: the `tool_call_id` and the record JSON.

Redis runs a script to completion without interleaving another client's command, so two
workers replaying the same event or snapshot cannot both pass the `SISMEMBER` check, and no
reader sees a snapshot whose index member is written but whose payload or messages are not.
Issued as separate commands, a worker that died between them would leave either a payload
nothing points at, or a claimed key whose snapshot was never written, which a retry would
then skip as a replay.

The idempotency key is recorded after the write rather than claimed before it, because
Redis does not roll back what a script already did when a later command errors: claimed
first, a failed write would leave the key behind and the retry would be dropped.

The prune keeps the newest `keep` members plus the newest `complete` one, which is the
retain set `_retained_seqs` computes (the newest overall is always within the newest
`keep`). Stored messages are left alone, as in `PostgresStepStore` and `MongoStepStore`,
and expire with the rest of the run.

Every write refreshes the TTL on every key of the run, so the run expires as a unit,
`expire_seconds` after its last write. `EXPIRE` on a key that does not exist is a no-op.
"""


@runtime_checkable
class RedisClient(Protocol):
    """The part of a Redis client `RedisStepStore` uses.

    `redis.asyncio.Redis` satisfies this, as does any wrapper or fake exposing
    the same commands. Arguments are passed positionally (except `set`'s `ex`
    and `nx`), so a client that names them differently still works.

    Declared as returning `Awaitable` rather than as `async def`, which would
    narrow the requirement to a `Coroutine`: `redis.asyncio.Redis` types its
    commands as `Awaitable`, so an `async def` protocol refuses the one client
    this exists to accept.
    """

    def get(self, name: str, /) -> Awaitable[object]: ...  # pragma: no cover

    def set(
        self, name: str, value: str, /, *, ex: int | None = None, nx: bool = False
    ) -> Awaitable[object]: ...  # pragma: no cover

    def expire(self, name: str, time: int, /) -> Awaitable[object]: ...  # pragma: no cover

    def sadd(self, name: str, /, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def srem(self, name: str, /, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def smembers(self, name: str, /) -> Awaitable[Iterable[str | bytes]]: ...  # pragma: no cover

    def lrange(self, name: str, start: int, end: int, /) -> Awaitable[Sequence[str | bytes]]: ...  # pragma: no cover

    def zrange(self, name: str, start: int, end: int, /) -> Awaitable[Sequence[object]]:
        """Members by ascending score.

        `Sequence[object]` rather than of strings because a driver types this
        command for the `withscores=True` shape too. The store never asks for
        scores, and `_as_text` rejects a member that is not a string.
        """
        ...  # pragma: no cover

    def hget(self, name: str, key: str, /) -> Awaitable[object]: ...  # pragma: no cover

    def hmget(self, name: str, keys: Sequence[str], /) -> Awaitable[Sequence[object]]: ...  # pragma: no cover

    def hgetall(self, name: str, /) -> Awaitable[Mapping[str | bytes, str | bytes]]: ...  # pragma: no cover

    def eval(self, script: str, numkeys: int, /, *keys_and_args: str) -> Awaitable[object]: ...  # pragma: no cover


def _as_text(value: object) -> str:
    """Decode one reply, whichever encoding the client is configured for.

    `redis.asyncio.Redis` hands back `bytes` unless built with
    `decode_responses=True`, and both configurations have to work.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    raise ValueError(f'expected a string reply, got {type(value).__name__}')


def _tag(run_id: str) -> str:
    """Wrap `run_id` as a Redis Cluster hash tag.

    Only the substring between the first `{` and the following `}` decides a key's
    slot, so tagging every key of one run co-locates them on one node, which is what
    lets `_WRITE_SCRIPT` touch all of them. On a standalone server the braces are
    inert. See the Redis cluster spec, "Hash tags".

    A `run_id` that is empty or starts with `}` is refused: the tag would be `{}`,
    which is not a hash tag, so each key would hash on its full name and the script
    would fail with `CROSSSLOT`. A `}` later in the id only shortens the tag, the
    same way for every key of the run.
    """
    if not run_id or run_id.startswith('}'):
        raise ValueError(f'run_id must be non-empty and must not start with `}}`, got {run_id!r}')
    return '{' + run_id + '}'


def _validate_prefix(value: str) -> None:
    """Reject a prefix whose braces would claim the hash tag ahead of `{run_id}`."""
    if '{' in value or '}' in value:
        raise ValueError(f'prefix must not contain `{{` or `}}`, got {value!r}')


def _validate_expire_seconds(value: object) -> None:
    """Reject a TTL Redis would read as an immediate delete.

    `EXPIRE` treats zero or negative as "delete now", the opposite of what a
    caller asking for retention means, and a non-`int` would only fail at the
    first write. Mirrors `_validate_max_snapshots`.
    """
    if value is None:
        return
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'expire_seconds must be an int >= 1 or None, got {value!r}')


def _parse_member(member: str) -> tuple[int, SnapshotState]:
    """Split a `<seq>:<state>` snapshot index member."""
    seq_text, _, state_text = member.partition(':')
    return int(seq_text), _snapshot_state(state_text)


class _SnapshotPayload(BaseModel):
    """The JSON one `snapshot_payloads` field holds, written and read.

    `strict=True` because the payload is data on the wire, not a user argument:
    a `step_index` of `"0"` is a payload written by something that is not this
    store, and coercing it would hide that.
    """

    model_config = ConfigDict(strict=True)

    step_index: int
    timestamp: datetime
    message_refs: list[str]
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    state: SnapshotState = 'complete'
    idempotency_key: str | None = None


class RedisStepStore:
    """Redis-backed store shared by every worker pointed at the same server.

    ```python
    from redis.asyncio import Redis

    from pydantic_ai_harness.step_persistence import RedisStepStore

    store = RedisStepStore(Redis.from_url('redis://localhost'))
    ```

    `prefix` namespaces every key, so one server can hold several deployments.
    It must not contain `{` or `}`, which would take over the run's hash tag.

    `expire_seconds` (default `None`) gives a run's keys a TTL that every write
    to the run refreshes, so the run expires as a unit that long after its last
    write. The `runs:*` index sets take it too, refreshed by every write whose
    ids they index, so a set expires with the last run it holds. A set busier
    than the runs in it still holds ids whose run has expired; `list_runs`
    drops those as it reads.

    `media_store` defaults to `None`, unlike the file, sqlite, Mongo, and
    Postgres stores: payloads stay inline, because moving large binary or text
    parts into an in-memory database is a choice the caller should make
    deliberately. Pass any `MediaStore` to externalize parts at or above
    `media_threshold_bytes`.

    `max_snapshots_per_run` (default `None`, unbounded) prunes to the retain set
    after each write, inside the same script as the write.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        prefix: str = _DEFAULT_PREFIX,
        expire_seconds: int | None = None,
        media_store: MediaStore | None = None,
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        _validate_prefix(prefix)
        _validate_max_snapshots(max_snapshots_per_run)
        _validate_expire_seconds(expire_seconds)
        self._client = client
        self._prefix = prefix
        self._expire_seconds = expire_seconds
        self._media_store = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._max_snapshots_per_run = max_snapshots_per_run

    def _key(self, kind: str, run_id: str) -> str:
        return f'{self._prefix}:{kind}:{_tag(run_id)}'

    def _run_keys(self, run_id: str) -> list[str]:
        """Every key of one run, in the order `_WRITE_SCRIPT` indexes `KEYS`."""
        return [
            self._key(kind, run_id)
            for kind in (
                'run',
                'events',
                'event_keys',
                'snapshots',
                'snapshots:seq',
                'snapshot_payloads',
                'snapshot_keys',
                'messages',
                'tool_effects',
            )
        ]

    # The index sets are the only keys not scoped to one run, so they carry no hash
    # tag and are written outside `_WRITE_SCRIPT`. All three sit under `runs:` as
    # siblings, so none is a prefix of another and a `SCAN` for one does not sweep up
    # the rest.
    @property
    def _runs_key(self) -> str:
        return f'{self._prefix}:runs:all'

    def _conversation_key(self, conversation_id: str) -> str:
        return f'{self._prefix}:runs:conversation:{conversation_id}'

    def _parent_key(self, parent_run_id: str) -> str:
        return f'{self._prefix}:runs:parent:{parent_run_id}'

    def _index_keys(self, conversation_id: str | None, parent_run_id: str | None) -> list[str]:
        """The index sets a write carrying these ids belongs to; `runs:all` always does."""
        keys = [self._runs_key]
        if conversation_id is not None:
            keys.append(self._conversation_key(conversation_id))
        if parent_run_id is not None:
            keys.append(self._parent_key(parent_run_id))
        return keys

    async def _index(self, run_id: str, conversation_id: str | None, parent_run_id: str | None) -> None:
        """Add `run_id` to its index sets and refresh their expiry.

        Every write does this, not only `register_run`, so a worker that died between
        registering a run and indexing it is repaired by the run's next write:
        `StepPersistence` skips `register_run` on a retry that finds the run key.
        """
        for key in self._index_keys(conversation_id, parent_run_id):
            await self._client.sadd(key, run_id)
            if self._expire_seconds is not None:
                await self._client.expire(key, self._expire_seconds)

    async def _write(self, run_id: str, *args: str) -> None:
        keys = self._run_keys(run_id)
        await self._client.eval(_WRITE_SCRIPT, len(keys), *keys, *args)

    async def register_run(self, record: RunRecord) -> None:
        """Store lineage for a run, refusing a `run_id` that is already here.

        `SET ... NX` enforces the single-shot `run_id` contract that
        `SqliteStepStore` gets from its primary key. `StepPersistence.before_run`
        pre-checks with `get_run`; this covers a direct caller and the race
        between that check and the write. The index sets are written after the
        run key, so a refused duplicate never lists the run under its ids.
        """
        created = await self._client.set(
            self._key('run', record.run_id),
            json.dumps(_run_to_dict(record)),
            ex=self._expire_seconds,
            nx=True,
        )
        if not created:
            raise ValueError(f'run_id {record.run_id!r} is already in the store')
        await self._index(record.run_id, record.conversation_id, record.parent_run_id)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        raw = await self._client.get(self._key('run', run_id))
        if raw is None:
            return None
        return _run_from_dict(_load_json_object(_as_text(raw)))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        """Return matching `RunRecord`s sorted by `started_at` ascending.

        Reads the narrowest index set that answers the query, so a filtered
        listing does not walk every run, then checks each record against both
        filters. Sorts on the parsed instant, not the stored ISO string, which
        would misorder mixed offsets.

        A member whose run key is gone (its run expired while other runs kept
        the set alive) is removed. The run key is read again after the removal,
        and the member put back if it is there: a run being registered under a
        recycled id may have written its key in between, and its own `SADD`
        may already have run.
        """
        if parent_run_id is not None:
            index_key = self._parent_key(parent_run_id)
        elif conversation_id is not None:
            index_key = self._conversation_key(conversation_id)
        else:
            index_key = self._runs_key
        records: list[RunRecord] = []
        for member in await self._client.smembers(index_key):
            run_id = _as_text(member)
            raw = await self._client.get(self._key('run', run_id))
            if raw is None:
                await self._client.srem(index_key, run_id)
                raw = await self._client.get(self._key('run', run_id))
                if raw is None:
                    continue
                await self._client.sadd(index_key, run_id)
            record = _run_from_dict(_load_json_object(_as_text(raw)))
            if (parent_run_id is None or record.parent_run_id == parent_run_id) and (
                conversation_id is None or record.conversation_id == conversation_id
            ):
                records.append(record)
        return sorted(records, key=lambda record: record.started_at)

    async def append_event(self, event: StepEvent) -> None:
        key = event.idempotency_key
        await self._write(
            event.run_id,
            'event',
            str(self._expire_seconds or 0),
            '0' if key is None else '1',
            key or '',
            json.dumps(_event_to_dict(event)),
        )
        await self._index(event.run_id, event.conversation_id, event.parent_run_id)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        raw_events = await self._client.lrange(self._key('events', run_id), 0, -1)
        return [_event_from_dict(_load_json_object(_as_text(raw))) for raw in raw_events]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        """Write one snapshot under a freshly allocated `seq`, then prune, in one script.

        Media is externalized first, since it goes to a separate store; a save
        that fails after that leaves content-addressed blobs nothing references,
        which the other stores can leave too.
        """
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        keyed = keyed_messages(_OBJECT_LIST_ADAPTER.validate_python(messages_json))
        payload = _SnapshotPayload(
            step_index=snapshot.step_index,
            timestamp=snapshot.timestamp,
            message_refs=[sha256 for sha256, _ in keyed],
            conversation_id=snapshot.conversation_id,
            parent_run_id=snapshot.parent_run_id,
            agent_name=snapshot.agent_name,
            state=snapshot.state,
            idempotency_key=snapshot.idempotency_key,
        )
        key = snapshot.idempotency_key
        await self._write(
            snapshot.run_id,
            'snapshot',
            str(self._expire_seconds or 0),
            '0' if key is None else '1',
            key or '',
            snapshot.state,
            payload.model_dump_json(),
            str(self._max_snapshots_per_run or 0),
            *(value for pair in keyed for value in pair),
        )
        await self._index(snapshot.run_id, snapshot.conversation_id, snapshot.parent_run_id)

    async def _snapshot_entries(self, run_id: str) -> list[tuple[int, SnapshotState]]:
        """The run's `(seq, state)` pairs, ascending, from the index alone."""
        members = await self._client.zrange(self._key('snapshots', run_id), 0, -1)
        return [_parse_member(_as_text(member)) for member in members]

    async def _load_snapshot(self, run_id: str, seq: int) -> ContinuableSnapshot | None:
        """Rebuild one snapshot, or `None` when a concurrent save pruned it after the index was read."""
        raw = await self._client.hget(self._key('snapshot_payloads', run_id), str(seq))
        if raw is None:
            return None
        payload = _SnapshotPayload.model_validate_json(_as_text(raw))
        stored: dict[str, str] = {}
        if payload.message_refs:
            found = await self._client.hmget(self._key('messages', run_id), payload.message_refs)
            stored = {
                ref: _as_text(message) for ref, message in zip(payload.message_refs, found) if message is not None
            }
        messages_json: object = json.loads(referenced_history_json(payload.message_refs, stored))
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=payload.step_index,
            messages=messages,
            conversation_id=payload.conversation_id,
            parent_run_id=payload.parent_run_id,
            agent_name=payload.agent_name,
            timestamp=payload.timestamp,
            state=payload.state,
            idempotency_key=payload.idempotency_key,
        )

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        """The newest snapshot passing the state gate, or `None`.

        Walks candidates newest-first, skipping one whose payload a concurrent
        prune removed after the index was read. Retention keeps the newest
        overall and the newest `complete`, so skipping still reaches the right
        snapshot.
        """
        for seq, state in reversed(await self._snapshot_entries(run_id)):
            if state == 'interrupted' and not include_interrupted:
                continue
            snapshot = await self._load_snapshot(run_id, seq)
            if snapshot is not None:
                return snapshot
        return None

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate. A snapshot that fails to load is
        skipped and logged, so one damaged value does not hide the rest of the
        run. Not part of the `StepStore` protocol: `conversation_search`
        consumes it through its narrower `SnapshotStore`.
        """
        snapshots: list[ContinuableSnapshot] = []
        for seq, state in await self._snapshot_entries(run_id):
            if state == 'interrupted' and not include_interrupted:
                continue
            try:
                snapshot = await self._load_snapshot(run_id, seq)
            except Exception:
                _logger.warning('Skipping unparsable snapshot for run %s', run_id, exc_info=True)
                continue
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._write(
            record.run_id,
            'tool_effect',
            str(self._expire_seconds or 0),
            record.tool_call_id,
            json.dumps(_tool_effect_to_dict(record)),
        )
        # `ToolEffectRecord` carries no lineage ids, so only `runs:all` is written here;
        # `StepPersistence` follows each tool-effect write with an event that carries them.
        await self._index(record.run_id, None, None)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        raw = await self._client.hget(self._key('tool_effects', run_id), tool_call_id)
        if raw is None:
            return None
        return _tool_effect_from_dict(_load_json_object(_as_text(raw)))

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        fields = await self._client.hgetall(self._key('tool_effects', run_id))
        records = [_tool_effect_from_dict(_load_json_object(_as_text(raw))) for raw in fields.values()]
        return [record for record in records if record.status == 'started']
