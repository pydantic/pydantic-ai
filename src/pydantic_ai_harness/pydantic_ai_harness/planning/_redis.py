"""Redis plan storage over a caller-owned redis.asyncio-compatible client.

Like `PostgresPlanStore`, this store never imports `redis`: it depends only on
the `RedisClient` protocol below, so the harness carries no Redis driver
dependency and the backend is testable with an in-memory fake client. Pass your
own `redis.asyncio.Redis` client (which already satisfies this protocol) at
construction.

The whole plan for a session is stored as one JSON document under a single key.
`set_items` is a single `SET` and so is atomic; the granular operations
(`add_item`, `update_item`, `remove_item`) are read-modify-write, so two tasks
mutating the same session concurrently can clobber each other. Drive one session
from one task, or use `set_items` if you need atomic whole-plan writes. The
`{session}` hash-tag keeps the key on one slot under Redis Cluster.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic_ai_harness.planning._events import PlanEventEmitter
from pydantic_ai_harness.planning._store import apply_updates, emit_created, emit_deleted, emit_mutation
from pydantic_ai_harness.planning._types import PlanItem, TaskStatus


@runtime_checkable
class RedisClient(Protocol):
    """The redis.asyncio-compatible client surface used by `RedisPlanStore`."""

    async def get(self, key: str) -> object:
        """Return the value at `key`, or `None`."""
        ...  # pragma: no cover

    async def set(self, key: str, value: str, *, ex: int | None = None) -> object:
        """Set `key` to `value`, optionally expiring it after `ex` seconds."""
        ...  # pragma: no cover


class RedisPlanStore:
    """Redis plan storage scoped to a `session` for multi-tenancy.

    Pass `expire_seconds` to give the session key a TTL (refreshed on every
    write), so plans for abandoned sessions expire instead of living forever.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        session: str = 'default',
        key_prefix: str = 'plan',
        expire_seconds: int | None = None,
        event_emitter: PlanEventEmitter | None = None,
    ) -> None:
        self._client = client
        self._session = session
        self._key_prefix = key_prefix
        self._expire_seconds = expire_seconds
        self._emitter = event_emitter

    @property
    def _key(self) -> str:
        # The `{session}` hash-tag keeps this key on a single Redis Cluster slot.
        return f'{self._key_prefix}:{{{self._session}}}'

    async def _load(self) -> list[PlanItem]:
        raw = await self._client.get(self._key)
        if raw is None:
            return []
        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        return [PlanItem.model_validate(entry) for entry in json.loads(text)]

    async def _save(self, items: list[PlanItem]) -> None:
        payload = json.dumps([item.model_dump(mode='json') for item in items])
        await self._client.set(self._key, payload, ex=self._expire_seconds)

    async def get_items(self) -> list[PlanItem]:
        """Return every step for this session in insertion order."""
        return await self._load()

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list for this session with `items`."""
        await self._save(list(items))

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id` for this session, or `None`."""
        return next((item for item in await self._load() if item.id == item_id), None)

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` for this session and return it."""
        items = await self._load()
        if any(existing.id == item.id for existing in items):
            raise ValueError(f'A step with id {item.id!r} is already in this plan.')
        items.append(item)
        await self._save(items)
        await emit_created(self._emitter, item)
        return item

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        items = await self._load()
        for item in items:
            if item.id == item_id:
                previous = item.model_copy() if self._emitter is not None else None
                apply_updates(
                    item,
                    content=content,
                    status=status,
                    active_form=active_form,
                    parent_id=parent_id,
                    depends_on=depends_on,
                )
                await self._save(items)
                await emit_mutation(self._emitter, item, previous)
                return item
        return None

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id` for this session; return whether it existed."""
        items = await self._load()
        for index, item in enumerate(items):
            if item.id == item_id:
                removed = items.pop(index)
                await self._save(items)
                await emit_deleted(self._emitter, removed)
                return True
        return False
