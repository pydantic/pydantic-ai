"""A remote Monty pool double that stores sessions the way a `monty-server` with an object store does.

`AsyncMontyRemote` against such a server gives each session a `session_id`, `dump()` writes the state
to a record that never changes and returns the record's ID, and `load_session`/`load_snapshot` start
a new session from an ID. Neither the repository nor `pydantic-monty` ships that server, so this
backs the same calls with local workers and keeps the records in `SERVER`, shared by every pool, as
the server's store is shared by every connection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pydantic_monty import (
    AsyncMonty,
    AsyncMontySession,
    AsyncSnapshot,
    MountDir,
    OsHandler,
    OSPolicy,
    PrintCallback,
    ResourceLimits,
)


@dataclass
class Server:
    """The store every pool shares. Tests change it to model a server that loses records or stops storing."""

    records: dict[bytes, bytes] = field(default_factory=dict[bytes, bytes])
    stores: bool = True
    drops_next_feed: Exception | None = None
    """An error the next `feed_start` raises, as a connection that drops mid-feed does."""

    def load(self, record_id: bytes) -> bytes:
        # Empty bytes are not a Monty dump, so loading a lost record raises Monty's own error.
        return self.records.get(record_id, b'')


SERVER = Server()


@dataclass
class StoredSession:
    """A checked-out local session whose `dump()` returns a record ID instead of the state."""

    session: AsyncMontySession
    stores: bool

    @property
    def session_id(self) -> bytes | None:
        return b'session' if self.stores else None

    async def dump(self) -> bytes:
        record_id = f'record-{len(SERVER.records)}'.encode()
        SERVER.records[record_id] = await self.session.dump()
        return record_id

    async def load_session(self, state: bytes) -> None:
        await self.session.load_session(SERVER.load(state))

    async def load_snapshot(
        self,
        state: bytes,
        *,
        mount: MountDir | list[MountDir] | None = None,
        print_callback: PrintCallback | None = None,
    ) -> AsyncSnapshot:
        return await self.session.load_snapshot(SERVER.load(state), mount=mount, print_callback=print_callback)

    async def feed_start(
        self,
        code: str,
        *,
        print_callback: PrintCallback | None = None,
        mount: MountDir | list[MountDir] | None = None,
        os: OsHandler | None = None,
        skip_type_check: bool = False,
    ) -> AsyncSnapshot:
        if (error := SERVER.drops_next_feed) is not None:
            SERVER.drops_next_feed = None
            raise error
        return await self.session.feed_start(
            code, print_callback=print_callback, mount=mount, os=os, skip_type_check=skip_type_check
        )


@dataclass
class StoredSessionPool:
    """Stands in for `AsyncMontyRemote`: an async context manager whose `checkout()` yields `StoredSession`s."""

    stores: bool = True
    """`False` models a server without a store; `SERVER.stores` is what a test flips mid-run."""
    _pool: AsyncMonty = field(default_factory=AsyncMonty)

    async def __aenter__(self) -> StoredSessionPool:
        await self._pool.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._pool.__aexit__(*args)

    @asynccontextmanager
    async def checkout(
        self,
        *,
        limits: ResourceLimits | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        os_policy: OSPolicy | None = None,
        ephemeral: bool | None = None,
    ) -> AsyncGenerator[StoredSession]:
        async with self._pool.checkout(
            limits=limits, type_check=type_check, type_check_stubs=type_check_stubs, os_policy=os_policy
        ) as session:
            # As on a storing server, an ephemeral session is never stored.
            yield StoredSession(session, self.stores and SERVER.stores and ephemeral is not True)
