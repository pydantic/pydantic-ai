from __future__ import annotations as _annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import logfire
from typing_extensions import LiteralString

P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, schema_sql: str, file: Path = Path('.chat_app_messages.sqlite')
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, schema_sql, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(schema_sql: str, file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(schema_sql)
        con.commit()
        return con

    async def execute(self, sql: LiteralString, *args: Any, commit: bool = False):
        await self._asyncify(self._execute, sql, *args, commit=True)
        if commit:
            await self._asyncify(self.con.commit)

    async def fetchall(self, sql: LiteralString, *args: Any) -> list[tuple[str, ...]]:
        c = await self._asyncify(self._execute, sql, *args)
        rows = await self._asyncify(c.fetchall)
        return [tuple(row) for row in rows]

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )
