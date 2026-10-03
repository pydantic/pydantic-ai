"""The connection surface the SQLite-backed stores use.

The stores open stdlib `sqlite3` themselves for a `database=` path, and that stays the default.
A caller-owned `connection=` must speak SQLite and provide connection-level `execute`, `commit`,
and `rollback` methods plus `in_transaction`; the step store additionally needs `executescript`.
Both stdlib `sqlite3` and `pyturso` provide those surfaces.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

SqliteParameters = Sequence[object] | Mapping[str, object]
SqliteRow = Sequence[object]


class SqliteCursor(Protocol):
    """The cursor surface the stores use."""

    @property
    def rowcount(self) -> int: ...  # pragma: no cover

    def fetchone(self) -> SqliteRow | None: ...  # pragma: no cover

    def fetchall(self) -> list[SqliteRow]: ...  # pragma: no cover

    def __iter__(self) -> Iterator[SqliteRow]: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover


class SqliteConnection(Protocol):
    """The SQLite connection surface used by the stores."""

    def execute(self, sql: str, parameters: SqliteParameters = ..., /) -> SqliteCursor: ...  # pragma: no cover

    def commit(self) -> None: ...  # pragma: no cover

    def rollback(self) -> None: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover

    @property
    def in_transaction(self) -> bool: ...  # pragma: no cover


class SqliteScriptConnection(SqliteConnection, Protocol):
    """A SQLite connection with the optional connection-level script extension."""

    def executescript(self, sql_script: str, /) -> SqliteCursor: ...  # pragma: no cover


def is_database_error(connection: SqliteConnection, error: Exception) -> bool:
    """Whether *error* belongs to the connection driver's DB-API database hierarchy."""
    database_error = getattr(connection, 'DatabaseError', None)
    if isinstance(database_error, type) and issubclass(database_error, Exception) and isinstance(error, database_error):
        return True
    for object_type in (*type(connection).__mro__, *type(error).__mro__):
        module = sys.modules.get(object_type.__module__)
        database_error = getattr(module, 'DatabaseError', None)
        if (
            isinstance(database_error, type)
            and issubclass(database_error, Exception)
            and isinstance(error, database_error)
        ):
            return True
    return False
