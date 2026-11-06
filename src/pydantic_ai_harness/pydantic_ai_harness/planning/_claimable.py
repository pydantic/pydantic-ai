"""Claimable plan stores: a plan several agents share, with atomic claims.

A `ClaimablePlanStore` is a `PlanStore` whose steps carry an `owner`. `claim` and
`release` read the whole board, decide, and write it back as one atomic operation,
so N agents racing for the same step produce exactly one winner, and completing a
step moves its dependents out of `blocked` exactly once.

Every backend implements the same two primitives on `BoardStore`: a read of the
whole board, and `_transact`, which applies a pure change to the board atomically.
The board rules (`_claim`, `_release`, `_replace`) are shared, so the backends can
only differ in how they make `_transact` atomic.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import anyio.to_thread
from pydantic import TypeAdapter

from pydantic_ai_harness.planning._store import PlanStore, apply_updates, validate_table_name
from pydantic_ai_harness.planning._toolset import dependency_block_transitions, is_blocked
from pydantic_ai_harness.planning._types import OwnedPlanItem, PlanItem, TaskStatus

if sys.platform == 'win32':  # pragma: no cover
    import msvcrt
else:
    import fcntl

_T = TypeVar('_T')

Board = list[OwnedPlanItem]
"""Every step of one board, in insertion order."""

BOARD_ADAPTER = TypeAdapter(Board)
"""Serializes a whole board to and from JSON, for the stores that keep it as one document."""


@runtime_checkable
class ClaimablePlanStore(PlanStore, Protocol):
    """A `PlanStore` that several agents share, with atomic claims on its steps.

    `claim` and `release` are each one atomic operation on the whole board, so
    concurrent claimers cannot both take a step, and a step's dependents move in
    the same operation as the step. Dependencies and `blocked` follow
    `Planning(enable_subtasks=True)` semantics: a step whose `depends_on` names an
    unresolved (not completed or cancelled) step is not available.

    The inherited `PlanStore` methods keep `owner` on every step id they leave in
    place, so a `set_items` or `update_item` racing a claim cannot undo it. Steps
    they add start unowned. They read and write plain `PlanItem`s; use
    `get_owned_items` and `get_owned_item` to read owners.
    """

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        """Return every step, with its owner, in insertion order, as detached copies."""
        ...  # pragma: no cover

    async def get_owned_item(self, item_id: str) -> OwnedPlanItem | None:
        """Return the step with `item_id`, with its owner, or `None`."""
        ...  # pragma: no cover

    async def claim(self, item_id: str | None, *, owner: str) -> OwnedPlanItem | None:
        """Atomically take a step for `owner` and move it to `in_progress`.

        A step can be claimed when it is `pending`, unowned or already owned by
        `owner`, and none of its dependencies is unresolved. With `item_id=None`,
        take the first such step in board order.

        Returns:
            The claimed step, or `None` when `item_id` is missing or not claimable,
            or no step is available.
        """
        ...  # pragma: no cover

    async def release(self, item_id: str, *, owner: str, status: TaskStatus) -> list[OwnedPlanItem]:
        """Atomically move `owner`'s step to `status` and apply its dependents' block transitions.

        `completed` and `cancelled` keep `owner` as a record of who finished the
        step; `pending` and `blocked` clear it, so the step can be claimed again.
        A dependent with an unresolved dependency becomes `blocked`, and a
        `blocked` dependent whose dependencies are all resolved returns to `pending`.

        Returns:
            Every step the release changed, in board order, so callers can emit one
            event each. Empty when the step is missing, not owned by `owner`, or
            already in that state.

        Raises:
            ValueError: `status` is `in_progress`; use `claim` to start a step.
        """
        ...  # pragma: no cover


def _plain(item: OwnedPlanItem) -> PlanItem:
    """`item` without its owner, for the `PlanStore` reads that promise plain `PlanItem`s."""
    return PlanItem.model_validate(item.model_dump(exclude={'owner'}))


def _owned(item: PlanItem, owner: str | None) -> OwnedPlanItem:
    """A detached copy of `item` owned by `owner`, whatever owner `item` itself carries."""
    return OwnedPlanItem.model_validate({**item.model_dump(), 'owner': owner})


def _find(board: Board, item_id: str) -> OwnedPlanItem | None:
    return next((item for item in board if item.id == item_id), None)


def _claimable(board: Board, item: OwnedPlanItem, owner: str) -> bool:
    return (
        item.status is TaskStatus.pending
        and item.owner in (None, owner)
        and not is_blocked(list[PlanItem](board), item)
    )


def _claim(board: Board, item_id: str | None, owner: str) -> OwnedPlanItem | None:
    candidates = board if item_id is None else [item for item in board if item.id == item_id]
    item = next((item for item in candidates if _claimable(board, item, owner)), None)
    if item is None:
        return None
    item.status = TaskStatus.in_progress
    item.owner = owner
    return item.model_copy(deep=True)


def _release(board: Board, item_id: str, owner: str, status: TaskStatus) -> list[OwnedPlanItem]:
    item = _find(board, item_id)
    if item is None or item.owner != owner:
        return []
    before = [entry.model_copy(deep=True) for entry in board]
    item.status = status
    if status in (TaskStatus.pending, TaskStatus.blocked):
        item.owner = None
    for dependent, new_status in list(dependency_block_transitions(list[PlanItem](board))):
        if item_id in dependent.depends_on:
            dependent.status = new_status
    return [entry.model_copy(deep=True) for entry, old in zip(board, before) if entry != old]


def _replace(board: Board, items: list[PlanItem]) -> None:
    ids = [item.id for item in items]
    if duplicates := sorted({item_id for item_id in ids if ids.count(item_id) > 1}):
        raise ValueError(f'Duplicate step ids: {", ".join(map(repr, duplicates))}.')
    owners = {item.id: item.owner for item in board}
    board[:] = [_owned(item, owners.get(item.id)) for item in items]


def _add(board: Board, item: PlanItem) -> PlanItem:
    if _find(board, item.id) is not None:
        raise ValueError(f'A step with id {item.id!r} is already in this plan.')
    board.append(_owned(item, None))
    return _plain(board[-1])


def _remove(board: Board, item_id: str) -> bool:
    item = _find(board, item_id)
    if item is None:
        return False
    board.remove(item)
    return True


def apply_change(board: Board, change: Callable[[Board], _T]) -> tuple[_T, bool]:
    """Apply `change` to `board` in place; return its result and whether the board changed."""
    before = [item.model_copy(deep=True) for item in board]
    result = change(board)
    return result, board != before


class BoardStore(ABC):
    """The shared `ClaimablePlanStore` implementation over two backend primitives."""

    @abstractmethod
    async def get_owned_items(self) -> list[OwnedPlanItem]:
        """Return every step, with its owner, in insertion order, as detached copies."""

    @abstractmethod
    async def _transact(self, change: Callable[[Board], _T]) -> _T:
        """Atomically read the board, apply `change` to it in place, and write it back if it changed.

        `change` is synchronous and pure apart from mutating the board, so a backend
        may run it more than once (optimistic retries) or on a worker thread. When it
        raises, nothing is written.
        """

    async def get_owned_item(self, item_id: str) -> OwnedPlanItem | None:
        """Return the step with `item_id`, with its owner, or `None`."""
        return _find(await self.get_owned_items(), item_id)

    async def get_items(self) -> list[PlanItem]:
        """Return every step in insertion order, without owners."""
        return [_plain(item) for item in await self.get_owned_items()]

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id` without its owner, or `None`."""
        item = await self.get_owned_item(item_id)
        return None if item is None else _plain(item)

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list with `items`, keeping the owner of every id that stays.

        Raises:
            ValueError: Two of `items` share an id.
        """
        await self._transact(lambda board: _replace(board, items))

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item`, unowned, and return it.

        Raises:
            ValueError: A step with the same id is already on the board.
        """
        return await self._transact(lambda board: _add(board, item))

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
        """Apply the non-`None` fields to `item_id`, keeping its owner; return the updated step or `None`."""

        def update(board: Board) -> PlanItem | None:
            item = _find(board, item_id)
            if item is None:
                return None
            apply_updates(
                item,
                content=content,
                status=status,
                active_form=active_form,
                parent_id=parent_id,
                depends_on=depends_on,
            )
            return _plain(item)

        return await self._transact(update)

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id`; return whether it existed."""
        return await self._transact(lambda board: _remove(board, item_id))

    async def claim(self, item_id: str | None, *, owner: str) -> OwnedPlanItem | None:
        """Atomically take `item_id`, or the first available step, for `owner`, and start it.

        See `ClaimablePlanStore.claim`.
        """
        return await self._transact(lambda board: _claim(board, item_id, owner))

    async def release(self, item_id: str, *, owner: str, status: TaskStatus) -> list[OwnedPlanItem]:
        """Atomically move `owner`'s step to `status` and update its dependents.

        See `ClaimablePlanStore.release`.
        """
        if status is TaskStatus.in_progress:
            raise ValueError("Cannot release a step to 'in_progress'; claim it instead.")
        return await self._transact(lambda board: _release(board, item_id, owner, status))


class InMemoryClaimablePlanStore(BoardStore):
    """In-process claimable board; state is lost on exit.

    Every operation applies its change without awaiting in between, so it is
    atomic with respect to every other task on the event loop. Not safe to share
    across threads or event loops.
    """

    def __init__(self) -> None:
        self._board: Board = []

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        """Return every step, with its owner, in insertion order, as detached copies."""
        return [item.model_copy(deep=True) for item in self._board]

    async def _transact(self, change: Callable[[Board], _T]) -> _T:
        board = [item.model_copy(deep=True) for item in self._board]
        result = change(board)
        self._board = board
        return result


class SqliteClaimablePlanStore(BoardStore):
    """SQLite-backed claimable board, scoped to a `session`.

    Each write runs as one `BEGIN IMMEDIATE` transaction on a worker thread, which
    takes SQLite's write lock before reading the board, so concurrent writers on
    the same file serialize across threads and processes. The table is its own
    (`claimable_plan_items` by default, with an `owner` column) and does not share
    rows with `SqlitePlanStore`.

    `':memory:'` is not supported because each operation opens its own connection.
    """

    def __init__(self, database: str | Path, *, session: str = 'default', table: str = 'claimable_plan_items') -> None:
        validate_table_name(table)
        if str(database) == ':memory:':
            raise ValueError(
                "SqliteClaimablePlanStore does not support ':memory:'; "
                'use InMemoryClaimablePlanStore or a file-backed database.'
            )
        self._database = str(database)
        self._session = session
        self._table = table
        self._lock = threading.Lock()
        self._ready = False

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self._database, isolation_level=None)
            try:
                if not self._ready:
                    connection.execute(
                        f'CREATE TABLE IF NOT EXISTS {self._table} ('
                        'session TEXT NOT NULL, seq INTEGER NOT NULL, id TEXT NOT NULL, '
                        'content TEXT NOT NULL, status TEXT NOT NULL, active_form TEXT NOT NULL, '
                        'parent_id TEXT, depends_on TEXT NOT NULL, owner TEXT, PRIMARY KEY (session, id))'
                    )
                    self._ready = True
                yield connection
            finally:
                connection.close()

    def _load(self, connection: sqlite3.Connection) -> Board:
        rows = connection.execute(
            f'SELECT id, content, status, active_form, parent_id, depends_on, owner '
            f'FROM {self._table} WHERE session = ? ORDER BY seq',
            (self._session,),
        ).fetchall()
        return [row_to_item(row) for row in rows]

    def _get_owned_items_sync(self) -> Board:
        with self._connect() as connection:
            return self._load(connection)

    def _transact_sync(self, change: Callable[[Board], _T]) -> _T:
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                board = self._load(connection)
                result, changed = apply_change(board, change)
                if changed:
                    connection.execute(f'DELETE FROM {self._table} WHERE session = ?', (self._session,))
                    connection.executemany(
                        f'INSERT INTO {self._table} '
                        '(session, seq, id, content, status, active_form, parent_id, depends_on, owner) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        [(self._session, seq, *item_to_row(item)) for seq, item in enumerate(board)],
                    )
            except BaseException:
                connection.execute('ROLLBACK')
                raise
            connection.execute('COMMIT')
            return result

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        """Return every step for this session, with its owner, in insertion order."""
        return await anyio.to_thread.run_sync(self._get_owned_items_sync)

    async def _transact(self, change: Callable[[Board], _T]) -> _T:
        return await anyio.to_thread.run_sync(self._transact_sync, change)


class FileClaimablePlanStore(BoardStore):
    """A claimable board kept as one JSON file, for a host that keeps its state in a directory.

    Each write holds an exclusive OS file lock (`flock`, or `msvcrt.locking` on
    Windows) on a `<name>.lock` file next to `path` while it reads the board and
    replaces the file, so writers serialize across threads and processes. The
    board is replaced with an atomic rename, so a reader never sees a partial
    write and takes no lock.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock_path = self.path.with_name(f'{self.path.name}.lock')

    def _load(self) -> Board:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return []
        return BOARD_ADAPTER.validate_json(data)

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open('a+b') as handle:
            if sys.platform == 'win32':  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _transact_sync(self, change: Callable[[Board], _T]) -> _T:
        with self._locked():
            board = self._load()
            result, changed = apply_change(board, change)
            if changed:
                temporary = self.path.with_name(f'{self.path.name}.tmp')
                temporary.write_bytes(BOARD_ADAPTER.dump_json(board))
                temporary.replace(self.path)
            return result

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        """Return every step, with its owner, in insertion order."""
        return await anyio.to_thread.run_sync(self._load)

    async def _transact(self, change: Callable[[Board], _T]) -> _T:
        return await anyio.to_thread.run_sync(self._transact_sync, change)


def item_to_row(item: OwnedPlanItem) -> tuple[object, ...]:
    """The column values of `item` after `session` and `seq`, for the SQL stores."""
    return (
        item.id,
        item.content,
        item.status.value,
        item.active_form,
        item.parent_id,
        json.dumps(item.depends_on),
        item.owner,
    )


def row_to_item(row: Sequence[object]) -> OwnedPlanItem:
    """A step from `id, content, status, active_form, parent_id, depends_on, owner` columns."""
    return OwnedPlanItem(
        id=str(row[0]),
        content=str(row[1]),
        status=TaskStatus(str(row[2])),
        active_form=str(row[3]),
        parent_id=None if row[4] is None else str(row[4]),
        depends_on=list(json.loads(str(row[5]))),
        owner=None if row[6] is None else str(row[6]),
    )
