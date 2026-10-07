"""Connection lifetime for explicit, sequential realtime runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING

import anyio
from anyio.abc import TaskGroup, TaskStatus

from ..exceptions import UserError
from ..models import ModelRequestParameters
from .codec import RealtimeConnection
from .model import RealtimeModel
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    from ._session import RealtimeSession


class RealtimeAttachment:
    """Keep transport entry and exit in the same task, independently of run-driving tasks."""

    def __init__(self, group: TaskGroup) -> None:
        self.group = group
        self.model: RealtimeModel | None = None
        self.requested_model: RealtimeModel | str | None = None
        self.session: RealtimeSession | None = None
        self.connection: RealtimeConnection | None = None
        self.settings: RealtimeModelSettings | None = None
        self.parameters: ModelRequestParameters | None = None
        self.instructions: str | None = None
        self.stopping = anyio.Event()
        self.stopped = anyio.Event()
        self.closed = False

    @asynccontextmanager
    async def connect(
        self,
        manager: AbstractAsyncContextManager[RealtimeConnection],
        *,
        model: RealtimeModel,
        settings: RealtimeModelSettings | None,
        parameters: ModelRequestParameters,
        instructions: str | None,
    ) -> AsyncGenerator[RealtimeConnection]:
        if self.closed:
            raise UserError('This realtime connection has closed.')
        if self.connection is None:
            self.connection = await self.group.start(self._hold, manager)
            assert self.connection is not None
            self.model = model
            self.settings = settings
            self.parameters = parameters
            self.instructions = instructions
        elif settings != self.settings or parameters != self.parameters or instructions != self.instructions:
            raise UserError(
                'This realtime connection cannot change its instructions, tool schemas or model settings between runs; '
                'open a new connection for the changed configuration.'
            )
        yield self.connection

    async def _hold(
        self,
        manager: AbstractAsyncContextManager[RealtimeConnection],
        *,
        task_status: TaskStatus[RealtimeConnection],
    ) -> None:
        try:
            with anyio.CancelScope() as scope:
                async with manager as connection:
                    scope.shield = True
                    task_status.started(connection)
                    await self.stopping.wait()
        finally:
            self.stopped.set()

    @asynccontextmanager
    async def run(self, session: RealtimeSession) -> AsyncGenerator[None]:
        if self.session is None:
            self.session = session
            session._persistent = True  # pyright: ignore[reportPrivateUsage]
            session._run.task_context = copy_context()  # pyright: ignore[reportPrivateUsage]
            await session.__aenter__()
        handle = session._run.handle  # pyright: ignore[reportPrivateUsage]
        assert handle is not None
        try:
            try:
                yield
                await handle._stop_inputs()  # pyright: ignore[reportPrivateUsage]
                await session._finish_run()  # pyright: ignore[reportPrivateUsage]
            except BaseException:
                # Normal exit drains sends, but an abort must cancel blocked sends before waiting
                # for their producers. The connection is then closed, never reused at an unknown frontier.
                with anyio.CancelScope(shield=True):
                    await handle._stop_inputs(abort=True)  # pyright: ignore[reportPrivateUsage]
                raise
            finally:
                await handle._close_streams()  # pyright: ignore[reportPrivateUsage]
        except BaseException as exc:
            session._closing_error = exc  # pyright: ignore[reportPrivateUsage]
            await self.close()
            raise

    async def close(self) -> None:
        self.closed = True
        with anyio.CancelScope(shield=True):
            try:
                if self.session is not None:
                    await self.session.close()
            finally:
                self.stopping.set()
                if self.connection is not None:
                    await self.stopped.wait()
