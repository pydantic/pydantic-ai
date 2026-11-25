"""A temporary directory for async code, made and removed off the event loop."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from anyio import CancelScope, to_thread


@asynccontextmanager
async def temporary_directory(*, prefix: str | None = None) -> AsyncGenerator[Path]:
    """A new temporary directory, removed with everything in it on exit, even when the caller is cancelled."""
    folder = await to_thread.run_sync(partial(TemporaryDirectory[str], prefix=prefix))
    try:
        yield Path(folder.name)
    finally:
        with CancelScope(shield=True):
            await to_thread.run_sync(folder.cleanup)
