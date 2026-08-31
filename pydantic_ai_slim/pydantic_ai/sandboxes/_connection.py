"""Private connection teardown shared by sandbox facades and policy wrappers."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast


async def close_backend_connection(backend: object) -> None:
    """Call the optional async `close(terminate=False)` contract when its signature accepts it."""
    close = getattr(backend, 'close', None)
    if close is None or not inspect.iscoroutinefunction(close):
        return
    try:
        inspect.signature(close).bind(terminate=False)
    except (TypeError, ValueError):
        return
    await cast(Callable[..., Awaitable[Any]], close)(terminate=False)
