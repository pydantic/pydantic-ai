"""Optional authoring support for sandbox backends backed by a native SDK object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Generic, TypeVar

import anyio

__all__ = ('LazySandbox',)

SandboxT = TypeVar('SandboxT')


class LazySandbox(Generic[SandboxT], ABC):
    """Share lazy acquisition and caching across provider backend implementations.

    Subclasses implement `create_or_attach` and their normal backend operations.
    Operations await `self.sandbox` to obtain the native SDK object. This helper
    does not change Pydantic AI's `SandboxBackend` protocol or manage remote lifetime.

    Concurrent acquisition is serialized per instance. A successful handle is cached;
    failed or cancelled acquisition is not cached, so a waiting or later caller can
    try again. Cancelling a waiter does not cancel another caller's acquisition.

    Args:
        sandbox: An already acquired native handle, if one is available.
    """

    def __init__(self, sandbox: SandboxT | None = None) -> None:
        """Retain an optional native handle without performing acquisition."""
        self._live = sandbox
        self._lock = anyio.Lock()

    @property
    def sandbox(self) -> Awaitable[SandboxT]:
        """The native SDK object, acquired when awaited and reused on later awaits.

        Each access is awaitable, including when a handle was passed to the constructor.
        A cached handle is not checked for remote liveness on subsequent access.
        """
        return self._resolve()

    async def _resolve(self) -> SandboxT:
        async with self._lock:
            if self._live is None:
                self._live = await self.create_or_attach()
            return self._live

    @abstractmethod
    async def create_or_attach(self) -> SandboxT:
        """Acquire a native SDK object or raise if it cannot be acquired.

        This is the provider implementation hook. Callers use `await self.sandbox`
        to get coordination and caching. The provider owns identity updates, SDK
        error translation, and cleanup of resources left by failed or cancelled
        acquisition. Returning successfully must leave the handle ready for operations.
        """
        ...
