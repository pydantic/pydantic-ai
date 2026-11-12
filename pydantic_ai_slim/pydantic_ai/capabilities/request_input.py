"""The deprecated `RequestInput` capability: `HandleDeferredToolCalls` with a handler that takes a resume token."""

# TODO(v3): remove this module and the `RequestInput` export from `capabilities/__init__.py`.

from __future__ import annotations

from datetime import timedelta

from typing_extensions import TypeIs, deprecated

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.tools import AgentDepsT

from .deferred_tool_handler import (
    _DEFAULT_WAIT_NAME,  # pyright: ignore[reportPrivateUsage]
    HandleDeferredToolCalls,
    _Handler,  # pyright: ignore[reportPrivateUsage]
    _TokenHandler,  # pyright: ignore[reportPrivateUsage]
)


@deprecated(
    '`RequestInput` is deprecated; use `HandleDeferredToolCalls`, whose handler gets a resume token when it '
    'takes a third parameter: `HandleDeferredToolCalls(ask, timeout=..., name=...)`.',
    category=PydanticAIDeprecationWarning,
)
class RequestInput(HandleDeferredToolCalls[AgentDepsT]):
    """Deprecated: use [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] with a handler that takes a resume token.

    `RequestInput(ask, timeout=..., name=...)` is `HandleDeferredToolCalls(ask, timeout=..., name=...)`,
    except that `ask` always gets the token, whatever its signature.
    """

    def __init__(
        self,
        ask: _TokenHandler[AgentDepsT],
        *,
        timeout: timedelta | None = None,
        name: str = _DEFAULT_WAIT_NAME,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        super().__init__(ask, timeout=timeout, name=name, id=id, description=description, defer_loading=defer_loading)

    @property
    def ask(self) -> _TokenHandler[AgentDepsT]:
        """The handler, which `RequestInput` always calls with the resume token."""
        handler = self.handler
        assert self._takes_token(handler)
        return handler

    def _takes_token(
        self, handler: _Handler[AgentDepsT] | _TokenHandler[AgentDepsT]
    ) -> TypeIs[_TokenHandler[AgentDepsT]]:
        """`RequestInput`'s `ask` always takes the token, as it did before `HandleDeferredToolCalls` took one."""
        return True
