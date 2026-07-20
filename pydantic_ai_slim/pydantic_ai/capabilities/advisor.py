from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import AdvisorModelName, AdvisorTool
from pydantic_ai.tools import AgentDepsT, RunContext

from .native_or_local import NativeOrLocalTool


@dataclass(init=False)
class Advisor(NativeOrLocalTool[AgentDepsT]):
    """Advisor capability.

    Lets a faster executor model consult a stronger advisor model mid-generation via the model's
    native advisor tool, and raises `UserError` on models that don't support it natively.

    There is no local fallback: consulting a stronger model mid-generation is a provider-native
    feature with no cross-provider equivalent. A subagent-based fallback is explicitly deferred.
    """

    model: AdvisorModelName | None
    """The advisor model to consult. Required unless an `AdvisorTool` instance is passed as `native`."""

    max_uses: int | None
    """Maximum number of advisor consultations per run. Requires native support."""

    max_tokens: int | None
    """Cap on the advisor's output tokens (minimum 1024). Requires native support."""

    caching: Literal['5m', '1h'] | None
    """Ephemeral caching TTL for the advisor context. Requires native support."""

    def __init__(
        self,
        *,
        native: AdvisorTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[AdvisorTool | None] | AdvisorTool | None]
        | bool = True,
        model: AdvisorModelName | None = None,
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
        self.local = None
        self.model = model
        self.max_uses = max_uses
        self.max_tokens = max_tokens
        self.caching = caching
        self.__post_init__()

    def _default_native(self) -> AdvisorTool:
        if self.model is None:
            raise UserError(
                'Advisor(native=True) requires an advisor `model` — pass `model=...` '
                "(e.g. `Advisor(model='claude-opus-4-8')`), or pass an `AdvisorTool` instance as `native`."
            )
        kwargs: dict[str, Any] = {'model': self.model}
        if self.max_uses is not None:
            kwargs['max_uses'] = self.max_uses
        if self.max_tokens is not None:
            kwargs['max_tokens'] = self.max_tokens
        if self.caching is not None:
            kwargs['caching'] = self.caching
        return AdvisorTool(**kwargs)
