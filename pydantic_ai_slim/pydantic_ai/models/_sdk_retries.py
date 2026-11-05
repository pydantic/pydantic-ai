"""Forward `ModelSettings['max_retries']` to the Stainless-generated SDK clients (OpenAI, Anthropic, Groq)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, Self, TypeVar

from ..settings import ModelSettings

__all__ = ('with_max_retries',)


class _StainlessClient(Protocol):
    @property
    def _client(self) -> Any: ...

    def with_options(self, *, http_client: Any, max_retries: int, _extra_kwargs: Mapping[str, Any]) -> Self: ...


_ClientT = TypeVar('_ClientT', bound=_StainlessClient)


def with_max_retries(
    client: _ClientT, model_settings: ModelSettings, *, carry_over: Mapping[str, Any] | None = None
) -> _ClientT:
    """The client to send one request with: `client` itself, or a copy carrying `ModelSettings['max_retries']`.

    These SDKs take `max_retries` only per client, not per request, so a request that sets it goes through a
    copy. The copy is handed the original's HTTP client explicitly: `AsyncAnthropicBedrock.with_options()`
    would otherwise build a fresh one, dropping the caller's transport and connection pool. `carry_over` passes
    other constructor arguments that a client's `with_options()` doesn't copy itself.
    """
    if (max_retries := model_settings.get('max_retries')) is None:
        return client
    return client.with_options(
        http_client=client._client,  # pyright: ignore[reportPrivateUsage]
        max_retries=max_retries,
        _extra_kwargs=carry_over or {},
    )
