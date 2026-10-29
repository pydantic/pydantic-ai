from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def non_static_auth_source(client: AsyncOpenAI) -> Literal['sdk', 'http'] | None:
    """Where credentials absent from the client's static headers are resolved.

    The SDK socket client bypasses HTTP authentication.
    The optional SDK attributes also cover Azure clients, which don't initialize every OpenAI client field.
    """
    if (
        getattr(client, '_api_key_provider', None) is not None
        or getattr(client, 'workload_identity', None) is not None
        or callable(getattr(client, '_azure_ad_token_provider', None))
    ):
        return 'sdk'
    http_client = getattr(client, '_client', None)
    if getattr(http_client, 'auth', None) is not None:
        return 'http'
    return None
