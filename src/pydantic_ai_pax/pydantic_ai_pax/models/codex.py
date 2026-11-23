"""Codex model discovery through the core provider's own client, as the Codex CLI reads its catalog."""

from openai import APIError
from pydantic import BaseModel, Field, ValidationError

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.providers.openai_codex import OpenAICodexProvider

CLIENT_VERSION = '0.170.0'
"""The Codex CLI release the catalog is asked for: it requires one, and hides models that need a newer client."""

TIMEOUT = 20
"""Seconds, as for Copilot model discovery: a sign-in waits on it once, unlike usage, which `/accounts` reloads."""


class _CodexModel(BaseModel):
    """One catalog entry; only what Pax reads."""

    slug: str = Field(min_length=1)
    visibility: str = 'list'


class _CodexCatalog(BaseModel):
    """The Codex backend's `GET /models` response, which is not the OpenAI `/v1/models` shape."""

    models: list[_CodexModel]


async def discover(provider: OpenAICodexProvider) -> list[str]:
    """The models the account's Codex catalog lists, in its order, skipping hidden ones.

    `provider` signs the request, so stale tokens refresh and persist as they do for model requests.
    """
    client = provider.client.with_options(max_retries=0, timeout=TIMEOUT)
    try:
        body = await client.get('/models', cast_to=bytes, options={'params': {'client_version': CLIENT_VERSION}})
    except (APIError, ModelAPIError) as exc:
        raise UserError(f'Codex model discovery failed: {exc.message}') from None
    try:
        catalog = _CodexCatalog.model_validate_json(body)
    except ValidationError:
        raise UserError('Codex returned an invalid model list.') from None
    names = list(dict.fromkeys(model.slug for model in catalog.models if model.visibility != 'hide'))
    if not names:
        raise UserError('Codex returned no models for this account.')
    return names
