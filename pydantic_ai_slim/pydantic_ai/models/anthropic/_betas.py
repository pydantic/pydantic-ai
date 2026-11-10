"""How the Anthropic betas a request asks for are read."""

from __future__ import annotations as _annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AnthropicModelSettings


def split_betas(header: str) -> set[str]:
    """The betas a comma-separated `anthropic-beta` header value lists."""
    return {stripped_beta for beta in header.split(',') if (stripped_beta := beta.strip())}


def beta_header_names(headers: Mapping[str, object]) -> list[str]:
    """The names of the `anthropic-beta` headers in `headers`, in any spelling, as the SDK folds them."""
    return [name for name in headers if name.lower() == 'anthropic-beta']


def requested_betas(model_settings: AnthropicModelSettings) -> set[str]:
    """The betas in the `anthropic_betas` setting and the `anthropic-beta` header of `extra_headers`, in any spelling."""
    betas = {str(beta) for beta in model_settings.get('anthropic_betas') or ()}
    extra_headers = model_settings.get('extra_headers', {})
    for name in beta_header_names(extra_headers):
        # A falsy value, such as `None` from an unset environment variable or the SDK's `omit`, names no betas.
        if header := extra_headers[name]:
            betas.update(split_betas(header))
    return betas


def extends_max_output(betas: Iterable[str]) -> bool:
    """Whether the betas include one that raises the maximum output, like `output-300k-2026-03-24` on batches.

    Then `anthropic_max_output_tokens` isn't the limit, so a larger `max_tokens` is left for the API to judge.
    """
    return any(beta.startswith('output-') for beta in betas)
