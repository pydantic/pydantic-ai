"""The Anthropic betas a request asks for itself."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AnthropicModelSettings


def requested_betas(model_settings: AnthropicModelSettings) -> set[str]:
    """The betas in the `anthropic_betas` setting and the `anthropic-beta` header of `extra_headers`."""
    betas = {str(beta) for beta in model_settings.get('anthropic_betas') or ()}
    if header := model_settings.get('extra_headers', {}).get('anthropic-beta'):
        betas.update(stripped for beta in header.split(',') if (stripped := beta.strip()))
    return betas


def extends_max_output(model_settings: AnthropicModelSettings) -> bool:
    """Whether the request asks for a beta that raises the maximum output, like `output-300k-2026-03-24` on batches.

    Then `anthropic_max_output_tokens` isn't the limit, so a larger `max_tokens` is left for the API to judge.
    """
    return any(beta.startswith('output-') for beta in requested_betas(model_settings))
