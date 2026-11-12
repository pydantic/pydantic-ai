"""Which Anthropic betas change what a request may ask for."""

from __future__ import annotations as _annotations

from collections.abc import Iterable


def extends_max_output(betas: Iterable[str]) -> bool:
    """Whether the betas include one that raises the maximum output, like `output-300k-2026-03-24` on batches.

    Then `anthropic_max_output_tokens` isn't the limit, so a larger `max_tokens` is left for the API to judge.
    """
    return any(beta.startswith('output-') for beta in betas)
