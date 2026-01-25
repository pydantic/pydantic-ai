from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolPolicy:
    """Policy for a single tool's usage limits.

    Set on individual tools via the `usage_policy` parameter when registering tools.
    See [soft tool usage limits](../tools-advanced.md#soft-tool-usage-limits) for examples.
    """

    max_uses: int | None = None
    """Maximum successful uses allowed across the entire agent run. `None` for unlimited."""

    max_uses_per_step: int | None = None
    """Maximum successful uses allowed within a single step. `None` for unlimited."""

    partial_execution: bool | None = None
    """If `False`, reject ALL calls when batch exceeds limits. Default (`None`) allows partial execution."""

    # If we add a mode sort of thing here then we can either raise hard errors or we can raise ModelRetries based on this setting which would be great?
    # It should not be present on the tool anyway, will scrap it off when I consider Douwe's suggestion to make it all args for tools
    # I am also not sure how much value add is partial_execution on the tool itself, not allowing a tool itself to be partially executable does make some sense in theory but how much practical value it adds for the messy code I have written needs some more thought
    mode: Literal['error', 'model_retry'] = (
        'model_retry'  # We can toggle behaviour from here instead of thinking about soft vs hard limits, now everything can be soft or hard LMAO
    )
