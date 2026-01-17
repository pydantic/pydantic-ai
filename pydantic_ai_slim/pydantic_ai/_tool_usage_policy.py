from __future__ import annotations

from dataclasses import dataclass, field


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

    partial_acceptance: bool | None = None
    """If `False`, reject ALL calls when batch exceeds limits. Default (`None`) allows partial acceptance."""


@dataclass
class ToolsPolicy(ToolPolicy):
    """Agent-wide policy for tool usage limits.

    Extends [`ToolPolicy`][pydantic_ai.ToolPolicy] with aggregate limits across all tools
    and run-time per-tool overrides. Set via the `tools_policy` parameter on
    [`Agent`][pydantic_ai.Agent] or pass to `run()` / `run_sync()` / `run_stream()`.
    See [soft tool usage limits](../tools-advanced.md#soft-tool-usage-limits) for examples.
    """

    per_tool: dict[str, ToolPolicy] = field(default_factory=dict)
    """Run-time per-tool overrides. Takes precedence over tool-level `usage_policy` settings."""
