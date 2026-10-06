"""The `policy` section of a managed agent config: tool-call rules, an MCP allowlist, and locked items."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PolicyMode = Literal['observe', 'enforce']
PolicyAction = Literal['deny', 'ask']


class PolicyMatch(BaseModel):
    """Which tool calls a rule applies to; every field set must match."""

    model_config = ConfigDict(extra='ignore')
    tool: str = '*'
    """Glob over the tool name."""
    command: str | None = None
    """Glob over the command of a shell tool call (its `command` argument)."""
    args: dict[str, str] | None = None
    """Globs over top-level argument values, by argument name."""


class PolicyRule(BaseModel):
    """One rule. `observe` records what it would have done; `enforce` does it."""

    model_config = ConfigDict(extra='ignore')
    name: str
    description: str = ''
    mode: PolicyMode = 'observe'
    action: PolicyAction = 'deny'
    match: PolicyMatch = Field(default_factory=PolicyMatch)
    monty: str | None = None
    """A Python snippet run in Monty with `tool_name` and `args`; it decides instead of `action`.

    It evaluates to `'allow'`, `'ask'` or `'deny'`, as its last expression or by assigning `decision`.
    """
    source: str | None = None
    proposal_id: str | None = None


class MCPPolicy(BaseModel):
    """Which MCP servers the user's own configuration may connect; servers Logfire pushes are always allowed."""

    model_config = ConfigDict(extra='ignore')
    allow: list[str] = Field(default_factory=list[str])
    """Server names or URL globs."""
    mode: PolicyMode = 'observe'


class Policy(BaseModel):
    """The whole section."""

    model_config = ConfigDict(extra='ignore')
    rules: list[PolicyRule] = Field(default_factory=list[PolicyRule])
    mcp: MCPPolicy | None = None
    locked: list[str] = Field(default_factory=list[str])
    """`kind:name` keys the user may not turn off, such as `skill:logfire-query` or `plugin:observability`."""
