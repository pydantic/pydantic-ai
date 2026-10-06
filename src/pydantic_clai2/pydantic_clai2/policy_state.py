"""The organization policy Logfire pushed, as the parts of CLAI outside the agent run need it.

Hackathon: the `observability` plugin owns the fleet config, but the `mcp` plugin (which servers to connect)
and the plugin loader (what may be disabled) act on it too. They read it here; the observability plugin
installs the reader when it loads and removes it when it unloads, so without that plugin nothing is gated.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai_harness.policy import Policy, PolicyDecision


@dataclass(kw_only=True)
class PolicySource:
    """What the observability plugin offers: the policy in force, and where decisions are recorded."""

    policy: Callable[[], Policy | None]
    record: Callable[[PolicyDecision], None]
    pushed_mcp_servers: Callable[[], frozenset[str]]
    """Names of MCP servers Logfire pushed, which the allowlist never applies to."""


_source: PolicySource | None = None


def install(source: PolicySource | None) -> None:
    """Set (or with `None`, clear) the policy source."""
    global _source
    _source = source


def current() -> PolicySource | None:
    """The policy source, if the observability plugin installed one."""
    return _source


def locked(key: str) -> bool:
    """Whether the organization locked `kind:name`, so the user cannot turn it off."""
    source = _source
    policy = source.policy() if source is not None else None
    return policy is not None and key in policy.locked


LOCKED_MESSAGE = 'locked by your organization'
