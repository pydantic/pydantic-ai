"""The organization policy Logfire pushed, as the parts of CLAI outside the agent run need it.

Hackathon: the `observability` plugin owns the fleet config, but the `mcp` plugin (which servers to connect)
and the plugin loader (what may be disabled) act on it too. They read it here; the observability plugin
installs the reader when it loads and removes it when it unloads, so without that plugin nothing is gated.
"""

import fnmatch
import weakref
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

_gated: 'weakref.WeakSet[object]' = weakref.WeakSet()
_recorded: set[tuple[str, bool]] = set()


def mark_gated(toolset: object) -> None:
    """Note an MCP toolset the allowlist already handled (or that Logfire pushed), so it is not checked twice."""
    _gated.add(toolset)


def is_gated(toolset: object) -> bool:
    """Whether `mark_gated` already saw this toolset."""
    return toolset in _gated


def mcp_allowed(name: str, url: str, *, subject: str) -> bool:
    """Whether the MCP allowlist lets this server be used; a server outside it is recorded once per session."""
    source = _source
    policy = source.policy() if source is not None else None
    if source is None or policy is None or policy.mcp is None:
        return True
    if any(name == allowed or (url and fnmatch.fnmatchcase(url, allowed)) for allowed in policy.mcp.allow):
        return True
    enforce = policy.mcp.mode == 'enforce'
    if (name, enforce) not in _recorded:
        _recorded.add((name, enforce))
        source.record(
            PolicyDecision(
                rule='mcp-allowlist',
                mode=policy.mcp.mode,
                action='deny',
                outcome='denied' if enforce else 'would_deny',
                tool_name=f'mcp:{name}',
                subject=subject[:500],
            )
        )
    return not enforce
