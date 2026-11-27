"""A plugin Pax doesn't ship, for the startup line that names such plugins and their tools."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


def port_scan(host: str) -> str:
    """Scan `host`."""
    # Named at startup, never called.
    return host  # pragma: no cover


def whois(domain: str) -> str:
    """Look `domain` up."""
    # Named at startup, never called.
    return domain  # pragma: no cover


@dataclass
class Scanner(AbstractCapability[object]):
    """Two function tools, which the startup line names."""

    def get_toolset(self) -> FunctionToolset[object]:
        return FunctionToolset([port_scan, whois])


@dataclass
class Unbound(AbstractCapability[object]):
    """A capability that can't build its toolset before `for_agent` binds it, failing other than by `assert`."""

    def get_toolset(self) -> FunctionToolset[object]:
        raise RuntimeError('no client until bound to an agent')
