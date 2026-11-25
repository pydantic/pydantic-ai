"""A plugin Pax doesn't ship, for the startup line that names such plugins and their tools."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


def port_scan(host: str) -> str:
    """Scan `host`."""
    return host  # pragma: no cover -- named at startup, never called


def whois(domain: str) -> str:
    """Look `domain` up."""
    return domain  # pragma: no cover -- named at startup, never called


@dataclass
class Scanner(AbstractCapability[object]):
    """Two function tools, which the startup line names."""

    def get_toolset(self) -> FunctionToolset[object]:
        return FunctionToolset([port_scan, whois])
