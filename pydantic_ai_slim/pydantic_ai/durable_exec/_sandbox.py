from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.sandboxes import SandboxConnector


@dataclass
class SandboxConnectorsCapability(AbstractCapability[Any]):
    """Internal capability used by deprecated durable-agent wrappers."""

    connectors: Sequence[SandboxConnector]

    def get_sandbox_connectors(self) -> Sequence[SandboxConnector]:
        return self.connectors


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a `get_sandbox` override.

    Durable integrations reject sandbox-contributing capabilities up front: entering the
    contributed context manager would run I/O in workflow code. The check is static, so a
    contributor only produced at run time by a dynamic capability function cannot be caught
    here — the workflow engine then blocks the I/O itself, less legibly.

    `WrapperCapability.get_sandbox` is a pure forwarder and
    `BaseDurabilityCapability.get_sandbox` only suppresses the framework default inside a
    durable context; neither is a user sandbox contribution. `apply()` also visits wrapped
    capabilities, so a real contributor behind a wrapper is still found. A durability
    subclass that overrides `get_sandbox` itself is still treated as a contributor.
    """
    # Import lazily so loading this lightweight workflow guard does not pull the durability
    # capability's model/toolset import graph into Temporal workflow sandbox validation.
    from ._base import BaseDurabilityCapability

    found = False

    def visit(leaf: AbstractCapability[Any]) -> None:
        nonlocal found
        get_sandbox = type(leaf).get_sandbox
        found = found or get_sandbox not in (
            AbstractCapability.get_sandbox,
            WrapperCapability.get_sandbox,
            BaseDurabilityCapability.get_sandbox,
        )

    capability.apply(visit)
    return found
