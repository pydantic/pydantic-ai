from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from pydantic_ai._instructions import Instruction
from pydantic_ai.tools import AgentDepsT


@dataclass
class DeferredCapabilityCatalogEntry:
    """Catalog metadata for a capability that can be loaded on demand."""

    capability_id: str
    description: str


@dataclass
class DeferredCapabilityOutputs(Generic[AgentDepsT]):
    """Loadable outputs for a deferred capability."""

    instructions: list[Instruction[AgentDepsT]]


@dataclass
class DeferredLoadingRegistry(Generic[AgentDepsT]):
    """Run-local catalog and outputs used by ``load_capability``."""

    catalog: dict[str, DeferredCapabilityCatalogEntry]
    outputs: dict[str, DeferredCapabilityOutputs[AgentDepsT]]
