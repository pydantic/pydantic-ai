"""`override(capabilities=...)` lists a durable engine must refuse, written once.

Each list holds the bound durability other than by an item of `root_capability.capabilities`, so it would go
through `for_agent` again and register the engine's durable units a second time. The refusal lives in
`BaseDurabilityCapability`; each engine's test module builds the agent with its durability capability and
parametrizes over `REBIND_CASES`.

This module must stay importable inside the Temporal workflow sandbox, which re-imports it: no engine imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability


@dataclass(frozen=True)
class RebindCase:
    """An `override(capabilities=...)` list holding the bound durability other than by an item of `root_capability`."""

    id: str
    capabilities: Callable[[Agent[object, str], AbstractCapability[object]], list[AbstractCapability[object]]]
    """Builds the list from the agent and its bound durability."""
    wrap_durability: bool = False
    """Whether the agent is constructed with its durability behind a `WrapperCapability`."""


REBIND_CASES = [
    RebindCase('nested', lambda agent, bound: [CombinedCapability([*agent.root_capability.capabilities])]),
    RebindCase('rewrapped', lambda agent, bound: [WrapperCapability(wrapped=bound)]),
    RebindCase('from_agent', lambda agent, bound: [bound], wrap_durability=True),
]
