"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pydantic_ai._agent_graph import EndStrategy
from pydantic_ai._spec import NamedSpec

CapabilitySpec = NamedSpec
"""The specification of a capability to be constructed.

Supports the same short forms as `EvaluatorSpec`:
* ``'MyCapability'`` — no arguments
* ``{'MyCapability': single_arg}`` — a single positional argument
* ``{'MyCapability': {k1: v1, k2: v2}}`` — keyword arguments
"""


class AgentSpec(BaseModel):
    """Specification for constructing an Agent from a dict/YAML/JSON."""

    model: str
    name: str | None = None
    description: str | None = None
    instructions: str | list[str] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int = 1
    output_retries: int | None = None
    end_strategy: EndStrategy = 'early'
    tool_timeout: float | None = None
    instrument: bool | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []
