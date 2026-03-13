"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

from pydantic import BaseModel

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
    capabilities: list[CapabilitySpec] = []
