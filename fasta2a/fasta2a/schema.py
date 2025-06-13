"""
This module re-exports the core schema types from the Google A2A SDK.

By using the SDK's types, FastA2A ensures compliance with the A2A specification
and leverages the robust Pydantic models provided by the SDK.
"""

from __future__ import annotations as _annotations

from a2a.types import (
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Task,
    TaskState,
)

# Alias for backward compatibility
Skill = AgentSkill
Provider = AgentProvider

__all__ = [
    "AgentCard",
    "Provider",
    "Skill",
    "Artifact",
    "Message",
    "Part",
    "Task",
    "TaskState",
]
