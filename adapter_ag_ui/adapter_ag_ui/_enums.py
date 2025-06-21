"""Enums for AG-UI protocol."""

from __future__ import annotations

from enum import Enum


# TODO(steve): Remove this and all uses once https://github.com/ag-ui-protocol/ag-ui/pull/49 is merged.
class Role(str, Enum):
    """Enum for message roles in AG-UI protocol."""

    ASSISTANT = 'assistant'
    USER = 'user'
    DEVELOPER = 'developer'
    SYSTEM = 'system'
    TOOL = 'tool'
