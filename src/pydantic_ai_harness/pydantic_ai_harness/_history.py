"""How the harness loads message history it saved itself."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STORED_HISTORY_CONTEXT: Mapping[str, Any] = {'pydantic_ai': {'rebuild_unrecorded_tool_files': True}}
"""The validation context the harness loads its own saved runs and conversations with.

A snapshot saved before tool results recorded their files carries no record. The harness wrote it to
continue or fork the run, so it loads as it did then rather than losing the files its tools returned.
"""
