"""StackOne capability: actions on a linked SaaS account (HRIS, ATS, CRM, and more) for agents.

Requires the `stackone` extra: `uv add "pydantic-ai-harness[stackone]"`.
"""

from pydantic_ai_harness.stackone._capability import StackOne
from pydantic_ai_harness.stackone._toolset import (
    STACKONE_API_KEY_ENV,
    STACKONE_BASE_URL,
    StackOneToolset,
    ToolMode,
)

__all__ = [
    'STACKONE_API_KEY_ENV',
    'STACKONE_BASE_URL',
    'StackOne',
    'StackOneToolset',
    'ToolMode',
]
