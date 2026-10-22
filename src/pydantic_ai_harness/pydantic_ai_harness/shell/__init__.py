"""Shell capability: gives agents configurable command execution."""

from pydantic_ai_harness.shell._capability import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._events import (
    SHELL_EVENTS,
    CommandFinishedEvent,
    CommandOutputEvent,
    CommandStartedEvent,
)
from pydantic_ai_harness.shell._persistent import MAX_FOREGROUND_WAIT
from pydantic_ai_harness.shell._toolset import RUN_SCOPED_TOOL_NAMES, SHELL_TOOL_NAMES, ShellToolset

__all__ = [
    'LLM_API_KEY_ENV_PATTERNS',
    'MAX_FOREGROUND_WAIT',
    'RUN_SCOPED_TOOL_NAMES',
    'SHELL_EVENTS',
    'SHELL_TOOL_NAMES',
    'CommandFinishedEvent',
    'CommandOutputEvent',
    'CommandStartedEvent',
    'Shell',
    'ShellToolset',
]
