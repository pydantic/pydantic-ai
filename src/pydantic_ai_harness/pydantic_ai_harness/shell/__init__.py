"""Shell capability: gives agents configurable command execution."""

from pydantic_ai_harness.shell._capability import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._toolset import ShellToolset

__all__ = ['LLM_API_KEY_ENV_PATTERNS', 'Shell', 'ShellToolset']
