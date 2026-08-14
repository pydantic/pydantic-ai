"""Complete coding-agent harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.coder._capability import DEFAULT_ALLOWED_COMMANDS, Coder

if TYPE_CHECKING:
    from pydantic_ai_harness.coder._agent import coder_agent

__all__ = ['DEFAULT_ALLOWED_COMMANDS', 'Coder', 'coder_agent']


def __getattr__(name: str) -> object:
    if name == 'coder_agent':
        from pydantic_ai_harness.coder._agent import coder_agent

        return coder_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
