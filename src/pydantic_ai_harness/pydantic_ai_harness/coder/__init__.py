"""Complete coding-agent harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.coder._capability import Coder
from pydantic_ai_harness.coder._events import ShellFinishedEvent, ShellOutputEvent, ShellStartedEvent

if TYPE_CHECKING:
    from pydantic_ai_harness.coder._agent import coder_agent

__all__ = ['Coder', 'ShellFinishedEvent', 'ShellOutputEvent', 'ShellStartedEvent', 'coder_agent']


def __getattr__(name: str) -> object:
    if name == 'coder_agent':
        from pydantic_ai_harness.coder._agent import coder_agent

        return coder_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
