"""Complete research-agent harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.researcher._capability import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher

if TYPE_CHECKING:
    from pydantic_ai_harness.researcher._agent import researcher_agent

__all__ = ['DEFAULT_RESEARCHER_INSTRUCTIONS', 'Researcher', 'researcher_agent']


def __getattr__(name: str) -> object:
    if name == 'researcher_agent':
        from pydantic_ai_harness.researcher._agent import researcher_agent

        return researcher_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
