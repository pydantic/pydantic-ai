"""Complete coding-agent harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.coder._capability import FILE_TOOL_NAMES, BenchmarkInstructions, Coder
from pydantic_ai_harness.coder._instructions import Autonomy
from pydantic_ai_harness.coder._model_guidance import ModelGuidance, is_openai_family, resolve_model_guidance

if TYPE_CHECKING:
    from pydantic_ai_harness.coder._agent import coder_agent

__all__ = [
    'FILE_TOOL_NAMES',
    'Autonomy',
    'BenchmarkInstructions',
    'Coder',
    'ModelGuidance',
    'coder_agent',
    'is_openai_family',
    'resolve_model_guidance',
]


def __getattr__(name: str) -> object:
    if name == 'coder_agent':
        from pydantic_ai_harness.coder._agent import coder_agent

        return coder_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
