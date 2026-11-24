"""Complete coding-agent harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.coder._capability import FILE_TOOL_NAMES, MAX_IMAGE_SIDE, BenchmarkInstructions, Coder
from pydantic_ai_harness.coder._instructions import Autonomy
from pydantic_ai_harness.coder._model_guidance import MODEL_GUIDANCE, Fragment, ModelGuidance

if TYPE_CHECKING:
    from pydantic_ai_harness.coder._agent import coder_agent

__all__ = [
    'FILE_TOOL_NAMES',
    'MAX_IMAGE_SIDE',
    'MODEL_GUIDANCE',
    'Autonomy',
    'BenchmarkInstructions',
    'Coder',
    'Fragment',
    'ModelGuidance',
    'coder_agent',
]


def __getattr__(name: str) -> object:
    if name == 'coder_agent':
        from pydantic_ai_harness.coder._agent import coder_agent

        return coder_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
