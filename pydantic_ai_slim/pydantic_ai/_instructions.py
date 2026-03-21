from collections.abc import Sequence

from pydantic_ai._template import TemplateStr
from pydantic_ai.tools import AgentDepsT

from . import _system_prompt

AgentInstructions = (
    TemplateStr[AgentDepsT]
    | str
    | _system_prompt.SystemPromptFunc[AgentDepsT]
    | Sequence[TemplateStr[AgentDepsT] | str | _system_prompt.SystemPromptFunc[AgentDepsT]]
    | None
)


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]:
    if instructions is None:
        return []
    # Note: TemplateStr is callable (__call__) so it's handled by the callable branch
    if isinstance(instructions, str) or callable(instructions):
        return [instructions]
    return list(instructions)
