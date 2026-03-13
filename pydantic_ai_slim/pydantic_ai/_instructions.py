from collections.abc import Sequence

from pydantic_ai.tools import AgentDepsT

from . import _system_prompt

Instructions = (
    str
    | _system_prompt.SystemPromptFunc[AgentDepsT]
    | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
    | None
)


def normalize_instructions(
    instructions: Instructions[AgentDepsT],
) -> list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]:
    if instructions is None:
        return []
    if isinstance(instructions, str) or callable(instructions):
        return [instructions]
    return list(instructions)
