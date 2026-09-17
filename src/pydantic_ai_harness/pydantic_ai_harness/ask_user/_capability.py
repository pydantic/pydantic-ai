"""The `AskUser` capability: one tool, one answerer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.ask_user._toolset import TOOL_NAME, AskUserToolset
from pydantic_ai_harness.ask_user._types import Answerer

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    f'When the task is ambiguous and the answer is not in the workspace, call `{TOOL_NAME}` with '
    'concrete options rather than guessing or burying the question in prose. Batch related questions '
    'into one call. If the user declines, make a reasonable choice and say which one you made.'
)


@dataclass(kw_only=True)
class AskUser(AbstractCapability[AgentDepsT]):
    """Let the model ask the user multiple-choice questions mid-run.

    The capability owns the question schema and validation; the `answerer` owns the person.
    Whatever it is, a terminal menu, a web form, or a scripted function in a test, the run waits
    for it inside the tool call and the model gets the picked labels back. There is no default:
    a capability that reads stdin would be unusable from a server.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import AskUser
    from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse


    async def pick_first(request: AskUserRequest) -> AskUserResponse:
        answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
        return AskUserResponse(answers=tuple(answers))


    agent = Agent('anthropic:claude-fable-5', capabilities=[AskUser(answerer=pick_first)])
    ```
    """

    answerer: Answerer
    """Presents each `AskUserRequest` to the user and returns their `AskUserResponse`."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on when to ask."""
        return _INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The `ask_user_question` tool bound to this capability's answerer."""
        return AskUserToolset[AgentDepsT](answerer=self.answerer)
