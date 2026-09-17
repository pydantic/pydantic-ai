"""The `ask_user_question` tool: validate, hand off to the answerer, report back to the model."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.ask_user._events import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_ai_harness.ask_user._types import (
    MAX_QUESTIONS,
    Answerer,
    AskUserRequest,
    Question,
    check_response,
    require_unique_headers,
)

TOOL_NAME = 'ask_user_question'

DECLINED = 'The user declined to answer. Continue without the answer, or ask differently if it is essential.'
"""The tool result when the answerer reports a cancelled response."""

Questions = Annotated[
    list[Question],
    Field(min_length=1, max_length=MAX_QUESTIONS),
    AfterValidator(require_unique_headers),
]


class AskUserToolset(FunctionToolset[AgentDepsT]):
    """Registers `ask_user_question` bound to one answerer."""

    def __init__(self, *, answerer: Answerer) -> None:
        super().__init__()
        self._answerer = answerer
        self.add_function(self.ask_user_question, name=TOOL_NAME)

    async def ask_user_question(self, ctx: RunContext[AgentDepsT], questions: Questions) -> dict[str, list[str]] | str:
        """Ask the user one or more multiple-choice questions and wait for the answers.

        Use it when the task is ambiguous and the answer cannot be found in the workspace.
        Offer concrete options; the user can only pick from them. The result maps each
        question's `header` to the labels the user picked, or explains that the user declined.

        Args:
            ctx: Framework-provided run context.
            questions: One to ten questions, each with a unique `header`, two to six options,
                and `multi_select` when several answers are allowed.
        """
        request = AskUserRequest(questions=tuple(questions))
        await ctx.emit(AskUserRequestedEvent(request=request))
        response = await self._answerer(request)
        # Observers waiting since the request event are released whether or not the response fits.
        await ctx.emit(AskUserAnsweredEvent(request_id=request.id, response=response))
        check_response(request, response)
        if response.cancelled:
            return DECLINED
        return {answer.header: list(answer.selected) for answer in response.answers}
