"""The `ask_user_question` tool: validate, hand off to the answerer, report back to the model."""

from __future__ import annotations

import anyio

from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.ask_user._events import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_ai_harness.ask_user._types import Answerer, AskUserRequest, AskUserResponse, Questions, check_response

TOOL_NAME = 'ask_user_question'

DECLINED = 'The user declined to answer. Continue without the answer, or ask differently if it is essential.'
"""The tool result when the answerer reports a cancelled response."""

TIMED_OUT = (
    'The user did not answer in time. Continue without the answer: make a reasonable choice and say which one you made.'
)
"""The tool result when the answerer does not return within `AskUser.timeout`."""


def ask_user_result(request: AskUserRequest, response: AskUserResponse) -> dict[str, list[str]] | str:
    """Check `response` against `request` and render it as the `ask_user_question` tool result.

    To answer a deferred call, pass this in `DeferredToolResults.calls` under the call's ID.
    Raises `ValueError` when the response does not fit the request, as `check_response` does.
    """
    check_response(request, response)
    if response.cancelled:
        return DECLINED
    return {
        answer.header: [answer.custom_answer] if answer.custom_answer is not None else list(answer.selected)
        for answer in response.answers
    }


class AskUserToolset(FunctionToolset[AgentDepsT]):
    """Registers `ask_user_question` bound to one answerer, or deferring every call when there is none."""

    def __init__(self, *, answerer: Answerer | None, timeout: float | None = None) -> None:
        super().__init__()
        self._answerer = answerer
        self._timeout = timeout
        self.add_function(self.ask_user_question, name=TOOL_NAME)

    async def ask_user_question(self, ctx: RunContext[AgentDepsT], questions: Questions) -> dict[str, list[str]] | str:
        """Ask the user one or more multiple-choice questions and wait for the answers.

        Use it when the task is ambiguous and the answer cannot be found in the workspace.
        Offer concrete options; the answerer may also accept a custom answer. The result maps
        each question's `header` to picked labels or a one-item list containing the custom
        answer, or explains that the user declined.

        Args:
            ctx: Framework-provided run context.
            questions: One to ten questions, each with a unique `header`, two to six options,
                and `multi_select` when several answers are allowed.
        """
        if self._answerer is None:
            # Keyed by the call (always set inside a tool) so `AskUserRequest.from_tool_call` rebuilds the same request.
            request = AskUserRequest(questions=tuple(questions), id=ctx.tool_call_id or '')
            await ctx.emit(AskUserRequestedEvent(request=request))
            raise CallDeferred
        request = AskUserRequest(questions=tuple(questions))
        await ctx.emit(AskUserRequestedEvent(request=request))
        response = AskUserResponse(cancelled=True)  # What observers see if the answerer times out.
        with anyio.move_on_after(self._timeout) as scope:
            response = await self._answerer(request)
        # Observers waiting since the request event are released whether or not the response fits.
        await ctx.emit(AskUserAnsweredEvent(request_id=request.id, response=response, timed_out=scope.cancelled_caught))
        if scope.cancelled_caught:
            return TIMED_OUT
        return ask_user_result(request, response)
