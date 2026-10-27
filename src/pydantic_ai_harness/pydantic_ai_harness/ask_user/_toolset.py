"""The `ask_user_question` tool: validate, hand off to the answerer, report back to the model."""

from __future__ import annotations

import anyio

from pydantic_ai._framing import close_tag, neutralize_tags, open_tag
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.messages import TextContent, ToolReturn, UserContent
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.ask_user._events import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_ai_harness.ask_user._types import (
    TOOL_NAME,
    Answerer,
    AskUserRequest,
    AskUserResponse,
    AskUserResult,
    Questions,
    check_response,
)

DECLINED = 'The user declined to answer. Continue without the answer, or ask differently if it is essential.'
"""The tool result when the answerer reports a cancelled response."""

TIMED_OUT = (
    'The user did not answer in time. Continue without the answer: make a reasonable choice and say which one you made.'
)
"""The tool result when the answerer does not return within `AskUser.timeout`."""


_ANSWER_NOTE_ORIGIN = 'pydantic-ai-harness.ask_user.answer_note.v1'


def ask_user_result(request: AskUserRequest, response: AskUserResponse) -> AskUserResult:
    """Check `response` against `request` and render it as the `ask_user_question` tool result.

    Without notes, returns the answer dictionary or decline text. With notes, returns a `ToolReturn`
    whose `return_value` is the same dictionary and whose `content` sends the notes in the user channel,
    framed by question header. Attachments remain multimodal, in their original order.

    To answer a deferred call, pass this in `DeferredToolResults.calls` under the call's ID.
    Raises `ValueError` when the response does not fit the request, as `check_response` does.
    """
    check_response(request, response)
    if response.cancelled:
        return DECLINED
    answers = {
        answer.header: [answer.custom_answer] if answer.custom_answer is not None else list(answer.selected)
        for answer in response.answers
    }
    content: list[UserContent] = []
    for answer in response.answers:
        if not answer.note:
            continue
        metadata = {'origin': _ANSWER_NOTE_ORIGIN, 'request_id': request.id, 'header': answer.header}
        content.append(TextContent(open_tag('answer_note', header=answer.header), metadata=metadata))
        for item in answer.note:
            if isinstance(item, str):
                content.append(neutralize_tags(item))
            elif isinstance(item, TextContent):
                # Escaped text must not retain a StructuredContent value that bypasses that escaping.
                content.append(TextContent(neutralize_tags(item.content), metadata=item.metadata))
            else:
                content.append(item)
        content.append(TextContent(close_tag('answer_note'), metadata=metadata))
    return ToolReturn(return_value=answers, content=content) if content else answers


class AskUserToolset(FunctionToolset[AgentDepsT]):
    """Registers `ask_user_question` bound to one answerer, or deferring every call when there is none."""

    def __init__(self, *, answerer: Answerer | None, timeout: float | None = None) -> None:
        super().__init__()
        self._answerer = answerer
        self._timeout = timeout
        self.add_function(self.ask_user_question, name=TOOL_NAME)

    async def ask_user_question(self, ctx: RunContext[AgentDepsT], questions: Questions) -> AskUserResult:
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
        response = AskUserResponse(cancelled=True)  # What observers see if the answerer times out or fails.
        try:
            with anyio.move_on_after(self._timeout) as scope:
                response = await self._answerer(request)
        except Exception:
            await ctx.emit(AskUserAnsweredEvent(request_id=request.id, response=response, failed=True))
            raise
        # Observers waiting since the request event are released whether or not the response fits.
        await ctx.emit(AskUserAnsweredEvent(request_id=request.id, response=response, timed_out=scope.cancelled_caught))
        if scope.cancelled_caught:
            return TIMED_OUT
        return ask_user_result(request, response)
