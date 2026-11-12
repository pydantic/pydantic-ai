"""The `ask_user_question` tool: validate, defer the question, and render the answer for the model."""

from __future__ import annotations

from pydantic_core import to_jsonable_python

from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.framing import close_tag, neutralize_tags, open_tag, register_tag
from pydantic_ai.messages import TextContent, ToolReturn, UserContent
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._events import event_ctx
from pydantic_ai_harness.ask_user._events import AskUserRequestedEvent
from pydantic_ai_harness.ask_user._types import (
    TOOL_NAME,
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
"""The tool result when no answer arrives within `AskUser.timeout`, or the wait for one fails."""


_ANSWER_NOTE_TAG = register_tag('answer_note')
"""Frames a user's note accompanying an answer, keyed by the question header."""

# Harness-prefixed origins mark generated data in compaction; these notes are the user's own words.
_ANSWER_NOTE_ORIGIN = 'user.pydantic-ai-harness.ask_user.answer_note.v1'


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
        content.append(TextContent(open_tag(_ANSWER_NOTE_TAG, header=answer.header), metadata=metadata))
        for item in answer.note:
            content.append(_note_item_content(item))
        content.append(TextContent(close_tag(_ANSWER_NOTE_TAG), metadata=metadata))
    return ToolReturn(return_value=answers, content=content) if content else answers


def _note_item_content(item: UserContent) -> UserContent:
    if isinstance(item, str | TextContent):
        # Neither a structured value nor user-supplied provenance metadata may bypass this text.
        text = item if isinstance(item, str) else item.content
        return neutralize_tags(text)
    return item


OWNER_METADATA_KEY = 'ask_user'
"""Metadata key naming the `AskUser` capability whose tool deferred a question, so only that one answers it."""

QUESTIONS_METADATA_KEY = 'questions'
"""Metadata key holding the questions as the tool received them, after any hook rewrote the call's arguments."""

REQUEST_ID_METADATA_KEY = 'request_id'
"""Metadata key holding the `AskUserRequest.id` the tool asked under, so `AskUser` answers under the same one."""


class AskUserToolset(FunctionToolset[AgentDepsT]):
    """Registers `ask_user_question`, which defers every call as a question for `AskUser` or the host to answer."""

    def __init__(self, *, capability_id: str | None = None, unique_request_ids: bool = False) -> None:
        # The capability's ID names the toolset's durable units, so `AskUser` works under a durable engine.
        super().__init__(id=capability_id)
        self._capability_id = capability_id
        self._unique_request_ids = unique_request_ids
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
        # Answered in the run, the ID is unique across runs, as an answerer serving several runs, such as a
        # web UI, may hold questions from all of them, and two runs can repeat a tool call ID. Answered by the
        # host, it is the call's ID (always set inside a tool), so `AskUserRequest.from_tool_call` rebuilds it.
        if self._unique_request_ids:
            request = AskUserRequest(questions=tuple(questions))
        else:
            request = AskUserRequest(questions=tuple(questions), id=ctx.tool_call_id or '')
        # Under Temporal the tool runs in an activity, which can't emit events yet, so the question is
        # still deferred and answered, without the event.
        if (events := event_ctx(ctx, 'AskUser')) is not None:
            await events.emit(AskUserRequestedEvent(request=request))
        # The questions as the tool received them, so `AskUser` asks those even when a hook such as an
        # argument guard rewrote the call's arguments, and the request's ID, journaled with the deferral
        # in a durable run so a replay asks under the same ID.
        raise CallDeferred(
            metadata={
                OWNER_METADATA_KEY: self._capability_id,
                QUESTIONS_METADATA_KEY: to_jsonable_python(questions),
                REQUEST_ID_METADATA_KEY: request.id,
            },
            kind='question',
        )
