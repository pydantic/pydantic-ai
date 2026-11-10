"""The `AskUser` capability: one tool, one answerer."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, overload

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs

from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.external_events import ExternalEventError, wait_for_external_event
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._callable import takes_context
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.ask_user._events import AskUserAnsweredEvent
from pydantic_ai_harness.ask_user._toolset import OWNER_METADATA_KEY, TIMED_OUT, AskUserToolset, ask_user_result
from pydantic_ai_harness.ask_user._types import (
    TOOL_NAME,
    Answerer,
    AnyAnswerer,
    AskUserRequest,
    AskUserResponse,
    AskUserResult,
    ContextAnswerer,
    TokenAnswerer,
    rebuild_request,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    f'When the task is ambiguous and the answer is not in the workspace, call `{TOOL_NAME}` with '
    'concrete options rather than guessing or burying the question in prose. Batch related questions '
    'into one call. If the user declines, make a reasonable choice and say which one you made.'
)

_RESPONSE_ADAPTER = TypeAdapter(AskUserResponse)


@dataclass(kw_only=True)
class AskUser(AbstractCapability[AgentDepsT]):
    """Let the model ask the user multiple-choice questions mid-run.

    The capability owns the question schema and validation; the `answerer` owns the person.
    Whatever it is, a terminal menu, a web form, or a scripted function in a test, the run waits
    for its answer and the model gets the picked labels back. Pass `answerer=None` to end the run
    with `DeferredToolRequests` instead, so the host answers in a later run.

    Each question is a deferred tool call marked as a question, and the answerer is called from
    `handle_deferred_tool_calls` through `wait_for_external_event`, as `RequestInput` calls its `ask`.
    So the wait is in process in a plain run, and recorded by the engine in a durable run on an
    engine that supports it, such as AWS Lambda durable functions.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import AskUser
    from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse


    async def pick_first(request: AskUserRequest) -> AskUserResponse:
        answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
        return AskUserResponse(answers=tuple(answers))


    agent = Agent('anthropic:claude-fable-5-2', capabilities=[AskUser(answerer=pick_first)])
    ```
    """

    answerer: AnyAnswerer[AgentDepsT] | None
    """Presents each `AskUserRequest` to the user and returns their `AskUserResponse`.

    An answerer with two required positional parameters receives the asking run's context before
    the request. If it also accepts just one, annotate the first as `RunContext[DepsT]` to select
    the context form. Request-only answerers receive just the request. An answerer with three
    required positional parameters also receives a resume token, and may return `None` after handing
    the token on: the answer then arrives when the token is resolved with an `AskUserResponse`.

    In a durable run, the answerer runs inside a durable unit that records its response, so it must
    return quickly: a person who may take a while should answer through the token.

    `None` defers every call: the run ends with `DeferredToolRequests` output, and the host answers
    in a later run with `ask_user_result` in `DeferredToolResults.calls`.
    """

    timeout: float | None = None
    """Seconds to wait for an answer before telling the model the user did not answer.

    In process, an async answerer still running then is cancelled. Must be positive; `None` waits
    indefinitely. Needs an answerer: a deferred call has nothing to wait on.
    """

    allow_non_durable_wait: bool = False
    """Ask in process inside a durable run whose engine doesn't support waiting for external events, instead of raising a `UserError`.

    The wait is then not recorded: when the engine replays the run, the answerer is called again, and
    when the process restarts, the wait and any answer to it are lost. It doesn't work on Temporal,
    whose workflow code can't be woken from outside the workflow.
    """

    id: str | None = 'ask_user'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    An agent has one `ask_user_question` tool, so two `AskUser` sharing this `id` are one answerer stated
    twice when they agree, and an error when they differ. To keep two, give each its own `id` and wrap them
    in `PrefixTools`, since their tool names are the same.
    """

    if TYPE_CHECKING:
        # Preserve request-only lambda inference while retaining the generated dataclass initializer.
        @overload
        def __init__(
            self,
            *,
            answerer: Answerer | None,
            timeout: float | None = None,
            allow_non_durable_wait: bool = False,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            answerer: ContextAnswerer[AgentDepsT] | TokenAnswerer[AgentDepsT],
            timeout: float | None = None,
            allow_non_durable_wait: bool = False,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        def __init__(
            self,
            *,
            answerer: AnyAnswerer[AgentDepsT] | None,
            timeout: float | None = None,
            allow_non_durable_wait: bool = False,
            id: str | None = 'ask_user',
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise UserError('`AskUser.timeout` must be positive, or `None` to wait indefinitely.')
        if self.answerer is None and self.timeout is not None:
            raise UserError('`AskUser.timeout` bounds the answerer, so it cannot be combined with `answerer=None`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one answerer stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on when to ask."""
        return _INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The `ask_user_question` tool, which defers each call as a question this capability answers."""
        return AskUserToolset[AgentDepsT](capability_id=self.id)

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Answer this capability's own questions with the answerer, one at a time; leave other calls to later handlers."""
        answerer = self.answerer
        if answerer is None:
            return None
        questions = [call for call in requests.calls if self._owns(requests, call.tool_call_id)]
        if not questions:
            return None
        return requests.build_results(
            calls={call.tool_call_id: await self._answer(answerer, ctx, call) for call in questions}
        )

    def _owns(self, requests: DeferredToolRequests, tool_call_id: str) -> bool:
        """Whether this capability's tool deferred the call, as opposed to another `AskUser` or another tool."""
        metadata = requests.metadata.get(tool_call_id, {})
        return (
            requests.kind(tool_call_id) == 'question'
            and OWNER_METADATA_KEY in metadata
            and metadata[OWNER_METADATA_KEY] == self.id
        )

    async def _answer(
        self, answerer: AnyAnswerer[AgentDepsT], run_ctx: RunContext[AgentDepsT], call: ToolCallPart
    ) -> AskUserResult:
        request = rebuild_request(call)
        # The context the tool call had, so the answerer and the events see which call asked.
        ctx = replace(run_ctx, tool_name=call.tool_name, tool_call_id=call.tool_call_id)
        try:
            # Current, so the wait can tell a durable run whose engine can't wait from a plain run.
            with set_current_run_context(ctx):
                payload = await wait_for_external_event(
                    TOOL_NAME,
                    ask=partial(_ask, answerer, ctx, request),
                    timeout=None if self.timeout is None else timedelta(seconds=self.timeout),
                    allow_non_durable_wait=self.allow_non_durable_wait,
                )
            response = _RESPONSE_ADAPTER.validate_python(payload)
        except ExternalEventError:
            await ctx.emit(
                AskUserAnsweredEvent(request_id=request.id, response=AskUserResponse(cancelled=True), timed_out=True)
            )
            return TIMED_OUT
        except Exception as error:
            # Observers waiting since the request event are released even when the run is about to fail.
            await ctx.emit(
                AskUserAnsweredEvent(request_id=request.id, response=AskUserResponse(cancelled=True), failed=True)
            )
            if isinstance(error, ValidationError):
                raise UserError(
                    f'The answer sent for question {request.id!r} is not an `AskUserResponse`: {error}'
                ) from error
            raise
        await ctx.emit(AskUserAnsweredEvent(request_id=request.id, response=response))
        return ask_user_result(request, response)


async def _ask(
    answerer: AnyAnswerer[AgentDepsT], ctx: RunContext[AgentDepsT], request: AskUserRequest, token: str
) -> JsonValue:
    """Call the answerer in the form it takes, and turn a response into the JSON form a resolver would send."""
    response: AskUserResponse | None
    if _takes_token(answerer):
        response = await answerer(ctx, request, token)
    elif _takes_context(answerer):
        response = await answerer(ctx, request)
    else:
        response = await answerer(request)
    return None if response is None else _RESPONSE_ADAPTER.dump_python(response, mode='json')


def _takes_token(answerer: AnyAnswerer[AgentDepsT]) -> TypeIs[TokenAnswerer[AgentDepsT]]:
    try:
        parameters = tuple(inspect.signature(answerer).parameters.values())
    except ValueError:
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3 and positional[2].default is inspect.Parameter.empty


def _takes_context(answerer: AnyAnswerer[AgentDepsT]) -> TypeIs[ContextAnswerer[AgentDepsT]]:
    return takes_context(answerer) is True
