"""Capability that resolves deferred tool calls with a handler, which answers in place or hands a resume token on."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, TypeAlias, overload

from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypeIs

from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.external_events import (
    ExternalEventError,
    _check_timeout,  # pyright: ignore[reportPrivateUsage]
    wait_for_external_event,
)
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied

from .abstract import AbstractCapability, CapabilityOrdering

_RESULTS_ADAPTER = TypeAdapter(DeferredToolResults)

_DEFAULT_WAIT_NAME = 'deferred_tool_calls'

_Handler: TypeAlias = Callable[
    [RunContext[AgentDepsT], DeferredToolRequests],
    DeferredToolResults | None | Awaitable[DeferredToolResults | None],
]
"""A handler that answers in place, or declines with `None`."""

_TokenHandler: TypeAlias = Callable[
    [RunContext[AgentDepsT], DeferredToolRequests, str],
    DeferredToolResults | None | Awaitable[DeferredToolResults | None],
]
"""A handler that answers in place, or hands the resume token on and returns `None` to wait for the answer."""


@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    """Resolve deferred tool calls inside the agent run, with a handler that answers them or hands them on.

    This capability adds no tool for the model. It resolves the deferred tool calls that other tools
    raise: tools that require approval, external tools and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred]
    calls, and questions a Pydantic AI Harness `AskUser(answerer=None)` asks. Without it, the run ends
    with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output; with it, the run
    continues with the approved calls executed, the denied ones reported to the model, and the
    external results filled in.

    The handler is called with the run context and the pending requests, and takes one of two forms:

    - `handler(ctx, requests)` answers in place by returning
      [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] for some or all pending calls, or
      returns `None` to decline: the next capability in the chain gets the calls, and calls nobody
      answers end the run as `DeferredToolRequests` output. It is called directly in the run, so in a
      durable run it is not journaled and runs again on replay.
    - `handler(ctx, requests, token)`, with a third required positional parameter, also gets a resume
      token. It answers in place the same way, or hands the token to whoever will answer, by posting a
      Slack message, sending an email or writing a database row, and returns `None`. The run then waits
      until code you write elsewhere resolves the token with the JSON form of `DeferredToolResults`.
      The wait is [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event]:
      in a plain run it happens in this process, and in a durable run on an engine that supports it,
      currently AWS Lambda durable functions with Pydantic AI Harness's `AWSLambdaDurability`, the
      engine records the handler's answer and suspends the run while it waits. Other durable engines
      refuse the wait with a [`UserError`][pydantic_ai.exceptions.UserError].

    See [Handle Deferred Tool Calls](../capabilities/handle-deferred-tool-calls.md).

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import HandleDeferredToolCalls
        from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext


        async def handle_deferred(
            ctx: RunContext, requests: DeferredToolRequests
        ) -> DeferredToolResults:
            # Auto-approve all tools that need approval
            return requests.build_results(approve_all=True)


        agent = Agent(
            'openai:gpt-5',
            capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
        )
        ```
    """

    handler: _Handler[AgentDepsT] | _TokenHandler[AgentDepsT]
    """Resolves the pending calls, or hands them and the resume token to whoever will.

    With two parameters, `(ctx, requests)`, it returns
    [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with results for some or all
    pending calls, or `None` to decline. It can be sync or async.

    With a third required positional parameter, `(ctx, requests, token)`, it returns the answer as
    `DeferredToolResults`, or `None` to wait until the token is resolved. It is called once per wait,
    and can be sync or async; a sync function runs in a worker thread, like a sync tool. An answer it
    returns goes through its JSON form, as a resolved answer does. In a durable run it runs inside a
    durable unit that records its answer, so a replay doesn't call it again, but the engine can retry
    it when it fails: return quickly, and make it idempotent, for example by keying a message or
    database row on the token.
    """

    _: KW_ONLY

    timeout: timedelta | None = None
    """How long to wait for the answer, or `None` to wait without a limit (in a durable run, as long as the engine allows).

    Needs a handler that takes a resume token. When it elapses, or the token is resolved with a
    failure instead of an answer, every pending approval is denied and every pending external call
    fails, with a message to the model saying why. In process, it also bounds the handler: an async
    handler still running when it elapses is cancelled. On AWS Lambda, it is also the timeout of the
    callback created for each wait, which is left open when the handler answers in place; set it so
    such callbacks expire.
    """

    name: str = _DEFAULT_WAIT_NAME
    """The wait's name in a durable engine's history, such as `'approve_refund'`.

    Needs a handler that takes a resume token. On AWS Lambda, the callback and the step that calls the
    handler are named after it (`{agent}__external_event.{name}` and `{agent}__external_event.{name}.ask`).
    Keep it stable across deployments, since a replay matches its history by name.
    """

    if TYPE_CHECKING:
        # `timeout` and `name` configure the wait, so only the form that takes a token accepts them.
        @overload
        def __init__(
            self,
            handler: _Handler[AgentDepsT],
            *,
            id: str | None = None,
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        @overload
        def __init__(
            self,
            handler: _TokenHandler[AgentDepsT],
            *,
            timeout: timedelta | None = None,
            name: str = _DEFAULT_WAIT_NAME,
            id: str | None = None,
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

        def __init__(
            self,
            handler: _Handler[AgentDepsT] | _TokenHandler[AgentDepsT],
            *,
            timeout: timedelta | None = None,
            name: str = _DEFAULT_WAIT_NAME,
            id: str | None = None,
            description: str | None = None,
            defer_loading: bool = False,
        ) -> None: ...

    def __post_init__(self) -> None:
        _check_timeout(self.timeout)
        if not self._takes_token(self.handler) and (self.timeout is not None or self.name != _DEFAULT_WAIT_NAME):
            raise UserError(
                '`timeout` and `name` configure the wait for an answer, so they need a handler that takes a '
                'resume token: `handler(ctx, requests, token)`.'
            )

    def get_ordering(self) -> CapabilityOrdering | None:
        """A handler that takes a token answers last: every handler that can resolve a call without waiting goes first."""
        return CapabilityOrdering(position='innermost') if self._takes_token(self.handler) else None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        handler = self.handler
        if self._takes_token(handler):
            return await self._wait(handler, ctx, requests)
        result = handler(ctx, requests)
        if inspect.isawaitable(result):
            return await result
        return result

    def _takes_token(
        self, handler: _Handler[AgentDepsT] | _TokenHandler[AgentDepsT]
    ) -> TypeIs[_TokenHandler[AgentDepsT]]:
        """Whether `handler` has a third required positional parameter, for the resume token."""
        try:
            parameters = inspect.signature(handler).parameters.values()
        except ValueError:
            return False
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 3 and positional[2].default is inspect.Parameter.empty

    async def _wait(
        self, handler: _TokenHandler[AgentDepsT], ctx: RunContext[AgentDepsT], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        """Hand the requests and a token to `handler`, and resolve the calls with the answer it or the token's resolver gives."""
        try:
            payload = await wait_for_external_event(
                self.name, ask=partial(_ask, handler, ctx, requests), timeout=self.timeout
            )
        except ExternalEventError as error:
            message = f'No decision on this tool call arrived: {error}'
            return requests.build_results(
                approvals={call.tool_call_id: ToolDenied(message) for call in requests.approvals},
                calls={call.tool_call_id: ToolFailed(message) for call in requests.calls},
            )
        try:
            decision = _RESULTS_ADAPTER.validate_python(payload)
            # `build_results` checks `approvals` and `calls` only; without this, a decision could set the
            # `tool_call_metadata` of a call an earlier handler already resolved.
            pending_ids = {call.tool_call_id for call in (*requests.approvals, *requests.calls)}
            for field_name, ids in (('metadata', decision.metadata), ('answered_by', decision.answered_by)):
                if extra_ids := set(ids) - pending_ids:
                    raise ValueError(f'`{field_name}` contains tool call IDs that are not pending: {sorted(extra_ids)}')
            return requests.build_results(
                approvals=decision.approvals,
                calls=decision.calls,
                metadata=decision.metadata,
                answered_by=decision.answered_by,
            )
        # Both a payload that is not `DeferredToolResults` (a `ValidationError`) and one naming calls
        # that are not pending, such as a decision sent for an earlier wait.
        except ValueError as error:
            raise UserError(f'The decision sent for the pending deferred tool calls is not valid: {error}') from error


async def _ask(
    handler: _TokenHandler[AgentDepsT], ctx: RunContext[AgentDepsT], requests: DeferredToolRequests, token: str
) -> JsonValue:
    """Call `handler`, and turn an answer into the JSON form a resolver would send, so both reach the run the same way."""
    # A plain bool, so the checker keeps the handler's declared return type in both branches.
    is_async: bool = is_async_callable(handler)
    if is_async:
        results = await await_maybe(handler(ctx, requests, token))
    else:
        results = await await_maybe(await run_in_executor(handler, ctx, requests, token))
    return None if results is None else _RESULTS_ADAPTER.dump_python(results, mode='json')
