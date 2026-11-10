"""Tests for waiting on an external event, and for `RequestInput`, which asks for a decision and waits for it.

Most tests install a recording `ExternalEventWaiter` with `external_event_scope`, the way a durable
engine does for each durable agent run, and drive `RequestInput` through `Agent.run`. `TestInProcess`
covers the default waiter used without a durable engine, resolved with `resolve_external_event`. The
AWS Lambda waiter, with suspension and replay, is tested in
`tests/harness/aws_lambda/test_aws_lambda_approvals.py`.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from datetime import timedelta
from functools import partial
from typing import Any

import anyio
import anyio.to_thread
import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import RequestInput, HandleDeferredToolCalls
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
    durable_unit_scope,
)
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.external_events import (
    ExternalEventError,
    ExternalEventWaiter,
    _pending_events,  # pyright: ignore[reportPrivateUsage]
    external_event_scope,
    fail_external_event,
    resolve_external_event,
    wait_for_external_event,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from .test_agent import READINESS_WAIT_TIMEOUT


class RecordingWaiter(ExternalEventWaiter):
    """Passes a fixed token to `ask`, then returns its answer, or else `outcome` as the payload, or raises it."""

    def __init__(self, outcome: JsonValue | ExternalEventError) -> None:
        self.outcome = outcome
        self.waits: list[tuple[str, timedelta | None]] = []

    async def wait(
        self, name: str, *, ask: Callable[[str], Awaitable[JsonValue]], timeout: timedelta | None
    ) -> JsonValue:
        self.waits.append((name, timeout))
        if (answer := await ask('token-1')) is not None:
            return answer
        if isinstance(self.outcome, ExternalEventError):
            raise self.outcome
        return self.outcome


def delete_and_look_up(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call an approval-required tool and an external one, then report what came back."""
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1'),
                ToolCallPart('look_up', {}, tool_call_id='call-2'),
            ]
        )
    return ModelResponse(parts=[TextPart('done')])


def build_agent(
    capabilities: list[Any], deleted: list[str], *, output_type: Any = str
) -> tuple[Agent[None, Any], list[tuple[str, list[str], list[str]]]]:
    asked: list[tuple[str, list[str], list[str]]] = []

    async def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
        asked.append(
            (token, [call.tool_call_id for call in requests.approvals], [call.tool_call_id for call in requests.calls])
        )

    agent = Agent(
        FunctionModel(delete_and_look_up),
        deps_type=type(None),
        output_type=output_type,
        capabilities=[RequestInput[None](ask, timeout=timedelta(hours=1)), *capabilities],
    )

    @agent.tool_plain(requires_approval=True)
    def delete(path: str) -> str:
        deleted.append(path)
        return f'deleted {path}'

    @agent.tool_plain
    def look_up() -> str:
        raise CallDeferred

    return agent, asked


def tool_returns(messages: list[ModelMessage]) -> list[tuple[str, object, str]]:
    return [
        (part.tool_call_id, part.content, part.outcome)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


async def test_decision_approves_with_override_args_and_supplies_external_result() -> None:
    deleted: list[str] = []
    agent, asked = build_agent([], deleted)
    waiter = RecordingWaiter(
        {
            'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}},
            'calls': {'call-2': {'answer': 42}},
        }
    )

    with external_event_scope(waiter):
        result = await agent.run('clean up')

    assert result.output == 'done'
    assert waiter.waits == [('deferred_tool_calls', timedelta(hours=1))]
    assert asked == [('token-1', ['call-1'], ['call-2'])]
    assert deleted == ['b.txt']
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'deleted b.txt', 'success'), ('call-2', {'answer': 42}, 'success')]
    )


async def test_decision_denies_with_and_without_a_message() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)
    waiter = RecordingWaiter(
        {
            'approvals': {'call-1': {'kind': 'tool-denied', 'message': 'Not that file.'}},
            'calls': {'call-2': {'kind': 'tool-failed', 'message': 'Lookup service is down.'}},
        }
    )

    with external_event_scope(waiter):
        result = await agent.run('clean up')

    assert deleted == []
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'Not that file.', 'denied'), ('call-2', 'Lookup service is down.', 'failed')]
    )

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': False}, 'calls': {'call-2': 'x'}})):
        result = await agent.run('clean up')

    assert tool_returns(result.all_messages())[0] == ('call-1', 'The tool call was denied.', 'denied')


async def test_decision_metadata_reaches_the_approved_tool() -> None:
    seen: list[dict[str, Any] | None] = []
    agent = Agent(
        FunctionModel(delete_and_look_up),
        deps_type=type(None),
        capabilities=[RequestInput[None](lambda ctx, requests, token: None)],
    )

    @agent.tool(requires_approval=True)
    def delete(ctx: RunContext[None], path: str) -> str:
        seen.append(ctx.tool_call_metadata)
        return 'deleted'

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    payload: JsonValue = {'approvals': {'call-1': True}, 'metadata': {'call-1': {'approved_by': 'alice'}}}
    with external_event_scope(RecordingWaiter(payload)):
        await agent.run('clean up')

    assert seen == [{'approved_by': 'alice'}]


async def test_no_decision_denies_approvals_and_fails_external_calls() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)

    with external_event_scope(RecordingWaiter(ExternalEventError('Callback timed out'))):
        result = await agent.run('clean up')

    assert deleted == []
    assert tool_returns(result.all_messages()) == snapshot(
        [
            ('call-1', 'No decision on this tool call arrived: Callback timed out', 'denied'),
            ('call-2', 'No decision on this tool call arrived: Callback timed out', 'failed'),
        ]
    )


async def test_partial_decision_leaves_the_rest_pending() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted, output_type=[str, DeferredToolRequests])

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': True}})):
        result = await agent.run('clean up')

    assert deleted == ['a.txt']
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_call_id for call in result.output.calls] == ['call-2']
    assert result.output.approvals == []


@pytest.mark.parametrize(
    ('payload', 'error'),
    [
        pytest.param(
            {'approvals': {'call-1': 'yes'}},
            r'(?s)The decision sent for the pending deferred tool calls is not valid: .*approvals\.call-1',
            id='not-deferred-tool-results',
        ),
        pytest.param(
            {'approvals': {'call-0': True}},
            r"not valid: `approvals` contains tool call IDs not in this `DeferredToolRequests.approvals`: \['call-0'\]",
            id='stale-tool-call-id',
        ),
        pytest.param(
            {'approvals': {'call-1': True}, 'metadata': {'call-0': {'approved_by': 'alice'}}},
            r"not valid: `metadata` contains tool call IDs that are not pending: \['call-0'\]",
            id='metadata-for-a-call-that-is-not-pending',
        ),
        pytest.param(
            {'approvals': {'call-1': True}, 'answered_by': {'call-0': 'alice'}},
            r"not valid: `answered_by` contains tool call IDs that are not pending: \['call-0'\]",
            id='answerer-for-a-call-that-is-not-pending',
        ),
    ],
)
async def test_invalid_decision_fails_the_run(payload: JsonValue, error: str) -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)

    with external_event_scope(RecordingWaiter(payload)), pytest.raises(UserError, match=error):
        await agent.run('clean up')

    assert deleted == []


async def test_other_handlers_resolve_calls_before_the_wait() -> None:
    """`RequestInput` is innermost, so a handler listed after it is still asked first."""
    deleted: list[str] = []

    def resolve_external(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(calls={call.tool_call_id: 'from handler' for call in requests.calls})

    agent, asked = build_agent([HandleDeferredToolCalls(resolve_external)], deleted)

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': True}})):
        result = await agent.run('clean up')

    assert asked == [('token-1', ['call-1'], [])]
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'deleted a.txt', 'success'), ('call-2', 'from handler', 'success')]
    )


async def test_a_sync_ask_runs_in_a_worker_thread() -> None:
    """Asking is often blocking I/O (Slack, email, a database row), so a sync `ask` runs off the event loop."""
    asked: list[tuple[str, int]] = []
    waiter = RecordingWaiter({'answer': 'yes'})

    with external_event_scope(waiter):
        payload = await wait_for_external_event(
            'review', ask=lambda token: asked.append((token, threading.get_ident()))
        )

    assert payload == {'answer': 'yes'}
    [(token, thread_id)] = asked
    assert token == 'token-1'
    assert thread_id != threading.get_ident()
    assert waiter.waits == [('review', None)]


async def test_wait_rejects_a_timeout_that_is_not_positive() -> None:
    with pytest.raises(UserError, match=r'`timeout` must be positive, got datetime.timedelta\(0\)'):
        await wait_for_external_event('review', ask=lambda token: None, timeout=timedelta(0))
    with pytest.raises(UserError, match=r'`timeout` must be positive, got datetime.timedelta\(days=-1'):
        RequestInput[None](lambda ctx, requests, token: None, timeout=timedelta(seconds=-1))


async def test_wait_inside_a_durable_unit_is_rejected() -> None:
    """A tool call runs in a durable unit whose result the engine records, so a wait can't be recorded inside it."""
    waiter = RecordingWaiter({'answer': 'yes'})

    with external_event_scope(waiter), durable_unit_scope():
        with pytest.raises(UserError, match=r'was called from inside a durable unit, such as a tool call'):
            await wait_for_external_event('review', ask=lambda token: None)

    assert waiter.waits == []


def test_not_spec_serializable() -> None:
    assert RequestInput.get_serialization_name() is None


class TestInProcess:
    """Without a durable engine, the wait happens in this process and `resolve_external_event` ends it."""

    async def test_an_approval_endpoint_resolves_the_token_and_the_run_continues(self) -> None:
        deleted: list[str] = []
        asked = anyio.Event()
        tokens: list[str] = []

        def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
            tokens.append(token)
            asked.set()

        agent = approval_agent(RequestInput[None](ask, timeout=timedelta(hours=1)), deleted)
        results: list[str] = []

        async def run() -> None:
            results.append((await agent.run('clean up')).output)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await asked.wait()
            # From another thread, as a sync web endpoint would.
            await anyio.to_thread.run_sync(resolve_external_event, tokens[0], {'approvals': {'call-1': True}})

        assert results == ['done']
        assert deleted == ['a.txt']
        assert len(tokens[0]) >= 32, 'the token is an unguessable ID'

    async def test_a_failure_denies_the_call_with_the_reason(self) -> None:
        deleted: list[str] = []

        def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
            fail_external_event(token, 'The approver rejected the whole batch.')

        result = await approval_agent(RequestInput[None](ask), deleted).run('clean up')

        assert deleted == []
        assert tool_returns(result.all_messages()) == snapshot(
            [('call-1', 'No decision on this tool call arrived: The approver rejected the whole batch.', 'denied')]
        )

    async def test_a_timeout_denies_the_call_and_forgets_the_token(self) -> None:
        tokens: list[str] = []
        agent = approval_agent(
            RequestInput[None](lambda ctx, requests, token: tokens.append(token), timeout=timedelta(milliseconds=1)),
            [],
        )

        result = await agent.run('clean up')

        assert tool_returns(result.all_messages()) == snapshot(
            [
                (
                    'call-1',
                    'No decision on this tool call arrived: No resolution arrived within 0:00:00.001000.',
                    'denied',
                )
            ]
        )
        with pytest.raises(UserError, match='No wait in this process has this token'):
            resolve_external_event(tokens[0], {'approvals': {'call-1': True}})

    async def test_a_token_resolves_once(self) -> None:
        def ask(token: str) -> None:
            resolve_external_event(token, 'first')
            with pytest.raises(UserError, match='No wait in this process has this token'):
                fail_external_event(token, 'second')

        assert await wait_for_external_event('review', ask=ask) == 'first'

    async def test_the_payload_arrives_in_its_json_form(self) -> None:
        """As on its way through a durable engine, so the wait sees the same value in process."""

        def ask(token: str) -> None:
            with pytest.raises(UserError, match='The payload for an external event must be JSON-serializable'):
                resolve_external_event(token, object())
            resolve_external_event(token, {'ids': ('call-1', 'call-2')})

        assert await wait_for_external_event('review', ask=ask) == {'ids': ['call-1', 'call-2']}

    async def test_a_resolution_that_lands_as_the_timeout_fires_still_counts(self) -> None:
        """The resolver was told it succeeded, so the wait returns its payload rather than timing out.

        The wait's wake-up is replaced by one that fails the way it does once the waiting loop has
        closed, so the wait only ends by its timeout, after the resolution was settled.
        """

        def closed_loop() -> None:
            raise RuntimeError('Event loop is closed')

        def ask(token: str) -> None:
            _pending_events[token].wake = closed_loop
            resolve_external_event(token, 'late')
            with pytest.raises(UserError, match='No wait in this process has this token'):
                resolve_external_event(token, 'again')

        assert await wait_for_external_event('review', ask=ask, timeout=timedelta(milliseconds=1)) == 'late'

    def test_an_unknown_token_is_rejected(self) -> None:
        with pytest.raises(
            UserError, match='No wait in this process has this token: it is wrong, from another process'
        ):
            resolve_external_event('not-a-token', None)

    async def test_a_cancelled_wait_forgets_its_token(self) -> None:
        tokens: list[str] = []
        asked = anyio.Event()

        def ask(token: str) -> None:
            tokens.append(token)
            asked.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(wait_for_external_event, 'review', ask=ask))
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await asked.wait()
            tg.cancel_scope.cancel()

        with pytest.raises(UserError, match='No wait in this process has this token'):
            resolve_external_event(tokens[0], None)

    async def test_two_waits_resolve_independently(self) -> None:
        tokens: dict[str, str] = {}
        both_asked = anyio.Event()
        payloads: dict[str, JsonValue] = {}

        async def wait(name: str) -> None:
            def ask(token: str) -> None:
                tokens[name] = token
                if len(tokens) == 2:
                    both_asked.set()

            payloads[name] = await wait_for_external_event(name, ask=ask)

        async with anyio.create_task_group() as tg:
            tg.start_soon(wait, 'first')
            tg.start_soon(wait, 'second')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await both_asked.wait()
            assert tokens['first'] != tokens['second']
            resolve_external_event(tokens['second'], 2)
            resolve_external_event(tokens['first'], 1)

        assert payloads == {'first': 1, 'second': 2}

    @pytest.mark.parametrize('anyio_backend', ['trio'])
    async def test_on_trio(self) -> None:
        tokens: list[str] = []
        asked = anyio.Event()

        async def ask(token: str) -> None:
            tokens.append(token)
            asked.set()

        payloads: list[JsonValue] = []

        async def wait() -> None:
            payloads.append(await wait_for_external_event('review', ask=ask))

        async with anyio.create_task_group() as tg:
            tg.start_soon(wait)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await asked.wait()
            await anyio.to_thread.run_sync(resolve_external_event, tokens[0], 'yes')

        assert payloads == ['yes']

    async def test_an_answer_from_ask_continues_the_run_without_a_wait(self) -> None:
        """`ask` answers in place, as a terminal prompt or a policy does: the answer goes through its JSON form, and no token stays pending."""
        deleted: list[str] = []
        tokens: list[str] = []

        async def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> DeferredToolResults:
            tokens.append(token)
            return requests.build_results(approvals={'call-1': True}, answered_by={'call-1': 'terminal'})

        result = await approval_agent(RequestInput[None](ask), deleted).run('clean up')

        assert result.output == 'done'
        assert deleted == ['a.txt']
        assert tokens[0] not in _pending_events

    async def test_the_timeout_also_bounds_ask(self) -> None:
        """An `ask` that puts the question to a person directly is cancelled when the timeout elapses."""
        cancelled = anyio.Event()

        async def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> DeferredToolResults | None:
            try:
                await anyio.sleep_forever()
            finally:
                cancelled.set()

        agent = approval_agent(RequestInput[None](ask, timeout=timedelta(milliseconds=1)), [])
        result = await agent.run('clean up')

        assert cancelled.is_set()
        assert tool_returns(result.all_messages())[0][2] == 'denied'

    async def test_an_answer_that_is_not_json_is_rejected(self) -> None:
        with pytest.raises(UserError, match='The payload for an external event must be JSON-serializable'):
            await wait_for_external_event('review', ask=lambda token: object())

    async def test_a_durable_engine_that_cannot_wait_can_opt_in_to_waiting_in_process(self) -> None:
        """With `allow_non_durable_wait`, the wait happens in process, unrecorded, instead of raising."""
        deleted: list[str] = []

        def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
            resolve_external_event(token, {'approvals': {'call-1': True}})

        agent = approval_agent(RequestInput[None](ask, allow_non_durable_wait=True), deleted, durable=True)

        assert (await agent.run('clean up')).output == 'done'
        assert deleted == ['a.txt']

    async def test_a_durable_engine_that_cannot_wait_is_rejected(self) -> None:
        """Waiting in process inside a durable run would not be recorded, so it is refused rather than done unsafely."""
        agent = approval_agent(RequestInput[None](lambda ctx, requests, token: None), [], durable=True)

        with pytest.raises(UserError, match="This agent's durable engine does not support waiting for external events"):
            await agent.run('clean up')


class _ImmediateBackend(JournalCallableOperationBackend[None]):
    def __init__(self, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(
            agent_name=agent_name,
            default_model_id=default_model_id,
            config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
        )

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        return await body()


class _AlwaysDurable(BaseDurabilityCapability[None]):
    """A durable engine that runs every operation immediately and implements no external-event waiter."""

    engine_spec = DurabilityEngineSpec(
        engine_name='Immediate', durable_unit_noun='operation', durable_container_noun='run', codec=JSON_CODEC
    )

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> _ImmediateBackend:
        return _ImmediateBackend(self.name, self.default_model_id)


def delete_only(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1')])
    return ModelResponse(parts=[TextPart('done')])


def approval_agent(capability: RequestInput[None], deleted: list[str], *, durable: bool = False) -> Agent[None, str]:
    agent = Agent(
        FunctionModel(delete_only),
        name='approver',
        deps_type=type(None),
        capabilities=[capability, *([_AlwaysDurable()] if durable else [])],
    )

    @agent.tool_plain(requires_approval=True)
    def delete(path: str) -> str:
        deleted.append(path)
        return f'deleted {path}'

    return agent
