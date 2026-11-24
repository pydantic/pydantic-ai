"""Tests for `SubAgents(forward_deferred_calls=True)`: a child's deferred tool calls reach the caller, and it resumes."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import timedelta

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, on_event
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition, ToolDenied
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.sub_agents import (
    DelegationFinishedEvent,
    InMemoryDeferredStateStore,
    SubAgent,
    SubAgents,
    TaskControl,
    delegated_requests,
    delegated_results,
)

from .._recording_durability import RecordingDurability
from ..conftest import CountingModel


class _Mailer:
    """A child that looks a contact up, then sends it and a copy, each send needing approval."""

    def __init__(self) -> None:
        self.model_calls = 0
        self.lookups = 0
        self.sent: list[str] = []
        self.seen: list[str] = []
        self.second_approval: set[str] = set()
        self.fail_for: set[str] = set()
        """Recipients whose next approved send fails."""
        """Recipients whose send asks for approval once more after it is approved."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.model_calls += 1
            returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            if any(part.tool_name == 'send_email' for part in returns):
                self.seen = sorted(f'{part.tool_call_id}={part.content}' for part in returns)
                return ModelResponse(parts=[TextPart('mail handled')])
            return ModelResponse(
                parts=[
                    ToolCallPart('lookup', {'name': 'ann'}, tool_call_id='l1'),
                    ToolCallPart('send_email', {'to': 'ann@example.com'}, tool_call_id='e1'),
                    ToolCallPart('send_email', {'to': 'cc@example.com'}, tool_call_id='e2'),
                ]
            )

        self.agent = Agent(FunctionModel(model_fn), name='mailer', output_type=[str, DeferredToolRequests])

        @self.agent.tool_plain
        def lookup(name: str) -> str:
            self.lookups += 1
            return f'{name}@example.com'

        @self.agent.tool
        def send_email(ctx: RunContext[object], to: str) -> str:
            if not ctx.tool_call_approved:
                raise ApprovalRequired
            if to in self.fail_for:
                self.fail_for.discard(to)
                raise RuntimeError('mail server down')
            if to in self.second_approval:
                self.second_approval.discard(to)
                raise ApprovalRequired({'reason': 'confirm again'})
            self.sent.append(to)
            return f'sent to {to}'


def _parent_model(agent_name: str = 'mailer', task: str = 'email ann') -> FunctionModel:
    """Delegates `task` to `agent_name` once, then reports what the delegation returned.

    Streams, because the `Listener` on the parent makes core stream the run.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                yield f'parent saw: {part.content}'
                return
        args = json.dumps({'agent_name': agent_name, 'task': task})
        yield {0: DeltaToolCall(name='delegate_task', json_args=args, tool_call_id='c1')}

    return FunctionModel(stream_function=stream)


@dataclass
class ApproveDelegations(AbstractCapability[object]):
    """An approval gate on the delegate tool: each delegation waits for the caller's approval before it runs."""

    async def before_tool_execute(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, object]
    ) -> dict[str, object]:
        if tool_def.name == 'delegate_task' and not ctx.tool_call_approved:
            raise ApprovalRequired
        return args


@dataclass
class Listener(AbstractCapability[object]):
    """Records how each delegation ended."""

    ended: list[DelegationFinishedEvent] = field(default_factory=list[DelegationFinishedEvent])

    @on_event(DelegationFinishedEvent)
    async def _on_finished(self, ctx: RunContext[object], event: DelegationFinishedEvent) -> None:
        self.ended.append(event)


def _parent(
    child: Agent[object, str | DeferredToolRequests],
    listener: Listener | None = None,
    *,
    task: str = 'email ann',
    store: InMemoryDeferredStateStore | None = None,
) -> Agent[object, str | DeferredToolRequests]:
    return Agent(
        _parent_model(child.name or '', task),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            SubAgents(agents=[SubAgent(child)], forward_deferred_calls=True, deferred_state_store=store),
            listener or Listener(),
        ],
    )


class TestSubAgentsForwardDeferredCalls:
    async def test_child_approvals_reach_the_caller_and_the_child_resumes(self) -> None:
        mailer = _Mailer()
        listener = Listener()
        parent = _parent(mailer.agent, listener)

        first = await parent.run('go')
        requests = first.output
        assert isinstance(requests, DeferredToolRequests)
        flat = delegated_requests(requests)
        assert flat.approvals == snapshot(
            [
                ToolCallPart(tool_name='send_email', args={'to': 'ann@example.com'}, tool_call_id='c1:e1'),
                ToolCallPart(tool_name='send_email', args={'to': 'cc@example.com'}, tool_call_id='c1:e2'),
            ]
        )
        assert flat.calls == []
        assert flat.metadata == snapshot({'c1:e1': {'agent_path': ['mailer']}, 'c1:e2': {'agent_path': ['mailer']}})
        # The child stopped after running the call that needed no approval, and nothing was sent.
        assert (mailer.model_calls, mailer.lookups, mailer.sent) == (1, 1, [])
        # The delegation ends here, so its outcome is reported, and the child's request counts into the parent's usage.
        assert [(event.outcome, event.output) for event in listener.ended] == snapshot(
            [('failed', "Sub-agent 'mailer' is waiting for answers to its deferred tool calls.")]
        )
        assert first.usage.requests == 2

        results = flat.build_results(approvals={'c1:e1': True, 'c1:e2': ToolDenied('not the copy')})
        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=delegated_results(requests, results)
        )

        assert second.output == 'parent saw: mail handled'
        # Resumed from its history: one more model request, the lookup not run again, only the approved call sent.
        assert (mailer.model_calls, mailer.lookups, mailer.sent) == (2, 1, ['ann@example.com'])
        assert mailer.seen == snapshot(['e1=sent to ann@example.com', 'e2=not the copy', 'l1=ann@example.com'])
        assert [event.outcome for event in listener.ended] == ['failed', 'ok']

    async def test_every_child_call_needs_an_answer(self) -> None:
        mailer = _Mailer()
        first = await _parent(mailer.agent).run('go')
        assert isinstance(first.output, DeferredToolRequests)

        with pytest.raises(ValueError, match=r"no answer for the calls \['c1:e2'\]"):
            delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True}))

    async def test_a_resumed_child_can_defer_again(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer.agent)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        mailer.second_approval.add('ann@example.com')
        results = DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True})
        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=delegated_results(first.output, results)
        )

        assert isinstance(second.output, DeferredToolRequests)
        again = delegated_requests(second.output)
        assert [call.tool_call_id for call in again.approvals] == ['c1:e1']
        assert again.metadata == snapshot({'c1:e1': {'reason': 'confirm again', 'agent_path': ['mailer']}})
        assert mailer.sent == ['cc@example.com']

        third = await parent.run(
            message_history=second.all_messages(),
            deferred_tool_results=delegated_results(second.output, again.build_results(approve_all=True)),
        )
        assert third.output == 'parent saw: mail handled'
        assert (mailer.model_calls, mailer.lookups, mailer.sent) == (2, 1, ['cc@example.com', 'ann@example.com'])

    async def test_approving_the_delegation_alone_does_not_approve_the_child_calls(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer.agent)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)

        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=first.output.build_results(approve_all=True)
        )
        # The task runs again, and its calls are asked about again.
        assert isinstance(second.output, DeferredToolRequests)
        assert [call.tool_call_id for call in delegated_requests(second.output).approvals] == ['c1:e1', 'c1:e2']
        assert (mailer.lookups, mailer.sent) == (2, [])

    async def test_an_approval_gate_on_the_delegate_tool_still_forwards(self) -> None:
        mailer = _Mailer()
        parent = Agent(
            _parent_model(),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True),
                Listener(),
                ApproveDelegations(),
            ],
        )
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        assert [call.tool_name for call in first.output.approvals] == ['delegate_task']
        assert delegated_requests(first.output) == first.output
        assert mailer.model_calls == 0

        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=first.output.build_results(approve_all=True)
        )
        assert isinstance(second.output, DeferredToolRequests)
        flat = delegated_requests(second.output)
        assert [call.tool_call_id for call in flat.approvals] == ['c1:e1', 'c1:e2']

        third = await parent.run(
            message_history=second.all_messages(),
            deferred_tool_results=delegated_results(second.output, flat.build_results(approve_all=True)),
        )
        assert third.output == 'parent saw: mail handled'
        assert (mailer.model_calls, mailer.lookups, sorted(mailer.sent)) == (
            2,
            1,
            ['ann@example.com', 'cc@example.com'],
        )

    def test_task_control_refuses_the_option(self) -> None:
        with pytest.raises(ValueError, match='not supported with `task_control`'):
            SubAgents(forward_deferred_calls=True, task_control=TaskControl())

    async def test_durable_execution_refuses_the_option(self) -> None:
        mailer = _Mailer()
        agent = Agent(
            _parent_model(),
            name='durable',
            capabilities=[
                SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True),
                RecordingDurability(),
            ],
        )
        with pytest.raises(UserError, match='not supported under durable execution'):
            await agent.run('go')

    async def test_nested_delegations_reach_the_caller_and_resume_through_every_level(self) -> None:
        mailer = _Mailer()

        def assistant_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for part in messages[-1].parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(f'assistant: {part.content}')])
            args = {'agent_name': 'mailer', 'task': 'email ann'}
            return ModelResponse(parts=[ToolCallPart('delegate_task', args, tool_call_id='m1')])

        assistant = Agent(
            FunctionModel(assistant_fn),
            name='assistant',
            output_type=[str, DeferredToolRequests],
            capabilities=[SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True)],
        )
        parent = _parent(assistant, task='handle the mail')

        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        flat = delegated_requests(first.output)
        assert [call.tool_call_id for call in flat.approvals] == ['c1:m1:e1', 'c1:m1:e2']
        assert flat.metadata == snapshot(
            {'c1:m1:e1': {'agent_path': ['assistant', 'mailer']}, 'c1:m1:e2': {'agent_path': ['assistant', 'mailer']}}
        )

        second = await parent.run(
            message_history=first.all_messages(),
            deferred_tool_results=delegated_results(first.output, flat.build_results(approve_all=True)),
        )
        assert second.output == 'parent saw: assistant: mail handled'
        assert (mailer.model_calls, mailer.lookups, sorted(mailer.sent)) == (
            2,
            1,
            ['ann@example.com', 'cc@example.com'],
        )

    async def test_nested_delegate_tools_with_other_names(self) -> None:
        mailer = _Mailer()

        def assistant_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for part in messages[-1].parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(f'assistant: {part.content}')])
            args = {'agent_name': 'mailer', 'task': 'email ann'}
            return ModelResponse(parts=[ToolCallPart('ask_mailer', args, tool_call_id='m1')])

        assistant = Agent(
            FunctionModel(assistant_fn),
            name='assistant',
            output_type=[str, DeferredToolRequests],
            capabilities=[
                SubAgents(agents=[SubAgent(mailer.agent)], tool_name='ask_mailer', forward_deferred_calls=True)
            ],
        )
        parent = _parent(assistant, task='handle the mail')
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        # Without its name, the inner delegate call is shown as it is.
        assert [call.tool_call_id for call in delegated_requests(first.output).approvals] == ['c1:m1']

        names = ('delegate_task', 'ask_mailer')
        flat = delegated_requests(first.output, tool_names=names)
        assert [call.tool_call_id for call in flat.approvals] == ['c1:m1:e1', 'c1:m1:e2']
        routed = delegated_results(first.output, flat.build_results(approve_all=True), tool_names=names)
        second = await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        assert second.output == 'parent saw: assistant: mail handled'

    async def test_external_calls_and_questions_keep_their_kind(self) -> None:
        def fetcher_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            if returns:
                return ModelResponse(parts=[TextPart(', '.join(sorted(str(part.content) for part in returns)))])
            return ModelResponse(
                parts=[
                    ToolCallPart('fetch', {'url': 'https://example.com'}, tool_call_id='f1'),
                    ToolCallPart('ask', {'question': 'Proceed?'}, tool_call_id='q1'),
                ]
            )

        fetcher = Agent(FunctionModel(fetcher_fn), name='fetcher', output_type=[str, DeferredToolRequests])

        @fetcher.tool_plain
        def fetch(url: str) -> str:
            raise CallDeferred

        @fetcher.tool_plain
        def ask(question: str) -> str:
            raise CallDeferred(kind='question')

        parent = _parent(fetcher)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        flat = delegated_requests(first.output)
        assert flat.approvals == []
        assert [(call.tool_call_id, flat.kind(call.tool_call_id)) for call in flat.calls] == [
            ('c1:f1', 'external'),
            ('c1:q1', 'question'),
        ]

        results = flat.build_results(calls={'c1:f1': 'the page', 'c1:q1': 'yes'})
        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=delegated_results(first.output, results)
        )
        assert second.output == 'parent saw: the page, yes'

    async def test_a_handler_on_the_parent_answers_through_the_helpers(self) -> None:
        mailer = _Mailer()

        def approve_all(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
            return delegated_results(requests, delegated_requests(requests).build_results(approve_all=True))

        parent = Agent(
            _parent_model(),
            capabilities=[
                SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True),
                HandleDeferredToolCalls(handler=approve_all),
                Listener(),
            ],
        )
        result = await parent.run('go')
        assert result.output == 'parent saw: mail handled'
        assert (mailer.model_calls, mailer.lookups, sorted(mailer.sent)) == (
            2,
            1,
            ['ann@example.com', 'cc@example.com'],
        )

    @pytest.mark.parametrize(
        ('tasks', 'collide'),
        [
            # Delegate `a` with child call `b:c`, and delegate `a:b` with child call `c`.
            pytest.param(('b:c', 'c'), True, id='collide'),
            # `a:b` is a child call of `a` and also the ID of the other delegate call, whose entries are dropped.
            pytest.param(('b', 'c'), False, id='nested-prefix'),
        ],
    )
    async def test_prefixed_ids(self, tasks: tuple[str, str], collide: bool) -> None:
        def clash_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompt = next(part.content for part in messages[0].parts if isinstance(part, UserPromptPart))
            assert isinstance(prompt, str)
            return ModelResponse(parts=[ToolCallPart('dangerous', {}, tool_call_id=prompt)])

        clash = Agent(FunctionModel(clash_fn), name='clash', output_type=[str, DeferredToolRequests])

        @clash.tool_plain(requires_approval=True)
        def dangerous() -> str:
            return 'done'  # pragma: no cover

        def parent_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart('delegate_task', {'agent_name': 'clash', 'task': tasks[0]}, tool_call_id='a'),
                    ToolCallPart('delegate_task', {'agent_name': 'clash', 'task': tasks[1]}, tool_call_id='a:b'),
                ]
            )

        parent = Agent(
            FunctionModel(parent_fn),
            output_type=[str, DeferredToolRequests],
            capabilities=[SubAgents(agents=[SubAgent(clash)], forward_deferred_calls=True)],
        )
        result = await parent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        if collide:
            with pytest.raises(ValueError, match=r"collide once prefixed with their delegate call ID: \['a:b:c'\]"):
                delegated_requests(result.output)
            with pytest.raises(ValueError, match='collide'):
                delegated_results(result.output, DeferredToolResults(approvals={'a:b:c': True}))
        else:
            flat = delegated_requests(result.output)
            assert [call.tool_call_id for call in flat.approvals] == ['a:b', 'a:b:c']
            assert sorted(flat.validated_args) == ['a:b', 'a:b:c']

    async def test_a_failed_resume_is_not_retried(self) -> None:
        mailer = _Mailer()
        store = InMemoryDeferredStateStore()
        parent = _parent(mailer.agent, store=store)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        routed = delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))

        mailer.fail_for.add('ann@example.com')
        with pytest.raises(RuntimeError, match='mail server down'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        # The other approved call may have run alongside the one that failed.
        sent = list(mailer.sent)
        assert 'ann@example.com' not in sent

        # The answers resume the sub-agent at most once, so a call that already ran doesn't run again.
        with pytest.raises(UserError, match='it was already resumed'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        assert mailer.sent == sent

    async def test_inherited_token_counting_does_not_rerun_approved_calls(self) -> None:
        # The child's model can't count tokens. A resumed child runs its approved calls before its first model
        # request, so starting it again without the flag would run them twice.
        mailer = _Mailer()
        parent = Agent(
            CountingModel(_parent_model()),
            output_type=[str, DeferredToolRequests],
            capabilities=[SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True), Listener()],
        )
        limits = UsageLimits(count_tokens_before_request=True)
        first = await parent.run('go', usage_limits=limits)
        assert isinstance(first.output, DeferredToolRequests)
        routed = delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))

        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=routed, usage_limits=limits
        )
        assert second.output == 'parent saw: mail handled'
        assert sorted(mailer.sent) == ['ann@example.com', 'cc@example.com']

    async def test_a_crash_before_any_deferral_propagates(self) -> None:
        def crash(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model down')

        broken = Agent(FunctionModel(crash), name='broken', output_type=[str, DeferredToolRequests])
        with pytest.raises(RuntimeError, match='model down'):
            await _parent(broken).run('go')

    async def test_the_in_memory_store_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryDeferredStateStore(max_states=1)
        await store.save('a', b'1')
        await store.save('b', b'2')
        assert await store.take('a') is None
        assert await store.take('b') == b'2'
        assert await store.take('b') is None

        with pytest.raises(ValueError, match='at least 1'):
            InMemoryDeferredStateStore(max_states=0)
        with pytest.raises(ValueError, match='must be positive'):
            InMemoryDeferredStateStore(max_age=timedelta(0))

        # The store's methods don't yield to the event loop, so the clock can be moved for them alone.
        now = [1000.0]
        monkeypatch.setattr(time, 'monotonic', lambda: now[0])
        expiring = InMemoryDeferredStateStore(max_age=timedelta(seconds=60))
        await expiring.save('a', b'1')
        await expiring.save('b', b'2')
        now[0] += 60
        await expiring.save('c', b'3')
        assert [await expiring.take('a'), await expiring.take('b'), await expiring.take('c')] == [None, None, b'3']
        await expiring.save('d', b'4')
        now[0] += 60
        assert await expiring.take('d') is None
        monkeypatch.undo()

    async def test_the_history_stays_on_the_server(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer.agent)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        state = first.output.metadata['c1']['sub_agent_run']
        # Only what the caller is asked about goes to it.
        assert sorted(state) == ['agent_name', 'requests', 'state_id']

        routed = delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))
        routed.metadata['c1']['sub_agent_run']['state_id'] = 'made-up'
        with pytest.raises(UserError, match='is not in this `SubAgents` store'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        assert mailer.sent == []

    async def test_answers_the_child_did_not_ask_for_are_rejected(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer.agent)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        routed = delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))
        routed.metadata['c1']['sub_agent_run']['results']['approvals']['l1'] = True

        with pytest.raises(UserError, match=r"was not waiting for these answers: \['l1'\]"):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)

    async def test_a_shared_store_resumes_in_another_instance(self) -> None:
        mailer = _Mailer()
        store = InMemoryDeferredStateStore()
        first = await _parent(mailer.agent, store=store).run('go')
        assert isinstance(first.output, DeferredToolRequests)
        answers = DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True})
        routed = delegated_results(first.output, answers)

        with pytest.raises(UserError, match='is not in this `SubAgents` store'):
            await _parent(mailer.agent).run(message_history=first.all_messages(), deferred_tool_results=routed)

        # A state in the same store, saved for another task, isn't accepted for this one.
        other = await _parent(mailer.agent, task='email bob', store=store).run('go')
        with pytest.raises(UserError, match='was saved for another delegation'):
            await _parent(mailer.agent, task='email bob', store=store).run(
                message_history=other.all_messages(), deferred_tool_results=routed
            )

        second = await _parent(mailer.agent, store=store).run(
            message_history=first.all_messages(), deferred_tool_results=routed
        )
        assert second.output == 'parent saw: mail handled'
        # The resumed state is deleted, so the same answers can't resume it twice.
        with pytest.raises(UserError, match='it was already resumed'):
            await _parent(mailer.agent, store=store).run(
                message_history=first.all_messages(), deferred_tool_results=routed
            )

    async def test_incomplete_state_is_rejected(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer.agent)
        first = await parent.run('go')
        requests = first.output
        assert isinstance(requests, DeferredToolRequests)
        routed = delegated_results(requests, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))
        del routed.metadata['c1']['sub_agent_run']['results']

        with pytest.raises(UserError, match='is not valid'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        broken = DeferredToolRequests(approvals=requests.approvals, metadata={'c1': {'sub_agent_run': {}}})
        with pytest.raises(UserError, match='is not valid'):
            delegated_requests(broken)

    async def test_a_state_on_another_tool_is_ignored(self) -> None:
        # A tool that puts what looks like a forwarded state in its own approval metadata is neither hidden nor approved.
        call = ToolCallPart('pay', {'amount': 1}, tool_call_id='p1')
        forged: dict[str, dict[str, object]] = {
            'sub_agent_run': {'state_id': 'x', 'agent_name': 'mailer', 'requests': {}}
        }
        requests = DeferredToolRequests(approvals=[call], metadata={'p1': forged})
        assert delegated_requests(requests) == requests
        results = DeferredToolResults(approvals={'p1': False})
        assert delegated_results(requests, results) == results

    async def test_requests_without_delegations_pass_through(self) -> None:
        call = ToolCallPart('pay', {'amount': 1}, tool_call_id='p1')
        requests = DeferredToolRequests(approvals=[call], calls=[replace(call, tool_call_id='x1')])
        assert delegated_requests(requests) == requests
        results = DeferredToolResults(approvals={'p1': True}, calls={'x1': 'done'}, metadata={'p1': {'k': 'v'}})
        assert delegated_results(requests, results) == results

    async def test_off_by_default(self) -> None:
        mailer = _Mailer()
        parent = Agent(
            _parent_model(),
            output_type=[str, DeferredToolRequests],
            capabilities=[SubAgents(agents=[SubAgent(mailer.agent)]), Listener()],
        )
        result = await parent.run('go')
        assert isinstance(result.output, str)
        assert result.output.startswith("parent saw: Sub-agent 'mailer' ended with `DeferredToolRequests`")
