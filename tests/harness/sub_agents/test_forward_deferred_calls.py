"""Tests for `SubAgents(forward_deferred_calls=True)`: a child's deferred tool calls reach the caller, and it resumes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext, ToolDenied
from pydantic_ai_harness.sub_agents import (
    DelegationFinishedEvent,
    SubAgent,
    SubAgents,
    delegated_requests,
    delegated_results,
)


class _Mailer:
    """A child that looks a contact up, then sends it and a copy, each send needing approval."""

    def __init__(self) -> None:
        self.model_calls = 0
        self.lookups = 0
        self.sent: list[str] = []
        self.seen: list[str] = []
        self.second_approval: set[str] = set()
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
            if to in self.second_approval:
                self.second_approval.discard(to)
                raise ApprovalRequired({'reason': 'confirm again'})
            self.sent.append(to)
            return f'sent to {to}'


def _parent_model() -> FunctionModel:
    """Delegates to `mailer` once, then reports what the delegation returned.

    Streams, because the `Listener` on the parent makes core stream the run.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                yield f'parent saw: {part.content}'
                return
        args = '{"agent_name": "mailer", "task": "email ann"}'
        yield {0: DeltaToolCall(name='delegate_task', json_args=args, tool_call_id='c1')}

    return FunctionModel(stream_function=stream)


@dataclass
class Listener(AbstractCapability[object]):
    """Records how each delegation ended."""

    ended: list[DelegationFinishedEvent] = field(default_factory=list[DelegationFinishedEvent])

    @on_event(DelegationFinishedEvent)
    async def _on_finished(self, ctx: RunContext[object], event: DelegationFinishedEvent) -> None:
        self.ended.append(event)


def _parent(mailer: _Mailer, listener: Listener | None = None) -> Agent[object, str | DeferredToolRequests]:
    return Agent(
        _parent_model(),
        output_type=[str, DeferredToolRequests],
        capabilities=[
            SubAgents(agents=[SubAgent(mailer.agent)], forward_deferred_calls=True),
            listener or Listener(),
        ],
    )


class TestForwardDeferredCalls:
    async def test_child_approvals_reach_the_caller_and_the_child_resumes(self) -> None:
        mailer = _Mailer()
        listener = Listener()
        parent = _parent(mailer, listener)

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
        assert [event.outcome for event in listener.ended] == ['deferred']
        assert first.usage.requests == 2

        results = flat.build_results(approvals={'c1:e1': True, 'c1:e2': ToolDenied('not the copy')})
        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=delegated_results(requests, results)
        )

        assert second.output == 'parent saw: mail handled'
        # Resumed from its history: one more model request, the lookup not run again, only the approved call sent.
        assert (mailer.model_calls, mailer.lookups, mailer.sent) == (2, 1, ['ann@example.com'])
        assert mailer.seen == snapshot(['e1=sent to ann@example.com', 'e2=not the copy', 'l1=ann@example.com'])
        assert [event.outcome for event in listener.ended] == ['deferred', 'ok']

    async def test_every_child_call_needs_an_answer(self) -> None:
        mailer = _Mailer()
        first = await _parent(mailer).run('go')
        assert isinstance(first.output, DeferredToolRequests)

        with pytest.raises(ValueError, match=r"no answer for the calls \['c1:e2'\]"):
            delegated_results(first.output, DeferredToolResults(approvals={'c1:e1': True}))

    async def test_a_resumed_child_can_defer_again(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer)
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
        parent = _parent(mailer)
        first = await parent.run('go')
        assert isinstance(first.output, DeferredToolRequests)

        second = await parent.run(
            message_history=first.all_messages(), deferred_tool_results=first.output.build_results(approve_all=True)
        )
        assert second.output == snapshot(
            "parent saw: Sub-agent 'mailer' ended with `DeferredToolRequests` because it needs tool approval or "
            'deferred tool results, which a delegating agent cannot provide. Do the task another way, or without '
            'this sub-agent.'
        )
        assert mailer.sent == []

    async def test_a_changed_child_call_is_rejected(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer)
        first = await parent.run('go')
        requests = first.output
        assert isinstance(requests, DeferredToolRequests)
        routed = delegated_results(requests, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))
        state = routed.metadata['c1']['sub_agent_run']
        response = next(m for m in reversed(state['messages']) if m['kind'] == 'response')
        response['parts'][1]['args'] = {'to': 'attacker@example.com'}

        with pytest.raises(UserError, match='does not match the call it was asked about'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)
        assert mailer.sent == []

    async def test_state_for_another_delegation_is_rejected(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer)
        first = await parent.run('go')
        requests = first.output
        assert isinstance(requests, DeferredToolRequests)
        routed = delegated_results(requests, DeferredToolResults(approvals={'c1:e1': True, 'c1:e2': True}))
        routed.metadata['c1']['sub_agent_run']['agent_name'] = 'other'

        with pytest.raises(UserError, match='was saved for another delegation'):
            await parent.run(message_history=first.all_messages(), deferred_tool_results=routed)

    async def test_incomplete_state_is_rejected(self) -> None:
        mailer = _Mailer()
        parent = _parent(mailer)
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
