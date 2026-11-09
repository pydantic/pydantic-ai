"""Tests for the `Team` gates, driven through real agent runs of a lead and of members."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, CapabilityEvent, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, AgentNode, Capability, Hooks, NodeResult
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness import Team
from pydantic_ai_harness.planning import InMemoryClaimablePlanStore, OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import MAX_EVENT_TEXT_CHARS, DelegationTasks
from pydantic_ai_harness.teams import (
    TeammateIdleEvent,
    TeamTaskCompletingEvent,
    TeamTaskCreatingEvent,
    check_task_completing,
    check_task_creating,
)
from pydantic_graph import End

pytestmark = pytest.mark.anyio


def _scripted(*responses: Sequence[ModelResponsePart]) -> FunctionModel:
    """A model that sends `responses` in order, streamed or not, then answers `'done'`."""
    steps = iter(responses)

    def next_parts() -> list[ModelResponsePart]:
        return list(next(steps, [TextPart('done')]))

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=next_parts())

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(next_parts()):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                    )
                }

    return FunctionModel(model, stream_function=stream)


def _transcript(messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
    """Each user prompt, tool call, tool return and text part of `messages`, as `(kind, text)`."""
    lines: list[tuple[str, str]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                lines.append(('retry' if part.outcome == 'retried' else 'return', str(part.content)))
            elif isinstance(part, UserPromptPart):
                lines.append(('user-prompt', str(part.content)))
            elif isinstance(part, TextPart):
                lines.append(('text', part.content))
            else:
                assert isinstance(part, ToolCallPart)
                lines.append(('call', f'{part.tool_name}({part.args_as_json_str()})'))
    return lines


def _keep_working_once(seen: list[TeammateIdleEvent], feedback: str = 'Run the tests first.') -> Hooks[object]:
    gate = Hooks[object]()

    @gate.on.event(TeammateIdleEvent)
    async def check_idle(ctx: RunContext[object], event: TeammateIdleEvent) -> None:
        if not seen:
            event.keep_working(feedback)
        seen.append(event)

    return gate


def _team(gates: Sequence[AbstractCapability[object]] = ()) -> Team[object]:
    """A team over an owner and board that no run here binds, so it adds only its gates and idle gate."""
    return Team[object](DelegationTasks(), board=InMemoryClaimablePlanStore(), gates=gates)


def _board(items: list[OwnedPlanItem], *, owner: str) -> Capability[object]:
    """Minimal board tools that call the team's task gates, as the team's own board tools will."""

    async def add_task(ctx: RunContext[object], content: str) -> str:
        item = OwnedPlanItem(id=str(len(items) + 1), content=content)
        await check_task_creating(ctx, item)
        items.append(item)
        return f'Added task {item.id}.'

    async def complete_task(ctx: RunContext[object], item_id: str) -> str:
        [item] = [item for item in items if item.id == item_id]
        await check_task_completing(ctx, item, owner=owner)
        item.status = TaskStatus.completed
        return f'Completed task {item_id}.'

    return Capability[object](tools=[add_task, complete_task])


class TestTeam:
    async def test_idle_gate_keeps_a_member_working(self):
        """An `asap` enqueue from the idle gate's `after_node_run` keeps the member's run going."""
        seen: list[TeammateIdleEvent] = []
        team = _team([_keep_working_once(seen)])
        agent = Agent(
            _scripted([TextPart('Fixed the bug.')], [TextPart('Tests pass.')]),
            capabilities=[team.member_capability('alice')],
        )

        result = await agent.run('Fix the bug.')

        assert result.output == 'Tests pass.'
        assert _transcript(result.all_messages()) == snapshot(
            [
                ('user-prompt', 'Fix the bug.'),
                ('text', 'Fixed the bug.'),
                ('user-prompt', 'Run the tests first.'),
                ('text', 'Tests pass.'),
            ]
        )
        assert [(event.member, event.final_output, event.feedback) for event in seen] == snapshot(
            [('alice', 'Fixed the bug.', ['Run the tests first.']), ('alice', 'Tests pass.', [])]
        )

    async def test_idle_gate_keeps_a_streamed_member_working(self):
        seen: list[TeammateIdleEvent] = []
        team = _team([_keep_working_once(seen)])
        agent = Agent(
            _scripted([TextPart('Fixed the bug.')], [TextPart('Tests pass.')]),
            capabilities=[team.member_capability('alice')],
        )

        async with agent.run_stream_events('Fix the bug.') as stream:
            streamed: list[AgentStreamEvent | AgentRunResultEvent[str]] = [event async for event in stream]

        assert [event.feedback for event in seen] == [['Run the tests first.'], []]
        idle = [event for event in streamed if isinstance(event, TeammateIdleEvent)]
        # Stream consumers read the decision the listeners made. The final idle event does not reach the
        # stream; see `test_stream_consumers_see_the_final_idle_event`.
        assert [event.feedback for event in idle] == [['Run the tests first.']]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'Core seam: a capability event emitted from `after_node_run` on the step that ends the run reaches '
            'immediate listeners but not the event stream, so stream consumers miss the idle event of a member '
            'that does go idle.'
        ),
    )
    async def test_stream_consumers_see_the_final_idle_event(self):
        agent = Agent(_scripted([TextPart('Done.')]), capabilities=[_team().member_capability('alice')])

        async with agent.run_stream_events('Go.') as stream:
            streamed = [event async for event in stream if isinstance(event, TeammateIdleEvent)]

        assert [event.final_output for event in streamed] == ['Done.']

    async def test_member_goes_idle_when_no_gate_objects(self):
        seen: list[TeammateIdleEvent] = []
        gate = Hooks[object]()

        @gate.on.event(TeammateIdleEvent)
        async def check_idle(ctx: RunContext[object], event: TeammateIdleEvent) -> None:
            seen.append(event)

        agent = Agent(_scripted([TextPart('Done.')]), capabilities=[_team([gate]).member_capability('bob')])

        result = await agent.run('Go.')

        assert result.output == 'Done.'
        assert [(event.member, event.final_output, event.truncated) for event in seen] == [('bob', 'Done.', False)]

    async def test_lead_run_has_no_idle_gate(self):
        emitted: list[CapabilityEvent] = []
        gate = Hooks[object]()

        @gate.on.event(CapabilityEvent)
        async def record(ctx: RunContext[object], event: CapabilityEvent) -> None:
            emitted.append(event)  # pragma: no cover -- the lead's run emits no team event.

        agent = Agent(_scripted([TextPart('Done.')]), capabilities=[_team([gate])])

        result = await agent.run('Go.')

        assert result.output == 'Done.'
        assert emitted == []

    async def test_final_output_is_bounded(self):
        seen: list[TeammateIdleEvent] = []
        team = _team([_keep_working_once(seen)])
        long_output = 'x' * (MAX_EVENT_TEXT_CHARS + 10)
        agent = Agent(_scripted([TextPart(long_output)]), capabilities=[team.member_capability('alice')])

        await agent.run('Go.')

        assert len(seen[0].final_output) == MAX_EVENT_TEXT_CHARS
        assert seen[0].truncated
        assert (seen[1].final_output, seen[1].truncated) == ('done', False)

    async def test_structured_output_reaches_the_gate_as_text(self):
        seen: list[TeammateIdleEvent] = []
        team = _team([_keep_working_once(seen, 'Double-check it.')])
        agent = Agent(
            _scripted(
                [ToolCallPart('final_result', {'response': 41})],
                [ToolCallPart('final_result', {'response': 42})],
            ),
            output_type=ToolOutput(int),
            capabilities=[team.member_capability('alice')],
        )

        result = await agent.run('Count.')

        assert result.output == 42
        assert [event.final_output for event in seen] == ['41', '42']

    async def test_member_is_not_idle_while_messages_are_queued(self):
        seen: list[TeammateIdleEvent] = []
        gate = Hooks[object]()

        @gate.on.event(TeammateIdleEvent)
        async def check_idle(ctx: RunContext[object], event: TeammateIdleEvent) -> None:
            seen.append(event)

        queued: list[str] = []

        class FollowUp(AbstractCapability[object]):
            """Queues one follow-up as the run first ends, like a background tool's result."""

            async def after_node_run(
                self, ctx: RunContext[object], *, node: AgentNode[object], result: NodeResult[object]
            ) -> NodeResult[object]:
                if isinstance(result, End) and not queued:
                    queued.append('Also update the changelog.')
                    ctx.enqueue(queued[0])
                return result

        # Listed after the member capability: the idle gate still runs its `after_node_run` last.
        agent = Agent(
            _scripted([TextPart('Fixed.')], [TextPart('Changelog updated.')]),
            capabilities=[_team([gate]).member_capability('alice'), FollowUp()],
        )

        result = await agent.run('Fix it.')

        assert result.output == 'Changelog updated.'
        assert [event.final_output for event in seen] == ['Changelog updated.']

    async def test_member_paused_for_approval_is_not_idle(self):
        seen: list[TeammateIdleEvent] = []
        team = _team([_keep_working_once(seen)])
        agent = Agent(
            _scripted([ToolCallPart('deploy', {})]),
            output_type=[str, DeferredToolRequests],
            capabilities=[team.member_capability('alice')],
        )

        @agent.tool_plain(requires_approval=True)
        def deploy() -> str:
            raise NotImplementedError

        result = await agent.run('Deploy.')

        assert isinstance(result.output, DeferredToolRequests)
        assert seen == []

    @pytest.mark.parametrize('run', ['lead', 'member'])
    async def test_task_creating_gate_turns_a_rejection_into_a_retry(self, run: str):
        seen: list[TeamTaskCreatingEvent] = []
        gate = Hooks[object]()

        @gate.on.event(TeamTaskCreatingEvent)
        async def require_acceptance_criteria(ctx: RunContext[object], event: TeamTaskCreatingEvent) -> None:
            seen.append(event)
            if 'done when' not in event.item.content:
                event.reject('Say when the task is done.')

        team = _team([gate])
        items: list[OwnedPlanItem] = []
        agent = Agent(
            _scripted(
                [ToolCallPart('add_task', {'content': 'Write docs'}, tool_call_id='first')],
                [ToolCallPart('add_task', {'content': 'Write docs, done when examples run'}, tool_call_id='second')],
            ),
            capabilities=[team if run == 'lead' else team.member_capability('alice'), _board(items, owner='alice')],
        )

        result = await agent.run('Plan the docs.')

        assert [item.content for item in items] == ['Write docs, done when examples run']
        assert [event.rejected for event in seen] == [True, False]
        assert _transcript(result.all_messages())[1:4] == snapshot(
            [
                ('call', 'add_task({"content":"Write docs"})'),
                ('retry', 'The task was not added to the board.\n\nSay when the task is done.'),
                ('call', 'add_task({"content":"Write docs, done when examples run"})'),
            ]
        )

    async def test_task_completing_gate_turns_a_rejection_into_a_retry(self):
        seen: list[tuple[str, str, TaskStatus]] = []
        tested: list[bool] = []
        gate = Hooks[object]()

        @gate.on.event(TeamTaskCompletingEvent)
        async def require_tests(ctx: RunContext[object], event: TeamTaskCompletingEvent) -> None:
            seen.append((event.owner, event.item.id, event.item.status))
            if not tested:
                event.reject('Run the tests before completing a task.')

        async def run_tests() -> str:
            tested.append(True)
            return 'All tests pass.'

        team = _team([gate])
        items = [OwnedPlanItem(id='1', content='Fix the bug', status=TaskStatus.in_progress)]
        agent = Agent(
            _scripted(
                [ToolCallPart('complete_task', {'item_id': '1'})],
                [ToolCallPart('run_tests', {})],
                [ToolCallPart('complete_task', {'item_id': '1'})],
            ),
            capabilities=[team.member_capability('alice'), _board(items, owner='alice')],
            tools=[run_tests],
        )

        result = await agent.run('Finish task 1.')

        assert items[0].status == TaskStatus.completed
        assert seen == [('alice', '1', TaskStatus.in_progress), ('alice', '1', TaskStatus.in_progress)]
        assert ('retry', 'The task was not marked completed.\n\nRun the tests before completing a task.') in (
            _transcript(result.all_messages())
        )

    async def test_feedback_from_several_gates_is_combined(self):
        first, second = Hooks[object](), Hooks[object]()

        @first.on.event(TeamTaskCreatingEvent)
        async def require_owner(ctx: RunContext[object], event: TeamTaskCreatingEvent) -> None:
            event.reject('Name an owner.')

        @second.on.event(TeamTaskCreatingEvent)
        async def require_estimate(ctx: RunContext[object], event: TeamTaskCreatingEvent) -> None:
            event.reject('Add an estimate.')

        items: list[OwnedPlanItem] = []
        agent = Agent(
            _scripted([ToolCallPart('add_task', {'content': 'Write docs'})], [TextPart('Gave up.')]),
            capabilities=[_team([first, second]), _board(items, owner='lead')],
        )

        result = await agent.run('Plan.')

        assert items == []
        assert _transcript(result.all_messages())[2] == (
            'retry',
            'The task was not added to the board.\n\nName an owner.\n\nAdd an estimate.',
        )

    async def test_gates_apply_to_every_member(self):
        seen: list[TeammateIdleEvent] = []
        gate = Hooks[object]()

        @gate.on.event(TeammateIdleEvent)
        async def check_idle(ctx: RunContext[object], event: TeammateIdleEvent) -> None:
            seen.append(event)

        team = _team([gate])
        for name in ('alice', 'bob'):
            agent = Agent(_scripted([TextPart(f'{name} done.')]), capabilities=[team.member_capability(name)])
            await agent.run('Go.')

        assert [(event.member, event.final_output) for event in seen] == [
            ('alice', 'alice done.'),
            ('bob', 'bob done.'),
        ]

    @pytest.mark.parametrize('include_content', [True, False])
    async def test_rejections_record_a_span(self, include_content: bool):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        idle_seen: list[TeammateIdleEvent] = []
        gate = _keep_working_once(idle_seen)

        @gate.on.event(TeamTaskCompletingEvent)
        async def reject_completion(ctx: RunContext[object], event: TeamTaskCompletingEvent) -> None:
            event.reject('Not yet.')

        items = [OwnedPlanItem(id='1', content='Fix the bug')]
        agent = Agent(
            _scripted([ToolCallPart('complete_task', {'item_id': '1'})], [TextPart('Stopped.')]),
            capabilities=[_team([gate]).member_capability('alice'), _board(items, owner='alice')],
        )
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=include_content)

        await agent.run('Finish task 1.')

        spans = [
            dict(span.attributes or {}) for span in exporter.get_finished_spans() if span.name == 'team gate rejected'
        ]
        if include_content:
            assert spans == snapshot(
                [
                    {
                        'team.gate': 'task_completing',
                        'team.feedback_count': 1,
                        'team.member': 'alice',
                        'team.item_id': '1',
                        'team.feedback': 'Not yet.',
                    },
                    {
                        'team.gate': 'teammate_idle',
                        'team.feedback_count': 1,
                        'team.member': 'alice',
                        'team.feedback': 'Run the tests first.',
                    },
                ]
            )
        else:
            assert all('team.feedback' not in span for span in spans)
            assert [span['team.gate'] for span in spans] == ['task_completing', 'teammate_idle']

    async def test_team_without_gates_changes_nothing(self):
        team = _team()
        agent = Agent(_scripted([TextPart('Done.')]), capabilities=[team, team.member_capability('a')])

        result = await agent.run('Go.')

        assert result.output == 'Done.'
        assert isinstance(result.all_messages()[-1], ModelResponse)
        assert isinstance(result.all_messages()[0], ModelRequest)
