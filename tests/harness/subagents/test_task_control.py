"""`SubAgents(task_control=True)`: background tasks, task-control tools, questions, events, remote agents.

Every scenario runs real agent runs on `FunctionModel`/`TestModel`, in process.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    CapabilityEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import WorkspaceBackend
from pydantic_ai_harness.subagents import (
    CANCELLED_ANSWER,
    AgentCreatedEvent,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    DynamicAgents,
    DynamicAgentSpec,
    InMemoryTransport,
    RemoteSubAgent,
    SubAgent,
    SubAgentMessage,
    SubAgents,
    SubAgentToolset,
    SubAgentTransport,
    SubAgentWorker,
    TaskAnswerEvent,
    TaskAssignedEvent,
    TaskBackgroundedEvent,
    TaskCancelledEvent,
    TaskCancelRequestEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskFollowUpEvent,
    TaskForceCancelEvent,
    TaskMessageEvent,
    TaskProgressEvent,
    TaskQuestionEvent,
    TaskRetryEvent,
    TaskStartedEvent,
    deserialize_message,
    serialize_message,
)

from .._recording_durability import RecordingDurability

WAIT = 10

Step = Callable[[list[ModelMessage]], ModelResponse]
AsyncStep = Callable[[list[ModelMessage]], Awaitable[ModelResponse]]


def scripted(*steps: Step | AsyncStep | ModelResponse) -> FunctionModel:
    """A model that answers its n-th request with the n-th step, and streams it too."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = steps[sum(isinstance(message, ModelResponse) for message in messages)]
        if isinstance(step, ModelResponse):
            return step
        response = step(messages)
        return await response if inspect.isawaitable(response) else response

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        response = await respond(messages, info)
        calls: DeltaToolCalls = {}
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                calls[index] = DeltaToolCall(
                    name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                )
        if calls:
            yield calls

    return FunctionModel(respond, stream_function=stream)


def call(tool_name: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name, args)])


def text(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content)])


def tool_returns(messages: Sequence[ModelMessage]) -> list[str]:
    """What the model saw of every tool result, in order."""
    return [
        part.model_response_str()
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def system_texts(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]


def user_texts(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def task_id_in(receipt: str) -> str:
    return receipt.split()[1]


def echo_last_return(messages: list[ModelMessage]) -> ModelResponse:
    return text(tool_returns(messages)[-1])


@dataclass
class Recorder(AbstractCapability[Any]):
    """Records every capability event the run emits, which in these runs are the `sub_agents` ones."""

    events: list[CapabilityEvent] = field(default_factory=list[CapabilityEvent])

    @on_event(CapabilityEvent)
    async def record(self, ctx: RunContext[Any], event: CapabilityEvent) -> None:
        self.events.append(event)

    def of(self, event_type: type[Any]) -> list[Any]:
        return [event for event in self.events if isinstance(event, event_type)]

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events if not isinstance(event, TaskProgressEvent)]


@dataclass
class Gate:
    """A child tool that blocks until the test opens it."""

    entered: anyio.Event = field(default_factory=anyio.Event)
    opened: anyio.Event = field(default_factory=anyio.Event)

    async def pause(self) -> str:
        """Wait for the gate."""
        self.entered.set()
        await self.opened.wait()
        return 'resumed'


def child(output: str = 'child output') -> Agent[Any, str]:
    return Agent(TestModel(call_tools=[], custom_output_text=output), name='researcher', description='Researches.')


def researcher(agent: Agent[Any, Any] | None = None, **kwargs: Any) -> SubAgent[Any]:
    return SubAgent(agent or child(), name='researcher', **kwargs)


def controlled(*agents: SubAgent[Any], **kwargs: Any) -> SubAgents[Any]:
    return SubAgents(agents=agents or [researcher()], task_control=True, **kwargs)


class TestSubAgentsTaskControl:
    async def test_a_foreground_task_returns_its_output_and_emits_its_steps(self) -> None:
        recorder = Recorder()
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Find x.'), echo_last_return),
            capabilities=[controlled(), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        [assigned] = recorder.of(TaskAssignedEvent)
        assert result.output == f'Task {assigned.task_id} (ok):\nchild output'
        # The run's own owner has no observer, so the delegation start and end events reach the run, as without one.
        # Stream events reach listeners as the stream is read, so only the immediate ones keep a fixed order.
        assert sorted(recorder.kinds()) == [
            'sub_agents.delegation_end',
            'sub_agents.delegation_start',
            'sub_agents.task_assigned',
            'sub_agents.task_completed',
            'sub_agents.task_started',
        ]
        assert assigned.prompt == 'Find x.'
        assert assigned.depth == 2
        assert assigned.parent_task_id is None
        assert assigned.can_ask_questions

    async def test_a_background_task_reports_back_and_keeps_the_run_going(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('slow result')), tools=[gate.pause])
        recorder = Recorder()

        def end_turn(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting for the report.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow work.', background=True),
                end_turn,
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        [receipt] = tool_returns(result.all_messages())
        assert 'is running in the background' in receipt
        report = system_texts(result.all_messages())[-1]
        assert report.startswith('Automated subagent task report')
        assert 'slow result' in report
        [backgrounded] = recorder.of(TaskBackgroundedEvent)
        assert backgrounded.reason == 'requested'

    async def test_a_slow_task_moves_to_the_background_after_auto_background_after(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('slow result')), tools=[gate.pause])
        recorder = Recorder()

        def end_turn(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow.'),
                end_turn,
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow), auto_background_after=0.01), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert 'is running in the background' in tool_returns(result.all_messages())[0]
        assert [event.reason for event in recorder.of(TaskBackgroundedEvent)] == ['auto']
        assert 'slow result' in system_texts(result.all_messages())[-1]

    async def test_a_question_moves_the_task_to_the_background_until_the_parent_answers(self) -> None:
        asker = Agent(
            scripted(
                call('ask_parent', question='Which year?'), lambda messages: text(f'used {tool_returns(messages)[-1]}')
            )
        )
        recorder = Recorder()

        def answer(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return call('answer_task', task_id=task_id, answer='1999')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Look it up.'),
                answer,
                text('Answered.'),
                text('Done.'),
            ),
            capabilities=[controlled(researcher(asker)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())
        assert 'is running in the background' in returns[0]
        assert returns[1].startswith('Answer sent to task ')
        question_text, report = system_texts(result.all_messages())
        assert question_text.startswith('Automated subagent question.')
        assert 'Which year?' in question_text
        assert 'used 1999' in report
        assert [event.reason for event in recorder.of(TaskBackgroundedEvent)] == ['question']
        [question] = recorder.of(TaskQuestionEvent)
        [answered] = recorder.of(TaskAnswerEvent)
        assert answered.question_id == question.question_id and answered.answer == '1999'

    async def test_a_listener_can_answer_a_question(self) -> None:
        asker = Agent(
            scripted(call('ask_parent', question='Which year?'), lambda messages: text(tool_returns(messages)[-1]))
        )

        @dataclass
        class Answerer(AbstractCapability[Any]):
            @on_event(TaskQuestionEvent)
            async def answer(self, ctx: RunContext[Any], event: TaskQuestionEvent) -> None:
                event.answer = '2001'

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Look it up.'), echo_last_return),
            capabilities=[controlled(researcher(asker)), Answerer()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(ok):\n2001')

    async def test_steering_reaches_a_running_task(self) -> None:
        gate = Gate()
        seen: list[str] = []

        def after_pause(messages: list[ModelMessage]) -> ModelResponse:
            seen.extend(user_texts(messages))
            return text('steered result')

        slow = Agent(scripted(call('pause'), after_pause), tools=[gate.pause])

        def steer(messages: list[ModelMessage]) -> ModelResponse:
            return call('message_task', task_id=task_id_in(tool_returns(messages)[-1]), message='Focus on y.')

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow.', background=True),
                steer,
                release,
                text('done'),
            ),
            capabilities=[controlled(researcher(slow))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert tool_returns(result.all_messages())[1].startswith('Message sent to task ')
        assert seen[-1] == 'Message from the agent that delegated this task to you:\nFocus on y.'

    async def test_cancel_task_stops_a_running_task(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('never')), tools=[gate.pause])
        recorder = Recorder()

        def cancel(messages: list[ModelMessage]) -> ModelResponse:
            return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]), force=True)

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow.', background=True),
                cancel,
                text('Waiting.'),
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]
        [cancelled] = recorder.of(TaskCancelledEvent)
        assert cancelled.forced
        assert len(recorder.of(TaskForceCancelEvent)) == 1

    async def test_a_resumed_task_continues_its_conversation(self) -> None:
        recorder = Recorder()
        seen: list[int] = []

        def respond(messages: list[ModelMessage]) -> ModelResponse:
            seen.append(len(messages))
            return text(f'answer {len(seen)}')

        remembering = Agent(scripted(respond, respond))

        def follow_up(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return call('delegate_task', agent_name='researcher', task='And then?', resume=task_id)

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='First.'), follow_up, echo_last_return),
            capabilities=[controlled(researcher(remembering)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nanswer 2')
        assert seen == [1, 3]
        [follow] = recorder.of(TaskFollowUpEvent)
        assert follow.task_id == recorder.of(TaskAssignedEvent)[0].task_id

    async def test_a_one_shot_task_cannot_be_resumed(self) -> None:
        def follow_up(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return call('delegate_task', agent_name='researcher', task='And then?', resume=task_id)

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='First.'), follow_up, echo_last_return),
            capabilities=[controlled(researcher(resumable=False))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert 'cannot resume; it is one-shot' in result.output

    async def test_list_tasks_shows_each_task_once_with_its_status(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('slow')), tools=[gate.pause])

        def listed(messages: list[ModelMessage]) -> ModelResponse:
            return call('list_tasks')

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text(tool_returns(messages)[-1])

        parent = Agent(
            scripted(
                call('list_tasks'),
                call('delegate_task', agent_name='researcher', task='x' * 100, background=True),
                listed,
                release,
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())
        assert returns[0] == 'You have not started any tasks.'
        task_id = task_id_in(returns[1])
        assert returns[2] == f"- {task_id} (researcher, running, background, resumable): '{'x' * 80}...'"

    async def test_list_tasks_is_capped(self) -> None:
        def many(messages: list[ModelMessage]) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': 'researcher', 'task': f't{i}'}) for i in range(52)]
            )

        parent = Agent(
            scripted(many, call('list_tasks'), echo_last_return),
            capabilities=[controlled()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.', usage_limits=UsageLimits(request_limit=None))
        lines = result.output.splitlines()
        assert len(lines) == 51
        assert lines[-1] == '(2 older finished tasks not shown.)'
        assert 'finished: ok' in lines[0]

    async def test_list_tasks_keeps_the_newest_running_tasks(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('slow')), tools=[gate.pause])
        listed: list[str] = []

        def many(messages: list[ModelMessage]) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart('delegate_task', {'agent_name': 'quick', 'task': 'q'}),
                    *(
                        ToolCallPart('delegate_task', {'agent_name': 'researcher', 'task': f't{i}', 'background': True})
                        for i in range(52)
                    ),
                ]
            )

        def release(messages: list[ModelMessage]) -> ModelResponse:
            listed.append(tool_returns(messages)[-1])
            gate.opened.set()
            return text('Waiting.')

        parent = Agent(
            scripted(many, call('list_tasks'), release, text('Done.'), text('Done.'), text('Done.')),
            capabilities=[controlled(researcher(slow), SubAgent(child(), name='quick'))],
        )
        with anyio.fail_after(WAIT):
            await parent.run('Go.', usage_limits=UsageLimits(request_limit=None))
        lines = listed[0].splitlines()
        assert len(lines) == 51
        # With no room left, the finished task left out is not called older: it may be newer than those shown.
        assert lines[-1] == '(2 older running tasks and 1 finished task not shown.)'
        assert "'t2'" in lines[0] and "'t51'" in lines[-2]

    async def test_max_running_tasks_refuses_another_task(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('slow')), tools=[gate.pause])

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='One.', background=True),
                call('delegate_task', agent_name='researcher', task='Two.', background=True),
                release,
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow), max_running_tasks=1)],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert '1 tasks are already running, the most allowed at once.' in tool_returns(result.all_messages())[1]

    async def test_listeners_can_rewrite_and_veto_task_messages(self) -> None:
        gate = Gate()
        asker = Agent(
            scripted(
                call('pause'), call('ask_parent', question='Which?'), lambda messages: text(tool_returns(messages)[-1])
            ),
            tools=[gate.pause],
        )

        @dataclass
        class Policy(AbstractCapability[Any]):
            @on_event(TaskAssignedEvent)
            async def assigned(self, ctx: RunContext[Any], event: TaskAssignedEvent) -> None:
                if event.prompt == 'Forbidden.':
                    event.veto('not allowed')
                event.prompt = event.prompt.upper()

            @on_event(TaskMessageEvent)
            async def message(self, ctx: RunContext[Any], event: TaskMessageEvent) -> None:
                event.veto()

            @on_event(TaskCancelRequestEvent)
            async def cancel(self, ctx: RunContext[Any], event: TaskCancelRequestEvent) -> None:
                event.veto('keep it')

            @on_event(TaskAnswerEvent)
            async def answer(self, ctx: RunContext[Any], event: TaskAnswerEvent) -> None:
                if event.answer == 'bad':
                    event.veto('rude')

            @on_event(TaskCompletedEvent)
            async def completed(self, ctx: RunContext[Any], event: TaskCompletedEvent) -> None:
                event.output = f'[checked] {event.output}'

        def control(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return ModelResponse(
                parts=[
                    ToolCallPart('message_task', {'task_id': task_id, 'message': 'hi'}),
                    ToolCallPart('cancel_task', {'task_id': task_id}),
                ]
            )

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        def answer(answer: str) -> Step:
            def step(messages: list[ModelMessage]) -> ModelResponse:
                task_id = task_id_in(tool_returns(messages)[1])
                return call('answer_task', task_id=task_id, answer=answer)

            return step

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Forbidden.'),
                call('delegate_task', agent_name='researcher', task='Allowed.', background=True),
                control,
                release,
                answer('bad'),
                answer('good'),
                text('Waiting.'),
                text('Done.'),
            ),
            capabilities=[controlled(researcher(asker)), Policy()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())
        assert returns[0] == 'The task was not sent: not allowed.'
        assert returns[2:4] == [
            'The message was not sent: a policy refused it.',
            'The cancellation was not sent: keep it.',
        ]
        assert returns[4:6] == ['The answer was not sent: rude.', returns[5]]
        assert returns[5].startswith('Answer sent to task ')
        assert system_texts(result.all_messages())[-1].endswith('<task-output>\n[checked] good\n</task-output>')

    async def test_a_soft_cancel_stops_the_task_before_its_next_request(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('never')), tools=[gate.pause])
        recorder = Recorder()

        def cancel(messages: list[ModelMessage]) -> ModelResponse:
            return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]))

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow.', background=True),
                cancel,
                release,
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert tool_returns(result.all_messages())[1].endswith('Its report goes to you. It can be resumed afterwards.')
        [cancelled] = recorder.of(TaskCancelledEvent)
        assert not cancelled.forced
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]

    async def test_a_soft_cancel_releases_a_task_waiting_for_an_answer(self) -> None:
        asker = Agent(
            scripted(call('ask_parent', question='Which?'), lambda messages: text(tool_returns(messages)[-1]))
        )

        def cancel(messages: list[ModelMessage]) -> ModelResponse:
            return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]))

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Ask.'), cancel, text('Done.'), text('Done.')),
            capabilities=[controlled(researcher(asker))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        # The task got no answer, and stopped before the model request that would have used it.
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]

    async def test_task_control_tools_refuse_tasks_that_are_not_theirs_to_control(self) -> None:
        def control(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return ModelResponse(
                parts=[
                    ToolCallPart('message_task', {'task_id': task_id, 'message': 'hi'}),
                    ToolCallPart('answer_task', {'task_id': task_id, 'answer': 'x'}),
                    ToolCallPart('cancel_task', {'task_id': task_id}),
                    ToolCallPart('cancel_task', {'task_id': 'unknown'}),
                ]
            )

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Quick.'), control, text('Done.')),
            capabilities=[controlled()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())[1:]
        assert 'has finished (ok), and messages only reach running tasks' in returns[0]
        assert 'is not waiting for an answer' in returns[1]
        assert returns[2].endswith('already finished (ok); there is nothing to cancel.')
        assert "Unknown task 'unknown'" in returns[3]

    async def test_an_application_owner_gets_every_task_event(self) -> None:
        observed: list[DelegationTaskEvent] = []

        async def observe(event: DelegationTaskEvent) -> None:
            observed.append(event)

        owner = DelegationTasks(observer=observe)
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Find x.'), echo_last_return),
            capabilities=[controlled()],
        )
        with anyio.fail_after(WAIT):
            async with owner.opened():
                with owner.bind():
                    result = await parent.run('Go.', conversation_id='session')
        assert result.output.endswith('(ok):\nchild output')
        kinds = [event.event.kind for event in observed if isinstance(event.event, CapabilityEvent)]
        assert kinds == [
            'sub_agents.task_assigned',
            'sub_agents.task_started',
            'sub_agents.delegation_start',
            'sub_agents.delegation_end',
            'sub_agents.task_completed',
        ]

    async def test_forward_events_streams_a_task_into_the_delegating_run(self) -> None:
        recorder = Recorder()
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Find x.'), echo_last_return),
            capabilities=[controlled(forward_events=True), recorder],
        )
        with anyio.fail_after(WAIT):
            await parent.run('Go.')
        assert recorder.of(TaskProgressEvent)

    async def test_a_nested_task_asks_its_own_parent(self) -> None:
        asker = Agent(
            scripted(call('ask_parent', question='Inner?'), lambda messages: text(tool_returns(messages)[-1])),
            name='inner',
        )
        inner_recorder = Recorder()

        def answer(messages: list[ModelMessage]) -> ModelResponse:
            return call('answer_task', task_id=task_id_in(tool_returns(messages)[-1]), answer='42')

        middle = Agent(
            scripted(
                call('delegate_task', agent_name='inner', task='Deep.'),
                answer,
                text('middle done'),
                text('middle done'),
            ),
            name='middle',
            capabilities=[SubAgents(agents=[SubAgent(asker)], task_control=True), inner_recorder],
        )
        outer_recorder = Recorder()
        parent = Agent(
            scripted(call('delegate_task', agent_name='middle', task='Go deep.'), echo_last_return),
            capabilities=[SubAgents(agents=[SubAgent(middle)], task_control=True), outer_recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nmiddle done')
        assert [event.question for event in inner_recorder.of(TaskQuestionEvent)] == ['Inner?']
        assert not outer_recorder.of(TaskQuestionEvent)
        [inner_assigned] = inner_recorder.of(TaskAssignedEvent)
        assert inner_assigned.depth == 3
        assert inner_assigned.parent_task_id == outer_recorder.of(TaskAssignedEvent)[0].task_id


class TestDynamicAgents:
    async def test_the_model_creates_and_delegates_to_an_agent(self) -> None:
        recorder = Recorder()
        parent = Agent(
            scripted(
                call('create_agent', name='poet', description='Writes poems.', instructions='Rhyme.', model='fast'),
                call('delegate_task', agent_name='poet', task='A haiku.'),
                echo_last_return,
            ),
            capabilities=[
                SubAgents(
                    models={'fast': TestModel(custom_output_text='old pond')},
                    dynamic_agents=DynamicAgents(models=['fast']),
                ),
                recorder,
            ],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        created_text = "Created sub-agent 'poet'. Delegate tasks to it with `delegate_task`."
        assert tool_returns(result.all_messages())[0] == created_text
        [created] = recorder.of(AgentCreatedEvent)
        assert (created.name, created.instructions, created.model) == ('poet', 'Rhyme.', 'fast')
        assert result.output == 'old pond'


class SerializingTransport(InMemoryTransport):
    """An in-memory transport that puts every message through the wire format, like a cross-process one."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    async def send(self, message: SubAgentMessage) -> None:
        self.sent.append(message.kind)
        await super().send(deserialize_message(serialize_message(message)))


@asynccontextmanager
async def serving(worker: SubAgentWorker[Any]) -> AsyncGenerator[None]:
    async with anyio.create_task_group() as group:
        group.start_soon(worker.serve)
        try:
            yield
        finally:
            group.cancel_scope.cancel()


def remote_parent(
    transport: SubAgentTransport, *steps: Step | ModelResponse, **kwargs: Any
) -> tuple[Agent[Any, str], Recorder]:
    recorder = Recorder()
    capability = SubAgents(
        task_control=True,
        remote_agents=[RemoteSubAgent(name='remote', description='Runs elsewhere.')],
        transport=transport,
        **kwargs,
    )
    return Agent(scripted(*steps), capabilities=[capability, recorder]), recorder


class TestRemoteSubAgents:
    async def test_a_remote_task_runs_on_its_worker_and_asks_questions(self) -> None:
        transport = SerializingTransport()
        asker = Agent(
            scripted(call('ask_parent', question='Which?'), lambda messages: text(f'got {tool_returns(messages)[-1]}')),
            name='remote',
        )
        worker = SubAgentWorker(agents=[SubAgent(asker)], transport=transport, deps=None)

        def answer(messages: list[ModelMessage]) -> ModelResponse:
            return call('answer_task', task_id=task_id_in(tool_returns(messages)[-1]), answer='this one')

        parent, recorder = remote_parent(
            transport,
            call('delegate_task', agent_name='remote', task='Work.'),
            answer,
            text('Waiting.'),
            text('Done.'),
            forward_events=True,
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert 'got this one' in system_texts(result.all_messages())[-1]
        kinds = recorder.kinds()
        assert kinds[:4] == [
            'sub_agents.task_assigned',
            'sub_agents.task_started',
            'sub_agents.task_question',
            'sub_agents.task_backgrounded',
        ]
        assert kinds[-1] == 'sub_agents.task_completed'
        assert recorder.of(TaskProgressEvent)
        [completed] = recorder.of(TaskCompletedEvent)
        assert completed.usage is not None and completed.usage.requests == 2
        assert result.usage.requests == 6
        assert transport.sent[:2] == ['sub_agents.task_assigned', 'sub_agents.task_started']

    async def test_a_remote_task_is_steered_resumed_and_cancelled(self) -> None:
        transport = InMemoryTransport()
        gate = Gate()
        seen: list[str] = []

        def after_pause(messages: list[ModelMessage]) -> ModelResponse:
            seen.extend(user_texts(messages))
            return text('first')

        remote = Agent(scripted(call('pause'), after_pause, text('second')), name='remote', tools=[gate.pause])
        worker = SubAgentWorker(agents=[SubAgent(remote)], transport=transport, deps=None)

        def steer(messages: list[ModelMessage]) -> ModelResponse:
            return call('message_task', task_id=task_id_in(tool_returns(messages)[-1]), message='Hurry.')

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        def resume(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[0])
            return call('delegate_task', agent_name='remote', task='Again.', resume=task_id)

        parent, recorder = remote_parent(
            transport,
            call('delegate_task', agent_name='remote', task='Work.', background=True),
            steer,
            release,
            resume,
            echo_last_return,
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert seen[-1] == 'Message from the agent that delegated this task to you:\nHurry.'
        assert result.output.endswith('(ok):\nsecond')
        assert len(recorder.of(TaskFollowUpEvent)) == 1

    async def test_a_remote_task_can_be_cancelled_softly_or_by_force(self) -> None:
        transport = InMemoryTransport()
        gate = Gate()
        remote = Agent(
            scripted(call('pause'), text('never'), call('pause'), text('never')), name='remote', tools=[gate.pause]
        )
        worker = SubAgentWorker(agents=[SubAgent(remote)], transport=transport, deps=None)

        def cancel(force: bool) -> Step:
            def step(messages: list[ModelMessage]) -> ModelResponse:
                return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]), force=force)

            return step

        def release(messages: list[ModelMessage]) -> ModelResponse:
            gate.opened.set()
            return text('Waiting.')

        parent, recorder = remote_parent(
            transport,
            call('delegate_task', agent_name='remote', task='Soft.', background=True),
            cancel(False),
            release,
            text('Done.'),
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]
        [cancelled] = recorder.of(TaskCancelledEvent)
        assert not cancelled.forced

        forced, forced_recorder = remote_parent(
            transport,
            call('delegate_task', agent_name='remote', task='Hard.', background=True),
            cancel(True),
            text('Waiting.'),
            text('Done.'),
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await forced.run('Go.')
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]
        # The worker's own report can't arrive once the task stopped, so the delegating side reports it.
        [cancelled] = forced_recorder.of(TaskCancelledEvent)
        assert cancelled.forced

    async def test_a_remote_task_nobody_picks_up_fails(self) -> None:
        transport = InMemoryTransport()
        parent, recorder = remote_parent(
            transport, call('delegate_task', agent_name='remote', task='Work.'), echo_last_return, accept_timeout=0.01
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert "No worker picked up the task within 0.01s, so sub-agent 'remote' is unavailable." in result.output
        assert [event.kind for event in recorder.of(TaskFailedEvent)] == ['sub_agents.task_failed']
        # The worker that eventually picks the task up is told to stop it.
        assert isinstance(await transport.receive('agent:remote'), TaskAssignedEvent)

    async def test_a_remote_agent_runs_on_its_own_model(self) -> None:
        parent, _ = remote_parent(
            InMemoryTransport(),
            call('delegate_task', agent_name='remote', task='Work.', model='fast'),
            echo_last_return,
            models={'fast': TestModel()},
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert "Sub-agent 'remote' runs on its worker's model; omit `model`." in result.output


class TestSubAgentWorker:
    async def test_retries_rerun_a_failing_task(self) -> None:
        attempts = 0

        def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('flaky')
            return text('recovered')

        transport = InMemoryTransport()
        worker = SubAgentWorker(
            agents=[SubAgent(Agent(FunctionModel(flaky)), name='remote')], transport=transport, deps=None, retries=1
        )
        parent, recorder = remote_parent(
            transport, call('delegate_task', agent_name='remote', task='Work.'), echo_last_return
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nrecovered')
        [retry] = recorder.of(TaskRetryEvent)
        assert (retry.attempt, retry.error) == (2, 'RuntimeError: flaky')

    async def test_a_task_that_cannot_finish_fails(self) -> None:
        def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('broken')

        async def noop() -> str:
            """Do nothing."""
            return 'ok'

        async def needs_approval() -> str:
            """Needs approval."""
            raise ApprovalRequired

        names = ('broken', 'limited', 'approving', 'modelless')
        agents = [
            SubAgent(Agent(FunctionModel(broken)), name='broken'),
            SubAgent(Agent(scripted(call('noop'), call('noop'), text('x')), tools=[noop]), name='limited'),
            SubAgent(
                Agent(
                    scripted(call('needs_approval')), output_type=[str, DeferredToolRequests], tools=[needs_approval]
                ),
                name='approving',
            ),
            SubAgent(Agent(), name='modelless'),
        ]
        transport = InMemoryTransport()
        worker = SubAgentWorker(agents=agents, transport=transport, deps=None)
        parent = Agent(
            scripted(
                ModelResponse(
                    parts=[ToolCallPart('delegate_task', {'agent_name': name, 'task': 'Work.'}) for name in names]
                ),
                text('Done.'),
            ),
            capabilities=[
                SubAgents(
                    task_control=True,
                    remote_agents=[RemoteSubAgent(name=name, can_ask_questions=False) for name in names],
                    transport=transport,
                    # The workers' usage stays theirs, so it doesn't count against the parent's own request limit.
                    forward_usage=False,
                )
            ],
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.', usage_limits=UsageLimits(request_limit=2))
        broken_out, limited_out, approving_out, modelless_out = tool_returns(result.all_messages())
        assert broken_out.endswith('(failed):\nRuntimeError: broken')
        assert '(failed):\nThe next request would exceed the request_limit of 2.' in limited_out
        assert 'needs tool approval or deferred tool results' in approving_out
        assert '(failed):\n`model` must either be set on the agent or included when calling it.' in modelless_out

    async def test_a_resume_the_worker_has_no_conversation_for_fails(self) -> None:
        transport = InMemoryTransport()
        worker = SubAgentWorker(
            agents=[SubAgent(child(), name='remote')], transport=transport, deps=None, max_conversations=0
        )

        def resume(messages: list[ModelMessage]) -> ModelResponse:
            return call(
                'delegate_task', agent_name='remote', task='Again.', resume=task_id_in(tool_returns(messages)[-1])
            )

        parent, _ = remote_parent(
            transport, call('delegate_task', agent_name='remote', task='Work.'), resume, echo_last_return
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert result.output.endswith('(failed):\nThis worker has no conversation to resume.')

    async def test_messages_sent_before_a_task_runs_reach_it(self) -> None:
        transport = InMemoryTransport()
        seen: list[str] = []

        def respond(messages: list[ModelMessage]) -> ModelResponse:
            seen.extend(user_texts(messages))
            return text('done')

        worker = SubAgentWorker(
            agents=[SubAgent(Agent(scripted(respond)), name='remote')], transport=transport, deps=None
        )
        start = TaskAssignedEvent(task_id='t1', agent_name='remote', prompt='Work.', can_ask_questions=False)
        await transport.send(TaskMessageEvent(task_id='t1', agent_name='remote', message='Early.'))
        await transport.send(TaskAnswerEvent(task_id='t1', agent_name='remote', question_id='unknown', answer='x'))
        await transport.send(start)
        with anyio.fail_after(WAIT):
            async with serving(worker):
                assert isinstance(await transport.receive('task:t1:parent'), TaskStartedEvent)
                assert isinstance(await transport.receive('task:t1:parent'), TaskCompletedEvent)
        assert seen == ['Work.', 'Message from the agent that delegated this task to you:\nEarly.']

    async def test_a_cancel_request_before_a_task_runs_cancels_it(self) -> None:
        transport = InMemoryTransport()
        worker = SubAgentWorker(agents=[SubAgent(child(), name='remote')], transport=transport, deps=None)
        await transport.send(TaskCancelRequestEvent(task_id='t1', agent_name='remote'))
        await transport.send(TaskAssignedEvent(task_id='t1', agent_name='remote', prompt='Work.'))
        with anyio.fail_after(WAIT):
            async with serving(worker):
                assert isinstance(await transport.receive('task:t1:parent'), TaskStartedEvent)
                settled = await transport.receive('task:t1:parent')
        assert isinstance(settled, TaskCancelledEvent) and not settled.forced

    async def test_a_user_error_fails_only_its_task(self) -> None:
        transport = InMemoryTransport()

        @dataclass
        class Misconfigured(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                raise UserError('misconfigured')

        broken = Agent(TestModel(), capabilities=[Misconfigured()])
        worker = SubAgentWorker(agents=[SubAgent(broken, name='remote')], transport=transport, deps=None)
        await transport.send(TaskAssignedEvent(task_id='t1', agent_name='remote', prompt='Work.'))
        with anyio.fail_after(WAIT):
            async with serving(worker):
                assert isinstance(await transport.receive('task:t1:parent'), TaskStartedEvent)
                failed = await transport.receive('task:t1:parent')
        assert isinstance(failed, TaskFailedEvent) and failed.error == 'misconfigured'

    def test_configuration_errors(self) -> None:
        transport = InMemoryTransport()
        with pytest.raises(ValueError, match='has no name'):
            SubAgentWorker(agents=[SubAgent(Agent())], transport=transport, deps=None)
        with pytest.raises(ValueError, match="Duplicate sub-agent name 'a'"):
            SubAgentWorker(
                agents=[SubAgent(Agent(), name='a'), SubAgent(Agent(), name='a')], transport=transport, deps=None
            )
        with pytest.raises(ValueError, match='`retries` must be zero or more'):
            SubAgentWorker(agents=[], transport=transport, deps=None, retries=-1)
        with pytest.raises(ValueError, match='`max_conversations` must be zero or more'):
            SubAgentWorker(agents=[], transport=transport, deps=None, max_conversations=-1)


class TestSubAgentMessages:
    def test_every_message_round_trips_and_has_its_mailbox(self) -> None:
        messages: list[SubAgentMessage] = [
            TaskAssignedEvent(task_id='t', agent_name='a', prompt='p', trace_context={'traceparent': 'x'}),
            TaskFollowUpEvent(task_id='t', agent_name='a', prompt='p'),
            TaskMessageEvent(task_id='t', agent_name='a', message='m'),
            TaskAnswerEvent(task_id='t', agent_name='a', question_id='q', answer='x'),
            TaskCancelRequestEvent(task_id='t', agent_name='a'),
            TaskForceCancelEvent(task_id='t', agent_name='a'),
            TaskStartedEvent(task_id='t', agent_name='a'),
            TaskQuestionEvent(task_id='t', agent_name='a', question_id='q', question='?'),
            TaskRetryEvent(task_id='t', agent_name='a', attempt=2, error='e'),
            TaskCompletedEvent(task_id='t', agent_name='a', output='o', usage=RunUsage(requests=1)),
            TaskFailedEvent(task_id='t', agent_name='a', error='e'),
            TaskCancelledEvent(task_id='t', agent_name='a', forced=True),
        ]
        for message in messages:
            assert deserialize_message(serialize_message(message)) == message
        assert [message.mailbox for message in messages[:3]] == ['agent:a', 'agent:a', 'task:t:worker']
        assert messages[6].mailbox == 'task:t:parent'

    def test_deserialize_message_rejects_other_events(self) -> None:
        event = TaskBackgroundedEvent(task_id='t', agent_name='a', reason='auto')
        data = TypeAdapter(CapabilityEvent).dump_json(event)
        with pytest.raises(
            ValueError, match=r"Expected a sub-agent message, got a 'sub_agents\.task_backgrounded' event\."
        ):
            deserialize_message(data)

    async def test_a_custom_transport_release_is_a_no_op_by_default(self) -> None:
        class Minimal(SubAgentTransport):
            async def send(self, message: SubAgentMessage) -> None:
                raise NotImplementedError

            async def receive(self, mailbox: str) -> SubAgentMessage:
                raise NotImplementedError

        assert await Minimal().release('task:t:parent') is None


class TestConfiguration:
    def test_task_control_fields_require_task_control(self) -> None:
        with pytest.raises(ValueError, match=r'`transport`, `max_running_tasks` require `task_control=True`'):
            SubAgents(agents=[researcher()], transport=InMemoryTransport(), max_running_tasks=2)

    def test_invalid_task_control_configurations(self) -> None:
        with pytest.raises(ValueError, match='need a `transport`'):
            SubAgents(task_control=True, remote_agents=[RemoteSubAgent(name='r')])
        with pytest.raises(ValueError, match="Duplicate sub-agent name 'researcher'"):
            controlled(remote_agents=[RemoteSubAgent(name='researcher')], transport=InMemoryTransport())
        with pytest.raises(ValueError, match="Duplicate sub-agent name 'r'"):
            controlled(
                remote_agents=[RemoteSubAgent(name='r'), RemoteSubAgent(name='r')], transport=InMemoryTransport()
            )
        with pytest.raises(ValueError, match="`tool_name` 'list_tasks' is the name of one of the task-control tools"):
            controlled(tool_name='list_tasks')
        with pytest.raises(ValueError, match="`tool_name` 'create_agent' is the name of the `dynamic_agents` tool"):
            SubAgents(tool_name='create_agent', dynamic_agents=DynamicAgents())
        with pytest.raises(ValueError, match='`max_running_tasks` must be at least 1'):
            controlled(max_running_tasks=0)
        with pytest.raises(ValueError, match='must be 1 to 64 letters'):
            RemoteSubAgent(name='no spaces')
        with pytest.raises(ValueError):
            SubAgents(dynamic_agents=DynamicAgents(models=['missing']))

    async def test_only_one_sub_agents_on_an_agent_can_control_tasks(self) -> None:
        parent = Agent(
            TestModel(),
            capabilities=[
                controlled(),
                SubAgents(agents=[researcher()], task_control=True, id='other', tool_name='other'),
            ],
        )
        with pytest.raises(UserError, match='Only one `SubAgents` on an agent can set `task_control`'):
            await parent.run('Go.')


class TestTaskControlEdges:
    async def test_a_question_queued_on_a_run_that_ended_is_asked_again(self) -> None:
        owner = DelegationTasks()
        async with owner.opened():

            async def run(record: DelegationTask) -> str:
                return await owner.ask(record.id, question_id='q1', question='Which?')

            receipt = await owner.delegate(
                agent_name='worker',
                prompt='p',
                conversation_id='root',
                model=None,
                background=True,
                resume=None,
                run=run,
            )
            task_id = task_id_in(receipt)
            with anyio.fail_after(WAIT):
                while owner.pending_question(task_id) is None:
                    await anyio.sleep(0)
            assert owner.pending_question(task_id) == ('q1', 'Which?')
            with owner.receiving(conversation_id='root', parent_id=None, enqueue=lambda part: None):
                pass
            parts: list[SystemPromptPart] = []

            def enqueue(part: SystemPromptPart) -> str:
                parts.append(part)
                return f'e{len(parts)}'

            with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue):
                owner.queue_reports(conversation_id='root', parent_id=None)
            # That run ended before delivering the question, so the next one gets it again.
            with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue):
                await owner.acknowledge('e2')
            assert len(parts) == 2 and '<task-question>\nWhich?\n</task-question>' in parts[0].content
            assert not owner.answer(task_id, question_id='other', answer='x')
            assert owner.answer(task_id, question_id='q1', answer='yes')
            assert not owner.answer(task_id, question_id='q1', answer='again')
            record = owner.records[task_id]
            with anyio.fail_after(WAIT):
                while record.status == 'running':
                    await anyio.sleep(0)
            assert record.output == 'yes'
            assert owner.pending_question(task_id) is None
            started = await owner.publish(record, TaskStartedEvent(task_id=task_id, agent_name='worker'), observe=False)
            assert started.capability_id is None

    async def test_a_queued_question_gets_no_answer_once_the_task_is_asked_to_stop(self) -> None:
        asker = Agent(
            scripted(
                ModelResponse(
                    parts=[
                        ToolCallPart('ask_parent', {'question': 'First?'}),
                        ToolCallPart('ask_parent', {'question': 'Second?'}),
                    ]
                ),
                text('never'),
            )
        )
        recorder = Recorder()

        def cancel(messages: list[ModelMessage]) -> ModelResponse:
            return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]))

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Ask.'), cancel, text('Done.'), text('Done.')),
            capabilities=[controlled(researcher(asker)), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert [event.question for event in recorder.of(TaskQuestionEvent)] == ['First?']
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]

    async def test_a_vetoed_listener_answer_leaves_the_question_to_the_model(self) -> None:
        asker = Agent(
            scripted(call('ask_parent', question='Which?'), lambda messages: text(tool_returns(messages)[-1]))
        )

        @dataclass
        class Policy(AbstractCapability[Any]):
            @on_event(TaskQuestionEvent)
            async def auto_answer(self, ctx: RunContext[Any], event: TaskQuestionEvent) -> None:
                event.answer = 'auto'

            @on_event(TaskAnswerEvent)
            async def no_auto(self, ctx: RunContext[Any], event: TaskAnswerEvent) -> None:
                if event.answer == 'auto':
                    event.veto()

        def answer(messages: list[ModelMessage]) -> ModelResponse:
            return call('answer_task', task_id=task_id_in(tool_returns(messages)[-1]), answer='manual')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Ask.'), answer, text('Waiting.'), text('Done.')
            ),
            capabilities=[controlled(researcher(asker)), Policy()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert '<task-output>\nmanual\n</task-output>' in system_texts(result.all_messages())[-1]

    async def test_two_answers_to_one_question_deliver_one(self) -> None:
        asker = Agent(
            scripted(call('ask_parent', question='Which?'), lambda messages: text(tool_returns(messages)[-1]))
        )

        def answer_twice(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return ModelResponse(
                parts=[
                    ToolCallPart('answer_task', {'task_id': task_id, 'answer': 'one'}),
                    ToolCallPart('answer_task', {'task_id': task_id, 'answer': 'two'}),
                ]
            )

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Ask.'),
                answer_twice,
                text('Waiting.'),
                text('Done.'),
            ),
            capabilities=[controlled(researcher(asker))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        first, second = tool_returns(result.all_messages())[1:3]
        assert first.startswith('Answer sent to task ')
        assert second.endswith('is no longer waiting for that answer.')
        assert '<task-output>\none\n</task-output>' in system_texts(result.all_messages())[-1]

    async def test_a_message_to_a_task_that_settles_meanwhile_is_not_delivered(self) -> None:
        gate = Gate()
        slow = Agent(scripted(call('pause'), text('done')), tools=[gate.pause])
        settled = anyio.Event()

        @dataclass
        class Policy(AbstractCapability[Any]):
            @on_event(TaskMessageEvent)
            async def let_it_finish(self, ctx: RunContext[Any], event: TaskMessageEvent) -> None:
                gate.opened.set()
                await settled.wait()

            @on_event(TaskCompletedEvent)
            async def completed(self, ctx: RunContext[Any], event: TaskCompletedEvent) -> None:
                settled.set()

        def steer(messages: list[ModelMessage]) -> ModelResponse:
            return call('message_task', task_id=task_id_in(tool_returns(messages)[-1]), message='Late.')

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Slow.', background=True),
                steer,
                text('Done.'),
                text('Done.'),
            ),
            capabilities=[controlled(researcher(slow)), Policy()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert tool_returns(result.all_messages())[1].endswith('finished before the message reached it.')

    async def test_tasks_of_tasks_are_listed_but_controlled_by_their_own_parent(self) -> None:
        asked = anyio.Event()
        listed = anyio.Event()

        @dataclass
        class QuestionSeen(AbstractCapability[Any]):
            @on_event(TaskQuestionEvent)
            async def seen(self, ctx: RunContext[Any], event: TaskQuestionEvent) -> None:
                asked.set()

        asker = Agent(
            scripted(call('ask_parent', question='Inner?'), lambda messages: text(tool_returns(messages)[-1])),
            name='inner',
        )
        quick = Agent(TestModel(call_tools=[], custom_output_text='quick'), name='quick')
        inner_ids: list[str] = []

        def resume_sibling(messages: list[ModelMessage]) -> ModelResponse:
            inner_ids.append(task_id_in(tool_returns(messages)[-1]))
            return call('delegate_task', agent_name='quick', task='x', resume=quick_ids[0])

        async def answer_after_listing(messages: list[ModelMessage]) -> ModelResponse:
            await listed.wait()
            return call('answer_task', task_id=inner_ids[0], answer='42')

        middle = Agent(
            scripted(
                call('delegate_task', agent_name='inner', task='Deep.', background=True),
                resume_sibling,
                answer_after_listing,
                text('Waiting.'),
                text('middle done'),
            ),
            name='middle',
            capabilities=[SubAgents(agents=[SubAgent(asker)], task_control=True), QuestionSeen()],
        )
        quick_ids: list[str] = []

        def start_middle(messages: list[ModelMessage]) -> ModelResponse:
            quick_ids.append(task_id_in(tool_returns(messages)[-1]))
            return call('delegate_task', agent_name='middle', task='Go deep.', background=True)

        async def list_when_asked(messages: list[ModelMessage]) -> ModelResponse:
            await asked.wait()
            return call('list_tasks')

        def control_grandchild(messages: list[ModelMessage]) -> ModelResponse:
            listed.set()
            return ModelResponse(
                parts=[
                    ToolCallPart('message_task', {'task_id': inner_ids[0], 'message': 'hi'}),
                    ToolCallPart('answer_task', {'task_id': inner_ids[0], 'answer': 'no'}),
                ]
            )

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='quick', task='Quick.'),
                start_middle,
                list_when_asked,
                control_grandchild,
                text('Waiting.'),
                text('Done.'),
            ),
            capabilities=[SubAgents(agents=[SubAgent(quick), SubAgent(middle)], task_control=True)],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())
        listing = returns[2].splitlines()
        assert listing[0].startswith(f'- {quick_ids[0]} (quick, finished: ok, resumable)')
        assert '(middle, running, background, resumable)' in listing[1]
        assert '(inner, running, background, started by task ' in listing[2]
        assert listing[3] == "  Waiting for an answer to: 'Inner?'"
        assert 'is steered by the task that started it' in returns[3]
        assert 'asks the task that started it' in returns[4]
        middle_report = system_texts(result.all_messages())[-1]
        assert 'middle done' in middle_report

    async def test_a_task_can_be_resumed_on_another_model_and_max_calls_still_apply(self) -> None:
        def resume(messages: list[ModelMessage]) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task',
                        {
                            'agent_name': 'researcher',
                            'task': 'Again.',
                            'resume': task_id_in(tool_returns(messages)[-1]),
                            'model': 'slow',
                        },
                    )
                ]
            )

        parent = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='First.', model='fast'),
                resume,
                call('delegate_task', agent_name='researcher', task='Third.'),
                echo_last_return,
            ),
            capabilities=[
                controlled(
                    researcher(Agent(name='researcher'), max_calls=2, can_ask_questions=False),
                    models={'fast': TestModel(custom_output_text='fast'), 'slow': TestModel(custom_output_text='slow')},
                )
            ],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        returns = tool_returns(result.all_messages())
        assert returns[0].endswith('(ok):\nfast')
        assert returns[1].endswith('(ok):\nslow')
        assert "Delegate budget for 'researcher' is exhausted for this run" in returns[2]

    async def test_a_task_that_fails_or_times_out_reports_a_failure(self) -> None:
        gate = Gate()
        failing = Agent(scripted(*[text('not a number')] * 3), output_type=int, name='failing')
        slow = Agent(scripted(call('pause'), text('late')), tools=[gate.pause], name='slow')
        recorder = Recorder()
        parent = Agent(
            scripted(
                ModelResponse(
                    parts=[
                        ToolCallPart('delegate_task', {'agent_name': 'failing', 'task': 'Count.'}),
                        ToolCallPart('delegate_task', {'agent_name': 'slow', 'task': 'Wait.'}),
                    ]
                ),
                text('Done.'),
            ),
            capabilities=[
                SubAgents(agents=[SubAgent(failing), SubAgent(slow, timeout_seconds=0.01)], task_control=True),
                recorder,
            ],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        failed_out, timeout_out = tool_returns(result.all_messages())
        assert "(failed):\nModelRetry: Sub-agent 'failing' failed" in failed_out
        assert "(timeout):\nSub-agent 'slow' exceeded its 0.01s time budget." in timeout_out
        assert len(recorder.of(TaskFailedEvent)) == 2

    async def test_a_directly_registered_toolset_sends_task_events_only_to_the_observer(self) -> None:
        observed: list[str] = []

        async def observe(event: DelegationTaskEvent) -> None:
            if isinstance(event.event, CapabilityEvent):
                observed.append(event.event.kind)

        toolset = SubAgentToolset[Any](
            agents={'researcher': researcher()},
            forward_usage=True,
            inherit_tools=False,
            shared_capabilities=[],
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=2,
            contain_errors=False,
            call_counts={},
            task_control=True,
            dynamic_agents=DynamicAgents(),
        )
        parent = Agent(
            scripted(
                call('create_agent', name='poet', description='Poems.', instructions='Rhyme.'),
                call('delegate_task', agent_name='researcher', task='Find x.'),
                echo_last_return,
            ),
            toolsets=[toolset],
        )
        owner = DelegationTasks(observer=observe)
        with anyio.fail_after(WAIT):
            async with owner.opened():
                with owner.bind():
                    result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nchild output')
        assert observed[0] == 'sub_agents.task_assigned' and observed[-1] == 'sub_agents.task_completed'

    async def test_a_remote_task_asked_to_stop_while_waiting_for_an_answer_stops(self) -> None:
        transport = InMemoryTransport()
        asker = Agent(scripted(call('ask_parent', question='Which?'), text('never')), name='remote')
        worker = SubAgentWorker(agents=[SubAgent(asker)], transport=transport, deps=None)

        def cancel(messages: list[ModelMessage]) -> ModelResponse:
            return call('cancel_task', task_id=task_id_in(tool_returns(messages)[-1]))

        parent, recorder = remote_parent(
            transport, call('delegate_task', agent_name='remote', task='Ask.'), cancel, text('Waiting.'), text('Done.')
        )
        with anyio.fail_after(WAIT):
            async with serving(worker):
                result = await parent.run('Go.')
        assert 'outcome: cancelled' in system_texts(result.all_messages())[-1]
        [cancelled] = recorder.of(TaskCancelledEvent)
        assert not cancelled.forced


class TestDynamicAgentLimits:
    async def run_one(self, capability: SubAgents[Any], **args: Any) -> str:
        parent = Agent(scripted(call('create_agent', **args), echo_last_return), capabilities=[capability])
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        return tool_returns(result.all_messages())[0]

    async def test_create_agent_refusals(self) -> None:
        base = {'description': 'd', 'instructions': 'i'}
        menu = {'fast': TestModel()}
        assert 'must be 1 to 64 letters' in await self.run_one(
            SubAgents(dynamic_agents=DynamicAgents()), name='no spaces', **base
        )
        assert "A sub-agent named 'self' already exists." in await self.run_one(
            SubAgents(include_self=True, dynamic_agents=DynamicAgents()), name='self', **base
        )
        assert "A sub-agent named 'researcher' already exists." in await self.run_one(
            SubAgents(agents=[researcher()], dynamic_agents=DynamicAgents()), name='researcher', **base
        )
        assert 'the most allowed in one run' in await self.run_one(
            SubAgents(dynamic_agents=DynamicAgents(max_agents=0)), name='poet', **base
        )
        assert "Model 'slow' is not allowed. Choose one of: fast." in await self.run_one(
            SubAgents(models=menu, dynamic_agents=DynamicAgents(models=['fast'])), name='poet', model='slow', **base
        )
        assert "Created sub-agent 'poet'" in await self.run_one(
            SubAgents(models=menu, dynamic_agents=DynamicAgents(models=['fast'])), name='poet', **base
        )
        assert 'You cannot choose a model' in await self.run_one(
            SubAgents(dynamic_agents=DynamicAgents()), name='poet', model='fast', **base
        )

    async def test_a_listener_can_veto_or_rewrite_a_created_agent(self) -> None:
        built: list[DynamicAgentSpec] = []

        def factory(spec: DynamicAgentSpec) -> Agent[Any, str]:
            built.append(spec)
            return Agent(TestModel(custom_output_text=spec.instructions), name=spec.name)

        @dataclass
        class Policy(AbstractCapability[Any]):
            @on_event(AgentCreatedEvent)
            async def check(self, ctx: RunContext[Any], event: AgentCreatedEvent) -> None:
                if event.name == 'rogue':
                    event.veto('not on my watch')
                event.instructions = f'Be careful. {event.instructions}'

        parent = Agent(
            scripted(
                call('create_agent', name='rogue', description='d', instructions='i'),
                call('create_agent', name='careful', description='d', instructions='Count.'),
                call('delegate_task', agent_name='careful', task='Go.'),
                echo_last_return,
            ),
            capabilities=[SubAgents(dynamic_agents=DynamicAgents(factory=factory)), Policy()],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert tool_returns(result.all_messages())[0] == 'The sub-agent was not created: not on my watch.'
        assert [spec.instructions for spec in built] == ['Be careful. Count.']
        assert result.output == 'Be careful. Count.'


class FailingTransport(InMemoryTransport):
    """An in-memory transport that can't send the messages of `kinds`, of any task or only of `task_id`."""

    def __init__(self, *kinds: str, task_id: str | None = None, release_fails: bool = False) -> None:
        super().__init__()
        self.kinds = set(kinds)
        self.task_id = task_id
        self.release_fails = release_fails
        self.released: list[str] = []

    async def send(self, message: SubAgentMessage) -> None:
        if message.kind in self.kinds and self.task_id in (None, message.task_id):
            raise ConnectionError(f'cannot send {message.kind}')
        await super().send(message)

    async def release(self, mailbox: str) -> None:
        if self.release_fails:
            raise ConnectionError(f'cannot release {mailbox}')
        self.released.append(mailbox)
        await super().release(mailbox)


class SandboxWorkspace(WorkspaceBackend):
    """A workspace like a sandbox the run owns, which a background task can't outlive."""

    @property
    def ref(self) -> None:
        return None

    async def working_dir(self) -> str:
        raise AssertionError('the workspace was touched')  # pragma: no cover


class TestTaskControlBoundaries:
    async def test_task_control_and_dynamic_agents_refuse_durable_execution(self) -> None:
        for capability in (controlled(), SubAgents(agents=[researcher()], dynamic_agents=DynamicAgents())):
            agent = Agent(TestModel(), name='durable', capabilities=[capability, RecordingDurability()])
            with pytest.raises(UserError, match='not supported under durable execution'):
                await agent.run('Go.')

    async def test_a_run_a_tool_starts_gets_an_owner_of_its_own(self) -> None:
        inner = Agent(
            scripted(
                call('delegate_task', agent_name='researcher', task='Inner.', background=True),
                text('Waiting.'),
                text('Reported.'),
            ),
            capabilities=[controlled()],
        )

        async def consult() -> str:
            """Ask the inner agent."""
            return (await inner.run('Inner go.')).output

        outer = Agent(scripted(call('consult'), echo_last_return), capabilities=[controlled()], tools=[consult])
        with anyio.fail_after(WAIT):
            result = await outer.run('Go.')
        # The inner run waited for its own task, whose report started its next request.
        assert result.output == 'Reported.'

    async def test_a_background_task_the_workspace_cannot_outlive_is_refused_before_it_is_announced(self) -> None:
        recorder = Recorder()
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Find x.', background=True), echo_last_return),
            capabilities=[controlled(), recorder],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.', workspace=SandboxWorkspace())
        assert 'cannot run background children' in result.output
        assert not recorder.of(TaskAssignedEvent)

    async def test_a_one_shot_delegate_is_one_shot_with_an_application_owner(self) -> None:
        def follow_up(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return call('delegate_task', agent_name='researcher', task='And then?', resume=task_id)

        owner = DelegationTasks()
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='First.'), follow_up, echo_last_return),
            capabilities=[SubAgents(agents=[researcher(resumable=False)])],
        )
        with anyio.fail_after(WAIT):
            async with owner.opened():
                with owner.bind():
                    result = await parent.run('Go.', conversation_id='session')
        assert 'cannot resume; it is one-shot' in result.output

    async def test_an_application_owner_gets_a_follow_up_once(self) -> None:
        observed: list[DelegationTaskEvent] = []

        async def observe(event: DelegationTaskEvent) -> None:
            observed.append(event)

        def follow_up(messages: list[ModelMessage]) -> ModelResponse:
            task_id = task_id_in(tool_returns(messages)[-1])
            return call('delegate_task', agent_name='researcher', task='And then?', resume=task_id)

        owner = DelegationTasks(observer=observe)
        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='First.'), follow_up, echo_last_return),
            capabilities=[controlled()],
        )
        with anyio.fail_after(WAIT):
            async with owner.opened():
                with owner.bind():
                    await parent.run('Go.', conversation_id='session')
        assert sum(isinstance(event.event, TaskFollowUpEvent) for event in observed) == 1

    async def test_a_remote_task_the_transport_cannot_send_fails(self) -> None:
        transport = FailingTransport('sub_agents.task_assigned')
        parent, recorder = remote_parent(
            transport, call('delegate_task', agent_name='remote', task='Work.'), echo_last_return
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        [failed] = recorder.of(TaskFailedEvent)
        assert failed.error == 'ConnectionError: cannot send sub_agents.task_assigned'
        assert result.output.endswith('(failed):\nConnectionError: cannot send sub_agents.task_assigned')

    async def test_a_worker_outlives_stray_messages_and_failing_tasks(self, caplog: pytest.LogCaptureFixture) -> None:
        class Stray(FailingTransport):
            strayed = False

            async def receive(self, mailbox: str) -> SubAgentMessage:
                if mailbox == 'agent:remote' and not self.strayed:
                    self.strayed = True
                    return TaskMessageEvent(task_id='t0', agent_name='remote', message='stray')
                return await super().receive(mailbox)

        transport = Stray('sub_agents.task_started', 'sub_agents.task_failed', task_id='t1')
        worker = SubAgentWorker(agents=[SubAgent(child(), name='remote')], transport=transport, deps=None)
        await transport.send(TaskAssignedEvent(task_id='t1', agent_name='remote', prompt='Work.'))
        await transport.send(TaskAssignedEvent(task_id='t2', agent_name='remote', prompt='Work.'))
        with caplog.at_level(logging.WARNING), anyio.fail_after(WAIT):
            async with serving(worker):
                assert isinstance(await transport.receive('task:t2:parent'), TaskStartedEvent)
                assert isinstance(await transport.receive('task:t2:parent'), TaskCompletedEvent)
                while 'Could not report the failure of sub-agent task t1.' not in caplog.text:
                    await anyio.sleep(0)  # pragma: lax no cover - the failure may already be logged
        assert "skipped a 'sub_agents.task_message' message" in caplog.text

    async def test_a_worker_reports_a_task_the_transport_failed(self) -> None:
        transport = FailingTransport('sub_agents.task_started')
        worker = SubAgentWorker(agents=[SubAgent(child(), name='remote')], transport=transport, deps=None)
        await transport.send(TaskAssignedEvent(task_id='t1', agent_name='remote', prompt='Work.'))
        with anyio.fail_after(WAIT):
            async with serving(worker):
                failed = await transport.receive('task:t1:parent')
        assert isinstance(failed, TaskFailedEvent)
        assert failed.error == 'ConnectionError: cannot send sub_agents.task_started'

    async def test_a_plain_sub_agents_run_a_tool_starts_has_no_owner(self) -> None:
        inner = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Inner.', background=True), echo_last_return),
            capabilities=[SubAgents(agents=[researcher()])],
        )

        async def consult() -> str:
            """Ask the inner agent."""
            return (await inner.run('Inner go.')).output

        outer = Agent(scripted(call('consult'), echo_last_return), capabilities=[controlled()], tools=[consult])
        with anyio.fail_after(WAIT):
            result = await outer.run('Go.')
        assert 'Background execution and resume require an open `DelegationTasks` owner' in result.output

    async def test_a_task_with_plain_sub_agents_reports_its_delegations_to_its_own_run(self) -> None:
        inner_recorder = Recorder()
        delegating = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Deeper.'), echo_last_return),
            name='middle',
            capabilities=[
                SubAgents(agents=[researcher()]),
                # A second plain `SubAgents` on the same run publishes alongside the first.
                SubAgents(agents=[SubAgent(child(), name='other')], tool_name='delegate_other', id='other'),
                inner_recorder,
            ],
        )
        parent = Agent(
            scripted(call('delegate_task', agent_name='middle', task='Go deeper.'), echo_last_return),
            capabilities=[controlled(SubAgent(delegating))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nchild output')
        assert sorted(inner_recorder.kinds()) == ['sub_agents.delegation_end', 'sub_agents.delegation_start']

    async def test_a_remote_task_keeps_its_error_when_cleanup_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = FailingTransport('sub_agents.task_assigned', 'sub_agents.task_force_cancel')
        parent, _ = remote_parent(transport, call('delegate_task', agent_name='remote', task='Work.'), echo_last_return)
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(failed):\nConnectionError: cannot send sub_agents.task_assigned')
        assert 'Could not stop remote sub-agent task' in caplog.text
        # The mailbox is released even though the worker could not be told to stop.
        [mailbox] = transport.released
        assert mailbox.startswith('task:') and mailbox.endswith(':parent')

        failing_release = FailingTransport('sub_agents.task_assigned', release_fails=True)
        parent, _ = remote_parent(
            failing_release, call('delegate_task', agent_name='remote', task='Work.'), echo_last_return
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(failed):\nConnectionError: cannot send sub_agents.task_assigned')
        assert 'Could not release the mailbox of remote sub-agent task' in caplog.text

    async def test_a_plain_run_under_another_owner_in_a_task_is_not_a_task(self) -> None:
        inner = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='Inner.'), echo_last_return),
            capabilities=[SubAgents(agents=[researcher()])],
        )

        async def nested() -> str:
            """Run the inner agent under an owner of its own."""
            other = DelegationTasks()
            async with other.opened():
                with other.bind():
                    return (await inner.run('Inner go.')).output

        middle = Agent(scripted(call('nested'), echo_last_return), name='middle', tools=[nested])
        parent = Agent(
            scripted(call('delegate_task', agent_name='middle', task='Go.'), echo_last_return),
            capabilities=[controlled(SubAgent(middle))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert result.output.endswith('(ok):\nchild output')

    async def test_a_task_asked_to_stop_while_putting_a_question_gets_no_answer_to_wait_for(self) -> None:
        owner = DelegationTasks()
        async with owner.opened():

            async def run(record: DelegationTask) -> str:
                owner.request_stop(record.id)
                return await owner.ask(record.id, question_id='q1', question='Which?')

            with anyio.fail_after(WAIT):
                result = await owner.delegate(
                    agent_name='worker',
                    prompt='p',
                    conversation_id='root',
                    model=None,
                    background=False,
                    resume=None,
                    run=run,
                )
        assert result.endswith(f'(ok):\n{CANCELLED_ANSWER}')

    async def test_a_plain_task_in_a_task_control_tree_cannot_resume_a_sibling(self) -> None:
        def resume_sibling(messages: list[ModelMessage]) -> ModelResponse:
            return call('delegate_task', agent_name='researcher', task='Mine now.', resume=user_texts(messages)[0])

        middle = Agent(
            scripted(resume_sibling, echo_last_return), name='middle', capabilities=[SubAgents(agents=[researcher()])]
        )

        def delegate_middle(messages: list[ModelMessage]) -> ModelResponse:
            return call('delegate_task', agent_name='middle', task=task_id_in(tool_returns(messages)[-1]))

        parent = Agent(
            scripted(call('delegate_task', agent_name='researcher', task='First.'), delegate_middle, echo_last_return),
            capabilities=[controlled(researcher(), SubAgent(middle))],
        )
        with anyio.fail_after(WAIT):
            result = await parent.run('Go.')
        assert 'only the agent that started it can resume it' in result.output


class TestDelegationTasksConcurrency:
    """A delegation persisting its acceptance races other users of the same owner."""

    @staticmethod
    async def accepting(owner: DelegationTasks, run: Callable[[DelegationTask], Awaitable[str]]) -> asyncio.Task[str]:
        """A background delegation, started and left while it persists its acceptance."""
        started = asyncio.create_task(
            owner.delegate(
                agent_name='worker',
                prompt='p',
                conversation_id='root',
                model=None,
                background=True,
                resume=None,
                run=run,
            )
        )
        while not owner.records:
            await asyncio.sleep(0)
        return started

    async def test_waiting_for_a_child_that_has_no_worker_yet(self, tmp_path: Path) -> None:
        owner = DelegationTasks(directory=tmp_path)

        async def run(record: DelegationTask) -> str:
            return 'done'

        async with owner.opened():
            started = await self.accepting(owner, run)
            [record] = owner.records.values()
            assert not started.done() and record.status == 'running', 'still persisting its acceptance'
            with anyio.fail_after(WAIT):
                await owner.wait_children(None)
                await started
            [record] = owner.records.values()
            assert (record.status, record.output) == ('finished', 'done')

    async def test_a_delegation_that_fails_to_persist_settles(self, tmp_path: Path) -> None:
        owner = DelegationTasks(directory=tmp_path)

        async def run(record: DelegationTask) -> str:
            raise AssertionError('not scheduled')  # pragma: no cover

        async with owner.opened():
            started = await self.accepting(owner, run)
            started.cancel()
            with anyio.fail_after(WAIT):
                await owner.wait_children(None)
                with pytest.raises(asyncio.CancelledError):
                    await started
            [record] = owner.records.values()
            assert (record.status, record.outcome) == ('finished', 'cancelled')

    async def test_a_child_stopped_before_its_worker_is_scheduled_ends_the_wait(self, tmp_path: Path) -> None:
        owner = DelegationTasks(directory=tmp_path)

        async def run(record: DelegationTask) -> str:
            raise AssertionError('not scheduled')  # pragma: no cover

        async with owner.opened():
            started = await self.accepting(owner, run)
            [record] = owner.records.values()
            waiting = asyncio.create_task(owner.wait_children(None))
            await asyncio.sleep(0)
            with anyio.fail_after(WAIT):
                await owner.cancel(record.id)
                await waiting
                assert 'stopped before execution' in await started
            assert (record.status, record.outcome) == ('finished', 'cancelled')

    async def test_closing_schedules_no_worker_for_a_delegation_persisting_its_acceptance(self, tmp_path: Path) -> None:
        owner = DelegationTasks(directory=tmp_path)
        ran: list[str] = []

        async def run(record: DelegationTask) -> str:
            ran.append(record.id)  # pragma: no cover
            return 'done'  # pragma: no cover

        accepting: list[asyncio.Task[str]] = []

        async def lingering(record: DelegationTask) -> str:
            try:
                await asyncio.Event().wait()
            finally:
                # Its cleanup outlasts the other delegation's save, so closing waits on it meanwhile.
                with anyio.CancelScope(shield=True):
                    await asyncio.wait(accepting)
            raise AssertionError('unreachable')  # pragma: no cover

        async with owner.opened():
            await owner.delegate(
                agent_name='worker',
                prompt='p',
                conversation_id='root',
                model=None,
                background=True,
                resume=None,
                run=lingering,
            )
            records = len(owner.records)
            started = asyncio.create_task(
                owner.delegate(
                    agent_name='worker',
                    prompt='p',
                    conversation_id='root',
                    model=None,
                    background=True,
                    resume=None,
                    run=run,
                )
            )
            accepting.append(started)
            while len(owner.records) == records:
                await asyncio.sleep(0)
        assert 'stopped before execution' in await started
        for _ in range(5):
            await asyncio.sleep(0)
        assert ran == []
        assert {(record.status, record.outcome) for record in owner.records.values()} == {('finished', 'cancelled')}

    async def test_an_older_receiver_leaves_a_newer_one_receiving(self) -> None:
        owner = DelegationTasks()
        older: list[str] = []
        newer: list[str] = []
        steered: list[str] = []

        def enqueue(into: list[str]) -> Callable[[SystemPromptPart], str]:
            def receive(part: SystemPromptPart) -> str:
                into.append(part.content)
                return f'{id(into)}-{len(into)}'

            return receive

        async with owner.opened():

            async def run(record: DelegationTask) -> str:
                return 'done'

            first = owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue(older))
            first.__enter__()
            with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue(newer)):
                first.__exit__(None, None, None)
                await owner.delegate(
                    agent_name='worker',
                    prompt='p',
                    conversation_id='root',
                    model=None,
                    background=True,
                    resume=None,
                    run=run,
                )
                with anyio.fail_after(WAIT):
                    while not newer:
                        await asyncio.sleep(0)
            assert older == [] and 'Automated subagent task report' in newer[0]

            async def steered_run(record: DelegationTask) -> str:
                with owner.receiving(
                    conversation_id='root', parent_id=record.id, enqueue=enqueue([]), steer=steered.append
                ):
                    inner = owner.receiving(
                        conversation_id='root', parent_id=record.id, enqueue=enqueue([]), steer=lambda text: None
                    )
                    inner.__enter__()
                    inner.__exit__(None, None, None)
                    assert owner.steer(record.id, 'go on')
                return 'done'

            receipt = await owner.delegate(
                agent_name='worker',
                prompt='p',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=steered_run,
            )
            assert 'done' in receipt
            assert steered == ['go on']

    async def test_a_report_queued_on_an_ended_receiver_goes_to_the_one_still_receiving(self) -> None:
        owner = DelegationTasks()
        older: list[str] = []
        newer: list[str] = []

        def enqueue(into: list[str]) -> Callable[[SystemPromptPart], str]:
            def receive(part: SystemPromptPart) -> str:
                into.append(part.content)
                return f'{id(into)}-{len(into)}'

            return receive

        async def run(record: DelegationTask) -> str:
            return 'done'

        async with owner.opened():
            with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue(newer)):
                with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue(older)):
                    await owner.delegate(
                        agent_name='worker',
                        prompt='p',
                        conversation_id='root',
                        model=None,
                        background=True,
                        resume=None,
                        run=run,
                    )
                    with anyio.fail_after(WAIT):
                        while not older:
                            await asyncio.sleep(0)
                assert len(older) == 1 and len(newer) == 1, 'the ended run never delivered it'
