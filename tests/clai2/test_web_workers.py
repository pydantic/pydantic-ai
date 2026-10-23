"""The web UI's Workers: delegated tasks streamed to browsers, their history, steering, and their controls."""

import asyncio
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.subagents import DelegationStartEvent, TaskQuestionEvent
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.builtin_plugins.web.protocol import (
    ItemAdded,
    ServerMessage,
    Snapshot,
    TaskChanged,
    TaskEntry,
    TaskItems,
    TextDelta,
    TurnEnded,
)
from pydantic_clai2.builtin_plugins.web.workers import Workers
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins import TaskEvent, TaskInfo
from tests.clai2.test_web import TOKEN, WIRE, MakeWeb, make_web, receive_until

__all__ = ['make_web']

CODER = PluginSettings(
    id='coder', factory='pydantic_clai2.builtin_plugins.coder', settings={'repo_context': False, 'sub_agents': True}
)
ORIGIN = {'origin': 'http://testserver'}


def prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def delegate(task: str, *, background: bool = False) -> DeltaToolCalls:
    args = f'{{"agent_name":"self","task":"{task}","background":{"true" if background else "false"}}}'
    return {0: DeltaToolCall(name='delegate_task', json_args=args)}


def task_changes(messages: list[ServerMessage]) -> list[TaskChanged]:
    return [message for message in messages if isinstance(message, TaskChanged)]


def until_finished(socket: WebSocketTestSession) -> tuple[list[ServerMessage], TaskEntry]:
    """Messages up to a task's row saying it finished, and that row."""
    received: list[ServerMessage] = []
    while True:
        messages, changed = receive_until(socket, TaskChanged)
        received += messages
        if changed.task.status == 'finished':
            return received, changed.task


def test_workers_stream_steer_and_control(make_web: MakeWeb) -> None:
    release, asking, blocking = asyncio.Event(), threading.Event(), threading.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        said, latest = prompts(messages), prompts(messages[-1:])
        if said[0] == 'child':
            if 'focus on tests' in said:
                yield 'steered'
            else:
                asking.set()
                await release.wait()
                yield 'child done'
        elif said[0] == 'blocker':
            blocking.set()
            await asyncio.Event().wait()
            yield 'unreachable'  # pragma: no cover
        elif latest == ['go']:
            yield delegate('child')
        elif latest == ['stop me']:
            yield delegate('blocker', background=True)
        else:
            yield 'parent done'

    web_ = make_web(create_stock_agent(FunctionModel(stream_function=stream)), extra=[CODER])
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers=ORIGIN) as socket:
        snapshot = WIRE.validate_json(socket.receive_text())
        assert isinstance(snapshot, Snapshot) and snapshot.tasks == [] and snapshot.task_items == {}
        assert web_.get('/api/tasks/nope').status_code == 404, 'no task owner runs before the prompt opens'
        web_.run()
        assert web_.post('/api/input', {'text': 'go'}).json() == {'menu': None}
        messages, items = receive_until(socket, TaskItems)
        [started] = task_changes(messages)
        task_id = started.task.id
        assert (started.task.agent, started.task.prompt, started.task.status) == ('general-purpose', 'child', 'running')
        [prompt] = items.messages
        assert isinstance(prompt, ItemAdded) and prompt.item.text == 'child' and prompt.item.id.startswith('task1-')

        # A browser that connects now sees the running task and its transcript so far.
        with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers=ORIGIN) as late:
            joined = WIRE.validate_json(late.receive_text())
            assert isinstance(joined, Snapshot)
            assert [task.id for task in joined.tasks] == [task_id]
            assert [item.text for item in joined.task_items[task_id]] == ['child']

        assert web_.post(f'/api/tasks/{task_id}/steer', {'text': '  '}).status_code == 400
        assert asking.wait(10), 'the child asks its model, which reads the message on its next request'
        assert web_.post(f'/api/tasks/{task_id}/steer', {'text': ' focus on tests '}).json() == {'steered': True}
        _, echoed = receive_until(socket, TaskItems)
        [steer] = echoed.messages
        assert isinstance(steer, ItemAdded) and (steer.item.kind, steer.item.text) == ('user', 'focus on tests')

        # Backgrounded, the parent goes on without it.
        assert web_.post(f'/api/tasks/{task_id}/background').status_code == 204
        messages, _ = receive_until(socket, TurnEnded)
        assert task_changes(messages)[-1].task.background
        web_.portal.call(release.set)
        messages, finished = until_finished(socket)
        assert (finished.outcome, finished.usage.requests) == ('ok', 2)
        streamed = [
            message
            for task_items in messages
            if isinstance(task_items, TaskItems)
            for message in task_items.messages
            if isinstance(message, TextDelta | ItemAdded)
        ]
        assert streamed and all(
            message.id.startswith('task1-') for message in streamed if isinstance(message, TextDelta)
        )

        saved = web_.get(f'/api/tasks/{task_id}').json()
        assert saved['task']['status'] == 'finished'
        assert [item['text'] for item in saved['items']] == ['child', 'child done', 'focus on tests', 'steered']
        assert web_.post(f'/api/tasks/{task_id}/steer', {'text': 'more'}).json() == {'steered': False}
        assert web_.get('/api/tasks/nope').status_code == 404

        # Resume asks the agent, as `/tasks resume` does.
        assert web_.post(f'/api/tasks/{task_id}/resume').status_code == 204
        receive_until(socket, TurnEnded)

        assert web_.post('/api/input', {'text': 'stop me'}).json() == {'menu': None}
        messages, _ = receive_until(socket, TaskItems)
        blocker = task_changes(messages)[-1].task
        assert blocker.prompt == 'blocker'
        assert blocking.wait(10), 'stopped while it waits on its model'
        assert web_.post(f'/api/tasks/{blocker.id}/stop').status_code == 204
        assert until_finished(socket)[1].outcome == 'cancelled'
        assert web_.post(f'/api/tasks/{blocker.id}/resume').status_code == 204, 'a user stop is lifted by asking'

        assert web_.post('/api/sessions/new').status_code == 200
        _, fresh = receive_until(socket, Snapshot)
        assert fresh.tasks == [] and fresh.task_items == {}
        assert web_.post('/api/input', {'text': '/exit'}).json() == {'menu': None}


def test_without_a_task_owner(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    assert web_.plugin.service.default().tasks() == []
    assert web_.get('/api/tasks/any').json() == {'detail': 'This CLAI runs no delegated tasks.'}
    assert web_.post('/api/tasks/any/stop').status_code == 404
    assert web_.post('/api/tasks/any/steer', {'text': 'hello'}).status_code == 404


def test_workers_echo_steering_only_for_a_live_task() -> None:
    published: list[ServerMessage] = []
    workers = Workers(publish=published.append, history=lambda task_id: [])
    workers.steered('gone', 'hello')
    assert published == []
    now = datetime.now(UTC)
    task = TaskInfo(
        id='t',
        parent_id=None,
        agent='Explore',
        model=None,
        prompt='look',
        status='running',
        outcome=None,
        started_at=now,
        finished_at=None,
        background=False,
        resumable=False,
        activity='starting',
        usage=RunUsage(),
    )
    start = DelegationStartEvent(agent_name='Explore', task='look', truncated=False, inherits_tools=False, model=None)
    workers.observe(TaskEvent(task=task, event=start))
    workers.observe(TaskEvent(task=task))
    assert [type(message) for message in published] == [TaskChanged, TaskItems], 'an unchanged row is not sent again'
    workers.steered('t', 'hello')
    assert [item.text for item in workers.items()['t']] == ['look', 'hello']


def test_a_resumed_or_joined_run_starts_from_the_tasks_history() -> None:
    saved: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('look')]),
        ModelResponse(parts=[TextPart('first run')]),
    ]
    published: list[ServerMessage] = []
    workers = Workers(publish=published.append, history=lambda task_id: saved)
    task = TaskInfo(
        id='t',
        parent_id=None,
        agent='general-purpose',
        model=None,
        prompt='look again',
        status='running',
        outcome=None,
        started_at=datetime.now(UTC),
        finished_at=None,
        background=False,
        resumable=True,
        activity='starting',
        usage=RunUsage(),
    )
    workers.observe(TaskEvent(task=task))
    assert workers.items() == {}, 'a change of the task alone, such as a question it waits on, starts no transcript'
    question = TaskQuestionEvent(task_id='t', agent_name='self', question_id='q', question='Which?')
    workers.observe(TaskEvent(task=task, event=question))
    assert workers.items() == {}, 'nor does its question, a task-control message'
    start = DelegationStartEvent(agent_name='self', task='look again', truncated=False, inherits_tools=True, model=None)
    start.task_id = 't'
    workers.observe(TaskEvent(task=task, event=start))
    assert [item.text for item in workers.items()['t']] == ['look', 'first run', 'look again'], (
        'resumed: and its prompt'
    )
    assert all(item.id.startswith('task1-') for item in workers.items()['t'])

    workers.reset()
    workers.observe(TaskEvent(task=task, event=PartStartEvent(index=0, part=TextPart('more'))))
    assert [item.text for item in workers.items()['t']] == ['look', 'first run', 'more'], (
        'joined midway: its own request'
    )
