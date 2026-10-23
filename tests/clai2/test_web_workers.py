"""The web UI's Workers: delegated tasks streamed to browsers, their history, steering, and their controls."""

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.subagents import DelegationTask
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
    TurnStarted,
)
from pydantic_clai2.builtin_plugins.web.workers import Workers
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins import TaskEvent, TaskInfo
from pydantic_clai2.runtime.tasks import resume_prompt
from tests.clai2.test_web import TOKEN, WIRE, MakeWeb, Web, make_web, receive_until

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
    continued: list[list[ModelMessage]] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        said, latest = prompts(messages), prompts(messages[-1:])
        if said[0] == 'child':
            if latest == ['and the docs?']:
                continued.append(messages)
                yield 'docs done'
            elif 'focus on tests' in said:
                yield 'steered'
            else:
                asking.set()
                await release.wait()
                yield 'child done'
        elif said[0] == 'blocker':
            if latest == ['try again']:
                yield 'unblocked'
            else:
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
        assert saved['task']['backgroundable'] is False, 'a finished task has no run to background'
        assert [item['text'] for item in saved['items']] == ['child', 'child done', 'focus on tests', 'steered']
        assert web_.post(f'/api/tasks/{task_id}/steer', {'text': 'more'}).json() == {'steered': False}
        assert web_.get('/api/tasks/nope').status_code == 404

        # Prompted once it has finished, it goes on from its saved history on its own: the session's
        # conversation gets no turn, no report, and no new messages.
        conversation = list(web_.shell.session.messages)
        assert web_.post(f'/api/tasks/{task_id}/prompt', {'text': ' '}).status_code == 400
        assert web_.post('/api/tasks/nope/prompt', {'text': 'hello'}).status_code == 404
        gone = finished_task(web_)
        gone.agent_name = 'gone'
        refused = web_.post(f'/api/tasks/{gone.id}/prompt', {'text': 'hello'})
        assert (refused.status_code, refused.json()['detail']) == (
            400,
            "Unknown sub-agent 'gone'. Available sub-agents: self.",
        )
        del web_.shell.tasks.owner.records[gone.id]
        assert web_.post(f'/api/tasks/{task_id}/prompt', {'text': ' and the docs? '}).status_code == 204
        messages, again = until_finished(socket)
        assert not [message for message in messages if isinstance(message, TurnStarted | TurnEnded)]
        restarted = task_changes(messages)[0].task
        assert (restarted.status, restarted.prompt, restarted.background) == ('running', 'and the docs?', False)
        assert (again.id, again.outcome) == (task_id, 'ok')
        [history] = continued
        assert prompts(history) == ['child', 'focus on tests', 'and the docs?'], 'the model sees what it did before'
        live = [
            message.item.text
            for task_items in messages
            if isinstance(task_items, TaskItems) and task_items.id == task_id
            for message in task_items.messages
            if isinstance(message, ItemAdded)
        ]
        assert live[:5] == ['child', 'child done', 'focus on tests', 'steered', 'and the docs?']
        assert web_.shell.session.messages == conversation
        saved = web_.get(f'/api/tasks/{task_id}').json()
        assert [item['text'] for item in saved['items']][-2:] == ['and the docs?', 'docs done']
        owner = web_.shell.tasks.owner
        assert owner.directory is not None
        assert 'docs done' in (owner.directory / f'{task_id}.json').read_text(), 'saved to its own history'

        # For HTTP callers, resume still asks the agent, as `/tasks resume` does; the page prompts instead.
        assert web_.post(f'/api/tasks/{task_id}/resume').status_code == 204
        receive_until(socket, TurnEnded)
        assert prompts(web_.shell.session.messages)[-1] == resume_prompt(owner.records[task_id])

        assert web_.post('/api/input', {'text': 'stop me'}).json() == {'menu': None}
        messages, _ = receive_until(socket, TaskItems)
        blocker = task_changes(messages)[-1].task
        assert blocker.prompt == 'blocker'
        assert blocking.wait(10), 'stopped while it waits on its model'
        assert web_.post(f'/api/tasks/{blocker.id}/stop').status_code == 204
        assert until_finished(socket)[1].outcome == 'cancelled'
        assert web_.post(f'/api/tasks/{blocker.id}/prompt', {'text': 'try again'}).status_code == 204, (
            'prompting lifts a user stop'
        )
        assert until_finished(socket)[1].outcome == 'ok'

        assert web_.post('/api/sessions/new').status_code == 200
        _, fresh = receive_until(socket, Snapshot)
        assert fresh.tasks == [] and fresh.task_items == {}
        assert web_.post('/api/input', {'text': '/exit'}).json() == {'menu': None}


def finished_task(web_: Web) -> DelegationTask:
    """A general-purpose task the session's conversation finished earlier, as one saved before a restart is."""
    record = DelegationTask(
        id=uuid4().hex,
        agent_name='self',
        prompt='child',
        conversation_id=web_.shell.session.summary.id,
        status='finished',
        outcome='ok',
        messages=[ModelRequest(parts=[UserPromptPart('child')]), ModelResponse(parts=[TextPart('child done')])],
    )
    web_.shell.tasks.owner.records[record.id] = record
    return record


def test_a_prompted_task_runs_beside_a_turn_with_code_mode_on(make_web: MakeWeb) -> None:
    """Code mode folds the delegate tool into `run_code`, except in the run that prompts a task."""
    holding, release, answering = threading.Event(), asyncio.Event(), asyncio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        said = prompts(messages)
        if said[0] == 'child':
            tools = {tool.name for tool in info.function_tools}
            assert 'run_code' in tools and 'delegate_task' not in tools, 'the task itself runs in code mode'
            await answering.wait()
            yield 'docs done'
        else:
            holding.set()
            await release.wait()
            yield 'parent done'

    web_ = make_web(create_stock_agent(FunctionModel(stream_function=stream)), extra=[CODER])
    web_.shell.context.set_setting(['run.code_mode', 'true'])
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers=ORIGIN) as socket:
        web_.run()
        assert web_.post('/api/input', {'text': 'hold'}).json() == {'menu': None}
        assert holding.wait(10)
        record = finished_task(web_)
        with ThreadPoolExecutor(1) as pool:
            prompted = pool.submit(web_.post, f'/api/tasks/{record.id}/prompt', {'text': 'and the docs?'})
            try:
                assert prompted.result(10).status_code == 204, 'answered once the task runs, not once it finishes'
            finally:
                web_.portal.call(answering.set)
        messages, finished = until_finished(socket)
        assert finished.outcome == 'ok', record.output
        assert not [message for message in messages if isinstance(message, TurnEnded)], 'the turn still runs'
        assert prompts(record.messages) == ['child', 'and the docs?']
        web_.portal.call(release.set)
        receive_until(socket, TurnEnded)
        assert prompts(web_.shell.session.messages) == ['hold']
        assert web_.post('/api/input', {'text': '/exit'}).json() == {'menu': None}


def test_a_task_takes_no_prompt_once_its_agent_delegates_no_more(make_web: MakeWeb) -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hi'

    web_ = make_web(create_stock_agent(FunctionModel(stream_function=stream)))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers=ORIGIN) as socket:
        web_.run()
        # A turn ending means the prompt is open, and with it the session's tasks.
        assert web_.post('/api/input', {'text': 'hello'}).json() == {'menu': None}
        receive_until(socket, TurnEnded)
        conversation = list(web_.shell.session.messages)
        record = finished_task(web_)
        record.user_stopped = True
        response = web_.post(f'/api/tasks/{record.id}/prompt', {'text': 'and the docs?'})
        assert response.status_code == 400
        assert response.json()['detail'] == (
            'This agent delegates no tasks now, so its tasks take no prompts. Turn sub-agents on first.'
        )
        assert record.user_stopped and record.status == 'finished', 'left as it was'
        assert prompts(record.messages) == ['child']
        assert web_.shell.session.messages == conversation
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
        backgroundable=True,
        resumable=False,
        activity='starting',
        usage=RunUsage(),
    )
    workers.observe(TaskEvent(task=task))
    workers.observe(TaskEvent(task=task))
    assert [type(message) for message in published] == [TaskChanged, TaskItems], 'an unchanged row is not sent again'
    assert isinstance(changed := published[0], TaskChanged) and changed.task.backgroundable
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
        backgroundable=True,
        resumable=True,
        activity='starting',
        usage=RunUsage(),
    )
    workers.observe(TaskEvent(task=task))
    assert [item.text for item in workers.items()['t']] == ['look', 'first run', 'look again'], (
        'resumed: and its prompt'
    )
    assert all(item.id.startswith('task1-') for item in workers.items()['t'])

    workers.reset()
    workers.observe(TaskEvent(task=task, event=PartStartEvent(index=0, part=TextPart('more'))))
    assert [item.text for item in workers.items()['t']] == ['look', 'first run', 'more'], (
        'joined midway: its own request'
    )
