"""The web UI's orchestrator chat over HTTP and the WebSocket: its own agent, conversation, questions, and file."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar

import anyio
from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    ItemAdded,
    OrchestratorEvent,
    OrchestratorMessage,
    OrchestratorState,
    QuestionAsked,
    Snapshot,
    ToolDone,
    TurnEnded,
    TurnStarted,
)
from pydantic_clai2.config import PluginSettings
from tests.clai2.test_web import TOKEN, WIRE, MakeWeb, Web, make_web, receive_until

__all__ = ('make_web',)

WORKSPACES = PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces')
QUESTION = {'questions': [{'question': 'Make one?', 'header': 'Make', 'options': [{'label': 'Yes'}, {'label': 'No'}]}]}
EventT = TypeVar('EventT')


def prompt_of(messages: list[ModelMessage]) -> str:
    return next(
        str(part.content)
        for message in reversed(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


async def orchestrating(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """The orchestrator's model: lists the workspaces and asks, works until stopped, or fails, as the prompt says."""
    prompt = prompt_of(messages)
    if 'stop' in prompt:
        yield 'Working on it'
        await anyio.sleep_forever()
    if 'fail' in prompt:
        raise RuntimeError('the model broke')
    if 'hello' in prompt:
        yield 'Hello.'
        return
    step = sum(isinstance(message, ModelResponse) for message in messages)
    if step == 0:
        yield {0: DeltaToolCall(name='workspace_list', json_args='{}')}
    elif step == 1:
        yield {0: DeltaToolCall(name='ask_user_question', json_args=json.dumps(QUESTION))}
    else:
        yield 'All set.'


def orchestrated(make: MakeWeb, *, workspaces: bool = True) -> Web:
    """A web UI whose agent model, which the orchestrator uses while no default is saved, is `orchestrating`."""
    agent = Agent(FunctionModel(stream_function=orchestrating), deps_type=type(None))
    return make(agent, extra=[WORKSPACES if workspaces else WORKSPACES.model_copy(update={'enabled': False})])


def opened(socket: WebSocketTestSession) -> OrchestratorState:
    """The orchestrator chat's state, sent after the session's snapshot when the WebSocket connects."""
    receive_until(socket, Snapshot)
    return orchestrator_until(socket, OrchestratorState)[1]


def orchestrator_until(socket: WebSocketTestSession, kind: type[EventT]) -> tuple[list[OrchestratorEvent], EventT]:
    """The orchestrator chat's messages up to and including the first `kind`; the session's are skipped."""
    seen: list[OrchestratorEvent] = []
    while True:
        message = WIRE.validate_json(socket.receive_text())
        event = message.message if isinstance(message, OrchestratorMessage) else None
        seen += [] if event is None else [event]
        if isinstance(event, kind):
            return seen, event


def test_the_orchestrator_chat_streams_asks_and_is_saved(make_web: MakeWeb, tmp_path: Path) -> None:
    """A message runs the orchestrator's own agent; its tool cards stream tagged for its chat, and it asks there."""
    web_ = orchestrated(make_web)
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        assert opened(socket) == OrchestratorState(
            enabled=True, running=False, model='function::orchestrating', items=[], questions=[]
        )
        assert web_.post('/api/orchestrator/input', {'text': '  '}).status_code == 400
        assert web_.post('/api/orchestrator/input', {'text': 'What is there?'}).status_code == 204
        seen, asked = orchestrator_until(socket, QuestionAsked)
        [started] = [event for event in seen if isinstance(event, TurnStarted)]
        assert started.item is not None and started.item.text == 'What is there?'
        # With it, the history the turn starts from, as another CLAI may have changed it since the page loaded.
        [starting] = [event for event in seen if isinstance(event, OrchestratorState)]
        assert starting.running and starting.items == [started.item]
        tools = [event.item for event in seen if isinstance(event, ItemAdded) and event.item.kind == 'tool']
        assert [tool.name for tool in tools] == ['workspace_list', 'ask_user_question']
        [listed] = [event for event in seen if isinstance(event, ToolDone)]
        assert listed.text == 'No workspaces yet.\nKinds: local-worktree.'
        request = asked.questions.id
        assert web_.get('/api/orchestrator/state').json()['questions'][0]['id'] == request
        # Only the orchestrator chat answers its questions: the session's route does not know them.
        assert web_.post(f'/api/questions/{request}', {'answers': None}).status_code == 404
        assert web_.post('/api/orchestrator/answer', {'id': 'nope', 'answers': None}).status_code == 404
        wrong = {'id': request, 'answers': [{'header': 'Make', 'selected': ['Maybe']}]}
        assert web_.post('/api/orchestrator/answer', wrong).status_code == 400
        right = {'id': request, 'answers': [{'header': 'Make', 'selected': ['Yes']}]}
        assert web_.post('/api/orchestrator/answer', right).status_code == 204
        _, end = orchestrator_until(socket, TurnEnded)
        assert end.outcome == 'completed' and end.usage is not None and end.usage.requests == 3
    state = web_.get('/api/orchestrator/state').json()
    assert (state['running'], [item['kind'] for item in state['items']]) == (
        False,
        ['user', 'tool', 'tool', 'assistant'],
    )
    assert state['items'][-1]['text'] == 'All set.'
    saved = tmp_path / 'state' / 'pydantic-clai2' / 'orchestrator.json'
    assert saved.is_file() and saved.stat().st_mode & 0o777 == 0o600
    # A restarted CLAI reads the same conversation; the session's own stays empty.
    again = Orchestrator(service=web_.plugin.service)
    assert [item.kind for item in web_.portal.call(again.state).items] == ['user', 'tool', 'tool', 'assistant']
    assert web_.get('/api/state').json()['items'] == []
    # Another CLAI starting a new chat starts one here too: the file is read again between turns.
    web_.portal.call(again.clear)
    assert web_.get('/api/orchestrator/state').json()['items'] == []


def test_a_conversation_it_cannot_save(make_web: MakeWeb) -> None:
    """The turn reports the failed save and leaves no temporary file; the chat keeps it until a save succeeds."""
    web_ = orchestrated(make_web)
    path = web_.plugin.service.orchestrator.path
    path.mkdir(parents=True)  # A folder where the file goes: it reads as no conversation, and cannot be replaced.
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        opened(socket)
        web_.post('/api/orchestrator/input', {'text': 'Say hello'})
        ended = orchestrator_until(socket, TurnEnded)[1]
        assert ended.outcome == 'failed' and ended.error is not None
        assert ended.error.startswith('The orchestrator chat could not be saved: ')
        assert list(path.parent.iterdir()) == [path]
        kept = web_.get('/api/orchestrator/state').json()['items']
        assert [item['kind'] for item in kept] == ['user', 'assistant']
        path.rmdir()
        web_.post('/api/orchestrator/input', {'text': 'Say hello again'})
        assert orchestrator_until(socket, TurnEnded)[1].outcome == 'completed'
    assert len(web_.portal.call(Orchestrator(service=web_.plugin.service).state).items) == 4


def test_stopping_failing_and_starting_over(make_web: MakeWeb, tmp_path: Path) -> None:
    web_ = orchestrated(make_web)
    orchestrator = web_.plugin.service.orchestrator
    assert web_.post('/api/orchestrator/interrupt').json() == {'interrupted': False}
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        opened(socket)
        assert web_.post('/api/orchestrator/input', {'text': 'Work until I stop you'}).status_code == 204
        orchestrator_until(socket, ItemAdded)
        busy = web_.post('/api/orchestrator/input', {'text': 'And this?'})
        assert (busy.status_code, busy.json()['detail']) == (
            409,
            'The orchestrator is still working. Wait for it, or stop it.',
        )
        assert web_.post('/api/orchestrator/clear').status_code == 409
        running = web_.get('/api/orchestrator/state').json()
        assert running['running'] and [item['kind'] for item in running['items']] == ['user', 'assistant']

        async def stop_twice() -> list[bool]:
            # Both on one tick of the loop, before the turn could end: the second stop does not cut its ending short.
            return [orchestrator.interrupt(), orchestrator.interrupt()]

        assert web_.portal.call(stop_twice) == [True, True]
        assert orchestrator_until(socket, TurnEnded)[1] == TurnEnded(outcome='cancelled')
        stopped = web_.get('/api/orchestrator/state').json()['items']
        assert stopped[0]['text'] == 'Work until I stop you', 'what ran before the stop is kept'

        assert web_.post('/api/orchestrator/input', {'text': 'Now fail'}).status_code == 204
        assert orchestrator_until(socket, TurnEnded)[1] == TurnEnded(outcome='failed', error='the model broke')

        assert web_.post('/api/orchestrator/clear').status_code == 204
        assert orchestrator_until(socket, OrchestratorState)[1].items == []
    assert not (tmp_path / 'state' / 'pydantic-clai2' / 'orchestrator.json').exists()

    # `/web stop` stops a running turn: no page is left to watch or answer it.
    web_.post('/api/orchestrator/input', {'text': 'Work until I stop you'})
    web_.portal.call(web_.plugin.stop)
    assert not orchestrator.running


def test_the_orchestrator_needs_workspaces(make_web: MakeWeb) -> None:
    """While `workspaces` is off the chat says so, and turning it on from the page lets it run."""
    web_ = orchestrated(make_web, workspaces=False)
    assert web_.get('/api/orchestrator/state').json()['enabled'] is False
    refused = web_.post('/api/orchestrator/input', {'text': 'Hi'})
    assert (refused.status_code, refused.json()['detail']) == (409, 'Turn the `workspaces` plugin on first.')
    assert web_.post('/api/plugins/workspaces/enable').status_code == 204
    assert web_.get('/api/orchestrator/state').json()['enabled'] is True


def test_the_session_agent_gets_no_workspace_tools(make_web: MakeWeb) -> None:
    """With `web` and `workspaces` on, the session's own agent has none of the orchestrator's tools."""
    offered: list[str] = []

    async def session_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        offered.extend(tool.name for tool in info.function_tools)
        yield 'Hello.'

    web_ = make_web(Agent(FunctionModel(stream_function=session_model), deps_type=type(None)), extra=[WORKSPACES])
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        running = web_.run()
        web_.post('/api/input', {'text': 'hi'})
        receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    assert 'ask_user_question' in offered
    assert not [name for name in offered if name.startswith(('workspace_', 'session_', 'sessions_'))]


def test_its_state_names_the_saved_default_model_not_the_sessions(make_web: MakeWeb) -> None:
    """The pane shows the model the orchestrator runs on: the saved default, not the session's own."""
    web_ = orchestrated(make_web)
    web_.shell.process.store.set('model', 'test')
    assert web_.get('/api/state').json()['status']['model'] == 'agent default'
    assert web_.get('/api/orchestrator/state').json()['model'] == 'test'
    # Named, not built: a saved chain since deleted still shows, and the page still connects.
    web_.shell.process.store.set('model', 'chain:gone')
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        assert opened(socket).model == 'chain:gone'


def test_a_conversation_it_cannot_read_or_without_a_model(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(deps_type=type(None)), extra=[WORKSPACES])
    orchestrator = web_.plugin.service.orchestrator
    orchestrator.path.parent.mkdir(parents=True)
    orchestrator.path.write_text('not json')
    assert web_.portal.call(orchestrator.load).messages == []
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        assert opened(socket).model is None
        assert web_.post('/api/orchestrator/input', {'text': 'Hi'}).status_code == 204
        ended = orchestrator_until(socket, TurnEnded)[1]
    assert ended == TurnEnded(outcome='failed', error='Choose a model with /model first.')


def test_a_turn_another_clai_overtook_is_not_kept(make_web: MakeWeb) -> None:
    """A turn whose file another CLAI changed meanwhile fails rather than writing over that CLAI's chat."""
    web_ = orchestrated(make_web)
    other = Orchestrator(service=web_.plugin.service)
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        opened(socket)
        web_.post('/api/orchestrator/input', {'text': 'Say hello'})
        assert orchestrator_until(socket, TurnEnded)[1].outcome == 'completed'
        web_.post('/api/orchestrator/input', {'text': 'Work until I stop you'})
        orchestrator_until(socket, ItemAdded)
        web_.portal.call(other.clear)
        web_.post('/api/orchestrator/interrupt')
        ended = orchestrator_until(socket, TurnEnded)[1]
    assert ended == TurnEnded(
        outcome='failed', error='Another CLAI changed the orchestrator chat during this turn, so it was not kept.'
    )
    assert web_.get('/api/orchestrator/state').json()['items'] == [], "the other CLAI's new chat stands"
