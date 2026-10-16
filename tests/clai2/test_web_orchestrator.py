"""The web UI's orchestrator chat over HTTP and the WebSocket: its own agent, conversation, questions, and file."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote

import anyio
import pytest
from inline_snapshot import snapshot
from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_clai2.builtin_plugins.web import orchestrator as orchestrator_module
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    ItemAdded,
    Notice,
    OrchestratorEvent,
    OrchestratorMessage,
    OrchestratorModel,
    OrchestratorState,
    QuestionAsked,
    ServerMessage,
    Snapshot,
    ToolDone,
    TurnEnded,
    TurnStarted,
)
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins import CommandEnd
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
    """A web UI whose session's model, which the orchestrator uses, is `orchestrating`."""
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
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        assert opened(socket) == OrchestratorState(enabled=True, running=False, items=[], questions=[], model=None)
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
    again = Orchestrator(host=web_.plugin.host, publish=lambda message: None)
    assert [item.kind for item in web_.portal.call(again.state).items] == ['user', 'tool', 'tool', 'assistant']
    assert web_.get('/api/state').json()['items'] == []
    # Another CLAI starting a new chat starts one here too: the file is read again between turns.
    web_.portal.call(again.clear)
    assert web_.get('/api/orchestrator/state').json()['items'] == []


def test_a_conversation_it_cannot_save(make_web: MakeWeb) -> None:
    """The turn reports the failed save and leaves no temporary file; the chat keeps it until a save succeeds."""
    web_ = orchestrated(make_web)
    path = web_.plugin.remote.orchestrator.path
    path.mkdir(parents=True)  # A folder where the file goes: it reads as no conversation, and cannot be replaced.
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
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
    assert len(web_.portal.call(Orchestrator(host=web_.plugin.host, publish=lambda message: None).state).items) == 4


def test_stopping_failing_and_starting_over(make_web: MakeWeb, tmp_path: Path) -> None:
    web_ = orchestrated(make_web)
    orchestrator = web_.plugin.remote.orchestrator
    assert web_.post('/api/orchestrator/interrupt').json() == {'interrupted': False}
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
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
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        running = web_.run()
        web_.post('/api/input', {'text': 'hi'})
        receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    assert 'ask_user_question' in offered
    assert not [name for name in offered if name.startswith(('workspace_', 'session_'))]


def test_a_conversation_it_cannot_read_or_without_a_model(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(deps_type=type(None)), extra=[WORKSPACES])
    orchestrator = web_.plugin.remote.orchestrator
    orchestrator.path.parent.mkdir(parents=True)
    orchestrator.path.write_text('not json')
    assert web_.portal.call(orchestrator.load).messages == []
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        opened(socket)
        assert web_.post('/api/orchestrator/input', {'text': 'Hi'}).status_code == 204
        ended = orchestrator_until(socket, TurnEnded)[1]
    assert ended == TurnEnded(outcome='failed', error='Choose a model for this session first.')


def test_a_turn_another_clai_overtook_is_not_kept(make_web: MakeWeb) -> None:
    """A turn whose file another CLAI changed meanwhile fails rather than writing over that CLAI's chat."""
    web_ = orchestrated(make_web)
    other = Orchestrator(host=web_.plugin.host, publish=lambda message: None)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
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


async def picked(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Picked.'


PICKED = FunctionModel(stream_function=picked, model_name='picked')


def picking(web_: Web) -> None:
    """Let the session's resolver build `other:model`, as it builds a saved model, chain, or profile."""
    web_.shell.session.resolve_model = lambda name: PICKED if name == 'other:model' else name


def last_answer(web_: Web) -> str:
    return web_.get('/api/orchestrator/state').json()['items'][-1]['text']


def test_the_orchestrator_has_its_own_model(make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch) -> None:
    """It follows the session's model until one is picked for it alone; the pick is saved, and the session keeps its own."""
    web_ = orchestrated(make_web)
    picking(web_)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        assert opened(socket).model is None
        assert web_.post('/api/orchestrator/model', {'name': 'other:model'}).status_code == 204
        assert orchestrator_until(socket, OrchestratorModel)[1] == OrchestratorModel(model='other:model')
        web_.post('/api/orchestrator/input', {'text': 'Say hello'})
        assert orchestrator_until(socket, TurnEnded)[1].outcome == 'completed'
        assert last_answer(web_) == 'Picked.'
        # Following the session again runs the session's model from the next turn.
        assert web_.post('/api/orchestrator/model', {'name': None}).status_code == 204
        assert orchestrator_until(socket, OrchestratorModel)[1] == OrchestratorModel(model=None)
        web_.post('/api/orchestrator/input', {'text': 'Say hello'})
        orchestrator_until(socket, TurnEnded)
        assert last_answer(web_) == 'Hello.'
    assert web_.shell.session.model is None, "the session's model is not the orchestrator's"
    assert web_.post('/api/orchestrator/model', {'name': ''}).status_code == 422
    # A restarted CLAI, or another one, reads the pick from its file beside the chat's.
    web_.post('/api/orchestrator/model', {'name': 'other:model'})
    again = Orchestrator(host=web_.plugin.host, publish=lambda message: None)
    assert web_.portal.call(again.state).model == 'other:model'
    assert web_.get('/api/state').json()['status']['model'] == 'agent default'

    def missing(name: str) -> None:
        raise ValueError(f'Cannot use {name}: its extra is not installed.')

    # Checked as the session's `/model NAME` checks a name.
    monkeypatch.setattr(orchestrator_module, 'check_installed', missing)
    refused = web_.post('/api/orchestrator/model', {'name': 'nope:model'})
    assert (refused.status_code, refused.json()['detail']) == (
        400,
        'Cannot use nope:model: its extra is not installed.',
    )
    settings = web_.plugin.remote.orchestrator.settings_path
    settings.unlink()
    settings.mkdir()
    unsaved = web_.post('/api/orchestrator/model', {'name': None})
    assert unsaved.status_code == 409 and unsaved.json()['detail'].startswith("The orchestrator's model could not be")
    assert web_.portal.call(again.state).model is None, 'a file it cannot read follows the session'


def notices(socket: WebSocketTestSession, count: int = 1) -> tuple[list[ServerMessage], list[Notice]]:
    """The session's messages, and the orchestrator chat's next `count` notices."""
    session: list[ServerMessage] = []
    found: list[Notice] = []
    while len(found) < count:
        message = WIRE.validate_json(socket.receive_text())
        if not isinstance(message, OrchestratorMessage):
            session.append(message)
        elif isinstance(message.message, Notice):
            found.append(message.message)
    return session, found


def test_the_orchestrator_chats_own_commands(make_web: MakeWeb) -> None:
    web_ = orchestrated(make_web)
    picking(web_)
    web_.shell.context.store.add_model(name='saved:model')
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        opened(socket)
        for text in ('/help', '/'):
            web_.post('/api/orchestrator/input', {'text': text})
            [shown] = notices(socket)[1]
            lines = shown.text.splitlines()
            assert (shown.command, shown.level, lines[0]) == (text, 'info', OWN_HELP)
            assert '/workspaces: ' in shown.text and '/plugins: ' in shown.text and '/fork' not in lines[-2]
        web_.post('/api/orchestrator/input', {'text': '/model'})
        assert notices(socket)[1][0].text == (
            "The orchestrator runs on agent default, the session's.\nSaved models: saved:model.\n"
            "/model NAME picks one for it alone; /model session follows the session's again."
        )
        web_.post('/api/orchestrator/input', {'text': '/model other:model'})
        assert notices(socket)[1][0].text == 'The orchestrator runs on other:model from its next turn.'
        web_.post('/api/orchestrator/input', {'text': '/model'})
        assert notices(socket)[1][0].text.startswith('The orchestrator runs on other:model.\n')
        web_.post('/api/orchestrator/input', {'text': '/model default'})
        assert notices(socket)[1][0].text == (
            "The orchestrator follows the session's model, agent default, from its next turn."
        )
        for text, error in [
            ('/model a b', 'Usage: /model [NAME|session]'),
            ('/model add openai:gpt-5', '/model add works in a session, not in the orchestrator chat.'),
            (
                '/set model openai:gpt-5',
                "/set model picks the session's model, in a session; /model NAME picks the orchestrator's.",
            ),
            ('/new now', 'Usage: /new'),
            ('/fork', '/fork works in a session, not in the orchestrator chat. See /help.'),
            ('/resume', '/resume works in a session, not in the orchestrator chat. See /help.'),
            ('/nonsense', 'Unknown command /nonsense. Use /help.'),
            ('/stop', 'The orchestrator is not working.'),
        ]:
            web_.post('/api/orchestrator/input', {'text': text})
            [shown] = notices(socket)[1]
            assert (shown.text, shown.command) == (error, text)
            assert shown.level == ('info' if text == '/stop' else 'error')
        # A path is a prompt, as in the session's composer.
        web_.post('/api/orchestrator/input', {'text': '/tmp/notes say hello'})
        assert orchestrator_until(socket, TurnEnded)[1].outcome == 'completed'
        # Mid-turn, `/stop` stops it and `/new` waits for it.
        web_.post('/api/orchestrator/input', {'text': 'Work until I stop you'})
        orchestrator_until(socket, ItemAdded)
        web_.post('/api/orchestrator/input', {'text': '/new'})
        assert notices(socket)[1][0].text == 'Stop the orchestrator before starting a new chat.'
        web_.post('/api/orchestrator/input', {'text': '/stop'})
        assert orchestrator_until(socket, TurnEnded)[1].outcome == 'cancelled'
        # Bare `clear` is `/clear`, an alias of `/new`, as at the prompt.
        assert web_.post('/api/orchestrator/input', {'text': 'clear'}).status_code == 204
        assert orchestrator_until(socket, OrchestratorState)[1].items == []


OWN_HELP = "/model: Show or pick the orchestrator's model; /model session follows the session's again"


def test_clais_own_commands_answer_in_the_orchestrator_chat(make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch) -> None:
    """`/workspaces` from the orchestrator chat runs at the prompt as from the session's, but answers in its chat."""
    web_ = orchestrated(make_web)
    orchestrator = web_.plugin.remote.orchestrator
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        opened(socket)
        running = web_.run()
        assert web_.post('/api/orchestrator/input', {'text': '/workspaces list'}).status_code == 204
        session, [shown] = notices(socket)
        assert shown == Notice(text=WORKSPACES_NONE, command='/workspaces list')
        assert not [message for message in session if isinstance(message, Notice)], 'not in the session'
        web_.post('/api/orchestrator/input', {'text': '/workspaces nope'})
        assert notices(socket)[1][0].level == 'error'
        # The same command from the session's composer answers there.
        web_.post('/api/input', {'text': '/workspaces list'})
        assert receive_until(socket, Notice)[1] == Notice(text=WORKSPACES_NONE, command='/workspaces list')

        def no_prompt(text: str) -> None:
            raise RuntimeError('This CLAI reads its prompts line by line, as its output is not a terminal.')

        monkeypatch.setattr(orchestrator.host, 'send_input', no_prompt)
        assert web_.post('/api/orchestrator/input', {'text': '/workspaces list'}).status_code == 409
        monkeypatch.undo()
        web_.post('/api/input', {'text': '/workspaces list'})
        assert receive_until(socket, Notice)[1].command == '/workspaces list', 'the refused one is not waited for'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    entered: list[str] = []
    monkeypatch.setattr(orchestrator.host, 'send_input', entered.append)
    web_.post('/api/orchestrator/input', {'text': '/workspaces list'})
    assert entered == ['/workspaces list']
    # One that ends with nothing to say shows nothing, as in the session; it is no longer waited for.
    assert orchestrator.command_ended(CommandEnd(text='/workspaces list'))
    assert not orchestrator.command_ended(CommandEnd(text='/workspaces list'))


WORKSPACES_NONE = 'No workspaces. Make one with /workspaces create KIND [REPO [REF]].'


def test_completion_in_the_orchestrator_chat(make_web: MakeWeb) -> None:
    """Its own commands and CLAI's that act on CLAI itself, never the session's."""
    web_ = orchestrated(make_web)
    web_.shell.context.store.add_model(name='saved:model')

    def texts(text: str) -> list[str]:
        response = web_.get(f'/api/orchestrator/complete?text={quote(text)}')
        return [found['text'] for found in response.json()]

    # The commands that change CLAI rather than the session's conversation, as they declare it.
    assert sorted(command.name for command in web_.shell.commands if not command.conversation) == snapshot(
        [
            'accounts',
            'config',
            'keys',
            'login',
            'plugins',
            'set',
            'settings',
            'spinner',
            'theme',
            'web',
            'workspaces',
        ]
    )
    offered = texts('/')
    assert {'model', 'new', 'clear', 'help', 'stop', 'workspaces', 'plugins', 'set', 'keys', 'login'} <= set(offered)
    assert not {'fork', 'compact', 'resume', 'tasks', 'monty', 'exit'} & set(offered)
    assert web_.get('/api/orchestrator/complete?text=/mo').json()[0] == {
        'text': 'model',
        'start': 2,
        'display': '/model',
        'meta': "Show or pick the orchestrator's model; /model session follows the session's again",
    }
    assert texts('/model ') == ['session', 'saved:model']
    assert texts('/model sa') == ['saved:model']
    assert 'list' in texts('/workspaces ')
    # `/set model` picks the session's model, so neither the key nor its values are offered.
    assert 'run.code_mode' in texts('/set ') and 'model' not in texts('/set ') + texts('/settings mo')
    assert texts('/set model ') == []
    assert texts('/fork ') == texts('/new ') == texts('/ ') == texts('hello') == []
    # The new routes take the token, as every other one does.
    assert web_.client.get('/api/orchestrator/complete?text=/').status_code == 403
    assert web_.client.post('/api/orchestrator/model', json={'name': None}).status_code == 403
