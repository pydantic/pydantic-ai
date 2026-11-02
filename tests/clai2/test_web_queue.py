"""Steering the running turn from the browser, and the queue of input waiting for it, kept in step with the prompt."""

from io import StringIO
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    ItemAdded,
    Notice,
    QueueChanged,
    QueuedItem,
    Snapshot,
    TurnEnded,
    TurnStarted,
    image_id,
)
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, QueuedInput
from pydantic_clai2.plugins.process_plugins import LiveRuntime
from pydantic_clai2.runtime.image_codec import decode_images
from pydantic_clai2.runtime.session_runtime import Inbox, create_shell
from tests.clai2.test_live_prompt import editor
from tests.clai2.test_web import HEADER, TOKEN, WIRE, MakeWeb, encoded, make_web, png, receive_until

__all__ = ['make_web']

ORIGIN = {'origin': 'http://testserver'}


NOTE = 'A context file the agent found.'


def waiting_agent() -> tuple[Agent[None, str], anyio.Event, anyio.Event]:
    """An agent whose first request calls `wait`, which holds the turn until released, then answers.

    `wait` enqueues `NOTE` for the model, as a capability such as `RepoContext` does, which is not the user's steering.
    """
    started, release = anyio.Event(), anyio.Event()
    tools = Capability[None]()

    @tools.tool
    async def wait(ctx: RunContext[None]) -> str:
        """Wait to be released."""
        ctx.enqueue(NOTE)
        started.set()
        await release.wait()
        return 'released'

    agent = Agent(TestModel(call_tools=['wait'], custom_output_text='done'), deps_type=type(None), capabilities=[tools])
    return agent, started, release


def queue_texts(queue: list[QueuedItem]) -> list[tuple[str, str]]:
    return [(item.text, item.kind) for item in queue]


def test_steer_queue_edit_and_remove_during_a_turn(make_web: MakeWeb) -> None:
    """While a turn runs, the browser steers it, and lists, edits, steers, and removes what queues behind it."""
    agent, started, release = waiting_agent()
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        snapshot = WIRE.validate_json(socket.receive_text())
        assert isinstance(snapshot, Snapshot) and snapshot.queue == []
        running = web_.run()
        web_.post('/api/input', {'text': 'go'})
        web_.portal.call(started.wait)

        for text in ('first', 'second', '/help'):
            assert web_.post('/api/input', {'text': text}).json() == {'menu': None}
        _, changed = receive_until(socket, QueueChanged)
        while len(changed.queue) < 3:
            _, changed = receive_until(socket, QueueChanged)
        assert queue_texts(changed.queue) == [('first', 'prompt'), ('second', 'prompt'), ('/help', 'command')]
        first, second, command = changed.queue
        assert [item['id'] for item in web_.get('/api/state').json()['queue']] == [first.id, second.id, command.id]
        assert web_.plugin.service.live_message().sessions[0].state == 'queued'

        # Edited in place, keeping its turn in the queue.
        edit = web_.client.put(f'/api/queue/{first.id}', json={'text': ' first, edited '}, headers={HEADER: TOKEN})
        assert edit.status_code == 204
        _, changed = receive_until(socket, QueueChanged)
        assert queue_texts(changed.queue)[0] == ('first, edited', 'prompt')
        assert changed.queue[0].id == first.id
        for text, detail in [
            ('  ', 'Type a prompt or a /command.'),
            ('!ls', 'Shell lines (!) run only in the terminal.'),
            ('/set', '/set opens its menu in the page; send it instead of queueing it.'),
        ]:
            refused = web_.client.put(f'/api/queue/{first.id}', json={'text': text}, headers={HEADER: TOKEN})
            assert (refused.status_code, refused.json()) == (400, {'detail': detail})
        missing = web_.client.put('/api/queue/nope', json={'text': 'x'}, headers={HEADER: TOKEN})
        assert (missing.status_code, missing.json()['detail']) == (
            404,
            'That message is no longer queued: it was sent or removed.',
        )

        assert web_.client.delete(f'/api/queue/{second.id}', headers={HEADER: TOKEN}).status_code == 204
        _, changed = receive_until(socket, QueueChanged)
        assert [item.id for item in changed.queue] == [first.id, command.id]
        assert web_.client.delete(f'/api/queue/{second.id}', headers={HEADER: TOKEN}).status_code == 404

        # Alt+Enter: the running turn takes a prompt now; a command never steers, and is entered as Enter would.
        assert web_.post('/api/steer', {'text': ' focus on tests '}).json() == {'steered': True, 'menu': None}
        assert web_.post('/api/steer', {'text': '/set'}).json() == {'steered': False, 'menu': 'set'}
        # Bare `clear` is `/clear`, as at the terminal's prompt, so it queues instead of steering.
        assert web_.post('/api/steer', {'text': 'clear'}).json() == {'steered': False, 'menu': None}
        _, changed = receive_until(socket, QueueChanged)
        cleared = changed.queue[-1]
        assert (cleared.text, cleared.kind) == ('/clear', 'command')
        assert web_.client.delete(f'/api/queue/{cleared.id}', headers={HEADER: TOKEN}).status_code == 204
        receive_until(socket, QueueChanged)
        assert web_.post(f'/api/queue/{command.id}/steer').json() == {'steered': False}
        assert web_.post(f'/api/queue/{first.id}/steer').json() == {'steered': True}
        _, changed = receive_until(socket, QueueChanged)
        assert [item.id for item in changed.queue] == [command.id], 'a steered prompt leaves the queue'
        assert web_.post('/api/queue/nope/steer').status_code == 404

        web_.portal.call(release.set)
        messages, end = receive_until(socket, TurnEnded)
        assert end.outcome == 'completed'
        steered = [
            message.item for message in messages if isinstance(message, ItemAdded) and message.item.kind == 'user'
        ]
        # Only the user's steering shows as theirs, not the note the tool enqueued.
        assert [(item.id, item.text) for item in steered] == [('steer1', 'focus on tests'), ('steer2', 'first, edited')]
        # The queued command runs once the turn ends, leaving the queue empty.
        messages, notice = receive_until(socket, Notice)
        assert notice.command == '/help'
        assert [message.queue for message in messages if isinstance(message, QueueChanged)][-1] == []
        state = web_.get('/api/state').json()
        # The saved history lists every prompt the model read, the note included, as it did before.
        assert [item['text'] for item in state['items'] if item['kind'] == 'user'] == [
            'go',
            NOTE,
            'focus on tests',
            'first, edited',
        ]

        # With no turn to take it, Alt+Enter enters the prompt as Enter does.
        assert web_.post('/api/steer', {'text': 'hello'}).json() == {'steered': False, 'menu': None}
        messages, _ = receive_until(socket, TurnStarted)
        assert [message.item.text for message in messages if isinstance(message, TurnStarted) and message.item] == [
            'hello'
        ]
        web_.portal.call(started.wait)
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


@pytest.mark.parametrize('command', ['/cost', '/usage'])
def test_usage_reports_answer_during_a_turn(make_web: MakeWeb, command: str) -> None:
    """`/cost` and `/usage` sent from the page answer while the turn runs, instead of queueing behind it."""
    agent, started, release = waiting_agent()
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'go'})
        web_.portal.call(started.wait)
        assert web_.post('/api/input', {'text': command}).json() == {'menu': None}
        assert web_.get('/api/state').json()['queue'] == []
        messages, notice = receive_until(socket, Notice)
        assert not any(isinstance(message, TurnEnded) for message in messages)
        # The running turn has no response yet, so there is no finished turn to report.
        assert (notice.command, notice.text.splitlines()[0]) == (command, 'No usage yet.')
        web_.portal.call(release.set)
        receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_steering_a_queued_prompt_with_an_image_shows_it(make_web: MakeWeb) -> None:
    """A prompt queued with an image steers with it, and the transcript shows its thumbnail once the model reads it."""
    agent, started, release = waiting_agent()
    web_ = make_web(agent)
    [red] = decode_images([encoded(png('red'))])
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'go'})
        web_.portal.call(started.wait)
        web_.post('/api/input', {'text': 'look', 'images': [encoded(png('red'))]})
        _, changed = receive_until(socket, QueueChanged)
        while not changed.queue or changed.queue[0].text == 'go':
            _, changed = receive_until(socket, QueueChanged)
        [queued] = changed.queue
        # Queued as the terminal's prompt holds it, its image by marker.
        assert queued.text.startswith('look [image:')
        assert web_.post(f'/api/queue/{queued.id}/steer').json() == {'steered': True}
        web_.portal.call(release.set)
        messages, _ = receive_until(socket, TurnEnded)
        [steered] = [
            message.item for message in messages if isinstance(message, ItemAdded) and message.item.kind == 'user'
        ]
        assert (steered.text, steered.images) == ('look', [image_id(red)])
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_queue_routes_of_one_session(make_web: MakeWeb) -> None:
    """A session's routes reach its own queue; refusals carry the prompt's own messages."""
    web_ = make_web(Agent(TestModel()))
    session = web_.plugin.service.default().id
    assert web_.post(f'/api/sessions/{session}/steer', {'text': '  '}).status_code == 400
    assert web_.post(f'/api/sessions/{session}/steer', {'text': '!ls'}).status_code == 400
    assert web_.client.delete(f'/api/sessions/{session}/queue/q0', headers={HEADER: TOKEN}).status_code == 404
    assert web_.post('/api/sessions/nope/steer', {'text': 'hi'}).status_code == 404


async def test_the_terminal_and_the_browser_share_one_queue(tmp_path: Path) -> None:
    """Input queues in the terminal's editor while it shows the session, else in its inbox, with the same IDs."""
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    changes: list[int] = []
    shell.process.live.subscribe(lambda: changes.append(shell.queued))
    shell.send_input('from the browser')
    [waiting] = shell.live_session.queue
    async with editor() as (live, _, _):
        live.queue_changed = shell.live_session.changed
        shell.bind(live)
        assert live.queued_messages == ('from the browser',)
        assert [entry.id for entry in shell.live_session.queue] == [waiting.id]
        # Typed in the terminal, the browser hears of it.
        live.buffer.replace('from the terminal')
        live.feed('enter')
        typed = shell.live_session.queue[-1]
        assert typed.text == 'from the terminal' and changes[-1] == 2
        # Edited and removed from the browser, the terminal's queue rows follow.
        shell.live_session.edit_queued(waiting.id, 'edited')
        assert live.queued_messages == ('edited', 'from the terminal')
        shell.live_session.remove_queued(typed.id)
        assert live.queued_messages == ('edited',) and changes[-1] == 1
        with pytest.raises(LookupError):
            shell.live_session.remove_queued(typed.id)
        with pytest.raises(ValueError, match='Type a prompt'):
            shell.live_session.edit_queued(waiting.id, ' ')
        # No turn runs, so nothing steers; commands and shell lines never do.
        assert not shell.live_session.steer('now')
        assert not shell.live_session.steer('/help')
        assert not shell.live_session.steer('!ls')
        shell.bind(None)
    # Back in the inbox, still the same entry.
    assert [(entry.id, entry.text) for entry in shell.live_session.queue] == [(waiting.id, 'edited')]
    shell.live_session.edit_queued(waiting.id, 'edited again')
    shell.live_session.remove_queued(waiting.id)
    assert shell.live_session.queue == [] and shell.queued == 0


def test_a_live_session_without_a_prompt_queues_nothing() -> None:
    """A `LiveRuntime` built without the session's prompt, as `PluginLoader` builds one in tests, holds no input."""
    host = PluginHost[None](name='', console=Console(file=StringIO()), settings={})
    session = LiveRuntime(live_id='x', host=host, running=lambda: False)
    assert (session.queue, session.steer('hi')) == ([], False)
    for change in (lambda: session.edit_queued('q1', 'x'), lambda: session.remove_queued('q1')):
        with pytest.raises(LookupError, match='no longer queued'):
            change()
    assert session.queued == 0


def test_a_live_session_counts_the_queue_its_prompt_holds() -> None:
    """With a prompt, `queued` counts its queue, so the count and the list cannot disagree."""

    class Inputs:
        queue = [QueuedInput(id='q1', text='first'), QueuedInput(id='q2', text='/help')]

        def edit_queued(self, input_id: str, text: str) -> None: ...  # pragma: no cover

        def remove_queued(self, input_id: str) -> None: ...  # pragma: no cover

        def steer(self, text: str) -> bool: ...  # pragma: no cover

    host = PluginHost[None](name='', console=Console(file=StringIO()), settings={})
    session = LiveRuntime(live_id='x', host=host, running=lambda: True, queued=lambda: 5, inputs=Inputs)
    assert session.queued == len(session.queue) == 2
    # Without a prompt, the count still comes from `queued`.
    assert LiveRuntime(live_id='y', host=host, running=lambda: True, queued=lambda: 3).queued == 3


async def test_the_former_inbox_still_reads_input_in_order() -> None:
    """`Inbox`, which `PromptInbox` replaced, keeps working for code that imports it."""
    inbox = Inbox()
    inbox.put('first')
    inbox.put('second')
    assert [await inbox.get(), await inbox.get()] == ['first', 'second']
    received: list[str] = []

    async def read() -> None:
        received.append(await inbox.get())

    async with anyio.create_task_group() as tg:
        tg.start_soon(read)
        await anyio.wait_all_tasks_blocked()  # `read` is now waiting on an empty inbox
        inbox.put('later')
    assert received == ['later']
