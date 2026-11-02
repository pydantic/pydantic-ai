"""Edit and Retry of an earlier prompt from the browser, which rewind the conversation as the terminal's Esc Esc does."""

import anyio
import pytest
from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    Notice,
    SessionsChanged,
    Snapshot,
    TurnEnded,
    TurnStarted,
    image_id,
)
from pydantic_clai2.commands import Command
from pydantic_clai2.runtime import image_codec
from pydantic_clai2.runtime.image_codec import decode_images
from tests.clai2.test_web import TOKEN, MakeWeb, encoded, make_web, png, receive_until
from tests.clai2.test_web_queue import ORIGIN, waiting_agent

__all__ = ['make_web']

BUSY = 'Wait for the running turn, command, or queued input to end.'


def settle(socket: WebSocketTestSession) -> None:
    """Wait for the turn to end and save, after which its history may be rewound."""
    receive_until(socket, TurnEnded)
    receive_until(socket, SessionsChanged)


def test_edit_and_retry_replace_a_prompt_and_every_later_turn(make_web: MakeWeb) -> None:
    """Send A, then B; editing A to A' leaves A' and its answer, with A's image, and B gone; Retry sends it again."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None)))
    [red] = decode_images([encoded(png('red'))])
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'A', 'images': [encoded(png('red'))]})
        _, started = receive_until(socket, TurnStarted)
        settle(socket)
        web_.post('/api/input', {'text': 'B'})
        settle(socket)

        def prompts() -> list[tuple[str, list[str]]]:
            items = web_.get('/api/state').json()['items']
            return [(item['text'], item['images']) for item in items if item['kind'] == 'user']

        first = next(item for item in web_.get('/api/state').json()['items'] if item['kind'] == 'user')
        assert started.item is not None and started.item.run_id == first['run_id'], 'live and saved prompts agree'
        route = f'/api/sessions/{web_.shell.session.conversation_id}/rewind/{first["run_id"]}'
        for text, status, detail in [
            ('/help', 400, 'Send a prompt in its place, not a /command.'),
            (' Clear ', 400, 'Send a prompt in its place, not a /command.'),
            ('!ls', 400, 'Shell lines (!) run only in the terminal.'),
        ]:
            refused = web_.post(route, {'text': text})
            assert (refused.status_code, refused.json()) == (status, {'detail': detail})
        missing = web_.post('/api/rewind/nope', {'text': 'x'})
        assert (missing.status_code, missing.json()) == (
            404,
            {'detail': 'That prompt can no longer be edited: it left the conversation or predates a compaction.'},
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(image_codec, 'MAX_PENDING_BYTES', 1)
            refused = web_.post(route, {'text': "A'"})
            assert (refused.status_code, refused.json()) == (
                400,
                {'detail': 'These images exceed the 32 MiB limit; send fewer at a time.'},
            )
            assert receive_until(socket, Snapshot)[1].items[-1].text == 'done', 'the browser gets the history back'
        assert prompts() == [('A', [image_id(red)]), ('B', [])], 'a refusal keeps the history'

        assert web_.post(route, {'text': " A' "}).status_code == 204
        assert receive_until(socket, Snapshot)[1].items == [], 'the browser starts over from the rewound history'
        settle(socket)
        assert prompts() == [("A'", [image_id(red)])]

        retried = next(item for item in web_.get('/api/state').json()['items'] if item['kind'] == 'user')
        assert web_.post(f'/api/rewind/{retried["run_id"]}', {'text': "A'"}).status_code == 204
        settle(socket)
        items = web_.get('/api/state').json()['items']
        assert [(item['kind'], item['text']) for item in items] == [('user', "A'"), ('assistant', 'done')]
        assert items[0]['run_id'] != retried['run_id']
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_rewinding_waits_for_the_running_turn(make_web: MakeWeb) -> None:
    """While a turn runs, its history is not rewound; steering, which starts no turn, offers no rewind."""
    agent, started, release = waiting_agent()
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'go'})
        _, turn = receive_until(socket, TurnStarted)
        web_.portal.call(started.wait)
        assert turn.item is not None and turn.item.run_id is not None
        busy = web_.post(f'/api/rewind/{turn.item.run_id}', {'text': 'again'})
        assert (busy.status_code, busy.json()) == (409, {'detail': BUSY})
        assert web_.post('/api/steer', {'text': 'focus'}).json()['steered']
        web_.portal.call(release.set)
        settle(socket)
        users = [item for item in web_.get('/api/state').json()['items'] if item['kind'] == 'user']
        assert [(item['text'], item['run_id'] is not None) for item in users] == [
            ('go', True),
            ('A context file the agent found.', False),
            ('focus', False),
        ]
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_rewinding_waits_for_a_command(make_web: MakeWeb) -> None:
    """A command such as `/compact` replaces the history too, so the history is not rewound while one runs."""
    web_ = make_web(Agent(TestModel(call_tools=[]), deps_type=type(None)))
    started, release = anyio.Event(), anyio.Event()

    async def hold(args: list[str]) -> str:
        started.set()
        await release.wait()
        return 'held'

    web_.shell.commands.register(Command(name='hold', description='Hold until released.', handler=hold))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'A'})
        _, turn = receive_until(socket, TurnStarted)
        settle(socket)
        assert turn.item is not None and turn.item.run_id is not None
        web_.post('/api/input', {'text': '/hold'})
        web_.portal.call(started.wait)
        busy = web_.post(f'/api/rewind/{turn.item.run_id}', {'text': 'again'})
        assert (busy.status_code, busy.json()) == (409, {'detail': BUSY})
        web_.portal.call(release.set)
        receive_until(socket, Notice)
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'
