"""Input requirements exercised through the composer, decoder and shared settings store."""

from pathlib import Path

import anyio
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from pydantic import ValidationError
from rich.text import Text

from pydantic_ai.messages import BinaryContent
from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption
from pydantic_clai2 import config
from pydantic_clai2.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.input_bindings import InputBindings, QuestionSubmit
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins.menus import CloseMenu
from pydantic_clai2.runtime.permissions import decision
from pydantic_clai2.ui.menus.input_menu import input_menu
from pydantic_clai2.ui.prompt.input_setup import KeyTest
from pydantic_clai2.ui.prompt.prompt_keys import PromptKeys
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs
from tests.clai2.test_live_prompt import editor
from tests.clai2.test_permissions import request as permission_request


def test_defaults_aliases_and_collisions() -> None:
    settings = Settings()
    assert (settings.keyboard_labels, settings.question_submit) == ('unset', 'ask')
    assert [settings.input_bindings.action(key) for key in ('enter', 'alt-enter', 'shift-enter', 'ctrl-j')] == [
        'send',
        'steer',
        'newline',
        'newline',
    ]
    assert InputBindings(send='ctrl-enter').action('enter') == 'send'
    assert InputBindings(steer='option-enter').action('alt-enter') == 'steer'
    for key in ('ctrl-enter', 'ctrl-shift-enter', 'ctrl-m', 'enter'):
        with pytest.raises(ValidationError, match='collide'):
            InputBindings(newline=key)
    for key in ('ctrl-c', 'ctrl-j', 'paste', 'escape', 'ctrl-x ctrl-s', 'a'):
        with pytest.raises(ValidationError, match='Choose a decoded key'):
            InputBindings(send=key)
    assert InputBindings().label('steer', keyboard='mac') == 'Option+Enter'
    assert InputBindings().label('steer', keyboard='pc') == 'Alt+Enter'
    assert InputBindings(steer='cmd-enter').label('steer', keyboard='mac') == 'Cmd+Enter'


async def test_remapped_composer_and_hints_use_effective_bindings() -> None:
    async with editor() as (live, pipe, _):
        live.bindings = lambda: InputBindings(send='ctrl-s', steer='alt-s', newline='enter')
        live.keyboard_labels = lambda: 'mac'
        pipe.send_text('one\rtwo\nthree\x13')
        assert await live.read() == 'one\ntwo\nthree'
        steered: list[str] = []

        def steer(text: str) -> bool:
            steered.append(text)
            return True

        live.steer = steer
        live.feed('paste', 'steering')
        live.feed('alt-s')
        assert steered == ['steering']
        assert live.queued_messages == ()


async def test_explicit_key_capture_never_executes_and_hides_draft() -> None:
    async with editor() as (live, pipe, _):
        live.buffer.replace('private draft')
        cursor = live.buffer.cursor
        live.key_test = KeyTest()
        for key in ('enter', 'alt-enter', 'ctrl-c', 'escape', 'ctrl-d', 'ctrl-j'):
            live.feed(key)
            assert live.queued_messages == ()
            assert live.buffer.text == 'private draft'
            assert live.buffer.cursor == cursor
            assert 'private draft' not in '\n'.join(live.frame())
        live.feed('enter')
        assert live.key_test is not None
        assert 'cannot be recovered' in live.key_test.report
        assert 'Ctrl+J' in live.key_test.report
        pipe.send_text('\x1b[200~typeahead\nnot submitted\x1b[201~\r\r')
        live.keys.read()
        assert live.queued_messages == ()
        assert 'typeahead\nnot submitted' in live.buffer.text
        live.feed('ctrl-g')
        live.feed('enter')
        assert live.queued_messages == ()
        await anyio.lowlevel.checkpoint()
        assert live.key_test is None
        assert 'private draft' in Text.from_ansi('\n'.join(live.frame())).plain


async def test_setup_actions_save_reset_and_cancel(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    context = CommandContext(settings=Settings(model=None), store=store, apply_setting=lambda key, settings: None)
    tested: list[bool] = []
    menu = input_menu(context, start_test=lambda: tested.append(True))
    rows = {row.key: row for row in menu.source.rows()}
    assert {'send', 'steer', 'newline', 'test', 'input.keyboard_labels', 'input.question_submit'} == rows.keys()
    assert menu.source.current(rows['send']) == 'enter'
    assert await menu.actions['send'](ScriptedDialogs([Back()])) == ()
    await menu.actions['send'](ScriptedDialogs([Pick(value='ctrl-s')]))
    assert context.settings.input_bindings.action('ctrl-s') == 'send'
    assert context.settings.input_bindings.action('enter') is None
    rejected = await menu.actions['newline'](ScriptedDialogs([Pick(value='ctrl-s')]))
    assert 'collide' in '\n'.join(rejected)
    assert context.settings.input_bindings.newline == 'shift-enter'
    result = await menu.actions['test'](ScriptedDialogs([]))
    assert isinstance(result, CloseMenu) and tested == [True]
    assert 'Saved' in menu.source.apply(rows['input.keyboard_labels'], 'mac')
    assert context.settings.keyboard_labels == 'mac'
    menu.source.reset(rows['send'])
    assert context.settings.input_bindings.send == 'enter'
    assert menu.source.reset(rows['test']) == 'No captured key is saved.'


def test_input_preferences_roundtrip_and_unknown_settings_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.set('display.thinking', False)
    previous = store.overrides()
    assert previous == {'display.thinking': False}
    assert store.load().input_bindings == InputBindings()
    store.set('input.keyboard_labels', 'mac')
    store.set('input.question_submit', 'answer')
    store.set('input.bindings', {'send': 'ctrl-s', 'steer': 'alt-enter', 'newline': 'shift-enter'})
    reopened = SettingsStore(store.path)
    assert reopened.load().keyboard_labels == 'mac'
    assert reopened.load().question_submit == 'answer'
    assert reopened.load().input_bindings.send == 'ctrl-s'
    saved = reopened.overrides()
    with pytest.raises(ValidationError, match='collide'):
        reopened.set('input.bindings', {'send': 'enter', 'newline': 'ctrl-enter'})
    assert reopened.overrides() == saved
    # An older build knows none of these names but must preserve their stored values.
    older = {key: field for key, field in config.SETTING_FIELDS.items() if not key.startswith('input.')}
    monkeypatch.setattr(config, 'SETTING_FIELDS', older)
    reopened.set('display.thinking', True)
    reopened.reset('display.thinking')
    assert {key: value for key, value in reopened.overrides().items() if key.startswith('input.')} == {
        key: value for key, value in saved.items() if key.startswith('input.')
    }


def question_request(*, batch: bool = False) -> AskUserRequest:
    question = Question(
        header='Approach', question='Which approach?', options=(QuestionOption(label='A'), QuestionOption(label='B'))
    )
    second = Question(
        header='Next', question='Which next step?', options=(QuestionOption(label='C'), QuestionOption(label='D'))
    )
    return AskUserRequest(id='request', questions=(question, second) if batch else (question,))


async def test_question_arrival_keeps_composer_and_explicit_route_moves_draft_once() -> None:
    async with editor() as (live, pipe, _):
        live.buffer.insert('draft')
        live.buffer.cursor = 2
        original = live.buffer
        responses: list[AskUserResponse] = []
        preference: list[QuestionSubmit] = ['ask']
        live.questions.preference = lambda: preference[-1]
        live.questions.remember = preference.append
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console)

        async def ask() -> None:
            responses.append(await answerer(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            assert live.questions.menu is not None
            assert not live.questions.focused
            assert live.buffer is original and (live.buffer.text, live.buffer.cursor) == ('draft', 2)
            typed = anyio.Event()

            def receive(key: str, data: str) -> None:
                live.feed(key, data)
                typed.set()

            live.keys.feed = receive
            pipe.send_text('X')
            await typed.wait()
            assert live.buffer.text == 'drXaft'
            live.feed('ctrl-z')
            assert live.buffer.text == 'draft'
            live.feed('enter')
            assert live.questions.routing and live.buffer.text == 'draft'
            assert 'Answer target: Approach' in '\n'.join(live.frame())
            live.feed('enter')  # Held Enter cannot select a routing action.
            await anyio.lowlevel.checkpoint()
            live.feed('escape')
            assert live.buffer is original and live.buffer.text == 'draft'
            live.feed('enter')
            await anyio.lowlevel.checkpoint()
            live.feed('r')
            live.feed('a')
            live.feed('enter')
        assert len(responses) == 1
        assert responses[0].answers[0].custom_answer == 'draft'
        assert live.buffer.text == '' and live.queued_messages == ()
        assert preference == ['ask', 'answer']
        assert live.questions.menu is None


async def test_follow_up_choice_does_not_answer_and_cancel_preserves_draft() -> None:
    async with editor() as (live, _, _):
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('paste', 'follow-up')
            live.feed('enter')
            await anyio.lowlevel.checkpoint()
            live.feed('q')
            assert live.queued_messages == ('follow-up',)
            assert live.questions.menu is not None and responses == []
            live.feed('paste', 'keep this draft')
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-c')
        assert responses[0].cancelled
        assert live.buffer.text == 'keep this draft'


async def test_batch_requires_explicit_next_question_intent_and_paste_never_selects() -> None:
    async with editor() as (live, _, _):
        live.questions.preference = lambda: 'answer'
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request(batch=True)))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('paste', 'first answer')
            live.feed('enter')
            live.feed('enter')
            await anyio.wait_all_tasks_blocked()
            assert live.questions.menu is not None and live.questions.menu.position == 2
            assert not live.questions.focused
            live.feed('paste', 'second answer')
            live.feed('enter')
            assert live.questions.routing  # Remembered answer does not consume the next question blindly.
            await anyio.lowlevel.checkpoint()
            live.feed('enter')
            assert not responses
            live.feed('a')
            live.feed('enter')
        assert [answer.custom_answer for answer in responses[0].answers] == ['first answer', 'second answer']


async def test_focused_question_uses_remapped_send_and_ctrl_j_only_inserts_newline() -> None:
    async with editor() as (live, _, _):
        live.bindings = lambda: InputBindings(send='ctrl-s', steer='alt-enter', newline='shift-enter')
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-j')
            assert live.questions.menu is not None and live.questions.menu.cursor == 0
            live.feed('down')
            live.feed('down')  # Other.
            live.feed('ctrl-s')
            live.feed('paste', 'line one\nline two')
            live.feed('ctrl-j')
            live.feed('paste', 'line three')
            live.feed('ctrl-s')
        assert responses[0].answers[0].custom_answer == 'line one\nline two\nline three'
        assert live.queued_messages == ()


@pytest.mark.parametrize('preference', ['ask', 'answer', 'follow-up'])
async def test_approval_requires_explicit_focus_regardless_of_question_preference(preference: QuestionSubmit) -> None:
    async with editor() as (live, _, _):
        live.questions.preference = lambda: preference
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console)
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await answerer.ask_approval(permission_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('paste', 'Approve')
            live.feed('enter')
            live.feed('enter')
            live.feed('alt-enter')
            assert live.questions.menu is not None and responses == []
            assert live.queued_messages == ('Approve',)
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-j')
            assert responses == []
            live.feed('down')
            live.feed('down')
            live.feed('down')  # Other, whose text always denies approval.
            live.feed('enter')
            live.feed('paste', 'Approve')
            live.feed('enter')
        result = decision(responses[0])
        assert not result.approved and not result.remember and result.reason == 'Approve'


async def test_visible_queue_edit_keeps_id_order_and_entire_earlier_draft() -> None:
    async with editor() as (live, _, _):
        live.submit('first')
        live.submit('second')
        first, second = live.queued_prompts()
        live.buffer.insert('earlier draft')
        live.buffer.cursor = 3
        earlier = live.buffer
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        assert 'e Edit | d Remove | s Steer' in '\n'.join(live.frame())
        live.feed('down')
        live.feed('e')
        assert [entry.id for entry in live.queued_prompts()] == [first.id, second.id]
        assert await live.read() == 'first'
        live.buffer.replace('edited second')
        live.feed('enter')
        assert live.buffer is earlier and live.buffer.cursor == 3
        assert live.buffer.text == 'earlier draft'
        assert await live.read() == 'edited second'
        assert live.queued_messages == ()
        live.buffer.undo()
        assert live.buffer.text == ''


async def test_visible_queue_remove_and_steer_cannot_execute_a_finished_entry() -> None:
    async with editor() as (live, _, _):
        steered: list[str] = []

        def steer(text: str) -> bool:
            steered.append(text)
            return True

        live.steer = steer
        live.submit('first')
        live.submit('second')
        live.buffer.insert('draft')
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        assert await live.read() == 'first'
        live.feed('s')  # Selected first already ran; never steer the next entry instead.
        assert steered == [] and live.queued_messages == ('second',)
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        live.feed('s')
        assert steered == ['second'] and live.queued_messages == ()
        assert live.buffer.text == 'draft'
        live.submit('remove me')
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        live.feed('d')
        assert live.queued_messages == () and live.buffer.text == 'draft'


async def test_explicit_focused_approval_still_authorizes_only_its_current_call() -> None:
    async with editor() as (live, _, _):
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console)
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await answerer.ask_approval(permission_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('f2')
            live.feed('enter')  # Same decoder batch as focus: cannot authorize.
            assert not responses
            await anyio.lowlevel.checkpoint()
            live.feed('enter')
        assert decision(responses[0]).approved
        assert not decision(responses[0]).remember


async def test_cancelled_question_keeps_images_and_queue_action_draft_inventory() -> None:
    async with editor() as (live, _, _):
        image = BinaryContent(data=b'image bytes', media_type='image/png')
        marker = live.images.attach([image])
        live.buffer.insert('caption ' + marker)
        original = live.buffer
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.questions.ask, question_request())
            await anyio.wait_all_tasks_blocked()
            live.feed('enter')
            await anyio.lowlevel.checkpoint()
            live.feed('a')  # Cannot silently drop an attached image into a text-only answer.
            assert live.questions.routing
            assert live.buffer is original and marker.strip() in live.buffer.text
            tasks.cancel_scope.cancel()
        assert live.questions.menu is None and live.buffer is original
        live.submit('queued')
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        live.feed('e')
        live.paste('edited')
        live.images.retain(live.drafts)
        assert live.images.resolve(original.text)[1] == [image]
        live.feed('escape')
        assert live.buffer is original
        assert live.images.resolve(live.buffer.text)[1] == [image]


@pytest.mark.parametrize(
    ('sequence', 'expected'),
    [
        ('\x1b[Q', 'f2'),
        ('\x1b[S', 'f4'),
        ('\x1bOQ', 'f2'),
        ('\x1bOS', 'f4'),
        ('\x1b[1;1Q', 'f2'),
        ('\x1b[1;1S', 'f4'),
    ],
)
@pytest.mark.parametrize('split', [False, True])
async def test_question_and_queue_focus_keys_decode_through_real_input(
    sequence: str, expected: str, split: bool
) -> None:
    received: list[str] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: received.append(key), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert received == [expected]
        finally:
            keys.stop()


async def test_calibration_preserves_paste_after_incomplete_escape_and_hides_unsupported_contents() -> None:
    async with editor() as (live, _, _):
        live.key_test = KeyTest()
        live.keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        live.keys.dispatch(KeyPress('[', '['))
        live.keys.dispatch(KeyPress(Keys.BracketedPaste, 'sensitive pasted payload'))
        assert live.buffer.text == 'sensitive pasted payload'
        assert 'sensitive' not in '\n'.join(live.frame())
        live.keys.unknown('\x1b[secret')
        assert live.key_test is not None
        assert 'Unsupported terminal key report' in live.key_test.report
        assert 'secret' not in live.key_test.report
        assert live.queued_messages == ()
