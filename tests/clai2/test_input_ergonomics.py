"""Input requirements exercised through the composer, decoder and shared settings store."""

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from pydantic import ValidationError
from rich.text import Text

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption
from pydantic_clai2 import config
from pydantic_clai2.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.input_bindings import BusyDelivery, InputBindings, KeyboardLabels, QuestionSubmit
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins.menus import CloseMenu
from pydantic_clai2.runtime.permissions import decision
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.ui.menus.input_menu import input_menu
from pydantic_clai2.ui.prompt.input_setup import KeyTest
from pydantic_clai2.ui.prompt.prompt_keys import PromptKeys
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs
from tests.clai2.test_live_prompt import editor
from tests.clai2.test_permissions import request as permission_request
from tests.clai2.test_steering import running
from tests.conftest import no_leaked_tasks


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
    assert InputBindings().label('steer', keyboard='mac') == 'Option+Return'
    assert InputBindings().label('steer', keyboard='pc') == 'Alt+Enter'
    assert InputBindings(steer='cmd-enter').label('steer', keyboard='mac') == 'Cmd+Return'


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
    assert {
        'send',
        'steer',
        'newline',
        'test',
        'input.keyboard_labels',
        'input.question_submit',
        'input.busy_delivery',
    } == rows.keys()
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
    assert store.load().busy_delivery == 'after-turn'
    store.set('input.busy_delivery', 'next-checkpoint')
    store.set('input.keyboard_labels', 'mac')
    store.set('input.question_submit', 'answer')
    store.set('input.bindings', {'send': 'ctrl-s', 'steer': 'alt-enter', 'newline': 'shift-enter'})
    reopened = SettingsStore(store.path)
    assert reopened.load().keyboard_labels == 'mac'
    assert reopened.load().question_submit == 'answer'
    assert reopened.load().busy_delivery == 'next-checkpoint'
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
        assert [entry.id for entry in live.queued_prompts()] == [first.id]
        assert await live.read() == 'first'
        live.buffer.replace('edited second')
        live.feed('enter')
        assert live.queued_prompts()[0].id == second.id
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


@pytest.mark.parametrize(
    'keyboard, enter, alt', [('mac', 'Return', 'Option'), ('pc', 'Enter', 'Alt'), ('neutral', 'Enter', 'Alt')]
)
async def test_physical_enter_labels_follow_effective_bindings(keyboard: KeyboardLabels, enter: str, alt: str) -> None:
    async with editor() as (live, _, _):
        live.keyboard_labels = lambda: keyboard
        assert live.bindings().label('send', keyboard=live.keyboard_labels()) == enter
        assert live.bindings().label('newline', keyboard=live.keyboard_labels()) == f'Shift+{enter}'
        assert f'{enter}: send' in live.input_hint()
        live.key_test = KeyTest()
        live.feed('enter')
        assert f'{enter}/Ctrl+M/Ctrl+{enter}' in live.key_test.report
        assert f'{alt}+{enter}' in live.key_test.report
        live.end_calibration()
        live.bindings = lambda: InputBindings(send='alt-shift-enter', steer='super-shift-enter', newline='enter')
        assert live.bindings().label('send', keyboard=live.keyboard_labels()) == f'{alt}+Shift+{enter}'
        assert f'{alt}+Shift+{enter}: send' in live.input_hint()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.questions.ask, question_request())
            await anyio.wait_all_tasks_blocked()
            assert f'{alt}+Shift+{enter}' in live.questions.hint(
                send_label=live.bindings().label('send', keyboard=live.keyboard_labels())
            )
            tasks.cancel_scope.cancel()


async def test_editing_is_not_pending_and_cancel_preserves_reserved_id_and_images() -> None:
    async with editor() as (live, _, _):
        image = BinaryContent(data=b'original image', media_type='image/png')
        marker = live.images.attach([image])
        live.submit('first')
        live.submit('second ' + marker)
        live.submit('third')
        first, second, third = live.queued_prompts()
        live.buffer.insert('earlier draft')
        earlier = live.buffer
        live.edit_queue_entry(second)
        live.buffer.replace('changed without image')
        live.images.retain(live.drafts)
        assert live.images.resolve(second.text)[1] == [image]
        assert live.queued_prompts() == [first, third]
        frame = Text.from_ansi('\n'.join(live.frame())).plain
        assert "Editing - not queued; won't send until saved" in frame
        assert 'queued: 2' in frame and 'queued: 3' not in frame
        assert 'Save/requeue' in frame and 'Cancel editing' in frame
        assert await live.read() == 'first'
        live.feed('escape')
        assert live.buffer is earlier
        assert [entry.id for entry in live.queued_prompts()] == [second.id, third.id]
        assert await live.read() == second.text
        assert await live.read() == 'third'
        assert live.queued_messages == ()


async def test_delivery_mode_is_persistent_and_reachable_without_queued_messages(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    context = CommandContext(settings=Settings(model=None), store=store, apply_setting=lambda key, settings: None)
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: context.settings.busy_delivery

        def remember(mode: BusyDelivery) -> object:
            return context.set_setting(['input.busy_delivery', mode])

        live.remember_busy_delivery = remember
        live.feed('f4')
        await anyio.lowlevel.checkpoint()
        assert 'New busy: after turn | m change' in '\n'.join(live.frame())
        live.feed('m')
        assert context.settings.busy_delivery == 'next-checkpoint'
        assert SettingsStore(store.path).load().busy_delivery == 'next-checkpoint'
        live.console.size = (40, 12)
        assert 'New busy: next checkpoint | m change' in '\n'.join(live.frame())
        live.feed('escape')
        menu = input_menu(context, start_test=lambda: None)
        row = next(row for row in menu.source.rows() if row.key == 'input.busy_delivery')
        assert row.choices == ('after-turn', 'next-checkpoint')
        menu.source.reset(row)
        assert context.settings.busy_delivery == 'after-turn'
        with pytest.raises(ValidationError):
            store.set('input.busy_delivery', 'interrupt')
        assert store.load().busy_delivery == 'after-turn'


@pytest.mark.parametrize('older_entries', [False, True])
@pytest.mark.parametrize('mode', ['after-turn', 'next-checkpoint'])
async def test_busy_delivery_waits_for_tools_and_reaches_next_feasible_model_request(
    mode: BusyDelivery, older_entries: bool
) -> None:
    started, release, completed = anyio.Event(), anyio.Event(), anyio.Event()
    other_started, other_release, other_completed = anyio.Event(), anyio.Event(), anyio.Event()
    seen: list[list[ModelMessage]] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        seen.append(list(messages))
        if len(seen) == 1:
            yield {0: DeltaToolCall(name='work', json_args='{}'), 1: DeltaToolCall(name='other_work', json_args='{}')}
        else:
            assert completed.is_set() and other_completed.is_set(), (
                'a model request must wait for all current tool work'
            )
            yield 'done'

    agent = Agent(FunctionModel(stream_function=model))

    @agent.tool_plain
    async def work() -> str:
        started.set()
        await release.wait()
        completed.set()
        return 'tool completed'

    @agent.tool_plain
    async def other_work() -> str:
        other_started.set()
        await other_release.wait()
        other_completed.set()
        return 'other tool completed'

    session = Session(agent, deps=None)
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: mode
        live.steer = session.steer

        async def run() -> None:
            await session.prompt('start')

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.interrupts.run, run())
            await started.wait()
            await other_started.wait()
            if older_entries:
                live.submit('older')
            live.feed('paste', 'new direction')
            live.feed('enter')
            assert not completed.is_set() and len(seen) == 1
            assert live.interrupts.active
            older = ('older',) if older_entries else ()
            assert live.queued_messages == (older if mode == 'next-checkpoint' else (*older, 'new direction'))
            if mode == 'next-checkpoint':
                if older_entries:
                    assert live.buffer.text == 'new direction'
                    assert 's Steer/promote: next checkpoint' in '\n'.join(live.frame())
                    await anyio.lowlevel.checkpoint()
                    live.feed('s')
                assert 'Next checkpoint (sent to active run): new direction' in live.notice
            release.set()
            await completed.wait()
            await anyio.wait_all_tasks_blocked()
            assert len(seen) == 1 and not other_completed.is_set()
            other_release.set()
        assert completed.is_set() and other_completed.is_set() and not session.running
        assert len(seen) == 2
        prompts = [
            part.content
            for message in seen[1]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert ('new direction' in prompts) is (mode == 'next-checkpoint')
        assert 'older' not in prompts
        assert any(
            isinstance(part, ToolReturnPart) and part.content == 'tool completed'
            for message in seen[1]
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if older_entries:
            assert await live.read() == 'older'
        if mode == 'after-turn':
            assert await live.read() == 'new direction'
        assert live.queued_messages == ()


@pytest.mark.parametrize('text', ['/help', '!echo explicit', 'clear', '!!interactive'])
async def test_checkpoint_mode_never_smuggles_commands_or_shell_lines(text: str) -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        steered: list[str] = []

        def steer(text: str) -> bool:
            return steered.append(text) or True

        live.steer = steer
        started, release = anyio.Event(), anyio.Event()

        async def work() -> None:
            started.set()
            await release.wait()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.interrupts.run, work())
            await started.wait()
            live.feed('paste', text)
            live.feed('enter')
            assert steered == []
            assert len(live.queued_messages) == 1
            release.set()
        assert await live.read() == ('/clear' if text == 'clear' else text)


async def test_checkpoint_mode_never_leapfrogs_queue_or_steers_a_saved_edit() -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        steered: list[str] = []

        def steer(text: str) -> bool:
            return steered.append(text) or True

        live.steer = steer
        started, release = anyio.Event(), anyio.Event()

        async def work() -> None:
            started.set()
            await release.wait()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.interrupts.run, work())
            await started.wait()
            live.submit('/help')
            live.feed('paste', 'behind command')
            live.feed('enter')
            assert steered == [] and live.queued_messages == ('/help',)
            assert live.buffer.text == 'behind command'
            assert 's Steer/promote: next checkpoint' in '\n'.join(live.frame())
            assert 'q Keep queue position: after turn' in '\n'.join(live.frame())
            await anyio.lowlevel.checkpoint()
            live.feed('q')
            entry = live.queued_prompts()[1]
            live.edit_queue_entry(entry)
            live.buffer.replace('saved edit')
            live.feed('enter')
            assert steered == [] and live.queued_messages == ('/help',)
            assert live.buffer.text == 'saved edit'
            assert 'q Keep queue position: after turn' in '\n'.join(live.frame())
            await anyio.lowlevel.checkpoint()
            live.feed('q')
            assert live.queued_prompts()[1].id == entry.id
            assert live.queued_messages == ('/help', 'saved edit')
            release.set()
        assert await live.read() == '/help'
        assert await live.read() == 'saved edit'


@pytest.mark.parametrize('submit', ['enter', 'alt-enter'])
async def test_unavailable_checkpoint_requires_explicit_queue_choice(submit: str) -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'

        def unavailable(text: str) -> bool:
            return False

        live.steer = unavailable
        async with running(live):
            live.feed('paste', 'during teardown')
            live.feed(submit)
            assert live.queued_messages == () and live.buffer.text == 'during teardown'
            live.console.size = (40, 12)
            assert 'Delivery: next checkpoint requested' in '\n'.join(live.frame())
            assert 'Cannot steer; nothing was sent.' in '\n'.join(live.frame())
            await anyio.lowlevel.checkpoint()
            live.feed('s')
            assert live.queued_messages == () and live.buffer.text == 'during teardown'
            assert 'nothing was sent' in live.notice
            live.feed('q')
            assert live.queued_messages == ('during teardown',)
            assert 'Follow-up (after turn): during teardown' in '\n'.join(live.frame())
            assert 'nothing was sent' not in '\n'.join(live.frame())
            live.feed('escape')
        assert live.queued_messages == () and live.buffer.text == 'during teardown'


@pytest.mark.parametrize('older_entries', [False, True])
async def test_checkpoint_delivery_and_remembered_answer_never_approve_waiting_tool(older_entries: bool) -> None:
    with no_leaked_tasks():
        async with editor() as (live, _, _):
            live.busy_delivery = lambda: 'next-checkpoint'
            live.questions.preference = lambda: 'answer'
            steered: list[str] = []

            def steer(text: str) -> bool:
                return steered.append(text) or True

            live.steer = steer
            responses: list[AskUserResponse] = []

            answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console)

            async def ask() -> None:
                responses.append(await answerer.ask_approval(permission_request()))

            async with running(live):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(ask)
                    await anyio.wait_all_tasks_blocked()
                    if older_entries:
                        live.submit('/help')
                    live.feed('paste', 'Approve')
                    live.feed('enter')
                    if older_entries:
                        assert steered == [] and not responses
                        await anyio.lowlevel.checkpoint()
                        live.feed('s')
                    assert steered == ['Approve'] and not responses
                    assert not live.questions.focused
                    assert live.queued_messages == (('/help',) if older_entries else ())
                    live.feed('f2')
                    await anyio.lowlevel.checkpoint()
                    live.feed('ctrl-c')
                assert not decision(responses[0]).approved
                live.feed('escape')


@pytest.mark.parametrize('editing', [False, True])
@pytest.mark.parametrize('choice', ['s', 'q', 'escape', 'ctrl-c'])
async def test_mixed_checkpoint_delivery_needs_explicit_choice_without_losing_state(editing: bool, choice: str) -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        steered: list[str] = []

        def steer(text: str) -> bool:
            steered.append(text)
            return True

        live.steer = steer
        live.submit('first')
        live.submit('second')
        live.submit('third')
        first, second, third = live.queued_prompts()
        image = BinaryContent(data=b'earlier image', media_type='image/png')
        marker = live.images.attach([image])
        live.buffer.insert('earlier ' + marker)
        live.buffer.cursor = 3
        earlier = live.buffer
        if editing:
            live.edit_queue_entry(second)
        live.buffer.replace('direction')
        draft = live.buffer
        async with running(live):
            live.feed('enter')
            assert steered == []
            assert live.buffer is draft and draft.text == 'direction'
            live.console.size = (40, 12)
            assert live.queued_prompts() == ([first, third] if editing else [first, second, third])
            assert 's Steer/promote: next checkpoint' in '\n'.join(live.frame())
            assert 'q Keep queue position: after turn' in '\n'.join(live.frame())
            live.console.size = (80, 24)
            live.images.retain(live.drafts)
            if editing:
                assert live.images.resolve(earlier.text)[1] == [image]
            await anyio.lowlevel.checkpoint()
            live.feed(choice)
            assert steered == (['direction'] if choice == 's' else [])
            if choice in ('escape', 'ctrl-c'):
                assert live.buffer is draft and draft.text == 'direction'
                assert 'Delivery: next checkpoint requested' not in '\n'.join(live.frame())
                if editing:
                    live.feed('escape')  # Cancel the edit separately, restoring its original entry.
                    assert live.buffer is earlier and live.buffer.cursor == 3
                else:
                    assert live.queued_prompts() == [first, second, third]
            elif choice == 's':
                assert live.queued_prompts() == ([first, third] if editing else [first, second, third])
                if editing:
                    assert live.buffer is earlier and live.buffer.cursor == 3
                    assert live.images.resolve(live.buffer.text)[1] == [image]
                else:
                    assert live.buffer.text == ''
                assert live.history.get_strings().count('direction') == 1
            else:
                assert steered == []
                assert live.queued_messages == (
                    ('first', 'direction', 'third') if editing else ('first', 'second', 'third', 'direction')
                )
                if editing:
                    assert live.queued_prompts()[1].id == second.id
                    assert live.buffer is earlier and live.buffer.cursor == 3
                assert 'Follow-up (after turn): direction' in Text.from_ansi('\n'.join(live.frame())).plain
            live.feed('escape')  # Stop the test's active work; no queued item runs on its own.
        assert live.queued_messages == ()


async def test_delivery_choice_cannot_consume_paste_or_same_batch_typeahead() -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        steered: list[str] = []

        def steer(text: str) -> bool:
            steered.append(text)
            return True

        live.steer = steer
        live.submit('older')
        async with running(live):
            live.feed('paste', 'draft')
            live.feed('enter')
            live.feed('s')  # Same decoder batch as Send: typeahead, not an action.
            live.feed('enter')
            assert live.buffer.text == 'drafts' and steered == []
            await anyio.lowlevel.checkpoint()
            live.feed('paste', 'q\ns\n')
            assert live.buffer.text == 'draftsq\ns\n' and steered == []
            live.feed('escape')
            assert live.buffer.text == 'draftsq\ns\n'
            assert live.queued_messages == ('older',)
            live.feed('escape')


async def test_held_edit_delivery_choice_survives_finish_and_restores_reserved_id() -> None:
    with no_leaked_tasks():
        async with editor() as (live, _, _):
            live.busy_delivery = lambda: 'next-checkpoint'
            live.submit('older')
            live.submit('edit me')
            _, held = live.queued_prompts()
            live.buffer.insert('earlier draft')
            earlier = live.buffer
            live.edit_queue_entry(held)
            live.buffer.replace('edited')
            live.feed('enter')  # A held edit still requires the delivery choice even if work just ended.
            assert live.queued_messages == ('older',)
            assert await live.read() == 'older'
            received: list[str] = []

            async def read_next() -> None:
                received.append(await live.read())

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(read_next)
                await anyio.wait_all_tasks_blocked()
                assert received == [] and live.queued_prompts() == []
                assert 'q Keep queue position: separate turn' in '\n'.join(live.frame())
                live.feed('s')  # No active run: must not silently enqueue or lose the edit.
                assert received == [] and live.buffer.text == 'edited'
                assert 'Cannot steer; nothing was sent.' in '\n'.join(live.frame())
                live.feed('q')
                assert live.queued_prompts()[0].id == held.id
                assert live.buffer is earlier and live.buffer.text == 'earlier draft'
                assert 'nothing was sent' not in '\n'.join(live.frame())
            assert received == ['edited'] and live.queued_messages == ()


async def test_delivery_choice_rechecks_eligibility_after_draft_changes() -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        steered: list[str] = []

        def steer(text: str) -> bool:
            steered.append(text)
            return True

        live.steer = steer
        live.submit('older')
        async with running(live):
            live.feed('paste', 'eligible')
            live.feed('enter')
            await anyio.lowlevel.checkpoint()
            live.buffer.replace('/help')
            assert 'Steer unavailable: commands/shell/blank' in '\n'.join(live.frame())
            live.feed('s')
            assert steered == [] and live.buffer.text == '/help'
            live.feed('q')
            assert live.queued_messages == ('older', '/help')
            live.feed('escape')


async def test_mixed_checkpoint_choice_remains_explicit_between_turns() -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        live.submit('older')
        live.buffer.insert('new direction')
        live.feed('enter')
        assert live.queued_messages == ('older',) and live.buffer.text == 'new direction'
        assert 'q Keep queue position: separate turn' in '\n'.join(live.frame())
        await anyio.lowlevel.checkpoint()
        live.feed('q')
        assert live.queued_messages == ('older', 'new direction')
        assert await live.read() == 'older'
        assert await live.read() == 'new direction'
