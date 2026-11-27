"""Input requirements exercised through the composer, decoder and shared settings store."""

from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from pydantic import ValidationError
from rich.console import RenderableType
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
from pydantic_ai_pax.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.input_bindings import BusyDelivery, InputBindings, KeyboardLabels, QuestionSubmit
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import ComposerAnswerer
from pydantic_ai_pax.plugins.menus import CloseMenu
from pydantic_ai_pax.runtime.permissions import decision
from pydantic_ai_pax.runtime.process import ClaiProcess
from pydantic_ai_pax.runtime.session import Session
from pydantic_ai_pax.ui.menus import dialogs
from pydantic_ai_pax.ui.menus.input_menu import input_menu
from pydantic_ai_pax.ui.prompt.input_setup import KeyTest
from pydantic_ai_pax.ui.prompt.live_prompt import LivePrompt
from pydantic_ai_pax.ui.prompt.prompt_keys import PromptKeys
from pydantic_ai_pax.ui.prompt.prompt_questions import PromptQuestions
from tests.conftest import no_leaked_tasks
from tests.pax.dialog_script import Back, Pick, ScriptedDialogs
from tests.pax.menu_script import Script, pick
from tests.pax.test_live_prompt import editor, highlighted, popup_shows
from tests.pax.test_permissions import request as permission_request
from tests.pax.test_session_runtime import runtime_over
from tests.pax.test_steering import running


def composer(live: LivePrompt) -> ComposerAnswerer:
    """The composer of the session `live` shows, as `SessionRuntime.ask_in_composer` reaches it once shown."""

    async def ask(request: AskUserRequest, /, *, approval: bool, preamble: RenderableType | None) -> AskUserResponse:
        return await live.questions.ask(request, approval=approval, preamble=preamble)

    return ask


class Steered(list[str]):
    """A `LivePrompt.steer` that accepts and records every prompt steered into the running turn."""

    def __call__(self, text: str) -> bool:
        self.append(text)
        return True


def test_defaults_aliases_and_collisions() -> None:
    settings = Settings()
    assert (settings.keyboard_labels, settings.question_submit) == ('neutral', 'ask')
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
        # Newline keeps naming Ctrl+J, which every terminal sends.
        live.input_changed = lambda: True
        hint = 'Ctrl+S: send - F4 Delivery - Ctrl+J newline - /input'
        assert live.input_hint() == hint
        assert live.key_label('steer') == 'Option+S'
        assert hint in Text.from_ansi('\n'.join(live.frame())).plain
        live.steer = steered = Steered()
        live.feed('paste', 'steering')
        live.feed('alt-s')
        assert steered == ['steering']
        assert live.queued_messages == ()


@pytest.mark.parametrize('key', ['ctrl-enter', 'ctrl-j'])
async def test_windows_ctrl_enter_follows_a_remapped_steer_binding(key: str) -> None:
    """Where Ctrl+Enter steers, it and the Ctrl+J older terminals send for it steer with the steer binding."""
    async with editor() as (live, _, _):
        live.ctrl_enter_steers = True
        live.bindings = lambda: InputBindings(steer='alt-s')
        live.steer = steered = Steered()
        live.feed('paste', 'steering')
        live.feed(key)
        assert steered == ['steering']
        assert live.key_label('steer') == 'Ctrl+Enter'
        # Ctrl+J steers here, so the hints name the newline binding instead.
        assert 'Shift+Enter newline' in live.input_hint()
        assert 'Ctrl+J' not in live.input_hint()


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
        'input.questions',
        'input.question_submit',
        'input.busy_delivery',
    } == rows.keys()
    assert menu.source.current(rows['send']) == 'enter'
    assert await menu.actions['send'](ScriptedDialogs([Back()])) == ()
    await menu.actions['send'](ScriptedDialogs([Pick(value='ctrl-s')]))
    assert context.settings.input_bindings.action('ctrl-s') == 'send'
    assert context.settings.input_bindings.action('enter') is None
    swapped = await menu.actions['newline'](ScriptedDialogs([Pick(value='ctrl-s')]))
    assert swapped[-1] == 'Newline is now ctrl-s; Send moved to shift-enter.'
    assert (context.settings.input_bindings.send, context.settings.input_bindings.newline) == ('shift-enter', 'ctrl-s')
    await menu.actions['send'](ScriptedDialogs([Pick(value='ctrl-s')]))
    result = await menu.actions['test'](ScriptedDialogs([]))
    assert isinstance(result, CloseMenu) and tested == [True]
    assert 'Saved' in menu.source.apply(rows['input.keyboard_labels'], 'mac')
    assert context.settings.keyboard_labels == 'mac'
    menu.source.reset(rows['send'])
    assert context.settings.input_bindings.send == 'enter'
    assert menu.source.reset(rows['test']) == 'No captured key is saved.'


async def test_choosing_a_key_another_action_holds_swaps_the_two(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    context = CommandContext(settings=Settings(model=None), store=store, apply_setting=lambda key, settings: None)
    menu = input_menu(context, start_test=lambda: None)
    # Enter for a newline and Alt+Enter to send: each step takes a key another action holds.
    assert (await menu.actions['newline'](ScriptedDialogs([Pick(value='enter')])))[-1] == (
        'Newline is now enter; Send moved to shift-enter.'
    )
    assert (await menu.actions['send'](ScriptedDialogs([Pick(value='alt-enter')])))[-1] == (
        'Send is now alt-enter; Steer moved to shift-enter.'
    )
    assert store.load().input_bindings == InputBindings(send='alt-enter', steer='shift-enter', newline='enter')
    assert context.settings.input_bindings.action('enter') == 'newline'


async def test_input_command_resets_every_input_setting_and_hands_key_tests_to_the_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = ClaiProcess.open(SettingsStore(tmp_path / 'config.db'))
    runtime, _ = runtime_over(process, tmp_path / 'project')
    chosen = {
        'input.bindings': '{"send": "ctrl-s"}',
        'input.keyboard_labels': 'mac',
        'input.question_submit': 'answer',
        'input.questions': 'composer',
        'input.busy_delivery': 'next-checkpoint',
    }
    for key, value in chosen.items():
        runtime.context.set_setting([key, value])
    assert {key for key in process.store.overrides() if key.startswith('input.')} == chosen.keys()
    assert await runtime.commands.execute_async('/input reset') == '\n'.join(f'Reset {key}. Applied.' for key in chosen)
    assert not any(key.startswith('input.') for key in process.store.overrides())
    settings = runtime.context.settings
    assert (
        settings.input_bindings,
        settings.keyboard_labels,
        settings.question_submit,
        settings.question_placement,
        settings.busy_delivery,
    ) == (InputBindings(), 'neutral', 'ask', 'picker', 'after-turn')
    with pytest.raises(ValueError, match=r'^Usage: /input \[test\|reset\]$'):
        await runtime.commands.execute_async('/input reset now')
    with pytest.raises(ValueError, match='Key testing needs the terminal composer'):
        await runtime.commands.execute_async('/input test')
    # The bare command opens setup, whose Test row hands capture to the same composer.
    monkeypatch.setattr(dialogs, 'RUNNERS', Script(lists=[pick('test')], choices=[], texts=[]).runners)
    async with editor() as (live, _, _):
        runtime.editor = live
        assert await runtime.commands.execute_async('/input test') == 'Test a key now. Ctrl+G closes.'
        assert live.key_test is not None
        live.end_calibration()
        assert await runtime.commands.execute_async('/input') == (
            'Test a key now. Ctrl+G closes; Ctrl+J is the newline fallback.'
        )
        assert live.key_test is not None


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
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))

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


async def test_a_focused_question_names_the_composer_keys_once() -> None:
    async with editor() as (live, _, _):
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            frame = Text.from_ansi('\n'.join(live.frame())).plain
            assert 'number/Enter select - Esc composer' in frame
            assert 'F2 composer - Ctrl+J newline - Ctrl+C decline' in frame
            live.feed('3')  # Other: the menu's own footer now names Ctrl+C.
            frame = Text.from_ansi('\n'.join(live.frame())).plain
            assert 'Enter submits - Esc back - Ctrl+C decline' in frame
            assert frame.count('Ctrl+C decline') == 1
            live.feed('ctrl-c')
        assert responses[0].cancelled


@pytest.mark.parametrize('preference', ['ask', 'answer'])
async def test_enter_on_a_highlighted_completion_completes_while_a_question_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preference: QuestionSubmit
) -> None:
    (tmp_path / 'notebook.md').touch()
    (tmp_path / 'notes.md').touch()
    monkeypatch.chdir(tmp_path)
    async with editor() as (live, _, _):
        live.questions.preference = lambda: preference
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            for key in 'look at @no':
                live.feed(key)
            await popup_shows(live, 'notes.md')
            live.feed('tab')
            assert highlighted(live)
            live.feed('enter')
            assert live.buffer.text in ('look at @notebook.md', 'look at @notes.md')
            assert not live.questions.routing and responses == []
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-c')
        assert responses[0].cancelled


async def test_the_hint_row_shows_after_an_input_change_or_while_input_waits() -> None:
    async with editor() as (live, _, _):

        def shown() -> str:
            return Text.from_ansi('\n'.join(live.frame())).plain

        assert 'Ctrl+J newline' not in shown(), "with no input setting changed, the frame is main's"
        live.input_changed = lambda: True
        assert 'Enter: send - F4 Delivery - Ctrl+J newline - /input' in shown()
        live.input_changed = lambda: False
        live.questions.preference = lambda: 'follow-up'
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            assert 'Enter: queue follow-up - F2 choices' in shown()
            live.feed('paste', 'next step')
            live.feed('enter')
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-c')
        assert responses[0].cancelled and live.queued_messages == ('next step',)
        assert 'Ctrl+J newline' not in shown(), 'queued entries keep main frame'
        live.edit_queue_entry(live.queued_prompts()[0])
        assert "Editing - won't send while you edit it" in shown()
        assert 'Ctrl+J newline' not in shown(), 'a held edit keeps main frame and its own row'


async def test_remembered_follow_up_queues_the_draft_without_routing() -> None:
    async with editor() as (live, _, _):
        live.questions.preference = lambda: 'follow-up'
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await live.questions.ask(question_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            assert live.input_hint().startswith('Enter: queue follow-up -')
            live.feed('paste', 'next step')
            live.feed('enter')
            assert not live.questions.routing
            assert live.queued_messages == ('next step',) and live.buffer.text == ''
            assert live.questions.menu is not None and responses == []
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-c')
        assert responses[0].cancelled
        assert live.queued_messages == ('next step',)


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
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))
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
        assert 'e Edit - d Remove - s Steer' in '\n'.join(live.frame())
        live.feed('down')
        live.feed('e')
        assert [entry.id for entry in live.runnable_prompts()] == [first.id]
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
        live.steer = steered = Steered()
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
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))
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


async def test_each_approval_shows_its_command_only_once_its_choices_are_the_ones_answered() -> None:
    shown: list[RenderableType] = []
    questions = PromptQuestions(
        paint=lambda: None, preference=lambda: 'ask', remember=lambda _: None, show=shown.append
    )
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(partial(questions.ask, question_request(), approval=True, preamble='rm -rf first'))
        tasks.start_soon(partial(questions.ask, question_request(), approval=True, preamble='rm -rf second'))
        await anyio.wait_all_tasks_blocked()
        assert shown == ['rm -rf first'], 'the second command waits until the first approval is answered'
        questions.submit(('Approve',))
        await anyio.wait_all_tasks_blocked()
        assert shown == ['rm -rf first', 'rm -rf second']
        questions.submit(None)


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
        # A late cursor-position reply is the terminal answering a query, not a key the user pressed.
        assert live.key_test is not None
        report = live.key_test.report
        live.keys.dispatch(KeyPress(Keys.CPRResponse, '\x1b[12;1R'))
        assert live.key_test.report == report
        # An unsupported CSI report reaches calibration through the decoder, without its contents.
        for char in '\x1b[999u':
            live.keys.dispatch(KeyPress(Keys.Escape if char == '\x1b' else char, char))
        assert 'Unsupported terminal key report' in live.key_test.report
        assert '999' not in live.key_test.report
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
        assert f'{enter}/Ctrl+M share' in live.key_test.report
        assert f'{alt}+{enter}' in live.key_test.report
        live.end_calibration()
        live.bindings = lambda: InputBindings(send='alt-shift-enter', steer='super-shift-enter', newline='enter')
        assert live.bindings().label('send', keyboard=live.keyboard_labels()) == f'{alt}+Shift+{enter}'
        assert f'{alt}+Shift+{enter}: send' in live.input_hint()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.questions.ask, question_request())
            await anyio.wait_all_tasks_blocked()
            assert f'{alt}+Shift+{enter}' in live.questions.hint(
                send_label=live.key_label('send'), newline_label=live.key_label('newline')
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
        assert live.runnable_prompts() == [first, third]
        frame = Text.from_ansi('\n'.join(live.frame())).plain
        assert "Editing - won't send while you edit it" in frame
        assert 'queued: 2' in frame and 'queued: 3' not in frame
        assert 'Stop editing' not in frame, 'the held edit keeps its own row without the hint row'
        live.input_changed = lambda: True
        frame = Text.from_ansi('\n'.join(live.frame())).plain
        assert 'Save/requeue' in frame and 'Stop editing' in frame
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
        frame = '\n'.join(live.frame())
        assert 'New busy: after turn - m change - Esc close' in frame and 'e Edit' not in frame
        live.feed('e')  # Nothing is queued, so nothing was ever queued to edit.
        assert live.notice == '' and 'm change' in '\n'.join(live.frame())
        live.feed('m')
        assert context.settings.busy_delivery == 'next-checkpoint'
        assert SettingsStore(store.path).load().busy_delivery == 'next-checkpoint'
        live.console.size = (40, 12)
        assert 'New busy: next checkpoint - m change' in '\n'.join(live.frame())
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
        live.steer = steered = Steered()
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
        live.steer = steered = Steered()
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
            assert steered == [] and [entry.text for entry in live.runnable_prompts()] == ['/help']
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
        # The interrupt leaves the queued entry to run next.
        assert live.queued_messages == ('during teardown',) and live.buffer.text == ''


@pytest.mark.parametrize('older_entries', [False, True])
async def test_checkpoint_delivery_and_remembered_answer_never_approve_waiting_tool(older_entries: bool) -> None:
    with no_leaked_tasks():
        async with editor() as (live, _, _):
            live.busy_delivery = lambda: 'next-checkpoint'
            live.questions.preference = lambda: 'answer'
            live.steer = steered = Steered()
            responses: list[AskUserResponse] = []

            answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))

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
        live.steer = steered = Steered()
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
            assert live.runnable_prompts() == ([first, third] if editing else [first, second, third])
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
                assert live.runnable_prompts() == ([first, third] if editing else [first, second, third])
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
            pending = live.queued_messages
            live.feed('escape')  # Stop the test's active work; the queue runs next, as after any interrupt.
        assert live.queued_messages == pending


async def test_delivery_choice_cannot_consume_paste_or_same_batch_typeahead() -> None:
    async with editor() as (live, _, _):
        live.busy_delivery = lambda: 'next-checkpoint'
        live.steer = steered = Steered()
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
            assert [entry.text for entry in live.runnable_prompts()] == ['older']
            assert await live.read() == 'older'
            received: list[str] = []

            async def read_next() -> None:
                received.append(await live.read())

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(read_next)
                await anyio.wait_all_tasks_blocked()
                assert received == [] and live.runnable_prompts() == []
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
        live.steer = steered = Steered()
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


@pytest.mark.parametrize('finish', ['enter', 'escape'])
@pytest.mark.parametrize('remove_all_predecessors', [False, True])
async def test_removing_edit_predecessors_preserves_reserved_order(finish: str, remove_all_predecessors: bool) -> None:
    async with editor() as (live, _, _):
        for text in ('first', 'second', 'third'):
            live.submit(text)
        first, second, third = live.queued_prompts()
        live.recall(backwards=True)
        live.discard(second)
        if remove_all_predecessors:
            live.discard(first)
        live.buffer.replace('edited third')
        live.feed(finish)
        expected = [third] if remove_all_predecessors else [first, third]
        assert live.queued_prompts() == expected
        assert third.text == ('edited third' if finish == 'enter' else 'third')


def test_binding_reset_collision_is_recoverable_and_preserves_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.set('input.bindings', {'send': 'ctrl-s', 'newline': 'enter'})
    context = CommandContext(settings=store.load(), store=store, apply_setting=lambda key, settings: None)
    menu = input_menu(context, start_test=lambda: None)
    row = next(row for row in menu.source.rows() if row.key == 'send')
    saved = store.overrides()
    assert 'collide' in menu.source.reset(row)
    assert store.overrides() == saved
    assert context.settings.input_bindings.send == 'ctrl-s'
    assert context.settings.input_bindings.newline == 'enter'


@pytest.mark.parametrize('boundary', ['route', 'question'])
async def test_image_answer_notice_clears_for_new_routing_or_question(boundary: str) -> None:
    async with editor() as (live, _, _):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(live.questions.ask, question_request(batch=True))
            await anyio.wait_all_tasks_blocked()
            marker = live.images.attach([BinaryContent(data=b'image', media_type='image/png')])
            live.buffer.replace('caption ' + marker)
            live.feed('enter')
            await anyio.lowlevel.checkpoint()
            live.feed('a')
            assert 'text only' in live.questions.notice
            await anyio.lowlevel.checkpoint()
            live.feed('escape')
            live.buffer.replace('text only answer')
            if boundary == 'route':
                live.feed('enter')
                assert live.questions.routing
            else:
                live.feed('f2')
                await anyio.lowlevel.checkpoint()
                live.feed('enter')
                await anyio.wait_all_tasks_blocked()
                assert live.questions.menu is not None and live.questions.menu.position == 2
            assert live.questions.notice == ''
            assert 'Queue the draft to keep its images' not in '\n'.join(live.frame())
            tasks.cancel_scope.cancel()


async def test_queue_edit_save_cannot_confirm_a_previously_focused_approval() -> None:
    async with editor() as (live, _, _):
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await answerer.ask_approval(permission_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.submit('queued')
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('f4')
            await anyio.lowlevel.checkpoint()
            live.feed('e')
            assert not live.questions.focused and not live.questions.routing
            live.buffer.replace('edited')
            live.feed('enter')
            await anyio.wait_all_tasks_blocked()
            assert responses == [] and live.queued_messages == ('edited',)
            tasks.cancel_scope.cancel()


@pytest.mark.parametrize('height', [6, 7, 8, 12, 24])
@pytest.mark.parametrize('editing', [False, True])
async def test_input_hints_fit_without_displacing_pinned_status_or_pending_work(height: int, editing: bool) -> None:
    async with editor() as (live, _, _):
        live.console.size = (100, height)
        live.pinned = lambda: 'PINNED ROW'
        live.input_changed = lambda: True

        def panel(glyph: str) -> list[str]:
            return ['pending work'] * 5

        live.panel = panel
        if editing:
            live.submit('held prompt')
            live.recall(backwards=True)
        frame = [Text.from_ansi(row).plain for row in live.frame()]
        assert len(frame) <= height - 2
        if height >= 7:
            assert any(row.startswith('─') for row in frame)
            if editing:
                assert "Editing - won't send while you edit it" in frame
        if height >= 12:
            assert frame[-2] == 'PINNED ROW'
            assert live.input_hint() in frame
        else:
            assert live.input_hint() not in frame


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


async def test_short_terminal_never_chooses_an_unseen_approval() -> None:
    async with editor() as (live, _, _):
        answerer = TerminalAnswerer(full_screen=live.suspended, console=live.console, composer=composer(live))
        responses: list[AskUserResponse] = []

        async def ask() -> None:
            responses.append(await answerer.ask_approval(permission_request()))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.wait_all_tasks_blocked()
            live.console.size = (40, 5)
            live.feed('f2')
            assert not live.questions.focused and 'Enlarge the terminal' in live.notice
            # Six rows leave the composer frame no room, so the focused approval takes the frame.
            live.console.size = (40, 6)
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            assert live.questions.focused
            assert any(row.startswith('> 1.') for row in Text.from_ansi('\n'.join(live.frame())).plain.splitlines())
            live.console.size = (40, 5)
            live.feed('enter')
            assert responses == [] and not live.questions.focused
            live.console.size = (80, 24)
            live.feed('f2')
            await anyio.lowlevel.checkpoint()
            live.feed('ctrl-c')
        assert not decision(responses[0]).approved


async def test_queue_edit_during_a_history_walk_keeps_the_walked_draft_images() -> None:
    async with editor() as (live, _, _):
        image = BinaryContent(data=b'walked image', media_type='image/png')
        marker = live.images.attach([image])
        live.submit('queued')
        (entry,) = live.queued_prompts()
        live.buffer.history.append('older prompt')
        live.buffer.insert('draft ' + marker)
        walking = live.buffer
        walking.recall(backwards=True)
        assert walking.text == 'older prompt'
        live.edit_queue_entry(entry)
        live.images.retain(live.drafts)
        assert live.images.resolve(walking.saved_draft)[1] == [image]


async def test_second_queue_edit_after_history_search_keeps_the_original_composer() -> None:
    async with editor() as (live, _, _):
        live.submit('a')
        live.submit('b')
        first, second = live.queued_prompts()
        live.buffer.history.append('alpha')
        live.buffer.insert('original')
        original = live.buffer
        live.edit_queue_entry(first)
        live.feed('ctrl-r')
        live.feed('l')  # The match is a new prompt, so the search ends the edit.
        assert live.queued_prompts() == [first, second]
        live.edit_queue_entry(second)
        live.feed('escape')
        assert live.buffer is original and live.buffer.text == 'original'
        assert live.queued_prompts() == [first, second]
