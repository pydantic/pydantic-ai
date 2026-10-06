"""UI telemetry: the shared chokepoints record what the user chose, nested, and typed text only as content."""

import io
import json
import threading
import time
from collections.abc import Generator
from functools import partial
from pathlib import Path

import anyio
import logfire
import pytest
from opentelemetry import propagate
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.api_keys import delete_key, prompt_api_key, rename_key, save_key
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.field_menu import FieldMenu, FieldRow, Runners, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker, worker_stopping
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from tests.clai2.test_plugin_loader import Harness

Recorded = tuple[str, dict[str, object]]


@pytest.fixture
def exporter(tmp_path: Path) -> Generator[InMemorySpanExporter]:
    """A local Logfire instance subscribed to UI telemetry without message content, exporting to memory."""
    yield from _subscribed(tmp_path, include_content=False)


@pytest.fixture
def content_exporter(tmp_path: Path) -> Generator[InMemorySpanExporter]:
    """The same, subscribed with message content, as the `observability` plugin does by default."""
    yield from _subscribed(tmp_path, include_content=True)


def _subscribed(tmp_path: Path, *, include_content: bool) -> Generator[InMemorySpanExporter]:
    spans = InMemorySpanExporter()
    propagator = propagate.get_global_textmap()
    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        metrics=False,
        config_dir=tmp_path / 'logfire',
        data_dir=tmp_path / 'logfire',
        additional_span_processors=[SimpleSpanProcessor(spans)],
        scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names),
        advanced=logfire.AdvancedOptions(emit_configuration_span=False),
    )
    propagate.set_global_textmap(propagator)
    unsubscribe = telemetry.subscribe(
        logfire.Logfire(config=instance.config, otel_scope=telemetry.SCOPE), include_content=include_content
    )
    try:
        yield spans
    finally:
        unsubscribe()
        unsubscribe()  # A second call is harmless.
        instance.shutdown(timeout_millis=3000)


def recorded(exporter: InMemorySpanExporter) -> list[Recorded]:
    """Each finished span or log as its message and the attributes a call site set."""
    return [(str(attributes(span)['logfire.msg']), _own(span)) for span in exporter.get_finished_spans()]


def attributes(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def _own(span: ReadableSpan) -> dict[str, object]:
    return {key: value for key, value in attributes(span).items() if not key.startswith(('logfire.', 'code.'))}


class Paint:
    title = 'Paint'

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def rows(self) -> tuple[FieldRow, ...]:
        return (
            FieldRow(key='color', description='', default='red', choices=('red', 'blue')),
            FieldRow(key='note', description='', default=''),
        )

    def current(self, row: FieldRow) -> str:
        return self.values.get(row.key, row.default)

    def problem(self, row: FieldRow, text: str) -> str | None:
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self.values[row.key] = raw
        return f'Saved {row.key}.'

    def reset(self, row: FieldRow) -> str:
        self.values.pop(row.key, None)
        return f'Reset {row.key}.'


def scripted(picks: list[str | None], choices: list[str], typed: list[str]) -> Runners:
    """Pick field rows by key (`None` closes the list), choose values, and type text, in order."""

    def run_list(menu: Menu) -> MenuResult:
        key = picks.pop(0)
        return MenuResult(cancelled=True) if key is None else MenuResult(item=MenuItem(key, value=key))

    def run_choice(menu: Menu) -> MenuResult:
        choice = choices.pop(0)
        return MenuResult(item=MenuItem(choice, value=choice))

    def run_text(widget: TextInput) -> TextInputResult:
        return TextInputResult(value=typed.pop(0))

    return Runners(run_list=run_list, run_choice=run_choice, run_text=run_text)


async def test_a_command_its_menu_and_the_fields_it_changes_nest(exporter: InMemorySpanExporter) -> None:
    source = Paint()
    runners = scripted(['color', 'note', None], ['blue'], ['my private note'])

    async def paint(args: list[str]) -> str:
        return '\n'.join(await run_worker(lambda: run_flow(FieldMenu(source), runners)))

    commands = Commands()
    commands.register(Command(name='paint', description='Paint', handler=paint))
    assert await commands.execute_async('/paint the fence') == 'Saved color.\nSaved note.'
    menu = 'tests.clai2.test_ui_telemetry:test_a_command_its_menu_and_the_fields_it_changes_nest.paint'
    assert recorded(exporter) == [
        ('Paint field color set', {'menu': 'Paint', 'field': 'color', 'choice': 'blue'}),
        ('Paint field note set', {'menu': 'Paint', 'field': 'note'}),
        (f'menu {menu}', {'menu': menu, 'result': 'list'}),
        ('command /paint', {'command': 'paint', 'arguments': 2}),
    ]
    color, note, opened, command = exporter.get_finished_spans()
    assert all(span.parent is not None for span in (color, note, opened))
    assert color.parent == note.parent == opened.context
    assert opened.parent == command.context
    assert all(
        span.instrumentation_scope and span.instrumentation_scope.name == 'clai2'
        for span in exporter.get_finished_spans()
    )
    assert all('logfire.tags' not in attributes(span) for span in exporter.get_finished_spans())
    assert 'my private note' not in json.dumps([_own(span) for span in exporter.get_finished_spans()])


async def test_reset_bare_commands_and_menu_results(exporter: InMemorySpanExporter) -> None:
    source = Paint()
    menu = FieldMenu(source)
    assert source.problem(menu.rows[0], 'red') is None
    assert menu.apply(menu.rows[0], '  ') == 'Reset color.'
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda args: 'help'))
    assert await commands.execute_async('/') == '/help: Help'
    with pytest.raises(ValueError):
        await commands.execute_async('/sk-secret-looking')
    with pytest.raises(ValueError, match='rejected'), telemetry.span('failing'):
        raise ValueError('rejected sk-secret-value')
    assert (await run_worker(lambda: MenuResult(cancelled=True))).cancelled
    assert await run_worker(partial(TextInputResult, value='typed secret')) == TextInputResult(value='typed secret')
    assert 'sk-secret' not in json.dumps([attributes(span) for span in exporter.get_finished_spans()], default=str)
    assert all(not span.events for span in exporter.get_finished_spans())
    assert recorded(exporter) == [
        ('Paint field color reset', {'menu': 'Paint', 'field': 'color'}),
        ('command /help', {'command': 'help', 'arguments': 0}),
        ('command /unknown', {'command': 'unknown', 'arguments': 0, 'error': 'ValueError'}),
        ('failing', {'error': 'ValueError'}),
        (
            'menu tests.clai2.test_ui_telemetry:test_reset_bare_commands_and_menu_results',
            {'menu': 'tests.clai2.test_ui_telemetry:test_reset_bare_commands_and_menu_results', 'cancelled': True},
        ),
        (
            'menu termflow.tui.textinput:TextInputResult',
            {'menu': 'termflow.tui.textinput:TextInputResult', 'cancelled': False},
        ),
    ]


async def test_a_menu_its_owner_cancels_says_so(exporter: InMemorySpanExporter) -> None:
    started = threading.Event()

    def wait() -> None:
        started.set()
        while not worker_stopping():
            time.sleep(0.01)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run_worker, wait)
        await anyio.to_thread.run_sync(started.wait)
        tasks.cancel_scope.cancel()
    [(message, own)] = recorded(exporter)
    assert message == 'menu tests.clai2.test_ui_telemetry:test_a_menu_its_owner_cancels_says_so.wait'
    assert own['closed_by'] == 'owner'


def test_operation_names_are_where_the_menu_was_written() -> None:
    assert telemetry.operation_name(run_flow) == 'ui.menus.field_menu:run_flow'
    assert telemetry.operation_name(partial(partial(run_flow))) == 'ui.menus.field_menu:run_flow'
    assert telemetry.operation_name(FieldMenu.build) == 'ui.menus.field_menu:FieldMenu.build'
    assert telemetry.operation_name(Paint()) == 'tests.clai2.test_ui_telemetry:Paint'
    assert telemetry.operation_name(len) == 'builtins:len'


def test_only_the_newest_subscriber_records(exporter: InMemorySpanExporter, tmp_path: Path) -> None:
    other = InMemorySpanExporter()
    propagator = propagate.get_global_textmap()
    newer = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        metrics=False,
        config_dir=tmp_path / 'newer',
        data_dir=tmp_path / 'newer',
        additional_span_processors=[SimpleSpanProcessor(other)],
        scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names),
        advanced=logfire.AdvancedOptions(emit_configuration_span=False),
    )
    propagate.set_global_textmap(propagator)
    unsubscribe = telemetry.subscribe(logfire.Logfire(config=newer.config, otel_scope=telemetry.SCOPE))
    try:
        with telemetry.span('command /{command}', command='session'):
            telemetry.record('inner')
    finally:
        unsubscribe()
        newer.shutdown(timeout_millis=3000)
    telemetry.record('after')
    assert recorded(other) == [('inner', {}), ('command /session', {'command': 'session'})]
    inner, command = other.get_finished_spans()
    assert inner.parent == command.context
    assert recorded(exporter) == [('after', {})]


def test_nothing_is_recorded_without_a_subscriber() -> None:
    telemetry.record('ignored', value=1)
    with telemetry.span('ignored') as span:
        span.set('value', 1)


async def test_settings_and_keys_record_names_not_secrets(
    exporter: InMemorySpanExporter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = CommandContext(
        settings=Settings(),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    context.set_setting(['display.thinking', 'false'])
    context.reset_setting('sessions.naming_model')
    context.set_setting(['display.spinner', 'puppy'])
    context.set_setting(['display.spinner', 'sk-pasted-secret'])
    save_key(name='OPENAI_API_KEY', value='first-secret')
    save_key(name='OPENAI_API_KEY', value='second-secret')
    rename_key(name='OPENAI_API_KEY', new_name='RENAMED')
    delete_key(name='RENAMED')

    class Typed:
        def __init__(self, value: str | None) -> None:
            self.value = value

        async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
            if self.value is None:
                raise EOFError
            return self.value

    assert await prompt_api_key(prompt=Typed('third-secret'), label='API key value') == 'third-secret'
    assert await prompt_api_key(prompt=Typed(None), label='API key value') is None
    assert recorded(exporter) == [
        ('setting display.thinking changed', {'setting': 'display.thinking', 'value': False}),
        ('setting sessions.naming_model changed', {'setting': 'sessions.naming_model', 'value': 'null'}),
        ('setting display.spinner changed', {'setting': 'display.spinner', 'value': 'puppy'}),
        ('setting display.spinner changed', {'setting': 'display.spinner', 'value': 'custom'}),
        ('key saved', {'key_name': 'OPENAI_API_KEY', 'replaced': False}),
        ('key saved', {'key_name': 'OPENAI_API_KEY', 'replaced': True}),
        ('key renamed', {'key_name': 'OPENAI_API_KEY', 'new_key_name': 'RENAMED'}),
        ('key deleted', {'key_name': 'RENAMED'}),
        ('key prompt', {'label': 'API key value', 'answer': 'typed'}),
        ('key prompt', {'label': 'API key value', 'answer': 'cancelled'}),
    ]
    assert 'secret' not in json.dumps([own for _, own in recorded(exporter)])


async def test_plugin_actions_are_recorded_as_requested(exporter: InMemorySpanExporter, tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('alpha')
    await harness.loader.load_all()
    await harness.loader.command(['disable', 'alpha'])
    await harness.loader.command(['enable', 'alpha'])
    await harness.loader.command(['reload', 'alpha'])
    await harness.loader.command(['remove', 'alpha'])
    with pytest.raises(ValueError, match='Unknown plugin'):
        await harness.loader.command(['disable', 'sk-secret-looking'])
    assert recorded(exporter) == [
        (f'plugin alpha {action}', {'plugin': 'alpha', 'action': action})
        for action in ('disable', 'enable', 'reload', 'remove')
    ]


async def test_conversations_cleared_and_resumed(exporter: InMemorySpanExporter, tmp_path: Path) -> None:
    session = Session(
        Agent(TestModel(custom_output_text='answer')),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        workspace=tmp_path,
    )
    await session.prompt('first')
    saved = session.summary.id
    session.clear()
    await session.resume(saved)
    assert recorded(exporter) == [
        ('conversation cleared', {'messages': 2}),
        ('conversation resumed', {'outcome': 'completed', 'messages': 2, 'other_workspace': False}),
    ]


async def test_prompt_submissions_interrupts_and_steering(exporter: InMemorySpanExporter) -> None:
    steered: list[str] = []
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda args: 'help'))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(5):
        live = LivePrompt(
            console=Console(file=io.StringIO(), force_terminal=True, width=80, height=24),
            commands=commands,
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Interrupts(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0,
            steer=lambda text: steered.append(text) is None,
        )
        async with live.opened():
            for text in ('a private prompt', '/help me', '/sk-secret', '!ls'):
                live.buffer.replace(text)
                live.feed('enter')
                assert await live.read() == text
            live.feed('up')
            live.feed('enter')
            assert await live.read() == '!ls'
            live.feed('ctrl-c')
            with pytest.raises(KeyboardInterrupt):
                await live.read()
            live.buffer.replace('steer this')
            live.feed('enter')
            live.feed('alt-enter')
            live.buffer.replace('steer that')
            live.feed('alt-enter')
    assert steered == ['steer this', 'steer that']
    assert recorded(exporter) == [
        ('prompt submitted', {'route': 'submitted', 'recalled': False, 'kind': 'prompt', 'chars': 16}),
        (
            'prompt submitted',
            {'route': 'submitted', 'recalled': False, 'kind': 'command', 'command': 'help', 'chars': 8},
        ),
        (
            'prompt submitted',
            {'route': 'submitted', 'recalled': False, 'kind': 'command', 'command': 'unknown', 'chars': 10},
        ),
        ('prompt submitted', {'route': 'submitted', 'recalled': False, 'kind': 'shell', 'chars': 3}),
        ('prompt submitted', {'route': 'submitted', 'recalled': True, 'kind': 'shell', 'chars': 3}),
        ('prompt interrupt', {'key': 'ctrl-c', 'cancelled_turn': False}),
        ('prompt submitted', {'route': 'submitted', 'recalled': False, 'kind': 'prompt', 'chars': 10}),
        ('prompt steer', {'steered': True, 'source': 'queue'}),
        ('prompt steer', {'steered': True, 'source': 'draft'}),
    ]


async def test_only_prompt_text_is_recorded_with_content(
    content_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With content on, prompts keep their words unscrubbed; `!` lines and command arguments stay out."""
    monkeypatch.setattr(telemetry, 'MAX_CONTENT_CHARS', 20)
    commands = Commands()
    commands.register(Command(name='plugins', description='Plugins', handler=lambda args: ''))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(5):
        live = LivePrompt(
            console=Console(file=io.StringIO(), force_terminal=True, width=80, height=24),
            commands=commands,
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Interrupts(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0,
        )
        async with live.opened():
            for text in (
                'fix the session bug',
                'a prompt longer than twenty characters',
                '!export TOKEN=sk-y',
                '/plugins add x m {"token": "sk-x"}',
            ):
                live.buffer.replace(text)
                live.feed('enter')
                assert await live.read() == text
    assert recorded(content_exporter) == [
        (
            'prompt submitted',
            {'route': 'submitted', 'recalled': False, 'kind': 'prompt', 'chars': 19, 'prompt': 'fix the session bug'},
        ),
        (
            'prompt submitted',
            {'route': 'submitted', 'recalled': False, 'kind': 'prompt', 'chars': 38, 'prompt': 'a prompt longer than'},
        ),
        ('prompt submitted', {'route': 'submitted', 'recalled': False, 'kind': 'shell', 'chars': 18}),
        (
            'prompt submitted',
            {'route': 'submitted', 'recalled': False, 'kind': 'command', 'command': 'plugins', 'chars': 34},
        ),
    ]
    assert 'sk-' not in json.dumps([own for _, own in recorded(content_exporter)])
