"""Main-footer layout, settings actions and conversation-owned text time."""

import asyncio
import io
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from inspect import signature
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.config import (
    DEFAULT_FOOTER_FIELDS,
    FOOTER_FIELDS,
    FooterField,
    PermissionMode,
    PluginSettings,
    Settings,
)
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import Plugin, SessionStart, TurnStart
from pydantic_ai_pax.runtime.session_runtime import create_shell
from pydantic_ai_pax.ui.menus.footer_menu import FOOTER_SETTING, FooterSource, footer_settings_menu
from pydantic_ai_pax.ui.menus.set_menu import settings_menu
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.status import TEXT_TIME, Status, StatusLine
from tests.pax.dialog_script import Back, Pick, ScriptedDialogs
from tests.pax.test_live_prompt import editor

REFERENCE = Path(__file__).parents[2] / 'src' / 'pydantic_ai_pax' / 'docs' / 'reference.md'


def test_reference_lists_the_default_footer_order() -> None:
    """Both places the reference page states the default footer layout match the shipped default."""
    reference = REFERENCE.read_text(encoding='utf-8')
    default_block = reference.split('The default order is:\n\n```json\n', 1)[1].split('\n```', 1)[0]
    assert json.loads(default_block) == list(DEFAULT_FOOTER_FIELDS)
    settings_row = next(line for line in reference.splitlines() if line.startswith('| `display.footer_fields` |'))
    assert json.loads(settings_row.split('`')[3]) == list(DEFAULT_FOOTER_FIELDS)


def test_default_footer_order_and_field_styles_survive_reordering() -> None:
    status = Status(
        model='main',
        workspace='/work',
        context_tokens=100,
        context_window=1000,
        context_alert=True,
        output_tokens=42,
        cost=Decimal('0.01'),
        activity='running: shell',
        status_segments=(lambda: 'plugin',),
        updates=lambda: 'update',
        live=lambda: '2 live',
        queued=lambda: 3,
    )
    for name in ('_text_response', '_streamed_text'):
        assert name not in signature(Status).parameters
        assert name not in repr(status)
    assert status.text() == (
        'main | /work | context: 100/1k tokens | 42 output tokens | $0.0100 | running: shell'
        ' | plugin | update | 2 live | queued: 3'
    )
    expected_styles = dict((text, role) for role, text in status.toolbar() if text != ' | ')
    visible: list[FooterField] = []
    status.fields = lambda: visible
    for field in FOOTER_FIELDS:
        visible[:] = [field]
        for role, text in status.toolbar():
            assert expected_styles[text] == role
    status.fields = lambda: ['activity', 'output', 'context', 'model']
    assert status.text() == 'running: shell | 42 output tokens | context: 100/1k tokens | main'
    assert (theme.WARNING, '100/1k') in status.toolbar()
    assert (theme.THINKING, 'shell') in status.toolbar()
    assert (theme.LITHIUM, '42') in status.toolbar()


async def test_conditional_fields_empty_layout_and_narrow_terminal() -> None:
    status = Status(fields=lambda: ['cost', 'workspace', 'plugins', 'updates', 'live', 'queued', 'assistant_age'])
    assert status.text() == ''
    status.fields = lambda: []
    assert status.text() == '' and status.toolbar() == []
    status.fields = lambda: ['activity', 'model']
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=8, height=3)
    async with StatusLine(console, status):
        pass
    assert 'ready' in Text.from_ansi(stream.getvalue()).plain
    assert 'agent default' not in Text.from_ansi(stream.getvalue()).plain


def test_the_footer_shows_the_permission_mode_it_reads_on_each_paint() -> None:
    mode: list[PermissionMode] = ['full']
    shown: list[FooterField] = ['model', 'workspace', 'permissions', 'activity']
    status = Status(fields=lambda: shown, model='opus', workspace='/work', permissions=lambda: mode[0])
    assert status.text() == 'opus | /work | full access | ready'
    assert (theme.WARNING, 'full access') in status.toolbar()
    mode[0] = 'auto'
    assert status.text() == 'opus | /work | auto mode | ready'
    assert (theme.MUTED, 'auto mode') in status.toolbar()
    mode[0] = 'read-only'
    assert status.text() == 'opus | /work | read-only | ready'
    # Without a mode, as for a host that has none, the field is hidden.
    assert Status(fields=lambda: shown, model='opus', workspace='/work').text() == 'opus | /work | ready'
    # A narrow row hides it after the plugins and before the context labels.
    shown[:] = ['model', 'workspace', 'permissions', 'context', 'activity', 'plugins']
    status.status_segments = (lambda: 'docker sandbox',)
    status.context_tokens, status.context_window = 61_000, 1_000_000
    assert ''.join(text for _, text in status.toolbar(width=60)) == 'opus | read-only | context: 61k/1m tokens | ready'
    assert ''.join(text for _, text in status.toolbar(width=45)) == 'opus | context: 61k/1m tokens | ready'


DEEP = '/home/dev/work/clients/acme/services/billing/jsonrc'
WIDE = '/srv/プロジェクト/日本語のリポジトリ'
REST = ' | running: shell | docker sandbox'


@pytest.mark.parametrize(
    ('workspace', 'width', 'expected'),
    [
        (
            DEEP,
            260,
            'opus | …rk/clients/acme/services/billing/jsonrc | context: 61k/1m tokens | 1,804 output tokens'
            f' | $0.0123{REST} | update ready | live: 2 | queued: 1',
        ),
        (
            DEEP,
            174,
            f'opus | …/jsonrc | context: 61k/1m tokens | 1,804 output tokens | $0.0123{REST} | update ready | live: 2'
            ' | queued: 1',
        ),
        (DEEP, 142, f'opus | …/jsonrc | context: 61k/1m tokens | $0.0123{REST} | update ready | live: 2 | queued: 1'),
        (DEEP, 120, f'opus | …/jsonrc | context: 61k/1m tokens{REST} | update ready | live: 2 | queued: 1'),
        (DEEP, 110, f'opus | …/jsonrc | context: 61k/1m tokens{REST} | update ready | live: 2'),
        (DEEP, 98, f'opus | …/jsonrc | context: 61k/1m tokens{REST} | live: 2'),
        (DEEP, 83, f'opus | …/jsonrc | context: 61k/1m tokens{REST}'),
        (DEEP, 73, f'opus | context: 61k/1m tokens{REST}'),
        (DEEP, 62, 'opus | context: 61k/1m tokens | running: shell'),
        (DEEP, 45, 'opus | 61k/1m | running: shell'),
        (DEEP, 29, '61k/1m | running: shell'),
        (WIDE, 120, f'opus | …/日本語のリポジトリ | context: 61k/1m tokens{REST} | update ready | live: 2'),
    ],
)
def test_a_narrow_footer_gives_way_by_priority_in_the_configured_order(
    workspace: str, width: int, expected: str
) -> None:
    status = Status(
        model='opus',
        workspace=workspace,
        context_tokens=61_000,
        context_window=1_000_000,
        output_tokens=1804,
        cost=Decimal('0.0123'),
        activity='running: shell',
        status_segments=(lambda: 'docker sandbox',),
        updates=lambda: 'update ready',
        live=lambda: 'live: 2',
        queued=lambda: 1,
    )
    assert ''.join(text for _, text in status.toolbar(width=width)) == expected
    assert cell_len(expected) <= width
    # The figure keeps its warning colour once its labels are gone.
    status.context_alert = True
    assert (theme.WARNING, '61k/1m') in status.toolbar(width=24)


@pytest.mark.parametrize('fields', [['bogus'], ['model', 'model'], 'model', [1]])
def test_invalid_layout_is_rejected(fields: object) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({'footer_fields': fields})


async def test_layout_menu_persists_visibility_order_and_layout_only_reset(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    context = CommandContext(settings=Settings(), store=store, apply_setting=lambda key, settings: None)
    context.set_setting(['display.thinking', 'false'])
    menu = settings_menu(context).menus[FOOTER_SETTING]()
    assert any(row.label == 'Reset layout' for row in menu.source.rows())
    assert menu.source.current(menu.source.rows()[-1]) == ''
    assert menu.source.current(menu.source.rows()[-2]) == 'hidden'
    # Show age, move it earlier, hide another field, then reorder while that field is hidden.
    for name, action in [
        ('assistant_age', 'toggle'),
        ('assistant_age', 'earlier'),
        ('model', 'toggle'),
        ('model', 'toggle'),
        ('model', 'later'),
    ]:
        await menu.actions[name](ScriptedDialogs([Pick(value=action)]))
    assert context.settings.footer_fields[0] == 'workspace'
    assert context.settings.footer_fields[1] == 'model'
    assert context.settings.footer_fields[-2:] == ['assistant_age', 'queued']
    store = SettingsStore(tmp_path / 'settings.db')
    assert store.load().footer_fields == context.settings.footer_fields
    menu = footer_settings_menu(context)
    assert await menu.actions['model'](ScriptedDialogs([Back()])) == []
    assert await menu.actions['workspace'](ScriptedDialogs([Pick(value='earlier')])) == []
    assert await menu.actions['queued'](ScriptedDialogs([Pick(value='later')])) == []
    await menu.actions['reset_layout'](ScriptedDialogs([]))
    assert context.settings.footer_fields == list(DEFAULT_FOOTER_FIELDS)
    assert context.settings.thinking is False
    assert FOOTER_SETTING not in context.store.overrides()
    first = menu.source.rows()[0]
    assert menu.source.problem(first, 'shown') is None
    assert menu.source.problem(first, 'wrong') == 'Choose shown or hidden.'
    with pytest.raises(ValueError, match='Choose shown or hidden'):
        menu.source.apply(first, 'wrong')
    menu.source.apply(first, 'hidden')
    menu.source.reset(first)
    assert first.key in context.settings.footer_fields


@pytest.mark.parametrize('visible', [False, True])
async def test_editor_queue_respects_layout_without_duplicate_counts(visible: bool) -> None:
    status = Status(fields=lambda: ['queued'] if visible else [])
    async with editor(toolbar=status.toolbar) as (live, _, _):
        status.queued = lambda: len(live.queued_messages)
        live.queue_in_toolbar = lambda: visible
        live.submit('queued user input')
        footer = Text.from_ansi(live.frame()[-1]).plain
        assert footer.count('queued: 1') == int(visible)
        live.notice = 'A notice replaces the toolbar ' * 10
        footer = Text.from_ansi(live.frame()[-1]).plain
        assert footer.startswith('A notice replaces the toolbar')
        # A notice wider than the row is cut before the queue count, not the count itself.
        assert footer.endswith(' | queued: 1') is visible
        assert footer.count('queued: 1') == int(visible)
        assert status.last_text_at is None


@pytest.mark.parametrize(
    ('saved', 'expected'),
    [(142, 142), (142.5, 142.5), (None, 100), (True, 100), ('142', 100), (float('nan'), 100), (float('inf'), 100)],
)
def test_text_age_uses_only_finite_numeric_local_metadata(saved: object, expected: float) -> None:
    response = ModelResponse(
        parts=[TextPart('saved text')],
        timestamp=datetime.fromtimestamp(100, UTC),
        metadata={TEXT_TIME: saved},
    )
    status = Status()
    status.restore_text_age([response])
    assert status.last_text_at == expected


def test_only_nonempty_assistant_text_refreshes_age() -> None:
    now = 100.0
    status = Status(fields=lambda: ['assistant_age'], clock=lambda: now)
    assert status.text() == ''
    excluded = [
        PartStartEvent(index=0, part=TextPart('')),
        PartDeltaEvent(index=0, delta=TextPartDelta('')),
        PartStartEvent(index=0, part=ThinkingPart('thinking')),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        PartStartEvent(index=0, part=ToolCallPart('shell', {})),
        FunctionToolCallEvent(ToolCallPart('shell', {}, 'call')),
        FunctionToolResultEvent(ToolReturnPart('shell', 'done', 'call')),
    ]
    for event in excluded:
        status.observe(event)
        assert status.last_text_at is None
    status.observe(PartStartEvent(index=0, part=TextPart('hello')))
    assert status.text() == 'last text: 0s'
    now = 142
    for event in excluded:
        status.observe(event)
    assert status.text() == 'last text: 42s'
    status.observe(PartDeltaEvent(index=0, delta=TextPartDelta(' world')))
    assert status.last_text_at == 142
    child = Status(last_text_at=999, model='child')
    status.subagent = lambda: child
    now = 150
    assert status.text() == 'last text: 8s'
    now = 100
    assert status.text() == 'last text: 0s'


def test_text_age_restores_rewinds_switches_and_keeps_live_delta_time() -> None:
    first = ModelResponse(parts=[TextPart('first')], timestamp=datetime.fromtimestamp(100, UTC))
    second = ModelResponse(parts=[TextPart('second')], timestamp=datetime.fromtimestamp(200, UTC))
    history: list[ModelMessage] = [
        first,
        second,
        ModelResponse(parts=[ThinkingPart('thought')], timestamp=datetime.fromtimestamp(250, UTC)),
        ModelRequest(parts=[UserPromptPart('queued', timestamp=datetime.fromtimestamp(299, UTC))]),
    ]
    status = Status(fields=lambda: ['assistant_age'], clock=lambda: 300)
    status.restore_text_age(history)
    assert status.text() == 'last text: 1m 40s'
    status.restore_text_age([first], keep_live=False)
    assert status.text() == 'last text: 3m 20s'
    status.clear_conversation()
    assert status.text() == ''
    status.restore_text_age(history)
    assert status.last_text_at == 200
    status.observe(PartDeltaEvent(index=0, delta=TextPartDelta('live')))
    third = ModelResponse(parts=[TextPart('live')], timestamp=datetime.fromtimestamp(290, UTC))
    status.restore_text_age([*history, third])
    assert status.last_text_at == 300
    status.restore_text_age([*history, third])
    assert status.last_text_at == 300
    status.restore_text_age([])
    assert status.text() == ''


async def test_shell_saves_last_streamed_text_time_and_restores_it_on_resume(tmp_path: Path) -> None:
    now = 100.0

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal now
        yield 'hello '
        now += 42
        yield 'world'

    shell = create_shell(
        Agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        load_plugins=False,
    )
    shell.status.clock = lambda: now
    shell.context.set_setting([FOOTER_SETTING, '["assistant_age"]'])
    try:
        result = await shell.run_turn(TurnStart(text='hello'))
        assert result.outcome == 'completed'
        assert shell.status.last_text_at == 142
        response = shell.session.messages[-1]
        assert isinstance(response, ModelResponse)
        assert response.metadata is not None and response.metadata[TEXT_TIME] == 142
        conversation_id = shell.session.conversation_id
        await shell.session.clear()
        assert shell.status.text() == ''
        now = 200
        await shell.session.resume(conversation_id)
        assert shell.status.text() == 'last text: 58s'
        shell.show_model()
        assert shell.status.text() == 'last text: 58s'
        await shell.sessions.start(resume=conversation_id, fork=True)
        assert shell.status.text() == 'last text: 58s'
        retained = list(shell.session.messages)
        result = await shell.run_turn(TurnStart(text='another turn'))
        assert result.outcome == 'completed'
        assert shell.status.last_text_at == 242
        shell.show_model()
        await shell.session.commit_messages(retained)
        shell.show_model()
        assert shell.status.last_text_at == 142
        await shell.session.resume(conversation_id)
        assert shell.status.last_text_at == 142
    finally:
        await shell.close('exit')


class PreferencesProbe(Plugin):
    async def on_session_start(self, event: SessionStart) -> None:
        if 'footer_fields' in event.settings.model_fields_set:
            return
        menu = self.host.command_menu('set')
        assert menu is not None
        row = next(row for row in menu.source.rows() if row.key == FOOTER_SETTING)
        candidate = json.dumps([*DEFAULT_FOOTER_FIELDS, 'assistant_age'])
        assert menu.source.problem(row, candidate) is None
        await anyio.to_thread.run_sync(menu.source.apply, row, candidate)
        assert self.host.setting(FOOTER_SETTING) == [*DEFAULT_FOOTER_FIELDS, 'assistant_age']


@pytest.mark.parametrize('explicit', [False, True])
async def test_public_startup_menu_can_activate_age_without_replacing_explicit_layout(
    tmp_path: Path, explicit: bool
) -> None:
    settings = Settings(model=None, footer_fields=['activity']) if explicit else Settings(model=None)
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=[],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=settings,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[PluginSettings(id='preferences_probe', factory=f'{__name__}:PreferencesProbe')],
        project=ProjectSettings(),
        headless=True,
    )
    try:
        await shell.loader.load_all()
        assert shell.context.settings.footer_fields == (
            ['activity'] if explicit else [*DEFAULT_FOOTER_FIELDS, 'assistant_age']
        )
        assert (FOOTER_SETTING in shell.context.store.overrides()) is (not explicit)
    finally:
        await shell.close('exit')


@pytest.mark.parametrize('enabled', [False, True])
async def test_child_text_never_refreshes_the_main_clock_or_borrows_it(tmp_path: Path, enabled: bool) -> None:
    now = 100.0
    child_started, release = asyncio.Event(), asyncio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        nonlocal now
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            now = 150
            yield 'child text'
            child_started.set()
            await release.wait()
        elif len(messages) == 1:
            yield 'main text'
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"self","task":"child"}')}
        else:
            now = 200
            yield 'done'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None, footer_fields=['model', 'assistant_age'] if enabled else ['model']),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        load_plugins=False,
    )
    shell.status.clock = lambda: now
    try:
        async with shell.tasks.router.opened(view=shell.tasks), asyncio.TaskGroup() as group:
            shell.session.delegations = shell.tasks.owner
            group.create_task(shell.run_turn(TurnStart(text='parent')))
            await asyncio.wait_for(child_started.wait(), timeout=10)
            assert shell.status.last_text_at == 100
            assert shell.status.subagent() is not None
            assert ('last text: 50s' in shell.status.text()) is enabled
            release.set()
        assert shell.status.last_text_at == 200
        record = shell.tasks.records()[0]
        history = await anyio.to_thread.run_sync(shell.tasks.history, record.id)
        assert all(
            TEXT_TIME not in (message.metadata or {}) for message in history if isinstance(message, ModelResponse)
        )
    finally:
        shell.session.delegations = None
        await shell.close('exit')


async def test_visible_moves_skip_hidden_neighbours_without_enabling_them(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    context = CommandContext(settings=Settings(), store=store, apply_setting=lambda key, settings: None)
    source = FooterSource(context)
    for name in ('workspace', 'permissions', 'context'):
        await source.configure(name, ScriptedDialogs([Pick(value='toggle')]))
    visible = set(context.settings.footer_fields)
    assert await source.configure('context', ScriptedDialogs([Pick(value='earlier')])) == [
        'Show this field before moving it.'
    ]
    await source.configure('output', ScriptedDialogs([Pick(value='earlier')]))
    assert context.settings.footer_fields[:2] == ['output', 'model']
    assert set(context.settings.footer_fields) == visible
    await source.configure('output', ScriptedDialogs([Pick(value='later')]))
    assert context.settings.footer_fields[:2] == ['model', 'output']
    assert set(context.settings.footer_fields) == visible
    assert store.load().footer_fields == context.settings.footer_fields
    await source.reset_layout(ScriptedDialogs([]))
    assert [row.key for row in source.rows()][:-1] == [
        *DEFAULT_FOOTER_FIELDS,
        *(name for name in FOOTER_FIELDS if name not in DEFAULT_FOOTER_FIELDS),
    ]


@pytest.mark.parametrize('initial_enabled', [False, True])
async def test_observer_identity_and_metadata_follow_live_opt_in_across_turns_and_conversations(
    tmp_path: Path, initial_enabled: bool
) -> None:
    class Metadata(AbstractCapability[None]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            response.metadata = {'unrelated': {'keep': True}, TEXT_TIME: 25}
            return response

    now = 100.0

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello '
        yield 'world'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Metadata()],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None, footer_fields=['assistant_age'] if initial_enabled else []),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        load_plugins=False,
    )
    shell.status.clock = lambda: now
    try:
        bound = None
        for enabled in (initial_enabled, True, False, True):
            shell.context.set_setting([FOOTER_SETTING, '["assistant_age"]' if enabled else '[]'])
            result = await shell.run_turn(TurnStart(text='hello'))
            assert result.outcome == 'completed', result.error
            if bound is None:
                bound = shell.session.agent
            assert shell.session.agent is bound
            response = shell.session.messages[-1]
            assert isinstance(response, ModelResponse)
            assert response.metadata == {'unrelated': {'keep': True}, TEXT_TIME: now if enabled else 25}
            now += 50
        conversation = shell.session.conversation_id
        retained = list(shell.session.messages)
        await shell.session.clear()
        assert shell.status.last_text_at is None
        await shell.session.resume(conversation)
        assert shell.status.last_text_at == now - 50
        assert shell.session.messages == retained
    finally:
        await shell.close('exit')


async def test_default_off_turn_does_not_add_text_time_to_history(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel(custom_output_text='hello')),
        deps=None,
        plugins=[],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        load_plugins=False,
    )
    try:
        for _ in range(2):
            result = await shell.run_turn(TurnStart(text='hello'))
            assert result.outcome == 'completed', result.error
        responses = [message for message in shell.session.messages if isinstance(message, ModelResponse)]
        assert all(TEXT_TIME not in (response.metadata or {}) for response in responses)
        conversation = shell.session.conversation_id
        await shell.session.clear()
        await shell.session.resume(conversation)
        assert shell.status.last_text_at == responses[-1].timestamp.timestamp()
        shell.context.set_setting([FOOTER_SETTING, '["assistant_age"]'])
        assert 'last text:' in shell.status.text()
        assert all(TEXT_TIME not in (response.metadata or {}) for response in responses)
    finally:
        await shell.close('exit')
