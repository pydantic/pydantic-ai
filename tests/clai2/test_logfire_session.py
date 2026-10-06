"""Session roots group UI and agent traces without spreading identity tags."""

import asyncio
import io
import json
from pathlib import Path

import anyio
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import DEFAULT_PLUGINS, chat
from pydantic_clai2._app import create_shell
from pydantic_clai2.builtin_plugins import logfire_session
from pydantic_clai2.cli import headless
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, TurnEnd, TurnStart
from pydantic_clai2.runtime._session import Session, current_session_id
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from tests.clai2.test_forks import Model
from tests.clai2.test_logfire import Recorder, close, load_logfire, make_host, operation, recorder as recorder

TEAM: dict[str, JsonValue] = {'name': 'LOGFIRE_TOKEN_TEAM'}
SIGNED_IN: dict[str, JsonValue] = {'email': 'signed-in@example.com', 'token': TEAM}


@pytest.mark.parametrize('ui_events', [False, True])
@pytest.mark.parametrize('email', [None, 'developer@example.com'])
@pytest.mark.parametrize('multiple_plugins', [False, True])
async def test_session_root_groups_turns_tools_and_nested_runs(
    recorder: Recorder, ui_events: bool, email: str | None, multiple_plugins: bool
) -> None:
    signed_in: dict[str, JsonValue] = {'token': TEAM, 'account': {'email': email, 'token': TEAM} if email else None}
    previous = load_logfire(make_host(ui_events=ui_events, **signed_in)) if multiple_plugins else None
    plugin = load_logfire(
        PluginHost(
            name='observability',
            console=Console(file=io.StringIO()),
            settings={'ui_events': ui_events, **signed_in},
            session_id=lambda: 'session-123',
        )
    )
    capabilities = [*(previous.capabilities if previous else ()), *plugin.capabilities]
    agent = Agent(TestModel(), deps_type=type(None), name='parent')
    nested = Agent(TestModel(), deps_type=type(None), name='child')

    @agent.tool_plain
    async def delegate() -> str:
        result = await nested.run('nested', capabilities=capabilities)
        return result.output

    # Neither construction nor startup leaves a current span attached to the caller.
    original = trace.get_current_span()
    provider = TracerProvider(shutdown_on_exit=False)
    with provider.get_tracer('caller').start_as_current_span('unrelated caller') as caller:
        if previous is not None:
            await previous.dispatch(SessionStart(agent=agent, settings=Settings()))
        await plugin.dispatch(SessionStart(agent=agent, settings=Settings()))
        try:
            for prompt in ('first', 'second'):
                with telemetry.span('command'):
                    telemetry.record('menu choice')
                    await agent.run(prompt, capabilities=capabilities)
                await plugin.dispatch(TurnEnd(text=prompt, outcome='completed'))
            assert trace.get_current_span().get_span_context() == caller.get_span_context()
        finally:
            await close(plugin)
            if previous is not None:
                await close(previous)
    provider.shutdown()
    assert trace.get_current_span() is original

    spans = recorder.exporters[-1].get_finished_spans()
    root = next(span for span in spans if span.name == 'CLAI session')
    assert root.parent is None
    assert root.context is not None
    assert (root.attributes or {})['agent_session_id'] == 'session-123'
    assert root.instrumentation_scope is not None and root.instrumentation_scope.name == 'clai2'
    assert (root.attributes or {})['logfire.tags'] == ((email,) if email else ())
    assert (root.attributes or {}).get('user.email') == email
    assert (root.attributes or {})['reason'] == 'exit'
    children = [span for span in spans if span is not root]
    assert children
    assert all(span.context is not None and span.context.trace_id == root.context.trace_id for span in children)
    assert all(span.parent is not None for span in children)
    assert all({'logfire.tags', 'user.email'}.isdisjoint(span.attributes or {}) for span in children)
    if email:
        assert email not in json.dumps([dict(span.attributes or {}) for span in children])
    tools = [span for span in spans if operation(span) == 'execute_tool']
    nested_runs = [
        span
        for span in spans
        if (span.attributes or {}).get('gen_ai.agent.name') == 'child' and operation(span) == 'invoke_agent'
    ]
    assert len(nested_runs) == len(tools) == 2
    assert [span.parent for span in nested_runs] == [span.context for span in tools]
    ui = [
        span for span in children if span.instrumentation_scope and span.instrumentation_scope.name == telemetry.SCOPE
    ]
    assert bool(ui) is ui_events
    if ui_events:
        commands = [span for span in ui if span.name == 'command']
        choices = [span for span in ui if span.name == 'menu choice']
        assert all(span.parent == root.context for span in commands)
        assert [span.parent for span in choices] == [span.context for span in commands]
        parent_runs = [
            span
            for span in spans
            if operation(span) == 'invoke_agent' and (span.attributes or {}).get('gen_ai.agent.name') == 'parent'
        ]
        assert [span.parent for span in parent_runs] == [span.context for span in commands]


@pytest.mark.parametrize(
    ('settings', 'email'),
    [
        ({}, None),
        ({'token': TEAM, 'account': SIGNED_IN}, 'signed-in@example.com'),
        # The token changed since setup, for example by an older build sharing the settings: the email is stale.
        ({'account': SIGNED_IN}, None),
        ({'token': {'name': 'LOGFIRE_TOKEN_OTHER'}, 'account': SIGNED_IN}, None),
        ({'user_tag': 'git-email', 'token': TEAM, 'account': SIGNED_IN}, 'developer@example.com'),
        ({'user_tag': 'git-email'}, 'developer@example.com'),
        ({'user_tag': False, 'token': TEAM, 'account': SIGNED_IN}, None),
    ],
)
async def test_user_tag_chooses_the_root_email_and_only_git_email_runs_git(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, settings: dict[str, JsonValue], email: str | None
) -> None:
    looked_up = False

    async def configured_email() -> str:
        nonlocal looked_up
        looked_up = True
        return 'developer@example.com'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.logfire.git_email', configured_email)
    plugin = load_logfire(make_host(**settings))
    await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    await close(plugin)
    assert looked_up is (settings.get('user_tag') == 'git-email')
    root = next(span for span in recorder.spans() if span.name == 'CLAI session')
    assert (root.attributes or {})['logfire.tags'] == ((email,) if email else ())
    assert (root.attributes or {}).get('user.email') == email


async def test_clear_resume_and_reload_follow_saved_conversation_ids(recorder: Recorder, tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel(), deps_type=type(None)),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=io.StringIO()),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=tuple(
            plugin.model_copy(update={'settings': {'ui_events': True}})
            for plugin in DEFAULT_PLUGINS
            if plugin.id == 'observability'
        ),
    )
    original = trace.get_current_span()
    try:
        await shell.loader.load_all()
        first_id = shell.session.summary.id
        assert (await shell.run_turn(TurnStart(text='first'), headless=True)).outcome == 'completed'
        child = shell.fork_session(None, [])
        await child.prompt('fork')
        shell.session.clear()
        second_id = shell.session.summary.id
        assert (await shell.run_turn(TurnStart(text='second'), headless=True)).outcome == 'completed'
        with telemetry.span('resume command'):
            await shell.session.resume(first_id)
            telemetry.record('after resume')
        assert (await shell.run_turn(TurnStart(text='resumed'), headless=True)).outcome == 'completed'
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.reload, 'observability')
        assert (await shell.run_turn(TurnStart(text='reloaded'), headless=True)).outcome == 'completed'
    finally:
        await shell.loader.close('exit')
    assert trace.get_current_span() is original
    roots = [span for span in recorder.spans() if span.name == 'CLAI session']
    runs = [span for span in recorder.spans() if operation(span) == 'invoke_agent']
    assert len(roots) == 4
    assert len(runs) == 5
    assert first_id != second_id
    assert [(span.attributes or {})['agent_session_id'] for span in roots] == [
        first_id,
        child.summary.id,
        second_id,
        first_id,
    ]
    assert [span.parent for span in runs] == [roots[index].context for index in (0, 1, 2, 0, 3)]
    resumed = next(span for span in recorder.spans() if span.name == 'after resume')
    assert resumed.parent == roots[0].context
    assert roots[0].context != roots[3].context
    assert all(span.parent is None and span.end_time is not None for span in roots)
    assert all(exporter.closed for exporter in recorder.exporters)


@pytest.mark.parametrize(('mode', 'run_turn'), [('id', True), ('browser', True), ('headless', True), ('id', False)])
@pytest.mark.parametrize('ui_events', [False, True])
async def test_startup_resume_opens_only_the_saved_conversation_root(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, run_turn: bool, ui_events: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    agent = Agent(TestModel(call_tools=[], custom_output_text='answer'))
    saved = Session(
        agent, deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'), workspace=tmp_path
    )
    await saved.prompt('saved')

    def select(_browser: SessionBrowser) -> str:
        return saved.summary.id

    monkeypatch.setattr(SessionBrowser, 'run', select)
    store = SettingsStore(tmp_path / 'config.db')
    plugins = tuple(
        plugin.model_copy(update={'settings': {'ui_events': ui_events}})
        for plugin in DEFAULT_PLUGINS
        if plugin.id == 'observability'
    )
    if mode == 'headless':
        monkeypatch.setattr(headless, 'create_agent', lambda: agent)
        monkeypatch.setattr(headless, 'STOCK_PLUGINS', plugins)
        assert (
            await headless.run_headless(
                text='resumed',
                settings=Settings(model=None),
                store=store,
                project=ProjectSettings(),
                resume=saved.summary.id,
            )
            == 0
        )
    else:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(('resumed\n' if run_turn else '') + '/exit\n')
            await chat(
                agent,
                deps=None,
                console=Console(file=io.StringIO()),
                store=store,
                builtin_plugins=plugins,
                resume='' if mode == 'browser' else saved.summary.id,
            )
    roots = [span for span in recorder.spans() if span.name == 'CLAI session']
    assert [(span.attributes or {})['agent_session_id'] for span in roots] == [saved.summary.id]
    root_context = roots[0].context
    assert root_context is not None
    ui = [
        span
        for span in recorder.spans()
        if span is not roots[0] and span.instrumentation_scope and span.instrumentation_scope.name == telemetry.SCOPE
    ]
    assert bool(ui) is ui_events
    if ui_events:
        startup = next(span for span in ui if span.name == 'session started')
        resumed = next(span for span in ui if span.name == 'conversation resumed')
        assert startup.parent == resumed.parent == root_context
    assert all(span.context is not None and span.context.trace_id == root_context.trace_id for span in recorder.spans())


@pytest.mark.parametrize(('prompt', 'outcome'), [('hello', 'completed'), ('explode', 'failed'), ('block', 'cancelled')])
async def test_fork_turn_end_stays_under_its_saved_conversation_root(
    recorder: Recorder, tmp_path: Path, prompt: str, outcome: str
) -> None:
    model = Model()
    shell = create_shell(
        Agent(FunctionModel(stream_function=model.respond)),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=io.StringIO()),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=tuple(
            plugin.model_copy(update={'settings': {'ui_events': True}})
            for plugin in DEFAULT_PLUGINS
            if plugin.id == 'observability'
        ),
        headless=True,
    )
    try:
        await shell.loader.load_all()
        await shell.forks.start(prompt)
        (fork,) = shell.forks.records
        if outcome == 'cancelled':
            await model.started.wait()
            fork.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await fork.task
        else:
            await fork.task
        assert current_session_id() is None
    finally:
        await shell.loader.close('exit')
    roots = [span for span in recorder.spans() if span.name == 'CLAI session']
    assert len(roots) == 2
    child_root = next(span for span in roots if (span.attributes or {})['agent_session_id'] == fork.session_id)
    finished = next(span for span in recorder.spans() if span.name == 'turn {outcome}')
    assert (finished.attributes or {})['outcome'] == outcome
    assert finished.parent == child_root.context
    assert 'logfire.tags' not in (finished.attributes or {})


async def test_background_run_keeps_nested_agents_and_threaded_ui_in_its_conversation(recorder: Recorder) -> None:
    plugin = load_logfire(
        PluginHost(
            name='observability',
            console=Console(file=io.StringIO()),
            settings={'ui_events': True},
            session_id=lambda: current_session_id() or 'foreground',
        )
    )
    background = Agent(TestModel(), deps_type=type(None), name='background')
    nested = Agent(TestModel(), deps_type=type(None), name='nested')
    background_session = Session(background, deps=None, plugins=plugin.capabilities)

    @background.tool_plain
    async def delegate() -> str:
        await anyio.to_thread.run_sync(telemetry.record, 'background UI')
        return (await nested.run('nested', capabilities=plugin.capabilities)).output

    try:
        await plugin.dispatch(SessionStart(agent=background, settings=Settings()))
        await background_session.prompt('fork')
        telemetry.record('foreground UI')
    finally:
        await close(plugin)
    spans = recorder.spans()
    roots = [span for span in spans if span.name == 'CLAI session']
    assert [(span.attributes or {})['agent_session_id'] for span in roots] == [
        'foreground',
        background_session.summary.id,
    ]
    runs = [span for span in spans if operation(span) == 'invoke_agent']
    tool = next(span for span in spans if operation(span) == 'execute_tool')
    nested_run, background_run = runs
    assert nested_run.parent == tool.context
    assert background_run.parent == roots[1].context
    background_ui = next(span for span in spans if span.name == 'background UI')
    foreground_ui = next(span for span in spans if span.name == 'foreground UI')
    assert background_ui.parent == tool.context
    assert foreground_ui.parent == roots[0].context


@pytest.mark.parametrize('email', [None, '', 'developer@example.com'])
async def test_git_email_uses_git_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, email: str | None
) -> None:
    config = tmp_path / 'gitconfig'
    config.write_text(f'[user]\nemail = {email}\n' if email is not None else '')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    assert await logfire_session.git_email() == (email or None)


@pytest.mark.parametrize('unavailable', ['missing', 'timeout'])
async def test_git_email_is_optional(monkeypatch: pytest.MonkeyPatch, unavailable: str) -> None:
    async def run_process(command: list[str], *, check: bool) -> None:
        if unavailable == 'missing':
            raise FileNotFoundError('git')
        await anyio.sleep_forever()

    monkeypatch.setattr(logfire_session, 'run_process', run_process)
    if unavailable == 'timeout':

        def immediate_timeout(delay: float) -> anyio.CancelScope:
            return anyio.move_on_after(0)

        monkeypatch.setattr(logfire_session, 'move_on_after', immediate_timeout)
    assert await logfire_session.git_email() is None
