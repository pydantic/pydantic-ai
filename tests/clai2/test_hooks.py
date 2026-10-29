"""`hooks` tables run commands before and after tools and turns; a project's runs only once approved."""

import asyncio
import io
import json
import os
import signal
import sys
from pathlib import Path

import anyio
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent, ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.hooks import (
    RESULT_CHARS,
    HookOutcome,
    HooksPlugin,
    communicate,
    fill,
    summary,
)
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.hooks import HOOKS_FACTORY, PROJECT_HOOKS, HookCommand, HookEvent
from pydantic_clai2.config.project_settings import PROJECT_FILE, load_project_settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionStart, TurnEnd, TurnStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the hook commands here are POSIX shell commands')


def make_host(workspace: Path, settings: dict[str, JsonValue], output: io.StringIO) -> PluginHost[None]:
    return PluginHost(
        name='hooks',
        console=Console(file=output, width=200),
        settings=settings,
        session_id=lambda: 'session-1',
        workspace=lambda: workspace,
    )


def hooks_plugin(workspace: Path, settings: dict[str, JsonValue], output: io.StringIO) -> LoadedPlugin[None]:
    return load_plugin(HooksPlugin, make_host(workspace, settings, output))


def note_agent(plugin: LoadedPlugin[None], written: list[str]) -> Agent[None, str]:
    """An agent whose model calls `write_note(path='a', text='a')` once."""
    agent = Agent(TestModel(call_tools=['write_note']), deps_type=type(None), capabilities=list(plugin.capabilities))

    @agent.tool_plain
    def write_note(path: str, text: str) -> str:
        written.append(path)
        return f'wrote {path}'

    return agent


def tool_return(messages: list[ModelMessage]) -> object:
    return next(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'write_note'
    )


async def test_before_tool_exit_2_blocks_the_call_and_the_model_gets_stderr(tmp_path: Path) -> None:
    output = io.StringIO()
    guard = 'cat > payload.json; echo "notes are read-only" >&2; exit 2'
    plugin = hooks_plugin(
        tmp_path, {'before_tool': [{'tools': ['write_note'], 'command': ['sh', '-c', guard]}]}, output
    )
    written: list[str] = []
    result = await note_agent(plugin, written).run('write')
    assert written == [], 'the tool never ran'
    assert tool_return(result.all_messages()) == 'notes are read-only'
    assert 'before_tool hook blocked write_note: notes are read-only' in output.getvalue()
    payload = json.loads((tmp_path / 'payload.json').read_text())
    assert payload == {
        'event': 'before_tool',
        'session_id': 'session-1',
        'cwd': str(tmp_path),
        'tool_name': 'write_note',
        'args': {'path': 'a', 'text': 'a'},
    }


async def test_a_block_without_stderr_gets_a_generic_result(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'before_tool': [{'command': ['sh', '-c', 'exit 2']}]}, output)
    written: list[str] = []
    result = await note_agent(plugin, written).run('write')
    assert written == []
    assert tool_return(result.all_messages()) == 'A before_tool hook blocked write_note.'
    assert 'before_tool hook blocked write_note: sh -c exit 2' in output.getvalue()


async def test_an_interrupt_while_a_command_fails_to_start_still_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start's own failure must not replace the cancellation, or a stopped turn would carry on."""
    started, fail = anyio.Event(), anyio.Event()

    async def failing_start(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        started.set()
        await fail.wait()
        raise FileNotFoundError('missing-hook')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', failing_start)
    scopes: list[anyio.CancelScope] = []

    async def hook() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await communicate(['missing-hook'], b'', cwd=tmp_path)

    async with anyio.create_task_group() as group:
        group.start_soon(hook)
        await started.wait()
        scopes[0].cancel()
        await anyio.wait_all_tasks_blocked()
        fail.set()
    assert scopes[0].cancelled_caught


@pytest.mark.parametrize(('code', 'warned'), [(0, False), (1, True)])
async def test_before_tool_other_exits_allow_the_call(tmp_path: Path, code: int, warned: bool) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'before_tool': [{'command': ['sh', '-c', f'echo oops >&2; exit {code}']}]}, output)
    written: list[str] = []
    result = await note_agent(plugin, written).run('write')
    assert written == ['a'] and tool_return(result.all_messages()) == 'wrote a'
    assert ('before_tool hook failed' in output.getvalue()) is warned
    assert ('before_tool hook failed: sh -c echo oops >&2; exit 1: oops' in output.getvalue()) is warned


async def test_a_tool_name_filter_skips_other_tools(tmp_path: Path) -> None:
    plugin = hooks_plugin(
        tmp_path,
        {'before_tool': [{'tools': ['other_tool'], 'command': ['sh', '-c', 'touch ran; exit 2']}]},
        io.StringIO(),
    )
    written: list[str] = []
    await note_agent(plugin, written).run('write')
    assert written == ['a'] and not (tmp_path / 'ran').exists()


async def test_a_background_child_neither_delays_the_hook_nor_is_stopped(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'turn_end': [{'command': 'sleep 30 & echo $! > child', 'timeout': 20}]}, output)
    with anyio.fail_after(10):
        await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    assert output.getvalue() == ''
    child = int((tmp_path / 'child').read_text())
    os.kill(child, 0)
    os.kill(child, signal.SIGKILL)
    await gone(child)


async def test_a_timeout_stops_the_whole_command(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(
        tmp_path, {'turn_end': [{'command': 'sleep 30 & echo $! > child; wait', 'timeout': 0.5}]}, output
    )
    await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    assert 'turn_end hook timed out after 0.5s' in output.getvalue()
    await gone(int((tmp_path / 'child').read_text()))


async def test_an_interrupted_turn_stops_the_whole_command(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(
        tmp_path, {'turn_end': [{'command': 'sleep 30 & echo $! > child; wait', 'timeout': 2}]}, output
    )
    child = tmp_path / 'child'
    with anyio.CancelScope() as interrupt:
        async with anyio.create_task_group() as group:
            group.start_soon(plugin.dispatch, TurnEnd(text='hello', outcome='completed'))
            with anyio.fail_after(10):
                while not child.exists() or not child.read_text():
                    await anyio.sleep(0.01)
            interrupt.cancel()
    assert interrupt.cancelled_caught and output.getvalue() == ''
    await gone(int(child.read_text()))


async def test_stdout_is_discarded(tmp_path: Path) -> None:
    """Even more than a pipe holds: a pipe nobody read would leave the command waiting until its timeout."""
    output = io.StringIO()
    chatty: dict[str, JsonValue] = {'command': ['sh', '-c', 'echo leaked; head -c 200000 /dev/zero'], 'timeout': 20}
    plugin = hooks_plugin(tmp_path, {'turn_end': [chatty]}, output)
    with anyio.fail_after(10):
        await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    assert output.getvalue() == ''


async def gone(pid: int) -> None:
    """Wait until `pid` no longer exists; the killed child is reaped by init, which takes a moment."""
    with anyio.fail_after(5):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)


async def test_a_timeout_is_reported_and_never_blocks(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'before_tool': [{'command': ['sleep', '30'], 'timeout': 0.2}]}, output)
    written: list[str] = []
    await note_agent(plugin, written).run('write')
    assert written == ['a']
    assert 'before_tool hook timed out after 0.2s: sleep 30' in output.getvalue()


async def test_a_command_that_cannot_start_is_reported(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'after_tool': [{'command': ['./missing-hook']}]}, output)
    written: list[str] = []
    await note_agent(plugin, written).run('write')
    assert written == ['a'] and 'after_tool hook failed: ./missing-hook: ' in output.getvalue()


@pytest.mark.parametrize(('event', 'outcome'), [('before_tool', 'blocked'), ('after_tool', 'failed')])
@pytest.mark.parametrize(
    ('text', 'error'),
    [('a\x00b', 'embedded null byte'), ('x' * 2_000_000, '[Errno 7] Argument list too long')],
    ids=['nul', 'too-long'],
)
async def test_an_argument_the_os_cannot_take_keeps_a_guard_closed(
    tmp_path: Path, event: HookEvent, outcome: HookOutcome, text: str, error: str
) -> None:
    """The model chooses tool arguments, so it must not skip a `before_tool` guard with one the OS rejects."""
    output = io.StringIO()
    loaded = hooks_plugin(tmp_path, {}, output).plugin
    assert isinstance(loaded, HooksPlugin)
    command = HookCommand(command=['true', '{text}'])
    done = await loaded.run(event, command, {}, args={'text': text}, tool='write_note')
    assert done.outcome == outcome and done.stderr.startswith(f'The hook cannot take these arguments: {error}')
    assert f'The hook cannot take these arguments: {error}' in output.getvalue()
    assert len(output.getvalue()) < 2000, 'a warning shows the start of a long command'


async def test_after_tool_fills_placeholders_as_argv_and_never_blocks(tmp_path: Path) -> None:
    output = io.StringIO()
    settings: dict[str, JsonValue] = {
        'after_tool': [
            {'command': ['touch', 'formatted-{path}', '{missing}']},
            {'command': ['sh', '-c', 'cat > after.json; exit 2']},
        ]
    }
    plugin = hooks_plugin(tmp_path, settings, output)
    written: list[str] = []
    result = await note_agent(plugin, written).run('write')
    assert (tmp_path / 'formatted-a').exists() and (tmp_path / '{missing}').exists()
    assert tool_return(result.all_messages()) == 'wrote a', 'exit 2 after the call changes nothing'
    assert json.loads((tmp_path / 'after.json').read_text())['result'] == 'wrote a'
    assert 'after_tool hook failed: sh -c cat > after.json; exit 2: exit code 2' in output.getvalue()


async def test_a_string_command_runs_through_the_shell_without_placeholders(tmp_path: Path) -> None:
    plugin = hooks_plugin(tmp_path, {'after_tool': [{'command': 'echo $((1 + 1)) {path} > shell.txt'}]}, io.StringIO())
    await note_agent(plugin, []).run('write')
    assert (tmp_path / 'shell.txt').read_text() == '2 {path}\n'


async def test_turn_hooks_get_the_prompt_and_never_cancel(tmp_path: Path) -> None:
    output = io.StringIO()
    settings: dict[str, JsonValue] = {
        'turn_start': [{'command': ['sh', '-c', 'cat > start.json; exit 2']}],
        'turn_end': [{'command': 'cat > end.json'}],
    }
    plugin = hooks_plugin(tmp_path, settings, output)
    assert plugin.capabilities == (), 'turn hooks need no capability'
    start = TurnStart(text='hello')
    await plugin.dispatch(start)
    await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    assert not start.cancelled
    assert 'turn_start hook failed: sh -c cat > start.json; exit 2: exit code 2' in output.getvalue()
    assert json.loads((tmp_path / 'start.json').read_text()) == {
        'event': 'turn_start',
        'session_id': 'session-1',
        'cwd': str(tmp_path),
        'prompt': 'hello',
    }
    assert json.loads((tmp_path / 'end.json').read_text())['outcome'] == 'completed'


async def test_hook_runs_are_traced(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = hooks_plugin(tmp_path, {'turn_end': [{'command': 'true'}, {'command': 'exit 3'}]}, io.StringIO())
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    finally:
        Agent.instrument_all(False)
    ok, failed = exporter.get_finished_spans()
    assert ok.name == failed.name == 'clai hook turn_end'
    assert ok.attributes is not None and ok.attributes['clai.hook.outcome'] == 'ok'
    assert failed.attributes is not None and failed.attributes['clai.hook.outcome'] == 'failed'
    assert failed.attributes['clai.hook.command'] == 'exit 3' and failed.status.status_code == StatusCode.ERROR
    assert failed.status.description == 'turn_end hook failed: exit 3: exit code 3'


async def test_traces_leave_out_content_when_told_to(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    output = io.StringIO()
    settings: dict[str, JsonValue] = {
        'after_tool': [{'command': ['sh', '-c', 'echo "secret $0" >&2; exit 1', '{text}']}]
    }
    plugin = hooks_plugin(tmp_path, settings, output)
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider, include_content=False))
    try:
        await note_agent(plugin, []).run('write')
    finally:
        Agent.instrument_all(False)
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'clai hook after_tool']
    assert (
        span.attributes is not None
        and span.attributes['clai.hook.command'] == 'sh -c echo "secret $0" >&2; exit 1 {text}'
    )
    assert span.status.description == 'after_tool hook failed'
    assert 'after_tool hook failed: sh -c echo "secret $0" >&2; exit 1 a: secret a' in output.getvalue()


async def test_placeholders_never_choose_the_program(tmp_path: Path) -> None:
    output = io.StringIO()
    plugin = hooks_plugin(tmp_path, {'before_tool': [{'command': ['{path}', '{path}']}]}, output)
    written: list[str] = []
    await note_agent(plugin, written).run('write')
    assert written == ['a'] and 'before_tool hook failed: {path} a: ' in output.getvalue()


def test_fill_and_summary() -> None:
    args: dict[str, object] = {'path': 'a.py', 'line': 3, 'flag': True, 'none': None, 'nested': {}}
    assert fill('{path}:{line}:{flag}:{none}:{nested}', args) == 'a.py:3:True:{none}:{nested}'
    assert summary({'ok': True}) == '{"ok":true}'
    assert summary('x' * (RESULT_CHARS + 5)) == 'x' * RESULT_CHARS + '...'


def test_hooks_is_a_builtin_that_runs_nothing_until_configured() -> None:
    assert PluginSettings(id='hooks', factory=HOOKS_FACTORY) in DEFAULT_PLUGINS
    plugin = hooks_plugin(Path.cwd(), {}, io.StringIO())
    assert plugin.capabilities == ()


async def test_your_own_table_is_the_hooks_builtin_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    builtin = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'hooks']
    loader = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=builtin,
        workspace=lambda: tmp_path,
    )
    await loader.load_all()
    table = json.dumps({'turn_end': [{'command': ['touch', 'mine']}]})
    await loader.command(['add', 'hooks', HOOKS_FACTORY, table])
    await loader.fire(TurnEnd(text='hello', outcome='completed'))
    await loader.close('exit')
    assert (tmp_path / 'mine').exists()


# Project hooks


def repository(root: Path, name: str, hooks: dict[str, JsonValue] | None) -> Path:
    directory = root / name
    (directory / '.git').mkdir(parents=True)
    (directory / 'src').mkdir()
    if hooks is not None:
        write_hooks(directory, hooks)
    return directory


def write_hooks(directory: Path, hooks: dict[str, JsonValue]) -> None:
    path = directory / PROJECT_FILE
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'hooks': hooks}))


def touching(marker: str) -> dict[str, JsonValue]:
    return {'turn_end': [{'command': ['touch', marker]}]}


def project_loader(store: SettingsStore, workspace: Path, output: io.StringIO) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=output, width=200),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        project=load_project_settings(workspace).plugins,
        workspace=lambda: workspace,
    )


async def end_turn(loader: PluginLoader[None]) -> None:
    await loader.load_all()
    await loader.fire(TurnEnd(text='hello', outcome='completed'))
    await loader.close('exit')


def test_a_project_table_becomes_an_unapproved_declaration(tmp_path: Path) -> None:
    repo = repository(
        tmp_path, 'repo', {'before_tool': [{'tools': ['write_file'], 'command': 'make guard', 'timeout': 10}]}
    )
    assert load_project_settings(repo / 'src').plugins == (
        PluginSettings(
            id=PROJECT_HOOKS,
            factory=HOOKS_FACTORY,
            enabled=False,
            settings={
                'before_tool': [{'tools': ['write_file'], 'command': 'make guard', 'timeout': 10.0}],
                'project': str(repo),
            },
        ),
    )
    write_hooks(repo, {'turn_end': []})
    assert load_project_settings(repo).plugins == (), 'an empty table declares nothing'


BAD_PROJECT_FILES: list[dict[str, JsonValue]] = [
    {'hooks': {'turn_end': [{'tools': ['write_file'], 'command': 'true'}]}},
    {'hooks': {'before_tool': [{'command': []}]}},
    {'hooks': {'before_tool': [{'command': 'true', 'timeout': 0}]}},
    {'hooks': {'on_save': [{'command': 'true'}]}},
    {'hooks': {'turn_end': [{'command': 'true'}], 'project': '/elsewhere'}},
    {'plugins': [{'id': PROJECT_HOOKS, 'factory': 'anything'}]},
]


@pytest.mark.parametrize('content', BAD_PROJECT_FILES)
def test_bad_project_tables_fail_startup(tmp_path: Path, content: dict[str, JsonValue]) -> None:
    (tmp_path / '.clai').mkdir()
    (tmp_path / PROJECT_FILE).write_text(json.dumps(content))
    with pytest.raises(ValueError, match=r'settings\.json'):
        load_project_settings(tmp_path)


async def test_project_hooks_run_only_once_approved_and_again_after_a_change(tmp_path: Path) -> None:
    repo = repository(tmp_path, 'repo', touching('first'))
    store = SettingsStore(tmp_path / 'config.db')

    output = io.StringIO()
    await end_turn(project_loader(store, repo, output))
    assert not (repo / 'first').exists(), 'unapproved project hooks never run'

    loader = project_loader(store, repo, io.StringIO())
    await loader.load_all()
    assert [entry.origin for entry in loader.entries() if entry.name == PROJECT_HOOKS] == ['project']
    await loader.command(['enable', PROJECT_HOOKS])
    await loader.fire(TurnEnd(text='hello', outcome='completed'))
    await loader.close('exit')
    assert (repo / 'first').exists()

    write_hooks(repo, touching('second'))
    loader = project_loader(store, repo, io.StringIO())
    await end_turn(loader)
    assert not (repo / 'second').exists(), 'an edited table waits for approval again'
    [entry] = [entry for entry in loader.entries() if entry.name == PROJECT_HOOKS]
    assert entry.state == 'disabled' and entry.project

    loader = project_loader(store, repo, io.StringIO())
    await loader.load_all()
    await loader.command(['enable', PROJECT_HOOKS])
    await loader.fire(TurnEnd(text='hello', outcome='completed'))
    await loader.close('exit')
    assert (repo / 'second').exists()


async def test_your_own_plugin_named_project_hooks_is_never_replaced(tmp_path: Path) -> None:
    repo = repository(tmp_path, 'repo', touching('ran'))
    store = SettingsStore(tmp_path / 'config.db')
    mine = PluginSettings(id=PROJECT_HOOKS, factory='my_plugins.hooks')
    store.save_plugin(mine)
    loader = project_loader(store, repo, io.StringIO())
    [entry] = [entry for entry in loader.entries() if entry.name == PROJECT_HOOKS]
    assert entry.declaration == mine and not entry.project


async def test_an_approval_never_runs_in_another_repository(tmp_path: Path) -> None:
    """The approval is saved with your settings, which every repository shares."""
    table: dict[str, JsonValue] = {**touching('ran'), 'before_tool': [{'command': 'touch ran'}]}
    trusted = repository(tmp_path, 'trusted', table)
    other = repository(tmp_path, 'other', None)
    store = SettingsStore(tmp_path / 'config.db')
    loader = project_loader(store, trusted, io.StringIO())
    await loader.load_all()
    await loader.command(['enable', PROJECT_HOOKS])
    assert len(loader.capabilities()) == 1
    await loader.close('exit')

    loader = project_loader(store, other, io.StringIO())
    await loader.load_all()
    assert loader.capabilities() == [], 'no tool hooks either'
    await end_turn(loader)
    assert not (other / 'ran').exists() and not (trusted / 'ran').exists()


@pytest.mark.parametrize('program', ['./scripts/mark.sh', './mark.sh'])
async def test_a_project_program_path_is_relative_to_the_project_root(tmp_path: Path, program: str) -> None:
    repo = repository(tmp_path, 'repo', {'turn_end': [{'command': [program, 'marked']}]})
    script = repo / program
    script.parent.mkdir(exist_ok=True)
    script.write_text('#!/bin/sh\ntouch "$1"\n')
    script.chmod(0o755)
    [declaration] = load_project_settings(repo / 'src').plugins
    output = io.StringIO()
    plugin = hooks_plugin(repo / 'src', declaration.settings, output)
    await plugin.dispatch(TurnEnd(text='hello', outcome='completed'))
    assert (repo / 'src' / 'marked').exists(), 'the program from the root, run in the session folder'
    assert output.getvalue() == ''


async def test_removing_the_project_table_stops_an_approved_one(tmp_path: Path) -> None:
    repo = repository(tmp_path, 'repo', touching('ran'))
    store = SettingsStore(tmp_path / 'config.db')
    loader = project_loader(store, repo, io.StringIO())
    await loader.load_all()
    await loader.command(['enable', PROJECT_HOOKS])
    await loader.close('exit')

    (repo / PROJECT_FILE).write_text('{}')
    await end_turn(project_loader(store, repo, io.StringIO()))
    assert not (repo / 'ran').exists()


@pytest.mark.parametrize('factory', [HOOKS_FACTORY, f'{HOOKS_FACTORY}:HooksPlugin'])
async def test_a_redeclared_builtin_never_replaces_your_table(tmp_path: Path, factory: str) -> None:
    """A project cannot declare the hooks plugin itself, so it can neither switch yours off nor overwrite it."""
    (tmp_path / '.clai').mkdir()
    (tmp_path / PROJECT_FILE).write_text(json.dumps({'plugins': [{'id': 'hooks', 'factory': factory}]}))
    with pytest.raises(ValueError, match='declare hooks with the `hooks` key'):
        load_project_settings(tmp_path)
