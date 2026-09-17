"""Loading, unloading, reloading, and dispatching between turns."""

import io
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart, TurnEnd, TurnStart
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


RECORDER = """
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, TurnEnd, TurnStart

LOADS = globals().get('LOADS', 0) + 1


def activate(host: PluginHost) -> None:
    host.commands.register(Command(name='{command}', description='From plugin', handler=lambda _: 'ok'))

    @host.on('session_start')
    async def started(event: SessionStart) -> None:
        host.console.print('{name} started')

    @host.on('session_end')
    async def stopped(event: SessionEnd) -> None:
        host.console.print('{name} stopped ' + event.reason)
        {end_body}

    @host.on('turn_start')
    async def before(event: TurnStart) -> None:
        {start_body}

    @host.on('turn_end')
    async def after(event: TurnEnd) -> None:
        {turn_end_body}
"""


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        builtin: tuple[PluginSettings, ...] = (),
        project: tuple[PluginSettings, ...] = (),
    ) -> None:
        self.store = SettingsStore(tmp_path / 'config.db')
        self.store.plugins_dir.mkdir(exist_ok=True)
        self.output = io.StringIO()
        self.commands = Commands()
        self.commands.register(Command(name='help', description='Built in', handler=lambda _: ''))
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=builtin,
            project=project,
        )

    def write(
        self,
        name: str,
        *,
        command: str | None = None,
        end_body: str = 'pass',
        start_body: str = 'pass',
        turn_end_body: str = 'pass',
    ) -> Path:
        path = self.store.plugins_dir / f'{name}.py'
        path.write_text(
            RECORDER.format(
                name=name,
                command=command or name,
                end_body=end_body,
                start_body=start_body,
                turn_end_body=turn_end_body,
            )
        )
        return path

    @property
    def text(self) -> str:
        return self.output.getvalue()


async def test_folder_discovery_and_load_order(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('beta')
    harness.write('alpha')
    (harness.store.plugins_dir / '_private.py').write_text('raise AssertionError')
    (harness.store.plugins_dir / 'not-a-name.py').write_text('raise AssertionError')
    (harness.store.plugins_dir / 'pkg').mkdir()
    (harness.store.plugins_dir / 'pkg' / '__init__.py').write_text(
        'from pydantic_clai2.plugins import PluginHost\ndef activate(host: PluginHost) -> None:\n    pass\n'
    )
    (harness.store.plugins_dir / 'empty').mkdir()
    await harness.loader.load_all()
    assert [entry.name for entry in harness.loader.entries()] == ['alpha', 'beta', 'pkg']
    assert all(entry.host is not None for entry in harness.loader.entries())
    assert harness.text.index('alpha started') < harness.text.index('beta started')
    assert {command.name for command in harness.commands} == {'help', 'alpha', 'beta'}
    await harness.loader.close('eof')
    assert harness.text.index('beta stopped eof') < harness.text.index('alpha stopped eof')
    assert {command.name for command in harness.commands} == {'help'}
    assert harness.loader.capabilities() == []


async def test_declared_capability_class_and_activate_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / 'site' / 'clai_extras'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'caps.py').write_text(
        'from pydantic_ai.capabilities import Capability\n'
        'from pydantic_clai2.plugins import PluginHost\n'
        'class Greeter(Capability[None]):\n'
        "    def __init__(self, *, greeting: str = 'hi') -> None:\n"
        '        super().__init__(instructions=greeting)\n'
        'def install(host: PluginHost) -> None:\n'
        "    host.add(Greeter(greeting='yo'))\n"
        'NOT_A_PLUGIN = 3\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))  # pyright: ignore[reportUnknownMemberType]
    harness = Harness(tmp_path)
    assert (
        await harness.loader.command(['add', 'greeter', 'clai_extras.caps:Greeter', '{"greeting": "hello"}'])
        == 'Added and loaded greeter.'
    )
    assert (
        await harness.loader.command(['add', 'installer', 'clai_extras.caps:install']) == 'Added and loaded installer.'
    )
    assert len(harness.loader.capabilities()) == 2
    with pytest.raises(PluginError, match='no callable'):
        await harness.loader.command(['add', 'number', 'clai_extras.caps:NOT_A_PLUGIN'])
    with pytest.raises(PluginError, match='not a capability class'):
        await harness.loader.command(['add', 'wrong', 'pathlib:Path'])
    with pytest.raises(PluginError, match="no callable 'activate'"):
        await harness.loader.command(['add', 'noactivate', 'clai_extras'])
    with pytest.raises(PluginError, match='ModuleNotFoundError'):
        await harness.loader.command(['add', 'missing', 'clai_extras.nope'])
    listing = await harness.loader.command(['list'])
    assert 'greeter: clai_extras.caps:Greeter (enabled, loaded)' in listing
    assert 'missing: clai_extras.nope (enabled, failed: ModuleNotFoundError' in listing
    await harness.loader.command(['reload', 'greeter'])
    assert len(harness.loader.capabilities()) == 2
    assert await harness.loader.command(['remove', 'greeter']) == 'Removed greeter.'
    assert len(harness.loader.capabilities()) == 1


async def test_failed_load_leaves_nothing_registered(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('clash', command='help')
    harness.write('explode', end_body='pass', start_body='pass')
    (harness.store.plugins_dir / 'explode.py').write_text(
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import PluginHost, SessionStart\n'
        'def activate(host: PluginHost) -> None:\n'
        "    host.commands.register(Command(name='boom', description='x', handler=lambda _: ''))\n"
        "    @host.on('session_start')\n"
        '    async def started(event: SessionStart) -> None:\n'
        "        raise RuntimeError('start failed')\n"
    )
    (harness.store.plugins_dir / 'syntax.py').write_text('def activate(host):\n    return (\n')
    await harness.loader.load_all()
    assert {command.name for command in harness.commands} == {'help'}
    states = {entry.name: entry.state for entry in harness.loader.entries()}
    assert states['clash'].startswith('enabled, failed: ValueError')
    assert states['explode'] == 'enabled, failed: RuntimeError: start failed'
    assert states['syntax'].startswith('enabled, failed: SyntaxError')
    assert "Plugin 'clash'" in harness.text and "Plugin 'syntax'" in harness.text
    assert harness.loader.capabilities() == []
    with pytest.raises(PluginError, match='duplicate command: help'):
        await harness.loader.load('clash')


async def test_enable_disable_reload_persist_and_refresh_module(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('counter', end_body='pass')
    await harness.loader.load_all()
    await harness.loader.load('counter')
    assert await harness.loader.command(['disable', 'counter']) == 'Disabled counter.'
    assert 'counter stopped exit' in harness.text
    assert harness.loader.capabilities() == []
    assert [plugin.enabled for plugin in harness.store.plugins()] == [False]
    assert 'counter' not in {command.name for command in harness.commands}
    fresh = PluginLoader[None](
        store=harness.store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=harness.store.load()),
    )
    await fresh.load_all()
    assert fresh.entries()[0].state == 'disabled'
    assert await harness.loader.command(['enable', 'counter']) == 'Enabled counter.'
    assert harness.store.plugins()[0].enabled
    assert await harness.loader.command(['reload', 'counter']) == 'Reloaded counter.'
    assert harness.text.count('counter started') == 3
    message = await harness.loader.command(['remove', 'counter'])
    assert message.startswith('Disabled counter. Delete ')
    assert not harness.store.plugins()[0].enabled
    assert harness.loader.entries()[0].state == 'disabled'


async def test_fire_reports_observers_and_fails_closed_on_turn_start(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('rude', turn_end_body="raise RuntimeError('late')")
    harness.write('polite', start_body="event.text += '!'")
    await harness.loader.load_all()
    start = TurnStart(text='hi')
    await harness.loader.fire(start)
    assert start.text == 'hi!'
    await harness.loader.fire(TurnEnd(text='hi', outcome='completed'))
    assert "Plugin 'rude': RuntimeError: late" in harness.text
    harness.write('rude', start_body="raise RuntimeError('no')", end_body="raise RuntimeError('bye')")
    await harness.loader.reload('rude')
    with pytest.raises(PluginError, match="Plugin 'rude': RuntimeError: no"):
        await harness.loader.fire(TurnStart(text='again'))
    await harness.loader.unload('rude')
    assert "Plugin 'rude': RuntimeError: bye" in harness.text
    await harness.loader.unload('rude')


async def test_command_usage_and_unknown_names(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    assert await harness.loader.command([]) == 'No plugins.'
    with pytest.raises(ValueError, match='Usage'):
        await harness.loader.command(['enable'])
    with pytest.raises(ValueError, match='Unknown plugins action'):
        await harness.loader.command(['frobnicate', 'x'])
    with pytest.raises(ValueError, match='Unknown plugin: ghost'):
        await harness.loader.command(['enable', 'ghost'])
    with pytest.raises(ValueError, match='Usage'):
        await harness.loader.command(['add', 'only-name'])


async def test_loaded_entry_survives_file_removal_until_unloaded(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    path = harness.write('ephemeral')
    await harness.loader.load_all()
    path.unlink()
    entry = harness.loader.entries()[0]
    assert entry.name == 'ephemeral' and entry.host is not None
    await harness.loader.unload('ephemeral')
    assert harness.loader.entries() == []
    harness.store.save_plugin(PluginSettings(id='ephemeral', factory='ephemeral'))
    assert harness.loader.entries()[0].source == 'ephemeral'
    assert harness.loader.plugins_dir == harness.store.plugins_dir


BUILTIN = (
    PluginSettings(id='hello', factory='pydantic_ai.capabilities:Capability', settings={'instructions': 'Say hello.'}),
)


async def test_builtin_plugins_are_on_by_default_and_overridable(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert entry.name == 'hello' and entry.builtin
    assert entry.source == 'pydantic_ai.capabilities:Capability (built-in)'
    assert entry.state == 'enabled, loaded'
    assert len(harness.loader.capabilities()) == 1
    assert harness.store.plugins() == []

    assert await harness.loader.command(['disable', 'hello']) == 'Disabled hello.'
    assert harness.loader.capabilities() == []
    fresh = Harness(tmp_path, builtin=BUILTIN)
    await fresh.loader.load_all()
    assert fresh.loader.entries()[0].state == 'disabled'
    assert fresh.loader.entries()[0].builtin

    message = await fresh.loader.command(['remove', 'hello'])
    assert message == 'hello is built in; restored its defaults. Use /plugins disable hello to turn it off.'
    assert fresh.store.plugins() == []
    assert fresh.loader.entries()[0].state == 'enabled, loaded'


async def test_store_declaration_replaces_a_builtin(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    harness.write('hello')
    harness.store.save_plugin(
        PluginSettings(id='hello', factory='hello', path=str(harness.store.plugins_dir / 'hello.py'))
    )
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert not entry.builtin and entry.path is not None
    assert 'hello' in {command.name for command in harness.commands}
    assert (await harness.loader.command(['remove', 'hello'])).startswith('Disabled hello.')


async def test_add_replaces_a_builtin_and_remove_restores_it(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    await harness.loader.load_all()
    message = await harness.loader.command(
        ['add', 'hello', 'pydantic_ai.capabilities:Capability', '{"instructions": "Say goodbye."}']
    )
    assert message == 'Replaced built-in hello.'
    entry = harness.loader.entries()[0]
    assert not entry.builtin and entry.state == 'enabled, loaded'
    assert harness.store.plugins()[0].settings == {'instructions': 'Say goodbye.'}
    assert len(harness.loader.capabilities()) == 1

    message = await harness.loader.command(['remove', 'hello'])
    assert message == 'hello is built in; restored its defaults. Use /plugins disable hello to turn it off.'
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.state == 'enabled, loaded'
    assert harness.store.plugins() == []
    replace = ['add', 'hello', 'pydantic_ai.capabilities:Capability']
    assert await harness.loader.command(replace) == 'Replaced built-in hello.'
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(replace)


PROJECT = (
    PluginSettings(
        id='hello', factory='pydantic_ai.capabilities:Capability', enabled=False, settings={'instructions': 'Project.'}
    ),
)
"""As `load_project_settings` hands them over: declared by the repository, off until the user approves."""


async def test_project_declarations_sit_above_builtins_and_below_the_store(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN, project=PROJECT)
    await harness.loader.load_all()
    hello = harness.loader.entries()[0]
    assert hello.project and not hello.builtin and hello.declaration.settings == {'instructions': 'Project.'}
    assert hello.source == 'pydantic_ai.capabilities:Capability (project)'
    assert hello.state == 'disabled', 'repository code does not run until the user approves it'
    assert harness.loader.capabilities() == [] and harness.store.plugins() == []

    assert await harness.loader.command(['enable', 'hello']) == 'Enabled hello.'
    assert len(harness.loader.capabilities()) == 1
    fresh = Harness(tmp_path, builtin=BUILTIN, project=PROJECT)
    await fresh.loader.load_all()
    entry = fresh.loader.entries()[0]
    assert entry.project and entry.state == 'enabled, loaded', 'approval is remembered in the user store'

    message = await fresh.loader.command(['remove', 'hello'])
    assert (
        message == 'hello is declared by the project; restored its defaults. Use /plugins disable hello to turn it off.'
    )
    assert fresh.loader.entries()[0].state == 'disabled' and fresh.store.plugins() == []

    replace = ['add', 'hello', 'pydantic_ai.capabilities:Capability', '{"instructions": "Mine."}']
    assert await fresh.loader.command(replace) == 'Replaced project hello.'
    assert not fresh.loader.entries()[0].project and fresh.store.plugins()[0].settings == {'instructions': 'Mine.'}
    with pytest.raises(ValueError, match='already exists'):
        await fresh.loader.command(replace)


async def test_remove_restores_a_project_plugin_that_names_a_file(tmp_path: Path) -> None:
    path = tmp_path / 'repo' / 'filed.py'
    path.parent.mkdir()
    path.write_text(
        'from pydantic_clai2.plugins import PluginHost\ndef activate(host: PluginHost) -> None:\n    pass\n'
    )
    project = (PluginSettings(id='filed', factory='filed', path=str(path), enabled=False),)
    harness = Harness(tmp_path, project=project)
    await harness.loader.load_all()
    assert harness.loader.entries()[0].project and harness.loader.entries()[0].path == path
    assert await harness.loader.command(['enable', 'filed']) == 'Enabled filed.'

    message = await harness.loader.command(['remove', 'filed'])
    assert message.startswith('filed is declared by the project; restored its defaults.')
    assert harness.store.plugins() == [] and harness.loader.entries()[0].state == 'disabled'


async def test_repo_context_builtin_loads_the_workspace_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'AGENTS.md').write_text('Answer in haiku.\n')
    monkeypatch.chdir(workspace)
    repo_context = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'repo_context')
    harness = Harness(tmp_path, builtin=(repo_context,))
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.state == 'enabled, loaded'
    model = TestModel(call_tools=[])
    await Agent(model, deps_type=type(None), capabilities=harness.loader.capabilities()).run('hi')
    assert model.last_model_request_parameters is not None
    parts = model.last_model_request_parameters.instruction_parts or []
    assert sum('Answer in haiku.' in part.content for part in parts) == 1
    assert model.last_model_request_parameters.function_tools == []

    assert await harness.loader.command(['disable', 'repo_context']) == 'Disabled repo_context.'
    assert harness.loader.capabilities() == []

    knobs = ['add', 'repo_context', 'pydantic_clai2.repo_context', '{"inventory_tool": true, "walk_up": true}']
    assert await harness.loader.command(knobs) == 'Replaced built-in repo_context.'
    model = TestModel(call_tools=[])
    await Agent(model, deps_type=type(None), capabilities=harness.loader.capabilities()).run('hi')
    assert model.last_model_request_parameters is not None
    assert [tool.name for tool in model.last_model_request_parameters.function_tools] == ['inventory_agent_context']
    assert (await harness.loader.command(['remove', 'repo_context'])).startswith('repo_context is built in')
    with pytest.raises(PluginError, match='extra_forbidden'):
        await harness.loader.command(['add', 'repo_context', 'pydantic_clai2.repo_context', '{"filenames": []}'])
