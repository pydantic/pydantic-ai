"""Git installations use real local repositories, without network access or package installers."""

import os
import shlex
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Generator
from contextlib import closing, contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import anyio
import pytest

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.plugin_requirements import Requirements
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins._git import clone_repository, parse_repository
from pydantic_clai2.plugins.loader import PluginError

from .test_plugin_loader import Harness

READINESS_WAIT_TIMEOUT = 30

PLUGIN = """\
from collections.abc import Sequence

from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import NoSettings, Plugin

from .greeting import MESSAGE


class Greeting(Plugin[NoSettings, None]):
    def get_commands(self) -> Sequence[Command]:
        return [Command(name='git_hello', description='From Git', handler=lambda _: MESSAGE)]

    async def configure(self) -> str:
        return 'Configured greeting.'
"""


def git(repository: Path, *args: str) -> None:
    subprocess.run(
        [
            'git',
            '-C',
            str(repository),
            '-c',
            'user.name=Plugin Test',
            '-c',
            'user.email=plugin@example.com',
            '-c',
            'commit.gpgsign=false',
            '-c',
            'core.hooksPath=/dev/null',
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / 'demo-plugin.git'
    repository.mkdir()
    git(repository, 'init')
    (repository / '__init__.py').write_text(PLUGIN)
    (repository / 'greeting.py').write_text("MESSAGE = 'Hello from Git'\n")
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Create plugin')
    return repository


@pytest.mark.parametrize('entrypoint', ['__init__.py', 'plugin.py'])
async def test_add_git_plugin_lifecycle(tmp_path: Path, repository: Path, entrypoint: str) -> None:
    if entrypoint != '__init__.py':
        (repository / '__init__.py').rename(repository / entrypoint)
        git(repository, 'add', '.')
        git(repository, 'commit', '-m', 'Use a module entry point')
    harness = Harness(tmp_path)
    assert await harness.loader.command(['add', repository.as_uri()]) == (
        'Added and loaded demo_plugin.\nConfigured greeting.'
    )
    assert await harness.commands.execute_async('/git_hello') == 'Hello from Git'
    checkout = harness.store.plugins_dir / '_git' / 'demo_plugin'
    saved = harness.store.plugins()[0]
    assert saved == PluginSettings(id='demo_plugin', factory='demo_plugin', path=str(checkout / entrypoint))
    assert (checkout / '.git').is_dir()
    assert len(harness.loader.entries()) == 1
    await harness.loader.close('exit')

    fresh = Harness(tmp_path)
    await fresh.loader.load_all()
    assert await fresh.commands.execute_async('/git_hello') == 'Hello from Git'
    assert await fresh.loader.command(['disable', 'demo_plugin']) == 'Disabled demo_plugin.'
    assert 'git_hello' not in fresh.commands
    assert await fresh.loader.command(['enable', 'demo_plugin']) == 'Enabled demo_plugin.\nConfigured greeting.'
    (checkout / entrypoint).write_text(PLUGIN.replace('MESSAGE)]', "'Reloaded')]"))
    assert await fresh.loader.command(['reload', 'demo_plugin']) == 'Reloaded demo_plugin.'
    assert await fresh.commands.execute_async('/git_hello') == 'Reloaded'
    assert (await fresh.loader.command(['remove', 'demo_plugin'])).startswith('Disabled demo_plugin.')
    assert fresh.store.plugins()[0].enabled is False
    assert (checkout / entrypoint).exists()
    assert 'git_hello' not in fresh.commands
    await fresh.loader.close('exit')
    restarted = Harness(tmp_path)
    await restarted.loader.load_all()
    assert 'git_hello' not in restarted.commands


@pytest.mark.parametrize(
    ('source', 'url', 'name'),
    [
        ('https://example.com/team/my-plugin.git', 'https://example.com/team/my-plugin.git', 'my_plugin'),
        ('https://example.com/my.plugin/', 'https://example.com/my.plugin/', 'my_plugin'),
        ('ssh://git@example.com:2222/team/plugin.git', 'ssh://git@example.com:2222/team/plugin.git', 'plugin'),
        ('git@example.com:team/plugin.git', 'git@example.com:team/plugin.git', 'plugin'),
        ('example.com:team/plugin', 'example.com:team/plugin', 'plugin'),
        ('git+https://example.com/plugin.git', 'https://example.com/plugin.git', 'plugin'),
        ('git+ssh://git@example.com/plugin.git', 'ssh://git@example.com/plugin.git', 'plugin'),
        ('file:///tmp/plugin.git', 'file:///tmp/plugin.git', 'plugin'),
        ('https://example.com/logfire.git', 'https://example.com/logfire.git', 'observability'),
    ],
)
def test_repository_url(source: str, url: str, name: str) -> None:
    assert parse_repository(source) == (url, name)


@pytest.mark.parametrize(
    'source',
    [
        'only-name',
        '-x',
        'ext::command',
        'ftp://example.com/plugin.git',
        'http://example.com/plugin.git',
        'git://example.com/plugin.git',
        'https:///plugin.git',
        'https://example.com/plugin.git?query=1',
        'https://example.com/plugin.git#main',
        'https://example.com/',
        'https://example.com/..',
        'https://example.com/_private.git',
        'https://example.com/123.git',
        'https://example.com/bad%20name.git',
    ],
)
async def test_rejected_url_has_no_side_effects(tmp_path: Path, source: str) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(ValueError):
        await harness.loader.command(['add', source])
    assert harness.store.plugins() == []
    assert list(harness.store.plugins_dir.iterdir()) == []


async def test_add_usage_mentions_git_urls(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match='Usage: /plugins add GIT_URL'):
        await harness.loader.command(['add'])
    assert harness.store.plugins() == []


@pytest.mark.parametrize('origin', ['builtin', 'project', 'saved', 'dropin'])
async def test_existing_plugin_is_not_replaced(tmp_path: Path, repository: Path, origin: str) -> None:
    declaration = PluginSettings(id='demo_plugin', factory='existing_plugin')
    harness = Harness(
        tmp_path,
        builtin=(declaration,) if origin == 'builtin' else (),
        project=(declaration,) if origin == 'project' else (),
    )
    if origin == 'saved':
        harness.store.save_plugin(declaration)
    if origin == 'dropin':
        harness.write('demo_plugin')
    before = harness.store.plugins()
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(['add', repository.as_uri()])
    assert harness.store.plugins() == before
    assert not (harness.store.plugins_dir / '_git').exists()


@pytest.mark.parametrize('kind', ['directory', 'file', 'symlink'])
async def test_existing_checkout_is_not_changed(tmp_path: Path, repository: Path, kind: str) -> None:
    harness = Harness(tmp_path)
    destination = harness.store.plugins_dir / '_git' / 'demo_plugin'
    destination.parent.mkdir()
    if kind == 'directory':
        destination.mkdir()
    elif kind == 'file':
        destination.write_text('preserve me')
    else:
        destination.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='checkout already exists'):
        await harness.loader.command(['add', repository.as_uri()])
    assert destination.exists() or destination.is_symlink()
    if kind == 'file':
        assert destination.read_text() == 'preserve me'
    assert harness.store.plugins() == []


@pytest.mark.parametrize('entrypoint', ['missing', 'symlink'])
async def test_invalid_repository_is_cleaned_up(tmp_path: Path, repository: Path, entrypoint: str) -> None:
    (repository / '__init__.py').unlink()
    if entrypoint == 'symlink':
        (repository / '__init__.py').symlink_to('greeting.py')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Invalid plugin entry point')
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match='must contain a regular'):
        await harness.loader.command(['add', repository.as_uri()])
    assert harness.loader.entries() == []
    assert harness.store.plugins() == []
    assert list((harness.store.plugins_dir / '_git').iterdir()) == []


async def test_failed_clone_can_be_retried(tmp_path: Path, repository: Path) -> None:
    harness = Harness(tmp_path)
    missing = tmp_path / 'missing' / 'demo-plugin.git'
    with pytest.raises(ValueError, match='Could not clone'):
        await harness.loader.command(['add', missing.as_uri()])
    assert harness.store.plugins() == []
    assert list((harness.store.plugins_dir / '_git').iterdir()) == []
    await harness.loader.command(['add', repository.as_uri()])
    assert 'git_hello' in harness.commands
    await harness.loader.close('exit')


async def test_missing_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path)
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(ValueError, match='Git is required'):
        await harness.loader.command(['add', 'https://example.com/plugin.git'])
    assert harness.store.plugins() == []
    assert list((harness.store.plugins_dir / '_git').iterdir()) == []


@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_clone_is_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    harness = Harness(tmp_path)
    cloning = anyio.Event()

    async def interrupted_clone(url: str, destination: Path) -> tuple[int, bytes]:
        assert url == 'https://example.com/plugin.git'
        (destination / '__init__.py').write_text('raise AssertionError("Not installed yet")')
        assert harness.loader.entries() == []
        cloning.set()
        if cancel:
            await anyio.sleep_forever()
        raise TimeoutError

    monkeypatch.setattr('pydantic_clai2.plugins._git.clone_repository', interrupted_clone)
    if cancel:
        async with anyio.create_task_group() as group:
            group.start_soon(harness.loader.command, ['add', 'https://example.com/plugin.git'])
            await cloning.wait()
            group.cancel_scope.cancel()
    else:
        with pytest.raises(ValueError, match='timed out'):
            await harness.loader.command(['add', 'https://example.com/plugin.git'])
    assert harness.store.plugins() == []
    assert list((harness.store.plugins_dir / '_git').iterdir()) == []


@pytest.mark.parametrize('origin', ['saved', 'dropin'])
async def test_plugin_created_while_cloning_is_preserved(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    harness = Harness(tmp_path)
    other = SettingsStore(harness.store.path)
    existing = PluginSettings(id='demo_plugin', factory='another_plugin', settings={'keep': True})

    async def concurrent_clone(url: str, destination: Path) -> tuple[int, bytes]:
        result = await clone_repository(url, destination)
        if origin == 'saved':
            other.save_plugin(existing)
        else:
            harness.write('demo_plugin')
        return result

    monkeypatch.setattr('pydantic_clai2.plugins._git.clone_repository', concurrent_clone)
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(['add', repository.as_uri()])
    assert not (harness.store.plugins_dir / '_git' / 'demo_plugin').exists()
    if origin == 'saved':
        assert harness.store.plugins() == [existing]
    else:
        assert (harness.store.plugins_dir / 'demo_plugin.py').is_file()
        assert harness.store.plugins() == []


async def test_declaration_created_between_recheck_and_save_is_preserved(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path)
    other = SettingsStore(harness.store.path)
    existing = PluginSettings(id='demo_plugin', factory='another_plugin', settings={'keep': True})
    save = harness.store.save_plugin

    def concurrent_save(
        plugin: PluginSettings, *, requires: Requirements | None = None, overwrite: bool = True
    ) -> None:
        assert overwrite is False
        other.save_plugin(existing, requires={'keep': frozenset({'future-feature'})})
        save(plugin, requires=requires, overwrite=overwrite)

    monkeypatch.setattr(harness.store, 'save_plugin', concurrent_save)
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(['add', repository.as_uri()])
    assert harness.store.plugins() == [existing]
    assert harness.store.plugin_requirements('demo_plugin') == {'keep': ['future-feature']}
    assert not (harness.store.plugins_dir / '_git' / 'demo_plugin').exists()


@pytest.mark.parametrize('saved_id', ['logfire', 'observability'])
@pytest.mark.parametrize('new_id', ['logfire', 'observability'])
def test_exclusive_save_preserves_legacy_aliases(tmp_path: Path, saved_id: str, new_id: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    existing = PluginSettings(id=saved_id, factory='original', settings={'keep': True})
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO plugins VALUES (?, ?)', (saved_id, existing.model_dump_json()))
    with pytest.raises(ValueError, match='already exists'):
        store.save_plugin(PluginSettings(id=new_id, factory='replacement'), overwrite=False)
    assert store.plugins() == [existing.model_copy(update={'id': 'observability'})]


@pytest.mark.skipif(sys.platform == 'win32', reason='uses an SSH helper with POSIX process groups and a Unix socket')
@pytest.mark.parametrize('interrupt', ['cancel', 'timeout'])
async def test_clone_terminates_git_and_ssh_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str
) -> None:
    harness = Harness(tmp_path)
    scopes: list[anyio.CancelScope] = []

    @contextmanager
    def controlled_timeout(delay: float) -> Generator[anyio.CancelScope, None, None]:
        assert delay == 120
        with anyio.fail_after(None) as scope:
            scopes.append(scope)
            yield scope

    async def install() -> None:
        if interrupt == 'timeout':
            with pytest.raises(ValueError, match='timed out'):
                await harness.loader.command(['add', 'ssh://example.invalid/demo-plugin.git'])
        else:
            await harness.loader.command(['add', 'ssh://example.invalid/demo-plugin.git'])

    monkeypatch.setattr('pydantic_clai2.plugins._git.fail_after', controlled_timeout)
    git_pid: int | None = None
    ssh_pid: int | None = None
    try:
        # The shorter path fits macOS's Unix socket path limit, unlike pytest's per-test directory.
        with TemporaryDirectory(prefix='clai-git-') as directory:
            socket_path = str(Path(directory) / 'ready')
            monkeypatch.setenv(
                'GIT_SSH_COMMAND',
                shlex.join([sys.executable, str(Path(__file__).with_name('plugin_git_ssh.py')), socket_path]),
            )
            monkeypatch.setenv('GIT_SSH_VARIANT', 'ssh')
            listener = await anyio.create_unix_listener(socket_path)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with listener, anyio.create_task_group() as group:
                    group.start_soon(install)
                    async with await listener.accept() as ready:
                        data = bytearray()
                        async for chunk in ready:
                            data.extend(chunk)
                    git_pid, ssh_pid = map(int, data.decode().split())
                    if interrupt == 'cancel':
                        group.cancel_scope.cancel()
                    else:
                        scopes[0].deadline = anyio.current_time()
            assert git_pid is not None and ssh_pid is not None
            states = subprocess.run(
                ['ps', '-o', 'stat=', '-p', str(git_pid), '-p', str(ssh_pid)], capture_output=True, text=True
            )
            assert states.returncode in (0, 1)
            # A helper reparented to init may briefly be a zombie, but it must no longer be running.
            assert all(state.strip().startswith('Z') for state in states.stdout.splitlines())
    finally:
        if git_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(git_pid, signal.SIGKILL)
    assert harness.store.plugins() == []
    assert not (harness.store.plugins_dir / '_git' / 'demo_plugin').exists()


async def test_import_failure_keeps_checkout_for_repair(tmp_path: Path, repository: Path) -> None:
    (repository / '__init__.py').write_text('raise RuntimeError("broken plugin")')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Broken plugin')
    harness = Harness(tmp_path)
    with pytest.raises(PluginError, match='broken plugin'):
        await harness.loader.command(['add', repository.as_uri()])
    assert harness.loader.entries()[0].state == 'enabled, failed: RuntimeError: broken plugin'
    assert harness.store.plugins()[0].id == 'demo_plugin'
    entry = harness.store.plugins_dir / '_git' / 'demo_plugin' / '__init__.py'
    entry.write_text(PLUGIN)
    await harness.loader.command(['enable', 'demo_plugin'])
    assert await harness.commands.execute_async('/git_hello') == 'Hello from Git'
    await harness.loader.close('exit')
