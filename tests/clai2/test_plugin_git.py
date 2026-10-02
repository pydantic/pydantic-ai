"""Git installations use real local repositories, without network access or package installers."""

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import anyio
import pytest

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins._git import parse_repository
from pydantic_clai2.plugins.loader import PluginError

from .test_plugin_loader import Harness

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
        ('http://example.com/my.plugin/', 'http://example.com/my.plugin/', 'my_plugin'),
        ('ssh://git@example.com:2222/team/plugin.git', 'ssh://git@example.com:2222/team/plugin.git', 'plugin'),
        ('git@example.com:team/plugin.git', 'git@example.com:team/plugin.git', 'plugin'),
        ('example.com:team/plugin', 'example.com:team/plugin', 'plugin'),
        ('git+https://example.com/plugin.git', 'https://example.com/plugin.git', 'plugin'),
        ('git+ssh://git@example.com/plugin.git', 'ssh://git@example.com/plugin.git', 'plugin'),
        ('git://example.com/plugin.git', 'git://example.com/plugin.git', 'plugin'),
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

    async def interrupted_clone(
        command: Sequence[str], *, env: Mapping[str, str], start_new_session: bool, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[:8] == [
            'git',
            '-c',
            'credential.interactive=false',
            'clone',
            '--depth',
            '1',
            '--',
            'https://example.com/plugin.git',
        ]
        assert env['GIT_TERMINAL_PROMPT'] == '0'
        assert start_new_session is True
        assert check is False
        (Path(command[-1]) / '__init__.py').write_text('raise AssertionError("Not installed yet")')
        assert harness.loader.entries() == []
        cloning.set()
        if cancel:
            await anyio.sleep_forever()
        raise TimeoutError

    monkeypatch.setattr('pydantic_clai2.plugins._git.run_process', interrupted_clone)
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
