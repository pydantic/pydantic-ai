"""The composer's `@` picker: a session folder's files, matched as the user types after `@`."""

import subprocess
from pathlib import Path

import pytest
from anyio.from_thread import start_blocking_portal

from pydantic_clai2.builtin_plugins.web import files
from pydantic_clai2.builtin_plugins.web.files import WorkspaceFiles, best, list_paths
from pydantic_clai2.builtin_plugins.web.protocol import CompletionInfo

from .test_web_sessions import Fleet


def write(folder: Path, *names: str) -> None:
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def test_a_session_route_finds_files_in_its_git_folder(tmp_path: Path) -> None:
    with start_blocking_portal() as portal:
        fleet = Fleet(root=tmp_path, portal=portal, agents={})
        try:
            check_routes(fleet)
        finally:
            fleet.exit()


def check_routes(fleet: Fleet) -> None:
    repo = fleet.folder('repo')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    write(repo, 'README.md', 'src/package/module.py', 'src/other.py', 'build.log', '.gitignore')
    (repo / '.gitignore').write_text('*.log\n')
    conversation = fleet.new('repo')

    found = fleet.get(f'/api/sessions/{conversation}/files?q=READ').json()
    assert found[0] == {'text': '@README.md', 'start': 5, 'display': 'README.md', 'meta': None}
    nested = [CompletionInfo(**entry) for entry in fleet.get(f'/api/sessions/{conversation}/files?q=src/pa').json()]
    assert [entry.text for entry in nested] == ['@src/package/', '@src/package/module.py']
    assert (nested[0].display, nested[0].meta, nested[1].display, nested[1].meta) == (
        'package/',
        'src',
        'module.py',
        'src/package',
    )
    assert fleet.get(f'/api/sessions/{conversation}/files?q=build').json() == [], '.gitignore applies'
    assert fleet.get(f'/api/sessions/{conversation}/files?q=nope').json() == []
    assert fleet.get('/api/sessions/nope/files?q=READ').status_code == 404


def test_best_ranks_names_then_paths_then_scattered_letters() -> None:
    paths = (
        'docs/readme-extra.md',
        'README.md',
        'notes/old_readme.txt',
        'readers/a.py',
        'src/reader.py',
        'r/e/a/d.py',
        'unrelated.py',
        'readers/',
        'docs/',
    )
    assert best(paths, 'READ') == [
        'readers/',
        'README.md',
        'readers/a.py',
        'src/reader.py',
        'docs/readme-extra.md',
        'notes/old_readme.txt',
        'unrelated.py',
        'r/e/a/d.py',
    ]
    assert best(paths, 'readers/') == ['readers/a.py'], 'a folder typed in full offers what it holds'
    assert best(paths, '')[:4] == ['docs/', 'readers/', 'README.md', 'unrelated.py'], 'the top level first'
    assert best(tuple(f'file{n}.py' for n in range(100)), 'file') == [f'file{n}.py' for n in range(50)]


async def test_a_folder_outside_git_is_walked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))  # Even inside a checkout.
    write(tmp_path, 'README.md', '.env', 'src/app.py', '.git/config', '.venv/lib.py', 'web/node_modules/x.js')
    assert await list_paths(tmp_path) == ('.env', 'README.md', 'src/app.py', 'src/')
    monkeypatch.setenv('PATH', '')  # No Git at all.
    assert await list_paths(tmp_path) == ('.env', 'README.md', 'src/app.py', 'src/')
    monkeypatch.setattr(files, 'FILE_LIMIT', 2)
    assert await list_paths(tmp_path) == ('.env', 'README.md')


async def test_a_listing_is_reused_until_it_goes_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finder = WorkspaceFiles()
    write(tmp_path, 'first.py')
    assert [entry.text for entry in await finder.find(tmp_path, 'py')] == ['@first.py']
    write(tmp_path, 'second.py')
    assert [entry.text for entry in await finder.find(tmp_path, 'py')] == ['@first.py']
    monkeypatch.setattr(files, 'FRESH_FOR', -1)
    assert [entry.text for entry in await finder.find(tmp_path, 'py')] == ['@first.py', '@second.py']
