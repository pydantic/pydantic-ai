"""Working-directory arguments are deprecated and ignored: the working directory belongs to the workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset
from pydantic_ai_harness.macroscope import Macroscope, MacroscopeToolset
from pydantic_ai_harness.repo_context import RepoContext, RepoContextToolset
from pydantic_ai_harness.shell import Shell, ShellToolset

from ._tool_calls import call_tool

_ON_THE_WORKSPACE = r"set it on the workspace, e\.g\. `LocalWorkspace\('\./repo'\)`"


def _coder(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Coder[None](elsewhere)


def _file_system(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return FileSystem[None](cwd=elsewhere)


def _shell(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Shell[None](cwd=elsewhere)


def _macroscope(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return Macroscope[None](cwd=elsewhere, command=str(fake_cli))


def _repo_context(elsewhere: str, fake_cli: Path) -> AbstractCapability[None]:
    return RepoContext[None](workspace_dir=Path(elsewhere))


@pytest.mark.parametrize(
    ('build', 'warning', 'tool', 'arguments', 'expected'),
    [
        (
            _coder,
            r"`Coder\(workspace=\.\.\.\)` is deprecated and ignored: .* attach `LocalWorkspace\('elsewhere'\)`",
            'shell',
            {'command': 'ls marker.txt'},
            'marker.txt',
        ),
        (
            _file_system,
            rf'`FileSystem\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'read_file',
            {'path': 'marker.txt'},
            'in the working directory',
        ),
        (
            _shell,
            rf'`Shell\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'run_command',
            {'command': 'ls marker.txt'},
            'marker.txt',
        ),
        (
            _macroscope,
            rf'`Macroscope\(cwd=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'run_macroscope_review',
            {},
            "review_id='marker.txt'",
        ),
        (
            _repo_context,
            rf'`RepoContext\(workspace_dir=\.\.\.\)` is deprecated and ignored: .*{_ON_THE_WORKSPACE}',
            'inventory_agent_context',
            {},
            "root='.claude', exists=True",
        ),
    ],
    ids=['Coder', 'FileSystem', 'Shell', 'Macroscope', 'RepoContext'],
)
async def test_working_directory_argument_warns_and_has_no_effect(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    build: Callable[[str, Path], AbstractCapability[None]],
    warning: str,
    tool: str,
    arguments: dict[str, object],
    expected: str,
) -> None:
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / 'marker.txt').write_text('in the working directory\n')
    (tmp_path / '.claude').mkdir()
    fake_cli = tmp_path_factory.mktemp('bin') / 'macroscope'
    fake_cli.write_text('#!/bin/sh\nprintf \'review_id=%s\\n\' "$(ls marker.txt)" >&2\n')
    fake_cli.chmod(0o755)

    with pytest.warns(HarnessDeprecationWarning, match=warning):
        capability = build('elsewhere', fake_cli)

    assert expected in await call_tool([capability], tool, arguments, workspace=LocalWorkspaceBackend(tmp_path))


@pytest.mark.parametrize(
    ('build', 'warning'),
    [
        (
            lambda: FileSystemToolset[None](
                cwd=Path('elsewhere'),
                allowed_patterns=[],
                denied_patterns=[],
                max_read_lines=10,
                max_list_results=10,
                max_search_results=10,
                max_find_results=10,
            ),
            r'`FileSystemToolset\(cwd=\.\.\.\)` is deprecated and ignored',
        ),
        (
            lambda: ShellToolset[None](
                cwd=Path('elsewhere'),
                allowed_commands=[],
                denied_commands=[],
                denied_operators=[],
                default_timeout=1,
                max_output_chars=10,
                persist_cwd=False,
                allow_interactive=False,
            ),
            r'`ShellToolset\(cwd=\.\.\.\)` is deprecated and ignored',
        ),
        (
            lambda: MacroscopeToolset[None](cwd=Path('elsewhere'), command='macroscope', base=None, timeout=1),
            r'`MacroscopeToolset\(cwd=\.\.\.\)` is deprecated and ignored',
        ),
        (
            lambda: RepoContextToolset[None](['.claude'], 'inventory', workspace_dir=Path('elsewhere')),
            r'`RepoContextToolset\(workspace_dir=\.\.\.\)` is deprecated and ignored',
        ),
    ],
    ids=['FileSystemToolset', 'ShellToolset', 'MacroscopeToolset', 'RepoContextToolset'],
)
def test_toolset_working_directory_argument_warns(build: Callable[[], object], warning: str) -> None:
    with pytest.warns(HarnessDeprecationWarning, match=warning):
        build()
