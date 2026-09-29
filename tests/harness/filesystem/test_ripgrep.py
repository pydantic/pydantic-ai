"""The opt-in ripgrep-backed `list_files` and `grep` tools."""

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import CommandResult, LocalWorkspaceBackend, Workspace, WorkspaceCommand
from pydantic_ai_harness.filesystem import RIPGREP_TOOL_NAMES, FilesSearchedEvent, FileSystem, FileSystemToolset

from .._tool_calls import call_tool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'app.py').write_text('import os\n\n\ndef main():\n    return os.name\n')
    (tmp_path / 'notes.txt').write_text('Import notes\nos is a module\n')
    (tmp_path / '.hidden').write_text('import os\n')
    (tmp_path / 'ignored.log').write_text('import os\n')
    (tmp_path / '.ignore').write_text('*.log\n')
    return tmp_path


def toolset(workspace: Path, **settings: object) -> FileSystemToolset[None]:
    capability = FileSystem[None](root_dir=workspace, tools=RIPGREP_TOOL_NAMES, **settings)  # pyright: ignore[reportArgumentType]
    built = capability.get_toolset()
    assert isinstance(built, FileSystemToolset)
    return built


async def call(
    workspace: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    working_dir: Path | None = None,
    **settings: object,
) -> str:
    """Call a tool of a `FileSystem` bounded by `workspace`, run in `working_dir` (by default `workspace`)."""
    capability = FileSystem[None](root_dir=workspace, tools=RIPGREP_TOOL_NAMES, **settings)  # pyright: ignore[reportArgumentType]
    return await call_tool(
        [capability, *capabilities], name, arguments, workspace=LocalWorkspaceBackend(working_dir or workspace)
    )


def fake_rg(workspace: Path, script: str) -> LocalWorkspaceBackend:
    """A workspace at `workspace` whose `rg` is `script`, found first on its `PATH`."""
    fake = workspace / 'bin'
    fake.mkdir()
    (fake / 'rg').write_text(script)
    (fake / 'rg').chmod(0o755)
    return LocalWorkspaceBackend(workspace, env={'PATH': f'{fake}{os.pathsep}{os.environ["PATH"]}'})


class Recorder(AbstractCapability[None]):
    def __init__(self) -> None:
        self.events: list[FilesSearchedEvent] = []

    @on_event(FilesSearchedEvent)
    async def searched(self, ctx: RunContext[None], event: FilesSearchedEvent) -> None:
        self.events.append(event)


class TestRegistration:
    async def test_opt_in(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path)]).run(
            'Inspect tools', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        assert not {'list_files', 'grep'} & {t.name for t in model.last_model_request_parameters.function_tools}

        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path, tools=['read_file', 'grep'])]).run(
            'Inspect', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        assert [t.name for t in model.last_model_request_parameters.function_tools] == ['read_file', 'grep']

    def test_unknown_tool_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Unknown filesystem tools: bogus'):
            FileSystem(root_dir=tmp_path, tools=['bogus']).get_toolset()


class TestListFiles:
    async def test_respects_ignore_rules_and_hidden_files(self, workspace: Path) -> None:
        assert (await toolset(workspace).list_files(workspace=LocalWorkspaceBackend(workspace))).splitlines() == [
            'notes.txt',
            'src/app.py',
        ]

    async def test_glob(self, workspace: Path) -> None:
        assert (
            await toolset(workspace).list_files(glob='*.py', workspace=LocalWorkspaceBackend(workspace)) == 'src/app.py'
        )
        assert (
            await toolset(workspace).list_files('src', glob='*.txt', workspace=LocalWorkspaceBackend(workspace))
            == 'No files found.'
        )

    async def test_glob_overrides_ignore_files_but_not_hidden(self, workspace: Path) -> None:
        assert (
            await toolset(workspace).list_files(glob='*.log', workspace=LocalWorkspaceBackend(workspace))
            == 'ignored.log'
        )
        assert (
            await toolset(workspace).list_files(glob='**', workspace=LocalWorkspaceBackend(workspace))
        ).splitlines() == [
            'ignored.log',
            'notes.txt',
            'src/app.py',
        ]
        assert '.hidden' not in await toolset(workspace).grep(
            'import os', glob='**', workspace=LocalWorkspaceBackend(workspace)
        )

    async def test_denied_patterns_filter_entries(self, workspace: Path) -> None:
        assert (
            await toolset(workspace, denied_patterns=['src/*']).list_files(workspace=LocalWorkspaceBackend(workspace))
            == 'notes.txt'
        )

    async def test_cap(self, workspace: Path) -> None:
        listed = await toolset(workspace, max_find_results=1).list_files(workspace=LocalWorkspaceBackend(workspace))
        assert listed.splitlines() == ['notes.txt', '[... truncated at 1 files]']

    async def test_cap_counts_permitted_entries_only(self, workspace: Path) -> None:
        listed = await toolset(workspace, max_find_results=1, denied_patterns=['notes.txt']).list_files(
            workspace=LocalWorkspaceBackend(workspace)
        )
        assert listed == 'src/app.py'

    async def test_cap_resolves_only_the_candidates_it_needs(self, workspace: Path) -> None:
        many = workspace / 'many'
        many.mkdir()
        for index in range(1200):
            (many / f'{index:04}.txt').write_text('')

        class Counting(LocalWorkspaceBackend):
            resolves = 0

            async def run(
                self,
                command: WorkspaceCommand,
                *,
                shell: bool = False,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
            ) -> CommandResult:
                if isinstance(command, str) and 'realpath --' in command:
                    self.resolves += 1
                return await super().run(command, shell=shell, env=env, timeout=timeout)

        backend = Counting(workspace)
        listed = await toolset(workspace, max_find_results=1).list_files('many', workspace=backend)
        assert listed.splitlines() == ['many/0000.txt', '[... truncated at 1 files]']
        # Candidates are resolved a batch at a time, stopping once the cap is reached.
        assert backend.resolves == 1

    async def test_oversized_record_stops_the_search(self, workspace: Path) -> None:
        fake = fake_rg(workspace, f'#!{sys.executable}\nimport sys\nsys.stdout.write("x" * 2_000_000)\n')
        assert await toolset(workspace).list_files(workspace=fake) == '[... truncated at 1000 files]'

    async def test_oversized_complete_record_stops_the_search(self, workspace: Path) -> None:
        fake = fake_rg(workspace, f'#!{sys.executable}\nimport sys\nsys.stdout.write("x" * 2_000_000 + "\\0")\n')
        assert await toolset(workspace).list_files(workspace=fake) == '[... truncated at 1000 files]'

    async def test_search_that_dies_without_a_status_is_reported(self, workspace: Path) -> None:
        # Killing the shell that would report `rg`'s status leaves no status line to read.
        fake = fake_rg(workspace, '#!/bin/sh\nkill -9 $PPID\n')
        with pytest.raises(ModelRetry, match='ripgrep failed'):
            await toolset(workspace).list_files(workspace=fake)

    @pytest.mark.parametrize('cwd_name', ['.', 'src'])
    async def test_event(self, workspace: Path, cwd_name: str) -> None:
        recorder = Recorder()
        await call(workspace, 'list_files', {'glob': '*.py'}, capabilities=[recorder], working_dir=workspace / cwd_name)
        assert recorder.events[0].search == 'find'
        assert recorder.events[0].pattern == '*.py'
        assert recorder.events[0].match_count == 1
        assert recorder.events[0].path == cwd_name
        assert recorder.events[0].root_dir == str(workspace)

    @pytest.mark.parametrize('path', ['notes.txt', 'missing', '..'])
    async def test_rejects_non_directories(self, workspace: Path, path: str) -> None:
        assert await call(workspace, 'list_files', {'path': path})
        assert 'src/app.py' not in await call(workspace, 'list_files', {'path': path})


class TestGrep:
    async def test_matches_with_line_numbers(self, workspace: Path) -> None:
        assert (
            await toolset(workspace).grep('import os', workspace=LocalWorkspaceBackend(workspace))
            == 'src/app.py:1:import os'
        )

    async def test_options(self, workspace: Path) -> None:
        built = toolset(workspace)
        assert (
            await built.grep('import', ignore_case=True, glob='*.txt', workspace=LocalWorkspaceBackend(workspace))
            == 'notes.txt:1:Import notes'
        )
        assert (
            await built.grep('os', file_type='py', context=1, workspace=LocalWorkspaceBackend(workspace))
        ).splitlines() == [
            'src/app.py:1:import os',
            'src/app.py-2-',
            'src/app.py-4-def main():',
            'src/app.py:5:    return os.name',
        ]
        assert (
            await built.grep('return os.name', literal=True, path='src', workspace=LocalWorkspaceBackend(workspace))
            == 'src/app.py:5:    return os.name'
        )
        assert await built.grep('nothing', workspace=LocalWorkspaceBackend(workspace)) == 'No matches found.'

    async def test_long_lines_are_cut_by_ripgrep(self, workspace: Path) -> None:
        (workspace / 'minified.js').write_text('x' * 5000 + 'needle' + 'y' * 5000 + '\n')
        result = await toolset(workspace).grep('needle', workspace=LocalWorkspaceBackend(workspace))
        assert result.startswith('minified.js:1:xxxx') and result.endswith('[... omitted end of long line]')
        assert len(result) < 5000

    async def test_file_target(self, workspace: Path) -> None:
        assert (
            await toolset(workspace).grep('os', path='notes.txt', workspace=LocalWorkspaceBackend(workspace))
            == 'notes.txt:2:os is a module'
        )

    async def test_authorization_filters_records(self, workspace: Path) -> None:
        assert (
            await toolset(workspace, denied_patterns=['src/*']).grep('os', workspace=LocalWorkspaceBackend(workspace))
            == 'notes.txt:2:os is a module'
        )

    async def test_cap_counts_context_lines(self, workspace: Path) -> None:
        capped = await toolset(workspace, max_search_results=1).grep(
            'os', context=2, workspace=LocalWorkspaceBackend(workspace)
        )
        assert capped.splitlines() == ['notes.txt-1-Import notes', '[... truncated at 1 lines]']
        assert (
            await toolset(workspace, max_search_results=2).grep('os', workspace=LocalWorkspaceBackend(workspace))
        ).splitlines() == [
            'notes.txt:2:os is a module',
            'src/app.py:1:import os',
            '[... truncated at 2 lines]',
        ]

    @pytest.mark.parametrize('cwd_name,path', [('.', 'src'), ('src', '.')])
    async def test_event(self, workspace: Path, cwd_name: str, path: str) -> None:
        recorder = Recorder()
        await call(
            workspace,
            'grep',
            {'pattern': 'os', 'path': path},
            capabilities=[recorder],
            working_dir=workspace / cwd_name,
        )
        assert recorder.events[0].search == 'grep'
        assert recorder.events[0].pattern == 'os'
        assert recorder.events[0].path == 'src'
        assert recorder.events[0].root_dir == str(workspace)
        assert recorder.events[0].match_count == 2

    @pytest.mark.parametrize(
        'arguments,message',
        [
            ({'pattern': '(', 'context': 0}, 'ripgrep failed'),
            ({'pattern': 'os', 'context': 21}, 'context must be between'),
            ({'pattern': 'os', 'path': 'missing'}, 'not a file or directory'),
            ({'pattern': 'os', 'path': '../outside'}, 'is outside root_dir'),
        ],
    )
    async def test_retries(self, workspace: Path, arguments: dict[str, object], message: str) -> None:
        assert message in await call(workspace, 'grep', arguments)

    @pytest.mark.skipif(os.name == 'nt' or os.geteuid() == 0, reason='root reads a mode-000 file')
    async def test_unreadable_paths_leave_a_partial_result(self, workspace: Path) -> None:
        locked, locked_dir = workspace / 'locked.py', workspace / 'locked'
        locked.write_text('import os\n')
        locked_dir.mkdir()
        (locked_dir / 'inner.py').write_text('import os\n')
        locked.chmod(0)
        locked_dir.chmod(0)
        try:
            assert await call(workspace, 'grep', {'pattern': 'import os'}) == (
                'src/app.py:1:import os\n[2 paths could not be read (Permission denied): locked, locked.py]'
            )
            assert await call(workspace, 'list_files', {'glob': '*.py'}) == (
                'locked.py\nsrc/app.py\n[1 path could not be read (Permission denied): locked]'
            )
        finally:
            locked.chmod(0o644)
            locked_dir.chmod(0o755)

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX symlinks')
    async def test_symlink_outside_root_is_dropped(
        self, workspace: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp('outside') / 'secret.txt'
        outside.write_text('import os\n')
        (workspace / 'link.txt').symlink_to(outside)
        listed = await toolset(workspace).list_files(workspace=LocalWorkspaceBackend(workspace))
        assert 'link.txt' not in listed
        assert 'link.txt' not in await toolset(workspace).grep('import os', workspace=LocalWorkspaceBackend(workspace))


class _CountingProbes(LocalWorkspaceBackend):
    probes = 0

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if isinstance(command, str) and 'command -v rg' in command:
            self.probes += 1
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class TestWithoutRipgrep:
    """A workspace without `rg` uses the in-workspace POSIX search tools."""

    @pytest.fixture
    def without_rg(self, workspace: Path, no_rg_path: str) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(workspace, env={'PATH': no_rg_path})

    async def test_list_files(self, workspace: Path, without_rg: LocalWorkspaceBackend) -> None:
        ts = toolset(workspace)
        # Git-backed POSIX enumeration also applies a root .ignore outside a repository.
        assert (await ts.list_files(workspace=without_rg)).splitlines() == ['notes.txt', 'src/app.py']
        assert await ts.list_files(glob='*.py', workspace=without_rg) == 'src/app.py'
        assert await ts.list_files(glob='src/*.py', workspace=without_rg) == 'src/app.py'

    async def test_grep(self, workspace: Path, without_rg: LocalWorkspaceBackend) -> None:
        ts = toolset(workspace)
        assert (await ts.grep('^import', ignore_case=True, glob='*.txt', workspace=without_rg)) == (
            'notes.txt:1:Import notes'
        )
        assert await ts.grep('os.name', literal=True, workspace=without_rg) == 'src/app.py:5:    return os.name'
        with pytest.raises(ModelRetry, match='`file_type` needs ripgrep'):
            await ts.grep('os', file_type='py', workspace=without_rg)

    async def test_missing_rg_is_probed_once_per_workspace(self, workspace: Path, no_rg_path: str) -> None:
        backend = _CountingProbes(workspace, env={'PATH': no_rg_path})
        ts, ws = toolset(workspace), Workspace(backend)
        await ts.grep('os', workspace=ws)
        await ts.list_files(workspace=ws)
        await ts.grep('os', workspace=ws)
        assert backend.probes == 1
