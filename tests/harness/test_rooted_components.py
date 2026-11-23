"""Every component that reads or writes under a root, held to the same symlink cases.

Each component's containment check is its own code: a link it follows past its root is the defect fixed in
https://github.com/pydantic/pydantic-ai/pull/12044 and https://github.com/pydantic/pydantic-ai/pull/13632, and
again for other components in https://github.com/pydantic/pydantic-ai/pull/11639 and
https://github.com/pydantic/pydantic-ai/pull/13595. A new rooted component gets a `COMPONENTS` row.

A row's expectation for a case is what the component's docs or docstring promise, with `source` naming where.
Where they promise nothing, `source` is `UNDOCUMENTED` and the expectation records what the component does.
The cases go through each component's public API, not through `pydantic_ai.workspaces._confine`, so the suite
also runs against the code from before a fix.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceBackend, WorkspaceError
from pydantic_ai_harness.file_system import FileSystemToolset
from pydantic_ai_harness.memory import FileMemoryStore
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.smart_search import SmartSearch
from pydantic_ai_harness.tool_output_limits import LocalFileOverflowStore, WorkspaceOverflowStore
from tests.harness.file_system.conftest import tools_path

from ..workspace_fakes import RunOnlyWorkspaceBackend

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='`LocalWorkspaceBackend` and symlinks need a POSIX host')

SECRET = 'rooted-component-secret'
"""Text only the files outside each root hold."""

Case = Literal[
    'regular_file', 'file_link', 'directory_link', 'link_mid_path', 'symlinked_root', 'root_swapped', 'failing_realpath'
]
"""`regular_file` is the control: a file below the root that every component reads, so a refusal in another case
is the component's own and not a broken setup."""
Outcome = Literal['refuses', 'follows']
Backend = Literal['native', 'shell', 'no_readlink', 'host']
"""`native` resolves links with `os.path.realpath`; `shell` resolves them with `readlink` in the workspace's
shell; `no_readlink` is that shell with no `readlink`, so `Workspace.realpath` cannot read a link; `host` is a
component that works on this machine's paths, not through a workspace."""

UNDOCUMENTED = 'undocumented: what the component does on main'


@dataclass(frozen=True)
class Expect:
    outcome: Outcome
    source: str
    """Where the component promises `outcome`, or `UNDOCUMENTED`."""


@dataclass(frozen=True)
class Layout:
    """A root with one case's links, beside a directory `outside` it whose files hold `SECRET`."""

    work: Path
    """The workspace's working directory."""
    root: Path
    outside: Path
    backend: WorkspaceBackend


class Handle(Protocol):
    """One component bound to a layout, kept across the calls of a case."""

    async def read(self, relative: str) -> str:
        """What reaches the caller when it reads `relative` under the root; empty when the component refuses."""
        ...

    async def write(self, relative: str) -> None:
        """Write `relative` under the root; return quietly when the component refuses."""
        ...


@dataclass(frozen=True)
class RootedComponent:
    name: str
    root: Callable[[Path], Path]
    """Where the root sits, given the working directory."""
    entry: str
    """The file name the cases read under the root."""
    open: Callable[[Layout], Handle]
    reads: Mapping[Case, Expect]
    writes: Mapping[Case, Expect] = field(default_factory=dict[Case, Expect])
    """Expectations for writing through each link case; empty for a component that only reads."""
    prepare: Callable[[Path], None] = lambda root: None
    """Files a root needs before the component recognizes it, written into every directory a case uses as one."""
    host: bool = False
    """Whether the component works on this machine's paths, so the workspace backends do not apply."""


# --- FileSystem ---

_FILESYSTEM_DOCS = 'docs/harness/filesystem.md "Security model": Containment'
_REALPATH_DOCSTRING = '`Workspace.realpath` raises `WorkspaceError` on a link it cannot read'


@dataclass
class _FileSystem:
    """Calls share one `Workspace`, as the calls of a run share `ctx.workspace`, so the root is bound once."""

    layout: Layout
    toolset: FileSystemToolset[None] = field(
        default_factory=lambda: FileSystemToolset[None](
            allowed_patterns=[],
            denied_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
    )

    def __post_init__(self) -> None:
        self.workspace = Workspace(self.layout.backend)

    async def read(self, relative: str) -> str:
        try:
            return str(await self.toolset.read_file(relative, workspace=self.workspace))
        except (ModelRetry, ToolFailed):
            return ''

    async def write(self, relative: str) -> None:
        try:
            await self.toolset.write_file(relative, 'written', workspace=self.workspace)
        except ModelRetry:
            pass


# --- FileMemoryStore ---

_MEMORY_DOCS = 'docs/harness/memory.md: a memory path that resolves outside its store is refused'


@dataclass
class _FileMemory:
    layout: Layout

    def __post_init__(self) -> None:
        self.store = FileMemoryStore('memory', workspace=self.layout.backend)

    async def read(self, relative: str) -> str:
        try:
            found = await self.store.read(relative, max_chars=10_000)
        except (ValueError, WorkspaceError):  # `MemoryPathEscapeError` is a `ValueError`
            return ''
        return '' if found is None else found.content

    async def write(self, relative: str) -> None:
        try:
            await self.store.write(relative, 'written', expected_version=None)
        except (ValueError, WorkspaceError):  # `MemoryPathEscapeError` is a `ValueError`
            pass


# --- WorkspaceOverflowStore ---

_WORKSPACE_STORE_READ = '`WorkspaceOverflowStore.read`: a handle outside the store directory raises `PermissionError`'


@dataclass
class _WorkspaceStore:
    layout: Layout
    store: WorkspaceOverflowStore = field(default_factory=WorkspaceOverflowStore)

    async def read(self, relative: str) -> str:
        try:
            return (await self.store.read(Workspace(self.layout.backend), relative)).decode()
        except (PermissionError, WorkspaceError):
            return ''

    async def write(self, relative: str) -> None:
        try:
            await self.store.write(Workspace(self.layout.backend), relative, b'written')
        except (PermissionError, WorkspaceError):
            pass


# --- LocalFileOverflowStore ---

_LOCAL_STORE_DOCS = 'docs/harness/tool-output-limits.md `LocalFileOverflowStore`'


@dataclass
class _LocalStore:
    layout: Layout

    def __post_init__(self) -> None:
        self.store = LocalFileOverflowStore(base_dir=self.layout.root)

    async def read(self, relative: str) -> str:
        try:
            return (await self.store.read(relative)).decode()
        except PermissionError:
            return ''

    async def write(self, relative: str) -> None:
        try:
            await self.store.write(relative, b'written')
        except PermissionError:
            pass


def _private_root(root: Path) -> None:
    root.chmod(0o700)


# --- RepoContext ---

_REPO_CONTEXT_DOCS = 'docs/harness/repo-context.md: links outside the scanned directory are skipped'


@dataclass
class _RepoContext:
    layout: Layout

    async def read(self, relative: str) -> str:
        """The instructions the run's first request carries."""
        agent = Agent(TestModel(), deps_type=type(None), capabilities=[RepoContext[None](filenames=(relative,))])
        try:
            result = await agent.run('go', workspace=self.layout.backend)
        except WorkspaceError:
            return ''
        return '\n'.join(
            message.instructions or '' for message in result.all_messages() if isinstance(message, ModelRequest)
        )

    async def write(self, relative: str) -> None:  # pragma: no cover -- `RepoContext` only reads
        raise NotImplementedError


# --- Skills ---

_SKILLS_DOCS = 'docs/harness/skills.md: a path that leads outside the skill directory through a symlink is refused'


def _skill(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'SKILL.md').write_text('---\nname: guide\ndescription: A guide.\n---\nRead the notes.\n', encoding='utf-8')


def _skill_reader(results: list[str], relative: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'guide'}, tool_call_id='load')])
        if len(messages) == 3:
            return ModelResponse(
                parts=[ToolCallPart('read_skill_file', {'skill': 'guide', 'path': relative}, tool_call_id='read')]
            )
        results.extend(_returned(messages[-1]))
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _returned(message: ModelMessage) -> list[str]:
    assert isinstance(message, ModelRequest)
    return [str(part.content) for part in message.parts if isinstance(part, ToolReturnPart)]


@dataclass
class _Skills:
    layout: Layout

    async def read(self, relative: str) -> str:
        results: list[str] = []
        agent = Agent(
            _skill_reader(results, relative),
            deps_type=type(None),
            capabilities=[Skills[None]('skills', read_files=True)],
        )
        try:
            await agent.run('go', workspace=self.layout.backend)
        except WorkspaceError:
            return ''
        return '\n'.join(results)

    async def write(self, relative: str) -> None:  # pragma: no cover -- `read_skill_file` only reads
        raise NotImplementedError


# --- SmartSearch ---


def _searcher(directory: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('smart_search', {'query': 'secret', 'directory': directory})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _judge(seen: list[str]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(str(part.content) for part in messages[-1].parts if isinstance(part, UserPromptPart))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'entity': 0.9, 'operation': 0.9})])

    return FunctionModel(respond)


@dataclass
class _SmartSearch:
    layout: Layout

    async def read(self, relative: str) -> str:
        """What the judge was sent from a search of the directory holding `relative`."""
        seen: list[str] = []
        directory = os.path.dirname(relative) or '.'
        agent = Agent(_searcher(directory), deps_type=type(None), capabilities=[SmartSearch[None](model=_judge(seen))])
        try:
            await agent.run('go', workspace=self.layout.backend)
        except WorkspaceError:
            return ''
        return '\n'.join(seen)

    async def write(self, relative: str) -> None:  # pragma: no cover -- `smart_search` only reads
        raise NotImplementedError


def _source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


_LINKS: tuple[Case, ...] = ('file_link', 'directory_link', 'link_mid_path')
_CONTROL: dict[Case, Expect] = {'regular_file': Expect('follows', 'the control: a regular file below the root')}

COMPONENTS: tuple[RootedComponent, ...] = (
    RootedComponent(
        name='FileSystem',
        root=lambda work: work,
        entry='notes.txt',
        open=_FileSystem,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _FILESYSTEM_DOCS) for case in _LINKS},
            'symlinked_root': Expect('follows', UNDOCUMENTED),
            'root_swapped': Expect('refuses', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', _REALPATH_DOCSTRING),
        },
        writes={case: Expect('refuses', _FILESYSTEM_DOCS) for case in _LINKS},
    ),
    RootedComponent(
        name='FileMemoryStore',
        root=lambda work: work / 'memory',
        entry='notes.md',
        open=_FileMemory,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _MEMORY_DOCS) for case in _LINKS},
            'symlinked_root': Expect('follows', UNDOCUMENTED),
            'root_swapped': Expect('refuses', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', _REALPATH_DOCSTRING),
        },
        writes={case: Expect('refuses', _MEMORY_DOCS) for case in _LINKS},
    ),
    RootedComponent(
        name='WorkspaceOverflowStore',
        root=lambda work: work / '.pydantic-ai-harness' / 'tool-output',
        entry='payload',
        open=_WorkspaceStore,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _WORKSPACE_STORE_READ) for case in _LINKS},
            'symlinked_root': Expect('refuses', UNDOCUMENTED),
            'root_swapped': Expect('refuses', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', _REALPATH_DOCSTRING),
        },
        writes={case: Expect('refuses', UNDOCUMENTED) for case in _LINKS},
    ),
    RootedComponent(
        name='LocalFileOverflowStore',
        root=lambda work: work / 'overflow',
        entry='payload',
        open=_LocalStore,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _LOCAL_STORE_DOCS) for case in _LINKS},
            'symlinked_root': Expect('refuses', _LOCAL_STORE_DOCS),
            'root_swapped': Expect('refuses', _LOCAL_STORE_DOCS),
        },
        writes={case: Expect('refuses', _LOCAL_STORE_DOCS) for case in _LINKS},
        prepare=_private_root,
        host=True,
    ),
    RootedComponent(
        name='RepoContext',
        root=lambda work: work,
        entry='AGENTS.md',
        open=_RepoContext,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _REPO_CONTEXT_DOCS) for case in _LINKS},
            'symlinked_root': Expect('follows', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', UNDOCUMENTED),
        },
    ),
    RootedComponent(
        name='Skills',
        root=lambda work: work / 'skills' / 'guide',
        entry='notes.md',
        open=_Skills,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', _SKILLS_DOCS) for case in _LINKS},
            'symlinked_root': Expect('follows', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', _REALPATH_DOCSTRING),
        },
        prepare=_skill,
    ),
    RootedComponent(
        name='SmartSearch',
        root=lambda work: work,
        entry='notes.py',
        open=_SmartSearch,
        reads={
            **_CONTROL,
            **{case: Expect('refuses', UNDOCUMENTED) for case in _LINKS},
            'symlinked_root': Expect('follows', UNDOCUMENTED),
            'failing_realpath': Expect('refuses', _REALPATH_DOCSTRING),
        },
        prepare=_source,
    ),
)


def _backends(component: RootedComponent, case: Case) -> Sequence[Backend]:
    if component.host:
        return ('host',)
    return ('no_readlink',) if case == 'failing_realpath' else ('native', 'shell')


def _cells(kind: Literal['reads', 'writes']) -> list[object]:
    return [
        pytest.param(component, case, backend, id=f'{component.name}-{case}-{backend}')
        for component in COMPONENTS
        for case in getattr(component, kind)
        for backend in _backends(component, case)
    ]


def _relative(component: RootedComponent, case: Case) -> str:
    """The path a case reads under the root."""
    if case == 'directory_link' or case == 'failing_realpath':
        return f'linked/{component.entry}'
    if case == 'link_mid_path':
        return f'sub/linked/{component.entry}'
    if case == 'regular_file':
        return f'sub/{component.entry}'
    return component.entry


def _backend(work: Path, backend: Backend, bin_dir: Path) -> WorkspaceBackend:
    if backend == 'shell':
        return RunOnlyWorkspaceBackend(LocalWorkspaceBackend(work))
    if backend == 'no_readlink':
        path = tools_path(bin_dir, exclude=frozenset({'readlink'}))
        return RunOnlyWorkspaceBackend(LocalWorkspaceBackend(work, env={'PATH': path}))
    return LocalWorkspaceBackend(work)


def _secret(directory: Path, component: RootedComponent) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    component.prepare(directory)
    (directory / component.entry).write_text(SECRET, encoding='utf-8')


def _layout(component: RootedComponent, case: Case, backend: Backend, tmp_path: Path) -> Layout:
    work = tmp_path / 'work'
    root = component.root(work)
    outside = tmp_path / 'outside'
    _secret(outside, component)
    if case == 'symlinked_root':
        root.parent.mkdir(parents=True, exist_ok=True)
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir(parents=True, exist_ok=True)
        component.prepare(root)
        if case == 'file_link':
            (root / component.entry).symlink_to(outside / component.entry)
        elif case == 'directory_link' or case == 'failing_realpath':
            (root / 'linked').symlink_to(outside, target_is_directory=True)
        elif case == 'link_mid_path':
            (root / 'sub').mkdir()
            (root / 'sub' / 'linked').symlink_to(outside, target_is_directory=True)
        elif case == 'regular_file':
            (root / 'sub').mkdir()
            (root / 'sub' / component.entry).write_text(SECRET, encoding='utf-8')
        else:
            (root / component.entry).write_text('inside', encoding='utf-8')
    work.mkdir(parents=True, exist_ok=True)
    return Layout(work, root, outside, _backend(work, backend, tmp_path / 'bin'))


async def _swap_root(layout: Layout, handle: Handle, entry: str) -> None:
    """Bind the root with a read that succeeds, then replace the root with a link to `outside`."""
    assert 'inside' in await handle.read(entry)
    layout.root.rename(layout.root.with_name(layout.root.name + '-before'))
    layout.root.symlink_to(layout.outside, target_is_directory=True)


def _observed(text: str) -> Outcome:
    return 'follows' if SECRET in text else 'refuses'


@pytest.mark.parametrize(('component', 'case', 'backend'), _cells('reads'))
async def test_read(component: RootedComponent, case: Case, backend: Backend, tmp_path: Path) -> None:
    """A read either reaches the file the case plants outside the root or is refused, as the row expects."""
    if component.name == 'SmartSearch' and shutil.which('rg') is None:  # pragma: no cover
        pytest.skip('`SmartSearch` lists files with `rg`')
    layout = _layout(component, case, backend, tmp_path)
    handle = component.open(layout)
    if case == 'root_swapped':
        await _swap_root(layout, handle, component.entry)

    observed = _observed(await handle.read(_relative(component, case)))

    expected = component.reads[case]
    assert observed == expected.outcome, expected.source


@pytest.mark.parametrize(('component', 'case', 'backend'), _cells('writes'))
async def test_write(component: RootedComponent, case: Case, backend: Backend, tmp_path: Path) -> None:
    """A write through a link either lands outside the root or is refused, as the row expects."""
    layout = _layout(component, case, backend, tmp_path)
    handle = component.open(layout)
    relative = _relative(component, case)
    written = layout.outside / component.entry if case == 'file_link' else layout.outside / 'written'
    if case != 'file_link':
        relative = os.path.join(os.path.dirname(relative), 'written')

    await handle.write(relative)

    reached = written.exists() and written.read_text(encoding='utf-8') != SECRET
    expected = component.writes[case]
    assert ('follows' if reached else 'refuses') == expected.outcome, expected.source
