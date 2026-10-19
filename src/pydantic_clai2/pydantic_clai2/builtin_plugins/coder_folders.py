"""The Coder folder list and local directory browser, asked through `Dialogs`."""

from __future__ import annotations

import re
import textwrap
from contextlib import suppress
from functools import partial
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import TYPE_CHECKING, Generic

from anyio import to_thread

from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.menus import Dialogs, Option
from pydantic_clai2.ui.menus.field_menu import SAVE_AND_CLOSE

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.coder import CoderSource

_FOLDER_NAME = re.compile(r'[A-Za-z0-9_-]+')

_CHILD = 'open:'
"""Prefixes a subdirectory's name in the browser's option values, so no name collides with another option."""


def prose(text: str) -> str:
    """Sentences in short lines: the terminal cuts long ones off beside an 80-column list; the web UI joins them."""
    return '\n\n'.join(textwrap.fill(paragraph, width=36) for paragraph in text.split('\n\n'))


_ADD_HELP = prose(
    'Load sub-agent definitions (*.md and *.toml), not files for code search.\n\n'
    'Add a directory path, browse local directories, or add a folder name such as agents or global. '
    'Names search .agents, .claude and .codex in the project and home. '
    'Relative paths resolve in each project; absolute and ~ paths are the same everywhere.\n\n'
    'Changes save immediately. Removing an entry never deletes files.'
)


def is_folder_name(value: str) -> bool:
    """Whether an entry expands to project and home folders rather than a single path."""
    return _FOLDER_NAME.fullmatch(value) is not None


def is_project_relative(value: str) -> bool:
    """Whether a path entry resolves against whichever project CLAI runs in, rather than one fixed place."""
    return not value.startswith('~') and not Path(value).is_absolute()


def _display(text: str) -> str:
    """Escape control characters, so a name or path from disk cannot drive the terminal."""
    return ''.join(char if char.isprintable() else repr(char)[1:-1] for char in text)


def _display_lines(text: str) -> str:
    return '\n'.join(_display(line) for line in text.split('\n'))


def _safe_initial(text: str) -> str:
    """Offer a replacement instead of emitting control characters from an existing path."""
    return text if text.isprintable() else ''


def _directory_problem(path: Path) -> str | None:
    try:
        if not path.is_dir():
            return 'Directory does not exist or is not a directory.'
        next(path.iterdir(), None)
    except OSError as exc:
        return f'Cannot read directory: {exc.strerror or exc}'
    return None


def _same_directory(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


class DirectoryPicker:
    """Browse directories, including hidden ones, without changing the process directory."""

    def __init__(self, *, start: Path, project: Path) -> None:
        self.current = start
        self.project = project
        self.error = ''

    def options(self) -> tuple[list[Option], str]:
        """The browser's options and description for the current directory; reads the disk."""
        children: list[Path] = []
        problem = _directory_problem(self.current)
        if problem is None:
            try:
                children = sorted(
                    (child for child in self.current.iterdir() if child.is_dir()),
                    key=lambda child: child.name.casefold(),
                )
            except OSError as exc:
                problem = f'Cannot list directory: {exc.strerror or exc}'
        options = [Option(value='use', label='Use this directory', description=prose(problem or ''))]
        if self.current != self.current.parent:
            options.append(Option(value='parent', label='.. (parent directory)'))
        options += [
            Option(value='path', label='Go to path / view current...'),
            Option(value='project', label='Project directory', description=_display(str(self.project))),
            Option(value='home', label='Home directory', description=_display(str(Path.home()))),
            *[
                Option(value=_CHILD + child.name, label=_display(child.name) + '/', description=_display(str(child)))
                for child in children
            ],
        ]
        notice = problem or self.error or ('' if children else 'No subdirectories. This directory can be used.')
        description = (
            (f'{prose(_display(notice))}\n\n' if notice else '')
            + f'Current directory\n{_display(str(self.current))}\n\n'
            + prose(
                'Open a folder to look inside; Use this directory saves its absolute path, '
                'the same in every project. Esc returns without changing settings.'
            )
        )
        return options, description

    async def run(self, dialogs: Dialogs) -> Path | None:
        """The directory the user chose, or `None` when they backed out."""
        while True:
            options, description = await to_thread.run_sync(self.options)
            choice = await dialogs.choose(title='Browse local directories', options=options, description=description)
            if choice is None:
                return None
            self.error = ''
            if choice == 'use':
                self.error = await to_thread.run_sync(_directory_problem, self.current) or ''
                if not self.error:
                    return self.current
            elif choice == 'path':
                text = await dialogs.text(
                    title='Go to directory',
                    label='Path',
                    initial=_safe_initial(str(self.current)),
                    placeholder='Enter a path; control characters are not copied',
                    check=self.path_problem,
                )
                if text:
                    self.current = await to_thread.run_sync(self.path, text)
            elif choice == 'parent':
                self.current = self.current.parent
            elif choice == 'project':
                self.current = self.project
            elif choice == 'home':
                self.current = Path.home()
            else:
                self.current = self.current / choice.removeprefix(_CHILD)

    def path(self, text: str) -> Path:
        return (self.current / Path(text.strip()).expanduser()).resolve()

    def path_problem(self, text: str) -> str | None:
        if not text.strip():
            return 'Enter a directory path.'
        try:
            return _directory_problem(self.path(text))
        except (OSError, RuntimeError, ValueError) as exc:
            return f'Invalid directory: {exc}'


class FolderMenu(Generic[DepsT]):
    """Manage entries individually while retaining their order and stored representation."""

    def __init__(self, source: CoderSource[DepsT], *, project: Path) -> None:
        self.source = source
        self.project = project
        self.notice = ''

    def folders(self) -> list[str]:
        return self.source.agent_folders()

    def details(self, index: int) -> str:
        """What one entry searches, and whether it is readable here; reads the disk."""
        value = self.folders()[index]
        if is_folder_name(value):
            details = f'Folder name: {value}\n\nProject, then home:\n' + '\n'.join(self.source.expand_folder(value))
            return _display_lines(details + '\n\nMissing named locations are skipped.')
        try:
            path = self.path(value)
            problem = _directory_problem(path)
        except (OSError, RuntimeError, ValueError) as exc:
            return _display_lines(f'{value}\n\nCannot open directory: {exc}')
        if is_project_relative(value):
            details = f'Relative path\n{value}\n\nIn this project\n{path}\n\n'
            details += prose(f'{problem} Skipped in this project.' if problem else 'Directory is readable.')
            details += '\n\n' + prose('Resolved against each project CLAI runs in.')
        else:
            details = f'Directory path\n{value}\n\nResolved location\n{path}\n\n'
            details += prose(problem or 'Directory is readable.')
        return _display_lines(details)

    def options(self) -> tuple[list[Option], str]:
        """The list's options and description, notices first; reads the disk."""
        folders = self.folders()
        options = [
            Option(
                value=str(index),
                label=f'{index + 1}. {value if is_folder_name(value) else _display(Path(value).name) + "/"}',
                description=self.details(index),
            )
            for index, value in enumerate(folders)
        ]
        options += [
            Option(value='path', label='+ Add directory path...', description=_ADD_HELP),
            Option(value='browse', label='+ Browse local directories...', description=_ADD_HELP),
            Option(value='name', label='+ Add folder name...', description=_ADD_HELP),
            Option(
                value='close',
                label=SAVE_AND_CLOSE,
                description=prose('Changes are already saved. Return to Coder settings.'),
            ),
        ]
        notices = [
            _display(self.notice),
            '' if folders else 'No folders selected. Disk agents are off.',
            '' if self.source.sub_agents_enabled() else 'Sub-agents are disabled in Coder settings.',
        ]
        return options, '\n'.join(prose(notice) for notice in notices if notice)

    async def run(self, dialogs: Dialogs) -> list[str]:
        """Show the list until the user closes it, saving each change; the messages to show."""
        messages: list[str] = []
        while True:
            options, description = await to_thread.run_sync(self.options)
            choice = await dialogs.choose(title='Agent folders', options=options, description=description)
            if choice is None or choice == 'close':
                return messages
            index: int | None = None
            if choice.isdigit():
                index = int(choice)
                choice = await dialogs.choose(
                    title='Manage agent folder',
                    options=[
                        Option(value='edit', label='Edit name or path...'),
                        Option(value='browse', label='Replace by browsing...'),
                        Option(value='remove', label='Remove from search (keep files)'),
                        Option(value='back', label='Back'),
                    ],
                    description=await to_thread.run_sync(self.details, index),
                )
                if choice is None or choice == 'back':
                    continue
            self.notice = ''
            folders = self.folders().copy()
            if choice == 'remove':
                assert index is not None
                folders.pop(index)
            else:
                value = await self.edit(dialogs, kind=choice, index=index)
                if value is None:
                    continue
                if index is None:
                    folders.append(value)
                else:
                    folders[index] = value
            try:
                await to_thread.run_sync(self.source.save_agent_folders, folders)
            except (OSError, RuntimeError, ValueError, SQLiteError) as exc:
                self.notice = f'Could not save: {exc}'
                continue
            self.notice = 'Folder removed. Files were not changed.' if choice == 'remove' else 'Agent folder saved.'
            messages.append(self.notice)

    async def edit(self, dialogs: Dialogs, *, kind: str, index: int | None) -> str | None:
        """The new entry, or `None` when the user backed out or it is not acceptable (saying why in `notice`)."""
        current = self.folders()[index] if index is not None else ''
        named = kind == 'name' or (kind == 'edit' and is_folder_name(current))
        if kind == 'browse':
            start = self.project
            if current and not is_folder_name(current):
                # A broken entry like `~nosuchuser/agents` must still be replaceable from the project.
                with suppress(RuntimeError):
                    start = self.project / Path(current).expanduser()
            chosen = await DirectoryPicker(start=start, project=self.project).run(dialogs)
            if chosen is None:
                return None
            value = str(chosen)
        else:
            text = await dialogs.text(
                title='Edit agent folder' if index is not None else 'Add agent folder',
                label='Name' if named else 'Path',
                initial=_safe_initial(current),
                placeholder='agents or global' if named else './team-agents (in each project) or ~/my-agents',
                check=partial(self.problem, named=named, index=index),
            )
            # Empty text never removes an entry: removing is its own action.
            if not text:
                return None
            value = self.value(text, named=named)
        # Checked again: the disk may have changed since the dialog accepted it, and browsing checks nothing.
        self.notice = await to_thread.run_sync(partial(self.problem, value, named=named, index=index)) or ''
        return None if self.notice else value

    def value(self, text: str, *, named: bool) -> str:
        text = text.strip()
        if named:
            return text
        if text.startswith('~') and not text.startswith('~/'):
            return str(Path(text).expanduser())
        return f'./{text}' if is_folder_name(text) else text

    def problem(self, text: str, *, named: bool, index: int | None) -> str | None:
        if not text.strip():
            return 'Enter a folder name.' if named else 'Enter a directory path.'
        if not all(char.isprintable() for char in text):
            return 'Names and paths cannot contain control characters.'
        if named and not is_folder_name(text.strip()):
            return 'Use letters, numbers, underscores or hyphens. Add a directory path for other locations.'
        try:
            value = self.value(text, named=named)
            candidate = value if named else self.path(value)
            for position, existing in enumerate(self.folders()):
                if position == index:
                    continue
                try:
                    other = existing if is_folder_name(existing) else self.path(existing)
                except (OSError, RuntimeError, ValueError):
                    continue
                if candidate == other or (
                    isinstance(candidate, Path) and isinstance(other, Path) and _same_directory(candidate, other)
                ):
                    return 'This folder is already in the list.'
            # A relative path may exist only in other projects, so only fixed locations must exist now.
            if not named and not is_project_relative(value):
                return _directory_problem(self.path(value))
        except (OSError, RuntimeError, ValueError) as exc:
            return f'Invalid directory: {exc}'
        return None

    def path(self, value: str) -> Path:
        return (self.project / Path(value).expanduser()).resolve()


async def edit_agent_folders(source: CoderSource[DepsT], dialogs: Dialogs) -> list[str]:
    """The **Agent folders** row's action: the folder list for the directory the session works in."""
    return await FolderMenu(source, project=source.host.workspace).run(dialogs)
