"""The Coder folder list and local directory browser, run on the menu worker."""

from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Generic, Literal

from rich.console import Console
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size
from termflow.tui.textinput import TextInput

from pydantic_clai2.builtin_plugins.coder import CoderSettings, CoderSource, is_folder_name
from pydantic_clai2.plugins import DepsT
from pydantic_clai2.ui.menus.field_menu import (
    TERMINAL,
    FieldMenu,
    Runners,
    is_save_and_close,
    picked,
    run_flow,
    save_and_close_item,
)
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.rendering._rendering import markdown_style


def _display(text: str) -> str:
    return ''.join(char if char.isprintable() else repr(char)[1:-1] for char in text)


def _safe_initial(text: str) -> str:
    """Offer a replacement instead of emitting control characters from an existing path."""
    return text if text.isprintable() else ''


def _preview(text: str) -> str:
    """Wrap paths by terminal cells rather than silently clipping their ends."""
    columns = max(10, terminal_size()[0] - 1)
    width = max(12, columns - max(20, columns // 2) - 3)
    return '\n'.join(line.plain for line in Text(_display_lines(text)).wrap(Console(), width, overflow='fold'))


def _display_lines(text: str) -> str:
    return '\n'.join(_display(line) for line in text.split('\n'))


def _menu_size() -> tuple[int, int]:
    # Leave room for Termflow's final newline so a full page does not scroll off its title.
    columns, rows = terminal_size()
    return columns, max(1, rows - 1)


def _directory_problem(path: Path) -> str | None:
    try:
        if not path.is_dir():
            return 'Directory does not exist or is not a directory.'
        next(path.iterdir(), None)
    except OSError as exc:
        return f'Cannot read directory: {exc.strerror or exc}'
    return None


class DirectoryPicker:
    """Browse directories, including hidden ones, without changing the process directory."""

    def __init__(self, *, start: Path, project: Path) -> None:
        self.current = start
        self.project = project
        self.error = ''

    def build(self) -> Menu:
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
        items = [
            MenuItem('Use this directory', value=True, disabled=problem is not None),
            MenuItem('.. (parent directory)', value=self.current.parent, disabled=self.current == self.current.parent),
            MenuItem('Go to path / view current...', value='path'),
            MenuItem('Project directory', value=self.project),
            MenuItem('Home directory', value=Path.home()),
            *[MenuItem(_display(child.name) + '/', value=child) for child in children],
        ]
        if problem or self.error:
            items.append(MenuItem(_display(problem or self.error), disabled=True))
        elif not children:
            items.append(MenuItem('No subdirectories. This directory can be used.', disabled=True))

        def matches(query: str, item: MenuItem) -> bool:
            query = query.casefold()
            # Termflow only dispatches navigation shortcuts while filtered rows exist.
            return query in item.label.casefold() or (
                item.value == 'path' and not any(query in row.label.casefold() for row in items)
            )

        return (
            MenuBuilder('Browse local directories')
            .size(_menu_size)
            .style(markdown_style())
            .items(items)
            .searchable(True)
            .filter_fn(matches)
            .preview(
                lambda item: _preview(
                    (f'{problem or self.error}\n\n' if problem or self.error else '')
                    + f'Current directory\n{self.current}\n\n'
                    + (f'Open\n{item.value}' if isinstance(item.value, Path) else item.label)
                    + '\n\nEnter opens a folder; Use this directory selects it. Esc returns without changing settings.'
                )
            )
            .on_key('left', lambda _menu, _item: MenuResult(item=MenuItem('', value=self.current.parent)))
            .on_key('ctrl-l', lambda _menu, _item: MenuResult(item=MenuItem('', value='path')))
            .footer_hint('Enter open/use | Left up | Ctrl+L path | Esc back')
            .key_source(menu_key)
            .build()
        )

    def run(self, *, runners: Runners) -> Path | None:
        while True:
            item = picked(runners.run_choice(self.build()))
            if item is None:
                return None
            self.error = ''
            if isinstance(item.value, Path):
                self.current = item.value
            elif item.value is True:
                self.error = _directory_problem(self.current) or ''
                if not self.error:
                    return self.current
            elif item.value == 'path':
                editor = (
                    TextInputBuilder('Go to directory')
                    .style(markdown_style())
                    .prompt('Path: ')
                    .initial(_safe_initial(str(self.current)))
                    .placeholder('Enter a path; control characters are not copied')
                    .validator(self.path_problem)
                    .footer_hint('Enter open - Esc back')
                    .key_source(menu_key)
                    .build()
                )
                result = runners.run_text(editor)
                if not result.cancelled and result.value is not None:
                    self.current = self.path(result.value)

    def path(self, text: str) -> Path:
        return (self.current / Path(text.strip()).expanduser()).resolve()

    def path_problem(self, text: str) -> str | None:
        if not text.strip():
            return 'Enter a directory path.'
        try:
            return _directory_problem(self.path(text))
        except (OSError, RuntimeError, ValueError) as exc:
            return f'Invalid directory: {exc}'


@dataclass(frozen=True, kw_only=True)
class FolderAction:
    """An entry action returned to the loop before opening another widget."""

    kind: Literal['path', 'name', 'browse', 'edit', 'remove']
    index: int | None = None


class FolderMenu(Generic[DepsT]):
    """Manage entries individually while retaining their order and stored representation."""

    def __init__(self, source: CoderSource[DepsT], *, project: Path) -> None:
        self.source = source
        self.project = project
        self.notice = ''

    def folders(self) -> list[str]:
        return self.source.host.settings(CoderSettings).agent_folders

    def details(self, item: MenuItem) -> str:
        return _preview((f'{self.notice}\n\n' if self.notice else '') + self._details(item))

    def _details(self, item: MenuItem) -> str:
        if is_save_and_close(item):
            return 'Changes are already saved. Return to Coder settings.'
        if isinstance(item.value, int):
            value = self.folders()[item.value]
            paths = CoderSettings(agent_folders=[value]).folders(home=Path.home())
            if is_folder_name(value):
                details = f'Folder name: {value}\n\nProject, then home:\n' + '\n'.join(paths)
                details += '\n\nMissing named locations are skipped.'
            else:
                try:
                    path = self.path(value)
                    details = f'Directory path\n{value}\n\nResolved location\n{path}\n\n'
                    details += _directory_problem(path) or 'Directory is readable.'
                except (OSError, RuntimeError, ValueError) as exc:
                    details = f'{value}\n\nCannot open directory: {exc}'
            return details
        return (
            'Load sub-agent definitions (*.md and *.toml), not files for code search.\n\n'
            'Add a directory path, browse local directories, or add a folder name such as agents or global. '
            'Names search .agents, .claude and .codex in the project and home.\n\n'
            'Changes save immediately. Removing an entry never deletes files.'
        )

    def build(self, *, initial: int = 0) -> Menu:
        folders = self.folders()
        items = [
            MenuItem(
                f'{index + 1}. {_display(value) if is_folder_name(value) else _display(Path(value).name) + "/"}',
                value=index,
                description='folder name' if is_folder_name(value) else _display(value),
            )
            for index, value in enumerate(folders)
        ]
        if not folders:
            items.append(MenuItem('No folders selected. Disk agents are off.', disabled=True))
        if not self.source.host.settings(CoderSettings).sub_agents:
            items.append(MenuItem('Sub-agents are disabled in Coder settings.', disabled=True))
        items += [
            MenuItem('+ Add directory path...', value=FolderAction(kind='path')),
            MenuItem('+ Browse local directories...', value=FolderAction(kind='browse')),
            MenuItem('+ Add folder name...', value=FolderAction(kind='name')),
            save_and_close_item(),
        ]
        if self.notice:
            items.insert(0, MenuItem(_display(self.notice), disabled=True))
        return (
            MenuBuilder('Agent folders')
            .size(_menu_size)
            .style(markdown_style())
            .items(items)
            .initial_index(min(initial + bool(self.notice), len(items) - 1))
            .preview(self.details)
            .on_key('a', lambda _menu, _item: MenuResult(item=MenuItem('', value=FolderAction(kind='path'))))
            .on_key('b', lambda _menu, _item: MenuResult(item=MenuItem('', value=FolderAction(kind='browse'))))
            .on_key('e', lambda _, item: self.action(item, kind='edit'))
            .on_key('d', lambda _, item: self.action(item, kind='remove'))
            .footer_hint('Enter menu | a add/b browse | e edit/d rm | Esc')
            .key_source(menu_key)
            .build()
        )

    def action(self, item: MenuItem, *, kind: Literal['edit', 'remove']) -> MenuResult | None:
        if not isinstance(item.value, int):
            return None
        return MenuResult(item=MenuItem('', value=FolderAction(kind=kind, index=item.value)))

    def actions(self, index: int) -> Menu:
        return (
            MenuBuilder('Manage agent folder')
            .size(_menu_size)
            .style(markdown_style())
            .items(
                [
                    MenuItem('Edit name or path...', value=FolderAction(kind='edit', index=index)),
                    MenuItem('Replace by browsing...', value=FolderAction(kind='browse', index=index)),
                    MenuItem('Remove from search (keep files)', value=FolderAction(kind='remove', index=index)),
                    MenuItem('Back', value=None),
                ]
            )
            .preview(lambda _: self.details(MenuItem('', value=index)))
            .footer_hint('Enter select - Esc back')
            .key_source(menu_key)
            .build()
        )

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
                    duplicate = (
                        candidate.samefile(other)
                        if isinstance(candidate, Path) and isinstance(other, Path)
                        else candidate == other
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                if duplicate:
                    return 'This folder is already in the list.'
            if not named:
                return _directory_problem(self.path(value))
        except (OSError, RuntimeError, ValueError) as exc:
            return f'Invalid directory: {exc}'
        return None

    def path(self, value: str) -> Path:
        return (self.project / Path(value).expanduser()).resolve()

    def editor(self, *, named: bool, index: int | None) -> TextInput:
        return (
            TextInputBuilder('Edit agent folder' if index is not None else 'Add agent folder')
            .style(markdown_style())
            .prompt('Name: ' if named else 'Path: ')
            .initial(_safe_initial(self.folders()[index]) if index is not None else '')
            .placeholder('agents or global' if named else './team-agents or ~/my-agents (relative to project)')
            .validator(lambda text: self.problem(text, named=named, index=index))
            .footer_hint('Enter save | Esc cancel | Empty never removes')
            .key_source(menu_key)
            .build()
        )

    def run(self, *, runners: Runners = TERMINAL) -> list[str]:
        messages: list[str] = []
        cursor = 0
        while True:
            item = picked(runners.run_list(self.build(initial=cursor)))
            if item is None:
                return messages
            if isinstance(item.value, int):
                cursor = item.value
                item = picked(runners.run_choice(self.actions(item.value)))
            if item is None or not isinstance(item.value, FolderAction):
                continue
            action = item.value
            self.notice = ''
            folders = self.folders().copy()
            if action.index is not None:
                cursor = action.index
            try:
                if action.kind == 'remove':
                    assert action.index is not None
                    folders.pop(action.index)
                else:
                    value = self.edit(action, runners=runners)
                    if value is None:
                        continue
                    if action.index is None:
                        cursor = len(folders)
                        folders.append(value)
                    else:
                        folders[action.index] = value
                data = self.source.host.settings(CoderSettings).model_dump(mode='json')
                data['agent_folders'] = folders
                self.source.host.save_settings(CoderSettings.model_validate(data))
            except (OSError, RuntimeError, ValueError, SQLiteError) as exc:
                self.notice = f'Could not save: {exc}'
                continue
            self.notice = (
                'Folder removed. Files were not changed.' if action.kind == 'remove' else 'Agent folder saved.'
            )
            messages.append(self.notice)

    def edit(self, action: FolderAction, *, runners: Runners) -> str | None:
        current = self.folders()[action.index] if action.index is not None else ''
        named = action.kind == 'name' or (action.kind == 'edit' and is_folder_name(current))
        if action.kind == 'browse':
            start = (
                self.project if not current or is_folder_name(current) else self.project / Path(current).expanduser()
            )
            chosen = DirectoryPicker(start=start, project=self.project).run(runners=runners)
            if chosen is None:
                return None
            value = str(chosen)
        else:
            result = runners.run_text(self.editor(named=named, index=action.index))
            if result.cancelled or result.value is None:
                return None
            value = self.value(result.value, named=named)
        problem = self.problem(value, named=named, index=action.index)
        if problem:
            self.notice = problem
            return None
        return value


def run_coder_flow(source: CoderSource[DepsT], *, runners: Runners = TERMINAL) -> list[str]:
    """Keep scalar settings in the shared editor and delegate the folder list."""
    folders = FolderMenu(source, project=Path.cwd())
    return run_flow(FieldMenu(source), runners, submenus={'agent_folders': lambda: folders.run(runners=runners)})
