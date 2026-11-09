"""The team's roster, a teammate's live transcript, and the shared board, as live terminal menus.

Each menu reads the team again on every key poll, so it updates as teammates claim and finish work.
The menus run in a worker thread; everything they read or do runs on the application loop.
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from rich.console import Console
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size
from termflow.tui.textinput import TextInput

from pydantic_ai_harness.planning import OwnedPlanItem, TaskStatus
from pydantic_ai_harness.step_persistence.conversations import conversation_text
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_ai_harness.teams import TeamMember, TeamSnapshot
from pydantic_clai2.runtime.team import display_name, item_label
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.rendering.rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text

ActionKind = Literal['message', 'stop', 'assign', 'unclaim', 'cancel']
ResultT = TypeVar('ResultT')


@dataclass(frozen=True, kw_only=True)
class TeamAction:
    """Something the user does to the team from a menu: `target` is a teammate's name or a board item's ID."""

    kind: ActionKind
    target: str
    value: str = ''
    """The message's text, or the teammate an item is assigned to."""


class TeamControls(Protocol):
    """What the menus read and do, on the application loop."""

    def snapshot(self) -> TeamSnapshot:
        """The team as last seen."""
        ...

    def member(self, name: str) -> DelegationTask | None:
        """A display copy of a teammate's record, with what it has streamed so far."""
        ...

    async def act(self, action: TeamAction) -> str:
        """Do `action`, returning a notice; raises `ValueError` or `LookupError` when it cannot."""
        ...


@dataclass(frozen=True)
class _Open:
    """A menu closed to open another widget: a teammate's transcript, a message, or the assign picker."""

    kind: Literal['focus', 'message', 'assign', 'board']
    target: str = ''


_BOARD_ORDER = (
    TaskStatus.in_progress,
    TaskStatus.pending,
    TaskStatus.blocked,
    TaskStatus.completed,
    TaskStatus.cancelled,
)


def _close(target: _Open) -> MenuResult:
    return MenuResult(item=MenuItem(target.kind, value=target))


def member_label(member: TeamMember) -> str:
    """A teammate on one line: name, type, model, state, and the item it works on."""
    task = f' · #{member.current_task}' if member.current_task is not None else ''
    model = member.model or 'session model'
    return terminal_text(f'{member.name} · {display_name(member.agent_name)} · {model} · {member.state}{task}')


class TeamView:
    """The menus' copy of the team, refreshed on each key poll, and the notice of the last action."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], TeamSnapshot],
        member: Callable[[str], DelegationTask | None],
        act: Callable[[TeamAction], str],
        key_source: Callable[[], str] = menu_key,
    ) -> None:
        self._snapshot = snapshot
        self._member = member
        self._act = act
        self.key_source = key_source
        self.team = snapshot()
        self.notice = ''
        self.selected: str | None = None
        self.board_order: list[str] = []
        """The board's rows, by item ID, in the order they first showed while the menus are open."""

    def refresh(self) -> None:
        self.team = self._snapshot()

    def member(self, name: str) -> DelegationTask | None:
        return self._member(name)

    def act(self, action: TeamAction) -> None:
        """Do `action`, keeping its notice or why it could not be done for the menu to show."""
        try:
            self.notice = self._act(action)
        except (ValueError, LookupError) as exc:
            self.notice = str(exc)

    def transcript(self, name: str) -> str:
        """A teammate's state, task, and conversation so far, as plain text."""
        member = next((member for member in self.team.members if member.name == name), None)
        record = self.member(name)
        if member is None or record is None:
            return self.notice or f'No teammate {name!r}.'
        text = f'{member_label(member)}\n{self.notice}\n\nTask: {record.prompt}\n\n' + conversation_text(
            record.messages
        )
        return terminal_text(text)

    def read_key(self, menu: Menu, items: Callable[[], list[MenuItem]]) -> str:
        key = self.key_source()
        self.refresh()
        menu.replace_items(items())
        return key or 'refresh'


class RosterMenu:
    """The teammates, one row each: Enter opens one's live transcript, `m` messages it, `x` stops it."""

    def __init__(self, view: TeamView) -> None:
        self.view = view

    def items(self) -> list[MenuItem]:
        return [MenuItem(member_label(member), value=member.name) for member in self.view.team.members] or [
            MenuItem('No teammates yet: /team spawn NAME PROMPT starts one', disabled=True)
        ]

    def preview(self, item: MenuItem) -> str:
        if not isinstance(item.value, str):
            return self.view.notice
        self.view.selected = item.value
        return self.view.transcript(item.value)

    def _stop(self, item: MenuItem) -> None:
        if isinstance(item.value, str):
            self.view.act(TeamAction(kind='stop', target=item.value))

    def _message(self, item: MenuItem) -> MenuResult | None:
        return _close(_Open('message', item.value)) if isinstance(item.value, str) else None

    def build(self) -> Menu:
        builder = (
            MenuBuilder('Team · teammates')
            .style(markdown_style())
            .items(self.items())
            .preview(self.preview)
            .list_width(48)
            .initial_index(next((i for i, item in enumerate(self.items()) if item.value == self.view.selected), 0))
        )
        menu = slash_search(
            builder,
            footer='Enter transcript - m message - x stop - b board - Esc close',
            key_source=lambda: self.view.read_key(menu, self.items),
            hotkeys={
                'm': lambda menu, item: self._message(item),
                'x': lambda menu, item: self._stop(item),
                'b': lambda menu, item: _close(_Open('board')),
            },
        )
        return menu


class MemberFocus:
    """One teammate's transcript, live, full width: `m` messages it, `x` stops it, Esc goes back."""

    def __init__(self, view: TeamView, name: str, *, size: Callable[[], tuple[int, int]] = terminal_size) -> None:
        self.view = view
        self.name = name
        self.size = size

    def items(self) -> list[MenuItem]:
        # Account for Termflow's pointer, edge margin and deferred-wrap column.
        width = max(1, self.size()[0] - 5)
        lines = Text(self.view.transcript(self.name)).wrap(Console(width=width), width, overflow='fold')
        return [MenuItem(line.plain or ' ', value=self.name) for line in lines]

    def build(self) -> Menu:
        menu = (
            MenuBuilder(f'Teammate {self.name} · live transcript')
            .style(markdown_style())
            .items(self.items())
            .searchable(False)
            .size(self.size)
            .initial_index(max(0, len(self.items()) - 1))
            .footer_hint('↑↓ scroll - m message - x stop - Esc back')
            .on_key('enter', lambda menu, item: None)
            .on_key('m', lambda menu, item: _close(_Open('message', self.name)))
            .on_key('x', lambda menu, item: self.view.act(TeamAction(kind='stop', target=self.name)))
            .key_source(lambda: self.view.read_key(menu, self.items))
            .build()
        )
        return menu


class BoardMenu:
    """The shared board, with each item's status, owner and what it waits for: `a` assign, `u` unclaim, `c` cancel."""

    def __init__(self, view: TeamView) -> None:
        self.view = view

    def items(self) -> list[MenuItem]:
        """One row per item, by status when the board opened; rows never move while it is open.

        An item keeps its row as its status changes, so the highlighted row is still the item the next
        key acts on; an item added meanwhile joins at the end.
        """
        board = self.view.team.board
        by_id = {item.id: item for item in board}
        rank = {status: index for index, status in enumerate(_BOARD_ORDER)}
        order = self.view.board_order
        order.extend(item.id for item in sorted(board, key=lambda item: rank[item.status]) if item.id not in order)
        return [
            MenuItem(item_label(by_id[item_id], board=board), value=item_id) for item_id in order if item_id in by_id
        ] or [MenuItem('The board is empty: the lead adds items with board_add', disabled=True)]

    def _item(self, item: MenuItem) -> OwnedPlanItem | None:
        return next((entry for entry in self.view.team.board if entry.id == item.value), None)

    def preview(self, item: MenuItem) -> str:
        entry = self._item(item)
        if entry is None:
            return self.view.notice
        self.view.selected = entry.id
        after = ', '.join(f'#{dependency}' for dependency in entry.depends_on) or 'nothing'
        return terminal_text(
            f'#{entry.id} · {entry.status.value} · {entry.owner or "unowned"}\nAfter: {after}\n'
            f'{self.view.notice}\n\n{entry.content}'
        )

    def _act(self, item: MenuItem, kind: ActionKind) -> None:
        if (entry := self._item(item)) is not None:
            self.view.act(TeamAction(kind=kind, target=entry.id))

    def _assign(self, item: MenuItem) -> MenuResult | None:
        entry = self._item(item)
        return None if entry is None else _close(_Open('assign', entry.id))

    def _focus(self, item: MenuItem) -> MenuResult | None:
        entry = self._item(item)
        owner = None if entry is None else entry.owner
        if owner is None or owner not in {member.name for member in self.view.team.members}:
            return None
        return _close(_Open('focus', owner))

    def build(self) -> Menu:
        builder = (
            MenuBuilder('Team · board')
            .style(markdown_style())
            .items(self.items())
            .preview(self.preview)
            .list_width(56)
            .initial_index(next((i for i, item in enumerate(self.items()) if item.value == self.view.selected), 0))
        )
        menu = slash_search(
            builder,
            footer="Enter owner's transcript - a assign - u unclaim - c cancel - Esc close",
            key_source=lambda: self.view.read_key(menu, self.items),
            hotkeys={
                'enter': lambda menu, item: self._focus(item),
                'a': lambda menu, item: self._assign(item),
                'u': lambda menu, item: self._act(item, 'unclaim'),
                'c': lambda menu, item: self._act(item, 'cancel'),
            },
        )
        return menu


def _assignee_menu(view: TeamView, item_id: str) -> Menu:
    return (
        MenuBuilder(f'Assign #{item_id} to')
        .style(markdown_style())
        .items([MenuItem(member_label(member), value=member.name) for member in view.team.members])
        .searchable(False)
        .footer_hint('Enter assign - Esc back')
        .build()
    )


def _message_input(name: str) -> TextInput:
    return (
        TextInputBuilder(f'Message {name}')
        .style(markdown_style())
        .prompt('> ')
        .placeholder('It reads it at its next model request; an idle teammate wakes')
        .build()
    )


def _opened(result: MenuResult) -> _Open | None:
    if result.cancelled or result.item is None:
        return None
    value = result.item.value
    return value if isinstance(value, _Open) else _Open('focus', value) if isinstance(value, str) else None


def _message(view: TeamView, name: str, runners: Runners) -> None:
    typed = runners.run_text(_message_input(name))
    if not typed.cancelled and typed.value is not None and typed.value.strip():
        view.act(TeamAction(kind='message', target=name, value=typed.value))


def _focus(view: TeamView, name: str, runners: Runners) -> None:
    """A teammate's transcript until Esc, with `m` asking for a message and coming back."""
    while (opened := _opened(runners.run_choice(MemberFocus(view, name).build()))) is not None:
        _message(view, opened.target, runners)


def run_board(view: TeamView, *, runners: Runners = TERMINAL) -> None:
    """The board until Esc; assigning asks for the teammate, and Enter opens the owner's transcript."""
    while (opened := _opened(runners.run_list(BoardMenu(view).build()))) is not None:
        if opened.kind == 'assign':
            picked = runners.run_choice(_assignee_menu(view, opened.target))
            if not picked.cancelled and picked.item is not None and isinstance(picked.item.value, str):
                view.act(TeamAction(kind='assign', target=opened.target, value=picked.item.value))
        else:
            _focus(view, opened.target, runners)


def run_roster(view: TeamView, *, runners: Runners = TERMINAL) -> None:
    """The roster until Esc: Enter opens a transcript, `m` asks for a message, `b` opens the board."""
    while (opened := _opened(runners.run_list(RosterMenu(view).build()))) is not None:
        if opened.kind == 'board':
            run_board(view, runners=runners)
        elif opened.kind == 'message':
            _message(view, opened.target, runners)
        else:
            _focus(view, opened.target, runners)


async def open_team(
    controls: TeamControls,
    *,
    board: bool = False,
    runners: Runners = TERMINAL,
    key_source: Callable[[], str] = menu_key,
) -> str:
    """Open the roster, or with `board` the board, while the team works on; returns the last action's notice."""
    loop = asyncio.get_running_loop()

    def on_loop(call: Callable[[], Coroutine[object, object, ResultT]]) -> ResultT:
        return asyncio.run_coroutine_threadsafe(call(), loop).result()

    async def snapshot() -> TeamSnapshot:
        return controls.snapshot()

    async def member(name: str) -> DelegationTask | None:
        return controls.member(name)

    def run() -> str:
        view = TeamView(
            snapshot=lambda: on_loop(snapshot),
            member=lambda name: on_loop(lambda: member(name)),
            act=lambda action: on_loop(lambda: controls.act(action)),
            key_source=key_source,
        )
        (run_board if board else run_roster)(view, runners=runners)
        return view.notice

    return await run_worker(run)
