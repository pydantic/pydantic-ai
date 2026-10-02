"""Live task picker and child transcript inspection over the native terminal menu."""

import asyncio
from collections.abc import Callable, Sequence

from rich.console import Console
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size

from pydantic_ai_harness.step_persistence.conversations import conversation_text
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_clai2.runtime.tasks import Tasks, task_tree
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text


class TaskMenu:
    """Polling snapshots run on the application loop; the menu thread owns only copied state."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], Sequence[DelegationTask]],
        action: Callable[[str, str], None],
        key_source: Callable[[], str] = menu_key,
    ) -> None:
        self.snapshot = snapshot
        self.action = action
        self.key_source = key_source
        self.records = list(snapshot())
        self.order = [record.id for _, record in task_tree(self.records)]
        self.offset = 0
        self.notice = ''
        self.selected: str | None = None

    def items(self) -> list[MenuItem]:
        by_id = {record.id: (depth, record) for depth, record in task_tree(self.records)}
        self.order.extend(task_id for task_id in by_id if task_id not in self.order)
        return [
            MenuItem(
                f'{"  " * depth}{record.agent_name} [{record.id[:8]}] {record.outcome or record.status}',
                value=record.id,
            )
            for task_id in self.order
            if task_id in by_id
            for depth, record in [by_id[task_id]]
        ] or [MenuItem('No delegated tasks in this conversation', disabled=True)]

    def preview(self, item: MenuItem) -> str:
        record = next((record for record in self.records if record.id == item.value), None)
        if record is None:
            return self.notice
        if self.selected != record.id:
            self.selected, self.offset = record.id, 0
        text = (
            f'{record.agent_name} · {record.id}\n{record.outcome or record.status} · '
            f'{"background" if record.background else "foreground"}\n'
            f'{self.notice}\n\nTask: {record.prompt}\n\n'
            + conversation_text(record.messages)
            + (f'\n\nResult ({record.outcome}):\n{record.output}' if record.status == 'finished' else '')
        )
        return '\n'.join(terminal_text(text).splitlines()[self.offset :])

    def act(self, menu: Menu, item: MenuItem, action: str) -> MenuResult | None:
        if isinstance(item.value, str):
            try:
                self.action(action, item.value)
                self.notice = f'{action} requested for {item.value[:8]}'
            except (ValueError, KeyError) as exc:
                self.notice = str(exc)
        return None

    def scroll(self, amount: int) -> None:
        self.offset = max(0, self.offset + amount)

    def build(self) -> Menu:
        def read_key() -> str:
            key = self.key_source()
            self.records = list(self.snapshot())
            menu.replace_items(self.items())
            return key or 'refresh'

        menu = (
            MenuBuilder('Tasks · live child transcripts')
            .style(markdown_style())
            .items(self.items())
            .preview(self.preview)
            .list_width(42)
            .initial_index(next((index for index, item in enumerate(self.items()) if item.value == self.selected), 0))
            .footer_hint('Enter transcript · ↑↓ task · b background · x stop tree · Esc close')
            .on_key('b', lambda menu, item: self.act(menu, item, 'background'))
            .on_key('x', lambda menu, item: self.act(menu, item, 'stop'))
            .on_key(']', lambda menu, item: self.scroll(10))
            .on_key('[', lambda menu, item: self.scroll(-10))
            .key_source(read_key)
            .build()
        )
        return menu


class TaskDetail:
    """Full-width transcript rows rendered and scrolled by Termflow."""

    def __init__(
        self,
        *,
        picker: TaskMenu,
        task_id: str,
        size: Callable[[], tuple[int, int]] = terminal_size,
    ) -> None:
        self.picker = picker
        self.task_id = task_id
        self.size = size

    def items(self) -> list[MenuItem]:
        self.picker.records = list(self.picker.snapshot())
        self.picker.offset = 0
        text = self.picker.preview(MenuItem('', value=self.task_id))
        # Account for Termflow's pointer, edge margin and deferred-wrap column.
        width = max(1, self.size()[0] - 5)
        lines = Text(text).wrap(Console(width=width), width, overflow='fold')
        return [MenuItem(line.plain or ' ', value=self.task_id) for line in lines] or [
            MenuItem('Task transcript unavailable', value=self.task_id)
        ]

    def build(self) -> Menu:
        def read_key() -> str:
            key = self.picker.key_source()
            menu.replace_items(self.items())
            return key or 'refresh'

        menu = (
            MenuBuilder(f'Task {self.task_id[:8]} · transcript')
            .style(markdown_style())
            .items(self.items())
            .searchable(False)
            .size(self.size)
            .footer_hint('↑↓ scroll · Home/End · b background · x stop tree · Esc tasks')
            .on_key('enter', lambda menu, item: None)
            .on_key('b', lambda menu, item: self.picker.act(menu, item, 'background'))
            .on_key('x', lambda menu, item: self.picker.act(menu, item, 'stop'))
            .key_source(read_key)
            .build()
        )
        return menu


def run_tasks(picker: TaskMenu, *, runners: Runners = TERMINAL) -> None:
    """Drive picker and detail with injectable widget runners."""
    while True:
        result = runners.run_list(picker.build())
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return
        picker.selected = result.item.value
        runners.run_choice(TaskDetail(picker=picker, task_id=result.item.value).build())


async def open_tasks(tasks: Tasks, *, runners: Runners = TERMINAL) -> str:
    """Inspect live snapshots without holding up child execution."""
    loop = asyncio.get_running_loop()

    async def snapshot() -> tuple[DelegationTask, ...]:
        return tasks.snapshots()

    async def action(name: str, task_id: str) -> None:
        if name == 'stop':
            await tasks.owner.cancel(task_id)
            await tasks.owner.save(tasks.owner.records[task_id])
        else:
            tasks.owner.background(task_id)

    def run() -> None:
        picker = TaskMenu(
            snapshot=lambda: asyncio.run_coroutine_threadsafe(snapshot(), loop).result(),
            action=lambda name, task_id: asyncio.run_coroutine_threadsafe(action(name, task_id), loop).result(),
        )
        run_tasks(picker, runners=runners)

    await run_worker(run)
    return ''
