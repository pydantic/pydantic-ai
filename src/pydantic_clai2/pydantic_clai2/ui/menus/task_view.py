"""`/tasks`: a live view of this conversation's sub-agents, built on `termflow.live`.

A sidebar lists the tasks, the transcript follows the selected one as it streams, and the input steers it.
"""

import asyncio
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import accumulate
from textwrap import wrap
from typing import Literal

from termflow.live import (
    KeyEvent,
    KeyState,
    LiveApp,
    MarkdownView,
    Rect,
    Region,
    TextLog,
    Widget,
    Window,
    hsplit,
    vsplit,
)
from termflow.tui.keys import Key

from pydantic_ai.messages import TextPart, ToolCallPart, UserPromptPart
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_clai2.plugins import TaskInfo
from pydantic_clai2.runtime.tasks import Tasks, task_tree
from pydantic_clai2.ui.menus.menu_worker import run_worker, worker_stopping
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.history import prompt_text
from pydantic_clai2.ui.rendering.rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text

Action = Literal['steer', 'prompt', 'stop', 'background', 'resume']

_NEEDS: dict[Action, str] = {
    'steer': 'running',
    'stop': 'running',
    'background': 'running',
    'prompt': 'finished',
    'resume': 'finished',
}
_KEYS: dict[str, Action] = {'ctrl-x': 'stop', 'ctrl-b': 'background', 'ctrl-r': 'resume'}
_MOVES = {Key.UP: -1, Key.DOWN: 1}
_SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
_SETTLED = {'ok': ('✓', theme.SUCCESS), 'cancelled': ('■', theme.WARNING)}
_RESET = '\x1b[0m'
_FPS = 20.0
_REFRESH = 0.25
"""Seconds between snapshots of the tasks."""
_TAIL = 8_000
"""Characters of transcript kept on screen; Markdown is rendered again whenever it changes."""
_SIDEBAR = 34
_MIN_WIDTH = 72
"""Columns below which the sidebar gives its space to the transcript."""
_MIN_HEIGHT = 20
"""Rows below which the activity log gives its space to the sidebar."""


@dataclass(frozen=True, kw_only=True)
class Row:
    """One task in the sidebar: its depth under the task that started it, its record, and how plugins see it."""

    depth: int
    task: DelegationTask
    info: TaskInfo


def task_rows(tasks: Tasks) -> list[Row]:
    """This conversation's tasks as a tree, each with the text it is streaming."""
    return [Row(depth=depth, task=task, info=tasks.info(task)) for depth, task in task_tree(tasks.snapshots())]


async def perform(tasks: Tasks, action: Action, task_id: str, text: str = '') -> str:
    """Do `action` on a task and say what happened. `text` is the message a steer or prompt carries.

    Raises:
        LookupError: this conversation has no such task.
        ValueError: the task finished before it could read a steer, or cannot do `action`.
    """
    label = task_id[:8]
    if action == 'stop':
        await tasks.stop(task_id)
        return f'Stopped {label}.'
    if action == 'background':
        tasks.background(task_id)
        return f'Backgrounded {label}.'
    if action == 'resume':
        await tasks.resume(task_id)
        return f'Resuming {label}.'
    if action == 'prompt':
        await tasks.prompt(task_id, text)
        return f'Prompted {label}.'
    if not tasks.steer(task_id, text):
        raise ValueError(f'{label} just finished: press Enter again to prompt it.')
    return f'Steered {label}.'


def transcript_blocks(task: DelegationTask) -> list[str]:
    """`task`'s conversation as Markdown blocks: what it was told, quoted, what it said, and the tools it called."""
    parts = [part for message in task.messages for part in message.parts]
    blocks = [f'*Directory: {task.working_directory}*'] if task.working_directory is not None else []
    if not any(isinstance(part, UserPromptPart) for part in parts):
        blocks.append(_quote(task.prompt))
    for part in parts:
        if isinstance(part, UserPromptPart):
            blocks.append(_quote(prompt_text(part)))
        elif isinstance(part, TextPart):
            blocks.append(part.content)
        elif isinstance(part, ToolCallPart):
            blocks.append(f'- `{part.tool_name}` {_clip(part.args_as_json_str(), 96)}')
    if task.status == 'finished' and task.outcome != 'ok' and task.output:
        blocks.append(f'**{task.outcome}:** {task.output}')
    return [terminal_text(block) for block in blocks if block.strip()]


def _quote(text: str) -> str:
    return '\n'.join(f'> {line}' for line in text.splitlines())


def _clip(text: str, width: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= width else text[: width - 1] + '…'


def _age(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h{minutes:02}m'
    return f'{minutes}m{seconds:02}s' if minutes else f'{seconds}s'


def _layout(area: Rect) -> list[Rect]:
    """Transcript over the input, beside the sidebar over the activity log; a small terminal drops the last two."""
    if area.width < _MIN_WIDTH:
        return vsplit(area, max(1, area.height - 3), 3)
    side, main = hsplit(area, _SIDEBAR, area.width - _SIDEBAR)
    panes = vsplit(main, max(1, main.height - 3), 3)
    if area.height < _MIN_HEIGHT:
        return [*panes, side]
    return [*panes, *vsplit(side, side.height - 7, 7)]


class TaskView:
    """The live `/tasks` view: the tasks, the selected one's transcript, and the message being typed to it."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], Sequence[Row]],
        act: Callable[[Action, str, str], str],
        size: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """`snapshot` reads the tasks and `act` does an action on one, both from the view's thread."""
        self.snapshot = snapshot
        self.act = act
        self.rows: Sequence[Row] = ()
        self.selected: str | None = None
        self.draft = ''
        self.notice = ''
        self.elapsed = 0.0
        self.log = TextLog()
        self.transcript = Window('Transcript', MarkdownView())
        self.input = Window('Steer', _Input(self))
        self.sidebar = Window('Tasks', _Sidebar(self))
        self._tail = _Tail(self.transcript)
        self._states: dict[str, str] = {}
        self._refreshed = 0.0
        self.app = LiveApp(
            [self.transcript, self.input, self.sidebar, Window('Activity', self.log)],
            _layout,
            title='/tasks',
            fps=_FPS,
            size=size,
        )
        self.app.focus(self.input)
        self.refresh()

    @property
    def current(self) -> Row | None:
        """The selected task's row."""
        return next((row for row in self.rows if row.task.id == self.selected), None)

    @property
    def mode(self) -> Literal['steer', 'prompt'] | None:
        """What Enter does with the message: steer a running task, prompt a finished one that can resume."""
        row = self.current
        if row is None:
            return None
        if row.info.status == 'running':
            return 'steer'
        return 'prompt' if row.info.resumable else None

    @property
    def placeholder(self) -> str:
        """What the empty input shows: what typing would do to the selected task."""
        row = self.current
        if row is None:
            return 'No tasks yet: ask the agent to delegate work.'
        if self.mode is None:
            return f'{row.info.agent} cannot be resumed, so it takes no messages.'
        return f'Type to {self.mode} {row.info.agent}{"" if self.mode == "steer" else " again"}, then Enter.'

    def say(self, text: str) -> None:
        """Show `text` over the input until the selection moves, and keep it in the activity log."""
        self.notice = terminal_text(text, keep='')
        stamp = f'{theme.sgr(theme.MUTED)}{time.strftime("%H:%M:%S")}{_RESET} '
        self.log.write(stamp + ('\n' + ' ' * 9).join(wrap(self.notice, _SIDEBAR - 11)) + '\n')

    def tick(self, seconds: float) -> None:
        """Advance the view's clock, refreshing when a snapshot is due and closing when the owner asks."""
        self.elapsed += seconds
        if worker_stopping():
            self.app.quit()
        elif self.elapsed - self._refreshed >= _REFRESH:
            self.refresh()

    def refresh(self) -> None:
        """Read the tasks again, keep the selection on its task, and report each one that changed state."""
        self._refreshed = self.elapsed
        self.rows = self.snapshot()
        if self.current is None:
            self.selected = self.rows[0].task.id if self.rows else None
        for row in self.rows:
            state = row.info.outcome or row.info.status
            if self._states.setdefault(row.task.id, state) != state:
                self._states[row.task.id] = state
                self.say(f'{row.task.id[:8]} is now {state}.')
        running = sum(row.info.status == 'running' for row in self.rows)
        self.sidebar.title = f'Tasks · {running} running' if running else 'Tasks'
        self._show()

    def _show(self) -> None:
        row = self.current
        mode = self.mode or ('finished' if row else 'steer')
        self.input.title = ' · '.join(filter(None, [mode.capitalize(), self.notice]))
        self.app.hints = ' · '.join(['↑↓ tasks', 'PgUp/PgDn scroll', *_hints(row), 'Esc close'])
        if row is None:
            self.transcript.title = 'Transcript'
            self._tail.show(None, [])
            return
        info = row.info
        state = info.activity if info.status == 'running' else info.outcome
        title = ' · '.join(filter(None, [info.agent, info.id[:8], info.model, state]))
        self.transcript.title = terminal_text(title, keep='')
        self._tail.show(info.id, transcript_blocks(row.task))

    def move(self, delta: int) -> None:
        """Select the task `delta` rows away, stopping at either end."""
        ids = [row.task.id for row in self.rows]
        if self.selected in ids:
            self.selected = ids[max(0, min(len(ids) - 1, ids.index(self.selected) + delta))]
            self.notice = ''
            self._show()

    def run(self, action: Action, text: str = '') -> bool:
        """Do `action` on the selected task and say how it went. Resuming hands the terminal back to the agent."""
        row = self.current
        if row is None:
            return False
        done = False
        if row.info.status != _NEEDS[action]:
            self.say(f'{action.capitalize()} needs a {_NEEDS[action]} task.')
        else:
            try:
                self.say(self.act(action, row.task.id, text))
                done = True
            except (LookupError, ValueError, RuntimeError) as exc:
                self.say(str(exc))
        if done and action == 'resume':
            self.app.quit()
        self.refresh()
        return done

    def submit(self) -> None:
        """Send the typed message to the selected task, keeping it when it could not be sent."""
        text, mode = self.draft.strip(), self.mode
        if text and mode is not None and self.run(mode, text):
            self.draft = ''

    def escape(self) -> None:
        """Clear the typed message, or close the view when there is none."""
        if self.draft:
            self.draft = ''
        else:
            self.app.quit()


@dataclass
class _Tail:
    """The transcript window's Markdown: the selected task's newest blocks, appended to while they only grow."""

    window: Window
    task_id: str | None = None
    first: int = 0
    text: str | None = None
    view: MarkdownView | None = None

    def show(self, task_id: str | None, blocks: Sequence[str]) -> None:
        if task_id != self.task_id:
            self.task_id, self.first, self.text = task_id, 0, None
        if sum(map(len, blocks[self.first :])) > 2 * _TAIL:
            sizes = list(accumulate(len(block) for block in reversed(blocks)))
            self.first = len(blocks) - max(1, bisect_right(sizes, _TAIL))
        hidden = ['*… earlier messages are hidden …*'] if self.first else []
        text = '\n\n'.join([*hidden, *blocks[self.first :]])
        if self.view is not None and self.text is not None and text.startswith(self.text):
            self.view.append(text[len(self.text) :])
        else:
            self.window.widget = self.view = MarkdownView(text, style=markdown_style())
        self.text = text


class _Sidebar(Widget):
    """The task tree: two lines a task, with the selected one marked and kept on screen."""

    focusable = False

    def __init__(self, view: TaskView) -> None:
        self.view = view

    def draw(self, region: Region, focused: bool) -> None:
        view = self.view
        visible = max(1, region.height // 2)
        selected = next((index for index, row in enumerate(view.rows) if row.task.id == view.selected), 0)
        first = max(0, selected - visible + 1)
        frame = _SPINNER[int(view.elapsed * 10) % len(_SPINNER)]
        for line, row in enumerate(view.rows[first : first + visible]):
            head, detail = _row_lines(row, selected=row.task.id == view.selected, frame=frame, width=region.width)
            region.ansi(0, line * 2, head)
            region.ansi(0, line * 2 + 1, detail)


def _hints(row: Row | None) -> list[str]:
    """The control keys the selected task answers to."""
    if row is None:
        return []
    if row.info.status == 'running':
        return [*(['^B background'] if row.info.backgroundable else []), '^X stop']
    return ['^R resume'] if row.info.resumable else []


def _row_lines(row: Row, *, selected: bool, frame: str, width: int) -> tuple[str, str]:
    """A task's two sidebar lines: marker, state glyph, name, ID and age, then what it is doing or how it ended."""
    info = row.info
    running = info.status == 'running'
    glyph, role = (frame, theme.ACCENT) if running else _SETTLED.get(info.outcome or '', ('✗', theme.ERROR))
    indent = '  ' * row.depth
    marker = f'{theme.sgr(theme.ACCENT)}❯{_RESET}' if selected else ' '
    age = _age(((info.finished_at or datetime.now(UTC)) - info.started_at).total_seconds())
    name = terminal_text(f'{info.agent} {info.id[:8]}', keep='')
    gap = ' ' * max(1, width - len(indent) - len(name) - len(age) - 4)
    head = (
        f'{marker}{indent} {theme.sgr(role)}{glyph} {theme.sgr(theme.INFO, bold=selected)}{name}'
        f'{theme.sgr(theme.MUTED)}{gap}{age}{_RESET}'
    )
    state = info.activity if running else info.outcome or info.status
    mode = ' · bg' if running and info.background else ''
    tone = theme.sgr(theme.MUTED if running else role)
    return head, f'{indent}    {tone}{terminal_text(state, keep="")}{mode}{_RESET}'


class _Input(Widget):
    """The view's one focused widget: it holds the typed message and routes every key."""

    def __init__(self, view: TaskView) -> None:
        self.view = view

    def update(self, dt: float, keys: KeyState) -> None:
        self.view.tick(dt)

    def on_key(self, event: KeyEvent) -> bool:
        view = self.view
        if event.key in _MOVES:
            view.move(_MOVES[event.key])
        elif event.key in (Key.PAGE_UP, Key.PAGE_DOWN):
            view.transcript.widget.on_key(event)
        elif event.key == Key.ENTER:
            view.submit()
        elif event.key == Key.BACKSPACE:
            view.draft = view.draft[:-1]
        elif event.key == Key.ESCAPE:
            view.escape()
        elif event.key in _KEYS:
            view.run(_KEYS[event.key])
        elif event.text.isprintable():
            view.draft += event.text
        return True

    def draw(self, region: Region, focused: bool) -> None:
        view = self.view
        if view.draft:
            region.ansi(0, 0, f'{theme.sgr(theme.ACCENT)}› {_RESET}{view.draft[-max(1, region.width - 3) :]}▌')
        else:
            region.ansi(0, 0, f'{theme.sgr(theme.MUTED)}› {terminal_text(view.placeholder, keep="")}{_RESET}')


async def open_tasks(tasks: Tasks) -> str:
    """Show the live view until the user closes it; the tasks keep running on the loop meanwhile."""
    loop = asyncio.get_running_loop()

    async def snapshot() -> list[Row]:
        return task_rows(tasks)

    def act(action: Action, task_id: str, text: str) -> str:
        return asyncio.run_coroutine_threadsafe(perform(tasks, action, task_id, text), loop).result()

    def view() -> None:
        TaskView(snapshot=lambda: asyncio.run_coroutine_threadsafe(snapshot(), loop).result(), act=act).app.run()

    await run_worker(view)
    return ''
