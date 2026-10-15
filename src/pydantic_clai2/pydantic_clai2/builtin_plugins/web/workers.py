"""The delegated tasks as browsers see them: a row for each, and each running one's transcript so far."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_clai2.builtin_plugins.web.protocol import (
    EventTranslator,
    Item,
    ItemAdded,
    ItemLog,
    ItemMessage,
    ServerMessage,
    TaskChanged,
    TaskEntry,
    TaskItems,
    Usage,
)
from pydantic_clai2.plugins import TaskEvent, TaskInfo


def task_entry(task: TaskInfo) -> TaskEntry:
    """A task's row, as browsers receive it."""
    return TaskEntry(
        id=task.id,
        parent_id=task.parent_id,
        agent=task.agent,
        model=task.model,
        prompt=task.prompt,
        status=task.status,
        outcome=task.outcome,
        started_at=task.started_at,
        finished_at=task.finished_at,
        background=task.background,
        resumable=task.resumable,
        activity=task.activity,
        usage=Usage(
            input_tokens=task.usage.input_tokens, output_tokens=task.usage.output_tokens, requests=task.usage.requests
        ),
    )


@dataclass(kw_only=True)
class _Live:
    """One run of a task: its events' translator, what browsers were sent of it, and its item ID prefix."""

    translator: EventTranslator
    log: ItemLog
    scope: str
    steered: int = 0


class Workers:
    """The current conversation's delegated tasks, from `on_task_event`.

    Only running tasks keep their transcript here, for a browser that connects while they run; a
    finished one's saved history serves browsers from then on.
    """

    def __init__(self, *, publish: Callable[[ServerMessage], None]) -> None:
        self._publish = publish
        self._sent: dict[str, TaskEntry] = {}
        self._live: dict[str, _Live] = {}
        self._runs = 0

    def observe(self, event: TaskEvent) -> None:
        """Send a changed row, and what the task's own run did, to browsers."""
        task = event.task
        entry = task_entry(task)
        if self._sent.get(task.id) != entry:
            self._sent[task.id] = entry
            self._publish(TaskChanged(task=entry))
        if task.status == 'finished':
            self._live.pop(task.id, None)
            return
        live = self._live.get(task.id) or self._start(task)
        if event.event is not None:
            self._send(task.id, live, live.translator.translate(event.event))

    def _start(self, task: TaskInfo) -> _Live:
        # Each run, a resumed one too, gets its own IDs, as a browser may still show the last run's.
        self._runs += 1
        scope = f'task{self._runs}-'
        live = self._live[task.id] = _Live(translator=EventTranslator(scope=scope), log=ItemLog(), scope=scope)
        self._send(task.id, live, [ItemAdded(item=Item(id=f'{scope}prompt', kind='user', text=task.prompt))])
        return live

    def steered(self, task_id: str, text: str) -> None:
        """Show a message the user sent a running task, as the terminal echoes steering."""
        live = self._live.get(task_id)
        if live is not None:
            live.steered += 1
            item = Item(id=f'{live.scope}steer{live.steered}', kind='user', text=text)
            self._send(task_id, live, [ItemAdded(item=item)])

    def _send(self, task_id: str, live: _Live, messages: list[ItemMessage]) -> None:
        if messages:
            for message in messages:
                live.log.apply(message)
            self._publish(TaskItems(id=task_id, messages=messages))

    def items(self) -> dict[str, list[Item]]:
        """Each running task's transcript so far, for a browser that connects now."""
        return {task_id: list(live.log.items.values()) for task_id, live in self._live.items()}

    def reset(self) -> None:
        """Forget the tasks of a conversation the shell no longer holds."""
        self._sent.clear()
        self._live.clear()
