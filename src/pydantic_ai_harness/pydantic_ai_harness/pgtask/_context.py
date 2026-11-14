from __future__ import annotations

import importlib
import re
import weakref
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from pgtask import Task, get_current_task

ENGINE_NAME = 'pgtask'

# pgtask suspends a sleeping task by raising this from `Task.sleep_for`, but neither exports it nor declares it in its
# type stub, so it's looked up in the native module that defines it.
try:
    TaskSuspended: type[Exception] = vars(importlib.import_module('pgtask._native'))['TaskSuspended']
# Only reached on a pgtask version whose internals moved, which the supported range doesn't include.
except (ImportError, KeyError) as _error:  # pragma: no cover
    raise ImportError(
        "`PGTaskDurability` needs pgtask's internal `pgtask._native.TaskSuspended` error, which the installed pgtask "
        'does not define. Install a pgtask version supported by `pydantic-ai-harness[pgtask]`.'
    ) from _error

StepT = TypeVar('StepT')

_UNSUPPORTED_STEP_NAME_CHAR = re.compile(r'[^A-Za-z0-9._:-]')


@dataclass
class TaskSteps:
    """Runs steps for one attempt of a pgtask task, numbering repeated step names in encounter order.

    pgtask takes an explicit `occurrence` for a step name that recurs instead of counting encounters
    itself. Numbering them here maps each step of a replayed attempt to the same checkpoint, as long
    as the steps are reached in the same order.
    """

    task: Task
    occurrences: Counter[str]

    async def step(self, name: str, operation: Callable[[], Awaitable[StepT]]) -> StepT:
        name, occurrence = self._next(name)
        return await self.task.step(name, operation, occurrence=occurrence)

    async def sleep(self, name: str, seconds: float) -> None:
        """Sleep durably: pgtask suspends the task until the wake time, and a replayed sleep that's over returns."""
        name, occurrence = self._next(name)
        # pgtask rejects a negative delay.
        await self.task.sleep_for(name, max(seconds, 0), occurrence=occurrence)

    def _next(self, name: str) -> tuple[str, int]:
        """The name pgtask accepts for `name`, and the occurrence of it this call is."""
        # Step names are built from the agent, toolset, and tool names, which can hold characters
        # pgtask rejects (the agent's own toolset has the id `<agent>`).
        name = _UNSUPPORTED_STEP_NAME_CHAR.sub('_', name)
        occurrence = self.occurrences[name]
        self.occurrences[name] += 1
        return name, occurrence


# The counters are shared by every step of an attempt, wherever it runs, so they are keyed by the task
# object rather than held in a `ContextVar`: one set in a spawned subtask would not be seen by its
# siblings. pgtask makes one `Task` per attempt, so a retry starts from zero. A `Task` holds dicts and
# so is unhashable, which rules out a `WeakKeyDictionary`; the entry is dropped when the task is.
_occurrences: dict[int, Counter[str]] = {}


def current_task_steps() -> TaskSteps | None:
    """Return the steps of the pgtask task running in this call chain, or `None` outside a handler."""
    task = get_current_task()
    if task is None:
        return None
    key = id(task)
    occurrences = _occurrences.get(key)
    if occurrences is None:
        occurrences = _occurrences[key] = Counter[str]()
        weakref.finalize(task, _occurrences.pop, key, None)
    return TaskSteps(task, occurrences)
