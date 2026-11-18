"""Wait a model error's `retry_after` with a durable sleep before running a journaled model step again.

For durable engines that retry a whole task rather than one step, and journal steps and sleeps by name in the
order they are reached, like Absurd and pgtask. Their task retry waits the engine's own backoff, and their steps
take no retry policy, so a model step whose error asks for a wait records the wait as its result instead of
failing: the journal then holds the failed attempt, so a replay doesn't send the request again. The engine's
durable sleep waits it, suspending the task, and the step runs again under the same name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic_ai._utils import is_str_dict
from pydantic_ai.durable_exec import (
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    model_error_retry_after,
)

__all__ = ('MINIMUM_RETRY_AFTER_CAP', 'MODEL_OPERATION_IDS', 'step_waiting_retry_after')

MODEL_OPERATION_IDS = (ModelRequestId, ModelCompactMessagesId, ModelCancelSuspendedResponseId)
"""The operations that send a model request, whose error can ask to wait before the request is sent again."""

MINIMUM_RETRY_AFTER_CAP = 60.0
"""The cap on a model error's `retry_after`, in seconds, when the engine sets no longer one."""

_WAIT_KEY = '__pydantic_ai_harness_retry_after__'

Step = Callable[[str, Callable[[], Awaitable[object]]], Awaitable[object]]
"""Runs a body as a named step, returning its journaled result on a replay."""

Sleep = Callable[[str, float], Awaitable[None]]
"""Sleeps durably under a name for a number of seconds, suspending the task until then."""


async def step_waiting_retry_after(
    name: str,
    body: Callable[[], Awaitable[object]],
    *,
    step: Step,
    sleep: Sleep,
    max_waits: int | None,
    maximum: float,
) -> object:
    """Run `body` as the step `name`, and run it again after a durable sleep of each wait a model error asks for.

    A wait is capped at `maximum`. After `max_waits` waits, typically the task's remaining attempts, the step fails
    with the model error, into the engine's task retry, as it would without a wait. `None` allows any number.
    """
    waits = 0

    async def attempt() -> object:
        try:
            return await body()
        except Exception as error:
            if max_waits is not None and waits >= max_waits:
                raise
            if (delay := model_error_retry_after(error, maximum=maximum)) is None:
                raise
            return {_WAIT_KEY: delay}

    while True:
        result = await step(name, attempt)
        if not (is_str_dict(result) and isinstance(delay := result.get(_WAIT_KEY), int | float)):
            return result
        waits += 1
        await sleep(f'{name}.retry_after', delay)
