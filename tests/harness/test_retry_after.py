"""`step_waiting_retry_after`, against a journal that records steps and sleeps the way Absurd and pgtask do."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from pydantic_ai.exceptions import ModelHTTPError, ModelRateLimitError
from pydantic_ai_harness._retry_after import step_waiting_retry_after

_NAME = 'agent__model.request'


def _rate_limited(retry_after: float | None) -> ModelHTTPError:
    return ModelHTTPError.for_category(
        ModelRateLimitError, status_code=429, model_name='test-model', retry_after=retry_after
    )


class _Journal:
    """Records each step's result and each sleep by name, in order; a step that raises records nothing.

    Pass the `entries` of an earlier journal to replay them: a recorded step returns its result without running,
    and a recorded sleep doesn't wait again.
    """

    def __init__(self, recorded: list[tuple[str, object]] | None = None) -> None:
        self._recorded = list(recorded or [])
        self.entries: list[tuple[str, object]] = []
        self.ran: list[str] = []
        self.slept: list[float] = []

    def _replay(self, name: str) -> tuple[bool, object]:
        if len(self.entries) < len(self._recorded):
            recorded_name, value = self._recorded[len(self.entries)]
            assert recorded_name == name
            self.entries.append((name, value))
            return True, value
        return False, None

    async def step(self, name: str, body: Callable[[], Awaitable[object]]) -> object:
        replayed, value = self._replay(name)
        if replayed:
            return value
        self.ran.append(name)
        value = await body()
        self.entries.append((name, value))
        return value

    async def sleep(self, name: str, seconds: float) -> None:
        replayed, _ = self._replay(name)
        if not replayed:
            self.slept.append(seconds)
            self.entries.append((name, seconds))


def _model(*errors: Exception) -> tuple[Callable[[], Awaitable[object]], list[int]]:
    """A step body that raises `errors` in turn, then answers."""
    calls: list[int] = []

    async def body() -> object:
        calls.append(len(calls))
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return {'parts': ['answered']}

    return body, calls


async def test_waits_durably_and_runs_the_step_again() -> None:
    """The failed attempt is recorded as the step's result, so a replay neither waits nor sends it again."""
    body, calls = _model(_rate_limited(5.0))
    journal = _Journal()

    result = await step_waiting_retry_after(
        _NAME, body, step=journal.step, sleep=journal.sleep, max_waits=None, maximum=60.0
    )

    assert result == {'parts': ['answered']}
    assert journal.slept == [5.0]
    assert journal.entries == [
        (_NAME, {'__pydantic_ai_harness_retry_after__': 5.0}),
        (f'{_NAME}.retry_after', 5.0),
        (_NAME, {'parts': ['answered']}),
    ]

    replay = _Journal(journal.entries)
    result = await step_waiting_retry_after(
        _NAME, body, step=replay.step, sleep=replay.sleep, max_waits=None, maximum=60.0
    )
    assert result == {'parts': ['answered']}
    assert (replay.ran, replay.slept, len(calls)) == ([], [], 2)


async def test_caps_the_wait() -> None:
    body, _ = _model(_rate_limited(3600.0))
    journal = _Journal()

    await step_waiting_retry_after(_NAME, body, step=journal.step, sleep=journal.sleep, max_waits=None, maximum=60.0)

    assert journal.slept == [60.0]


async def test_fails_into_the_task_retry_after_max_waits() -> None:
    """Once the task's remaining attempts are used up by waits, the step fails as it would without one."""
    body, calls = _model(_rate_limited(5.0), _rate_limited(5.0))
    journal = _Journal()

    with pytest.raises(ModelRateLimitError):
        await step_waiting_retry_after(_NAME, body, step=journal.step, sleep=journal.sleep, max_waits=1, maximum=60.0)

    assert (journal.slept, len(calls)) == ([5.0], 2)


async def test_an_error_without_retry_after_fails_the_step() -> None:
    body, calls = _model(_rate_limited(None))
    journal = _Journal()

    with pytest.raises(ModelRateLimitError):
        await step_waiting_retry_after(_NAME, body, step=journal.step, sleep=journal.sleep, max_waits=3, maximum=60.0)

    assert (journal.entries, len(calls)) == ([], 1)
