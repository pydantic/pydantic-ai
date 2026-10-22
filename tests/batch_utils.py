"""A test model with a slow in-memory batch API, for the `BatchModelRequests` tests.

It lives outside the test modules so Temporal's workflow sandbox, which re-imports a test module, re-imports it like
any other helper, while activities keep using the instance the test module built.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import timedelta

import anyio
import anyio.lowlevel
from typing_extensions import TypedDict

from pydantic_ai.models.batch import Batch, BatchRequest, EncodedBatchRequest
from pydantic_ai.models.test import TestModel


class FastPolling(TypedDict):
    poll_interval: timedelta
    max_poll_interval: timedelta


class FastBatchSettings(FastPolling):
    max_wait: timedelta


FAST_POLLING: FastPolling = {'poll_interval': timedelta(milliseconds=1), 'max_poll_interval': timedelta(milliseconds=1)}
"""Tiny but nonzero poll intervals, so a test never waits out a real timer."""

FAST_BATCH_SETTINGS: FastBatchSettings = {**FAST_POLLING, 'max_wait': timedelta(milliseconds=1)}
"""`FAST_POLLING`, plus a tiny but nonzero `max_wait`, for `BatchModelRequests`."""

READINESS_WAIT_TIMEOUT = 10


class SlowBatchTestModel(TestModel):
    """Reports each batch as in progress for `pending_checks` status checks, and records every batch call.

    `on_batch_call` is called with the name of each batch method as it runs, so a test can check which durable
    boundary it ran in.
    """

    def __init__(self, pending_checks: int = 0, on_batch_call: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.pending_checks = pending_checks
        self.on_batch_call = on_batch_call
        self.encoded = 0
        self.created: list[Batch] = []
        self.status_checks = 0
        self.canceled: list[str] = []
        self._waiters: list[tuple[Callable[[], bool], anyio.Event]] = []

    def reset(self, pending_checks: int) -> None:
        """Clear recorded calls, for module-level reuse across tests."""
        self.pending_checks = pending_checks
        self.encoded = 0
        self.created = []
        self.status_checks = 0
        self.canceled = []

    async def wait_until(self, condition: Callable[[SlowBatchTestModel], bool]) -> None:
        """Wait until `condition` holds, checked after every batch call, failing the test if it never does."""
        event = anyio.Event()
        self._waiters.append((lambda: condition(self), event))
        self.notify_waiters()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await event.wait()

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        encoded = await super().batch_encode(request)
        # Counted on return: the request is queued before its task next yields, so a waiter woken now sees it queued.
        self.encoded += 1
        self.notify_waiters()
        return encoded

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        self._record('batch_create')
        batch = await super().batch_create(requests)
        self.created.append(batch)
        return replace(batch, status='in_progress') if self.pending_checks else batch

    async def batch_status(self, batch: Batch) -> Batch:
        self._record('batch_status')
        self.status_checks += 1
        self.notify_waiters()
        if self.status_checks <= self.pending_checks:
            return replace(batch, status='in_progress')
        return await super().batch_status(batch)

    async def batch_cancel(self, batch: Batch) -> Batch:
        # Several event loop iterations first, like a real request, so a cancellation that reaches the call stops it
        # before it's recorded.
        for _ in range(10):
            await anyio.lowlevel.checkpoint()
        self.canceled.append(batch.id)
        return await super().batch_cancel(batch)

    def notify_waiters(self) -> None:
        for condition, event in self._waiters:
            if condition():
                event.set()

    def _record(self, method: str) -> None:
        if self.on_batch_call is not None:
            self.on_batch_call(method)


UNRUN_MESSAGES = {
    'expired': r'The batch request was expired before the model ran it\.',
    'canceled': r'The batch was canceled before the model ran the request\.',
}
"""The error message of a request a batch ended without running, by its result status."""
