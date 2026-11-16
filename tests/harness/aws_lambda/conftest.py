"""A recording stand-in for `DurableContext`.

The real `DurableContext` talks to the durable-execution service, so tests drive a fake that
reproduces the two behaviours this capability depends on:

- a step result is checkpointed through the SDK's serializer, so a value that cannot be
  serialized fails here exactly as it would in production;
- a resumed execution serves a completed step from its checkpoint without running the body
  again, and a step whose body raised is recorded as failed;
- a callback's result suspends the execution until a test resolves the callback, and a resumed
  execution reads the resolution from the journal.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip('aws_durable_execution_sdk_python')

from aws_durable_execution_sdk_python.config import CallbackConfig, StepConfig
from aws_durable_execution_sdk_python.exceptions import CallbackError, SuspendExecution
from aws_durable_execution_sdk_python.serdes import deserialize as _deserialize, serialize as _serialize

# The SDK's helpers are generic over an unbound TypeVar when no SerDes is supplied.
deserialize: Callable[..., Any] = _deserialize
serialize: Callable[..., str] = _serialize

_ARN = 'arn:aws:lambda:us-east-1:000000000000:function:test'


@dataclass
class Operation:
    """One recorded durable operation."""

    name: str | None
    status: str
    payload: str | None = None
    error: BaseException | None = None
    config: StepConfig | None = None
    thread_id: int = 0
    callback_id: str | None = None
    callback_config: CallbackConfig | None = None


class StepContext:
    """Stand-in for the `StepContext` the SDK passes to a step body."""


class FakeCallback:
    """Stand-in for the SDK's `Callback`, reading its outcome from the recorded operation."""

    def __init__(self, operation: Operation) -> None:
        assert operation.callback_id is not None
        self.callback_id = operation.callback_id
        self._operation = operation

    def result(self) -> str | None:
        if self._operation.status == 'succeeded':
            return self._operation.payload
        if self._operation.error is not None:
            raise self._operation.error
        raise SuspendExecution('Callback result not received yet. Suspending execution while waiting for result.')


class FakeDurableContext:
    """Records the durable steps run against it.

    Pass `journal` (the `operations` of an earlier context) to replay: completed steps are
    served from their checkpoints and their bodies are not run.
    """

    def __init__(self, journal: list[Operation] | None = None) -> None:
        self.journal: list[Operation] = journal or []
        self.operations: list[Operation] = []
        self.invoked: list[str | None] = []
        self._cursor = 0

    def step(
        self,
        func: Callable[[Any], Any],
        name: str | None = None,
        config: StepConfig | None = None,
    ) -> Any:
        recorded = self._replay(name)
        if recorded is not None:
            if recorded.error is not None:
                raise recorded.error
            return self._load(recorded.payload)

        self.invoked.append(name)
        try:
            value = func(StepContext())
            payload = self._dump(value)
        except BaseException as exc:
            self.operations.append(Operation(name=name, status='failed', error=exc, config=config))
            raise
        self.operations.append(
            Operation(
                name=name,
                status='succeeded',
                payload=payload,
                config=config,
                thread_id=threading.get_ident(),
            )
        )
        return self._load(payload)

    def create_callback(self, name: str | None = None, config: CallbackConfig | None = None) -> FakeCallback:
        recorded = self._replay(name)
        if recorded is not None:
            return FakeCallback(recorded)

        operation = Operation(
            name=name, status='started', callback_id=f'callback-{len(self.operations)}', callback_config=config
        )
        self.operations.append(operation)
        return FakeCallback(operation)

    def _replay(self, name: str | None) -> Operation | None:
        """The journaled operation at the cursor, if this context is replaying one, after checking it is the same operation."""
        index = self._cursor
        self._cursor += 1
        if index >= len(self.journal):
            return None
        recorded = self.journal[index]
        if recorded.name != name:
            raise AssertionError(f'replay divergence at operation {index}: recorded {recorded.name!r}, got {name!r}')
        self.operations.append(recorded)
        return recorded

    def complete_callback(self, callback_id: str, result: str | None) -> None:
        """Resolve a callback the way `SendDurableExecutionCallbackSuccess` does."""
        operation = self._callback(callback_id)
        operation.status = 'succeeded'
        operation.payload = result

    def fail_callback(self, callback_id: str, message: str) -> None:
        """End a callback the way a timeout or `SendDurableExecutionCallbackFailure` does."""
        operation = self._callback(callback_id)
        operation.status = 'failed'
        operation.error = CallbackError(message)

    def _callback(self, callback_id: str) -> Operation:
        return next(operation for operation in self.operations if operation.callback_id == callback_id)

    @staticmethod
    def _dump(value: Any) -> str:
        return serialize(None, value, 'operation', _ARN)

    @staticmethod
    def _load(payload: str | None) -> Any:
        assert payload is not None
        return deserialize(None, payload, 'operation', _ARN)

    @property
    def step_names(self) -> list[str | None]:
        return [operation.name for operation in self.operations]

    @property
    def failed(self) -> list[Operation]:
        return [operation for operation in self.operations if operation.status == 'failed']
