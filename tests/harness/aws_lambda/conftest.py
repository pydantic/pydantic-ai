"""A recording stand-in for `DurableContext`.

The real `DurableContext` talks to the durable-execution service, so tests drive a fake that
reproduces the two behaviours this capability depends on:

- a step result is checkpointed through the SDK's serializer, so a value that cannot be
  serialized fails here exactly as it would in production;
- a resumed execution serves a completed step from its checkpoint without running the body
  again, and a step whose body raised is recorded as failed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip('aws_durable_execution_sdk_python')

from aws_durable_execution_sdk_python.config import StepConfig
from aws_durable_execution_sdk_python.serdes import deserialize as _deserialize
from aws_durable_execution_sdk_python.serdes import serialize as _serialize

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


class StepContext:
    """Stand-in for the `StepContext` the SDK passes to a step body."""


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
        index = self._cursor
        self._cursor += 1

        if index < len(self.journal):
            recorded = self.journal[index]
            if recorded.name != name:
                raise AssertionError(
                    f'replay divergence at operation {index}: recorded {recorded.name!r}, got {name!r}'
                )
            self.operations.append(recorded)
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
