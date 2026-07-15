from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .abstract import AbstractCapability

RunCapabilityValidator = Callable[[AbstractCapability[Any]], None]

_run_capability_validators: ContextVar[tuple[RunCapabilityValidator, ...]] = ContextVar(
    '_run_capability_validators', default=()
)


@contextmanager
def run_capability_validation(validator: RunCapabilityValidator) -> Generator[None]:
    """Apply an integration policy after dynamic capabilities resolve for a run."""
    token = _run_capability_validators.set((*_run_capability_validators.get(), validator))
    try:
        yield
    finally:
        _run_capability_validators.reset(token)


def validate_run_capabilities(capabilities: Sequence[AbstractCapability[Any]]) -> None:
    for capability in capabilities:
        for validator in _run_capability_validators.get():
            validator(capability)
