from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

# `dirty-equals` matchers are typed as `DirtyEquals[T]`, not `T`, so passing
# them where pydantic-ai expects concrete `str`/`datetime`/etc. fails pyright
# strict. Following pydantic-ai's own conftest, re-export with TYPE_CHECKING
# stubs that pretend the matchers return the concrete type. Tests should
# `from tests.conftest import IsStr, IsDatetime, ...` instead of importing
# from `dirty_equals` directly.
if TYPE_CHECKING:
    MatcherT = TypeVar('MatcherT')

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsInstance(expected_type: type[MatcherT], **kwargs: Any) -> MatcherT: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsPartialDict(*args: Any, **kwargs: Any) -> dict[Any, Any]: ...
else:
    from dirty_equals import IsDatetime, IsInstance, IsNow, IsPartialDict, IsStr

__all__ = ('IsDatetime', 'IsInstance', 'IsNow', 'IsPartialDict', 'IsStr', 'agent_run_names')

# Prevent accidental real model requests during tests.
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def test_model() -> TestModel:
    """A fresh ``TestModel`` instance for each test."""
    return TestModel()


@pytest.fixture
def test_agent(test_model: TestModel) -> Agent[None, str]:
    """A minimal agent wired to ``TestModel`` for capability tests."""
    return Agent(test_model, name='test-agent')


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Convenience alias for ``tmp_path`` (useful for store / session tests)."""
    return tmp_path


@pytest.fixture
def allow_model_requests() -> Iterator[None]:
    """Temporarily allow real model requests within a test."""
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture
def instrument_all_agents() -> Iterator[None]:
    """Instrument every `Agent` for the test, including ones a capability builds internally.

    Per-agent `instrument=` does not reach agents a capability constructs on its own, so
    this is the only way to observe their run spans.
    """
    Agent.instrument_all(True)
    try:
        yield
    finally:
        Agent.instrument_all(False)


def agent_run_names(capfire: CaptureLogfire) -> list[str]:
    """The `agent_name` of every agent run span, in export order.

    Capabilities that build an internal `Agent` must name it, otherwise core infers a name from
    the caller's frame locals and Logfire groups the run under something like `self`.
    """
    return [
        str(span['attributes']['agent_name'])
        for span in capfire.exporter.exported_spans_as_dict()
        if 'agent_name' in span['attributes']
    ]
