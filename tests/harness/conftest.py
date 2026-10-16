from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

import pydantic_ai.models
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

# `dirty-equals` matchers are typed as `DirtyEquals[T]`, not `T`, so passing
# them where pydantic-ai expects concrete `str`/`datetime`/etc. fails pyright
# strict. Following pydantic-ai's own conftest, re-export with TYPE_CHECKING
# stubs that pretend the matchers return the concrete type. Tests should
# `from tests.harness.conftest import IsStr, IsDatetime, ...` instead of importing
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

__all__ = (
    'IsDatetime',
    'IsInstance',
    'IsNow',
    'IsPartialDict',
    'IsStr',
    'agent_run_names',
)

# Temporal markers live in `tests/harness/_temporal.py`: modules defining workflows must not import
# this file, because the workflow sandbox would re-import it along with `dirty_equals`.

# Prevent accidental real model requests during tests.
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(scope='session')
def anyio_backend() -> str:
    """The harness suite is asyncio-only: capabilities lean on `asyncio.create_task`, Temporal, Monty and
    other asyncio-native pieces, so `--anyio-backend=trio` does not apply here.

    Session-scoped like the root override so module-scoped async fixtures (`temporal_env`, Modal `session`)
    can depend on it."""
    return 'asyncio'


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Not yet: the suite predates the detector, and inside a Temporal workflow it turns Code Mode's portal
    startup failure into a hang. https://github.com/pydantic/pydantic-ai/issues/8821"""
    return False


@pytest.fixture
def test_model() -> TestModel:
    """A fresh `TestModel` instance for each test."""
    return TestModel()


@pytest.fixture
def test_agent(test_model: TestModel) -> Agent[None, str]:
    """A minimal agent wired to `TestModel` for capability tests."""
    return Agent(test_model, name='test-agent')


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Convenience alias for `tmp_path` (useful for store / session tests)."""
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


@pytest.fixture(scope='session')
def session_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One loop for every sync test that calls `run_sync`, closed when the session ends."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def current_event_loop_for_sync_tests(
    request: pytest.FixtureRequest, session_event_loop: asyncio.AbstractEventLoop
) -> Iterator[None]:
    """Give sync tests a current event loop.

    anyio's runner unsets the current loop after each async test. Without one, `Agent.run_sync`
    creates a new loop per call and never closes it, and the leak surfaces as an unraisable
    `ResourceWarning` in an unrelated later test. Async tests are left to anyio's own runner.
    """
    if not inspect.iscoroutinefunction(request.function):
        asyncio.set_event_loop(session_event_loop)
    yield
