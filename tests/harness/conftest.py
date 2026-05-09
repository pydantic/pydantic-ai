from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# `dirty-equals` matchers are typed as `DirtyEquals[T]`, not `T`, so passing
# them where pydantic-ai expects concrete `str`/`datetime`/etc. fails pyright
# strict. Following pydantic-ai's own conftest, re-export with TYPE_CHECKING
# stubs that pretend the matchers return the concrete type. Tests should
# `from tests.conftest import IsStr, IsDatetime, ...` instead of importing
# from `dirty_equals` directly.
if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsPartialDict(*args: Any, **kwargs: Any) -> dict[Any, Any]: ...
else:
    from dirty_equals import IsDatetime, IsNow, IsPartialDict, IsStr

__all__ = ('IsDatetime', 'IsNow', 'IsPartialDict', 'IsStr')

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
