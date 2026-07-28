"""The old `pydantic_ai_harness.context` path still imports, with a `HarnessDeprecationWarning` to the new path."""

from __future__ import annotations

import sys

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.repo_context import RepoContext


def test_fresh_import_warns_and_aliases_old_path() -> None:
    sys.modules.pop('pydantic_ai_harness.context', None)
    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.repo_context`'):
        import pydantic_ai_harness.context

    assert pydantic_ai_harness.context.RepoContext is RepoContext
