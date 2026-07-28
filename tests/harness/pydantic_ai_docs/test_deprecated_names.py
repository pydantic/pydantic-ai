"""The old `pydantic_ai_harness.docs` path still imports, with a `HarnessDeprecationWarning` to the new path."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs, PydanticAIDocsToolset, PydanticAIDocsTopic


def test_shim_warns_and_aliases_old_names() -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module('pydantic_ai_harness.docs')

    with pytest.warns(HarnessDeprecationWarning, match=r'`pydantic_ai_harness\.docs` has been renamed'):
        importlib.reload(shim)

    assert shim.PyaiDocs is PydanticAIDocs
    assert shim.PyaiDocsToolset is PydanticAIDocsToolset
    assert shim.PyaiDocsTopic is PydanticAIDocsTopic


def test_fresh_import_warns() -> None:
    sys.modules.pop('pydantic_ai_harness.docs', None)
    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.pydantic_ai_docs`'):
        import pydantic_ai_harness.docs

    assert pydantic_ai_harness.docs.PyaiDocs is PydanticAIDocs
