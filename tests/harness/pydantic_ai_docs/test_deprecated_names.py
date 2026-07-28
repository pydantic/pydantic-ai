"""The old `pydantic_ai_harness.docs` path still imports, with a `HarnessDeprecationWarning` to the new path."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest
from pydantic_ai import Agent

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

    assert issubclass(shim.PyaiDocs, PydanticAIDocs)
    assert shim.PyaiDocsToolset is PydanticAIDocsToolset
    assert shim.PyaiDocsTopic is PydanticAIDocsTopic


def test_fresh_import_warns() -> None:
    sys.modules.pop('pydantic_ai_harness.docs', None)
    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.pydantic_ai_docs`'):
        import pydantic_ai_harness.docs

    assert issubclass(pydantic_ai_harness.docs.PyaiDocs, PydanticAIDocs)


class TestPreRenameSpecCompatibility:
    def test_serialization_names(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from pydantic_ai_harness.docs import PyaiDocs

        assert PyaiDocs.get_serialization_name() == 'PyaiDocs'
        assert PydanticAIDocs.get_serialization_name() == 'PydanticAIDocs'

    def test_old_spec_block_name_loads(self) -> None:
        # A spec saved before the rename uses the `PyaiDocs` block name; passing the
        # deprecated class through `custom_capability_types` must keep it loading.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from pydantic_ai_harness.docs import PyaiDocs

        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'PyaiDocs': {}}]},
            custom_capability_types=[PyaiDocs],
        )
        loaded = [c for c in agent.root_capability.capabilities if isinstance(c, PydanticAIDocs)]
        assert len(loaded) == 1

    def test_experimental_shim_exposes_the_same_class(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from pydantic_ai_harness.docs import PyaiDocs
            from pydantic_ai_harness.experimental.docs import PyaiDocs as experimental_pyai_docs

        assert experimental_pyai_docs is PyaiDocs
